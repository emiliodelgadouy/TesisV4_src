from __future__ import annotations

import tensorflow as tf


class PositiveMixup:
    """MixUp solo entre muestras positivas del batch (Beta(alpha, alpha), prob p)."""

    def __init__(self, *, alpha: float = 0.1, probability: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.probability = float(probability)

    @staticmethod
    def _labels_positive_mask(labels: tf.Tensor) -> tf.Tensor:
        """Mascara de positivos: binaria (>=0.5) o multilabel (cualquier canal >=0.5).

        Asume primer eje = batch (como en ``dataset.batch``).
        """
        labels = tf.cast(labels, tf.float32)
        batch_size = tf.shape(labels)[0]
        flat = tf.reshape(labels, [batch_size, -1])
        return tf.reduce_any(flat >= 0.5, axis=1)

    @staticmethod
    def _expand_lambda(lam: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Expande lambda [B] al rank de `target` (imagen o etiqueta)."""
        lam = tf.reshape(lam, [-1])
        target_rank = tf.rank(target)
        broadcast_shape = tf.concat(
            [[tf.shape(lam)[0]], tf.ones([target_rank - 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.reshape(lam, broadcast_shape)

    def apply_batch(self, images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Pensado para ``tf.data`` despues del ``map`` con flip/crop.

        Sin ``tf.cond`` (compatible con Autograph en ``tf.data.map``).
        """
        images_f32 = tf.cast(images, tf.float32)
        labels_f32 = tf.cast(labels, tf.float32)
        batch_size = tf.shape(images_f32)[0]
        pos_mask = self._labels_positive_mask(labels_f32)
        n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
        mix_gate = tf.cast(
            tf.logical_and(
                tf.less(tf.random.uniform([], dtype=tf.float32), self.probability),
                tf.greater_equal(n_pos, 2),
            ),
            tf.float32,
        )

        pos_indices = tf.boolean_mask(tf.range(batch_size), pos_mask)
        n = tf.shape(pos_indices)[0]
        order = tf.random.shuffle(tf.range(n))
        partner_order = tf.math.floormod(order + 1, tf.maximum(n, 1))
        partners = tf.gather(pos_indices, partner_order)
        partner_at = tf.tensor_scatter_nd_update(
            tf.range(batch_size),
            tf.expand_dims(pos_indices, axis=1),
            partners,
        )

        lam_a = tf.random.gamma([batch_size], self.alpha, beta=1.0, dtype=tf.float32)
        lam_b = tf.random.gamma([batch_size], self.alpha, beta=1.0, dtype=tf.float32)
        lam = lam_a / (lam_a + lam_b + 1e-8)

        partner_images = tf.gather(images_f32, partner_at)
        partner_labels = tf.gather(labels_f32, partner_at)

        lam_img = self._expand_lambda(lam, images_f32)
        lam_lbl = self._expand_lambda(lam, labels_f32)
        mixed_images = lam_img * images_f32 + (1.0 - lam_img) * partner_images
        mixed_labels = lam_lbl * labels_f32 + (1.0 - lam_lbl) * partner_labels

        pos_mask_f = tf.cast(pos_mask, tf.float32)
        pos_mask_img = self._expand_lambda(pos_mask_f, images_f32)
        pos_mask_lbl = self._expand_lambda(pos_mask_f, labels_f32)
        mixed_images = pos_mask_img * mixed_images + (1.0 - pos_mask_img) * images_f32
        mixed_labels = pos_mask_lbl * mixed_labels + (1.0 - pos_mask_lbl) * labels_f32

        out_images = (1.0 - mix_gate) * images_f32 + mix_gate * mixed_images
        out_labels = (1.0 - mix_gate) * labels_f32 + mix_gate * mixed_labels
        return tf.cast(out_images, images.dtype), out_labels

    @classmethod
    def apply(
        cls,
        images: tf.Tensor,
        labels: tf.Tensor,
        *,
        alpha: float = 0.1,
        probability: float = 0.5,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return cls(alpha=alpha, probability=probability).apply_batch(images, labels)
