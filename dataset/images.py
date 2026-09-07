from __future__ import annotations

import numpy as np
import tensorflow as tf


def _clahe_np_uint8(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    import cv2

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.astype(np.uint8).reshape(arr.shape[0], arr.shape[1])
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid), int(tile_grid)),
    )
    eq = clahe.apply(gray)
    return np.repeat(eq[..., None], 3, axis=-1).astype(np.float32)


class ImageDecoder:
    """Decodifica JPEG/PNG y aplica CLAHE sobre luminancia."""

    @staticmethod
    def _decode_fallback(raw: tf.Tensor) -> tf.Tensor:
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return img

    @classmethod
    def decode(cls, path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        lower = tf.strings.lower(path)
        is_jpeg = tf.strings.regex_full_match(lower, ".*\\.jpe?g")
        is_png = tf.strings.regex_full_match(lower, ".*\\.png")

        def _jpeg():
            img = tf.image.decode_jpeg(raw, channels=3, dct_method="INTEGER_FAST")
            img.set_shape([None, None, 3])
            return img

        def _png():
            img = tf.image.decode_png(raw, channels=3)
            img.set_shape([None, None, 3])
            return img

        img = tf.cond(
            is_jpeg,
            _jpeg,
            lambda: tf.cond(is_png, _png, lambda: cls._decode_fallback(raw)),
        )
        # Los decoders devuelven uint8; float16 representa exactamente todos los valores 0..255.
        return tf.cast(img, tf.float16)

    @staticmethod
    def apply_clahe(img: tf.Tensor, clip_limit: float = 2.0, tile_grid: int = 8) -> tf.Tensor:
        """CLAHE sobre luminancia; salida RGB float32 en [0, 255]."""
        img_u8 = tf.cast(tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 255.0), tf.uint8)
        out = tf.numpy_function(
            func=lambda x: _clahe_np_uint8(x, clip_limit, tile_grid),
            inp=[img_u8],
            Tout=tf.float32,
        )
        out.set_shape([None, None, 3])
        return out
