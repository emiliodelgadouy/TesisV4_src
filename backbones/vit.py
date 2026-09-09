from __future__ import annotations

from typing import override

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import Backbone, DEFAULT_WEIGHTS, mode_batch_sizes

# AugReg ViT-S/16 (Steiner et al.): ImageNet-21k + fine-tune ImageNet-1k a 224.
_VITS16_WEIGHTS_URL = "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
_VITS16_WEIGHTS_MD5 = "63b5673b77637d178c870b62b0b186b6"
_VITS16_WEIGHTS_FNAME = "vits16_augreg_i21k_i1k_224.npz"


def vit_preprocess_input(x):
    # AugReg usa value_range(-1, 1) sobre imagenes [0, 255].
    x = tf.cast(x, tf.float32)
    return (x / 127.5) - 1.0


def _interpolate_posemb(posemb: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Reescala el grid de pos-embeddings (cls + parches) al tamaño de entrada."""
    n_tokens = grid_h * grid_w + 1
    if posemb.shape[1] == n_tokens:
        return posemb
    token, grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(round(np.sqrt(len(grid))))
    if gs_old * gs_old != len(grid):
        raise ValueError(f"posemb con {len(grid)} parches no forma una grilla cuadrada")
    dim = grid.shape[-1]
    grid = tf.image.resize(grid.reshape(1, gs_old, gs_old, dim), (grid_h, grid_w), method="bilinear")
    grid = np.asarray(grid).reshape(1, grid_h * grid_w, dim)
    return np.concatenate([token, grid], axis=1).astype(posemb.dtype, copy=False)


class ViTEmbeddings(layers.Layer):
    """Concatena el token CLS y suma los positional embeddings."""

    def build(self, input_shape):
        hidden = int(input_shape[-1])
        n_patches = int(input_shape[1])
        self.cls = self.add_weight(shape=(1, 1, hidden), initializer="zeros", trainable=True, name="cls")
        self.posemb = self.add_weight(shape=(1, n_patches + 1, hidden), initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True, name="posemb")
        super().build(input_shape)

    def call(self, patch_tokens):
        batch = tf.shape(patch_tokens)[0]
        cls = tf.tile(self.cls, [batch, 1, 1])
        return tf.concat([cls, patch_tokens], axis=1) + self.posemb


class TransformerBlock(layers.Layer):
    """Bloque pre-norm de ViT: LN + MHSA residual, LN + MLP residual."""

    def __init__(self, hidden_size, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} no es divisible por num_heads={num_heads}")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads, name="attn")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp_dense1 = layers.Dense(mlp_dim, name="mlp_dense1")
        self.mlp_dense2 = layers.Dense(hidden_size, name="mlp_dense2")

    def call(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y)
        y = self.norm2(x)
        y = self.mlp_dense1(y)
        y = keras.activations.gelu(y, approximate=True)
        y = self.mlp_dense2(y)
        return x + y


class PatchTokensToGrid(layers.Layer):
    """Tira el CLS y rearma los parches como mapa espacial [B, Gh, Gw, D] para GAP."""

    def __init__(self, grid_h, grid_w, **kwargs):
        super().__init__(**kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)

    def build(self, input_shape):
        n_patches = int(input_shape[1]) - 1
        expected = self.grid_h * self.grid_w
        if n_patches != expected:
            raise ValueError(f"se esperaban {expected} parches, hay {n_patches}")
        self._hidden = int(input_shape[-1])
        super().build(input_shape)

    def call(self, tokens):
        patches = tokens[:, 1:, :]
        return tf.reshape(patches, (-1, self.grid_h, self.grid_w, self._hidden))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.grid_h, self.grid_w, input_shape[-1])


def _download_vits16_weights() -> str:
    return keras.utils.get_file(
        fname=_VITS16_WEIGHTS_FNAME,
        origin=_VITS16_WEIGHTS_URL,
        file_hash=_VITS16_WEIGHTS_MD5,
        cache_subdir="vit",
    )


def _load_augreg_weights(model: keras.Model, npz_path: str, *, grid_h: int, grid_w: int, num_layers: int) -> None:
    with np.load(npz_path) as data:
        params = {key: data[key].copy() for key in data.files}

    model.get_layer("patch_embed").set_weights([params["embedding/kernel"], params["embedding/bias"]])
    posemb = _interpolate_posemb(params["Transformer/posembed_input/pos_embedding"], grid_h, grid_w)
    model.get_layer("embeddings").set_weights([params["cls"], posemb])

    for index in range(num_layers):
        block = model.get_layer(f"encoderblock_{index}")
        prefix = f"Transformer/encoderblock_{index}"
        attn_prefix = f"{prefix}/MultiHeadDotProductAttention_1"
        block.norm1.set_weights([params[f"{prefix}/LayerNorm_0/scale"], params[f"{prefix}/LayerNorm_0/bias"]])
        block.attn.set_weights(
            [
                params[f"{attn_prefix}/query/kernel"],
                params[f"{attn_prefix}/query/bias"],
                params[f"{attn_prefix}/key/kernel"],
                params[f"{attn_prefix}/key/bias"],
                params[f"{attn_prefix}/value/kernel"],
                params[f"{attn_prefix}/value/bias"],
                params[f"{attn_prefix}/out/kernel"],
                params[f"{attn_prefix}/out/bias"],
            ]
        )
        block.norm2.set_weights([params[f"{prefix}/LayerNorm_2/scale"], params[f"{prefix}/LayerNorm_2/bias"]])
        block.mlp_dense1.set_weights([params[f"{prefix}/MlpBlock_3/Dense_0/kernel"], params[f"{prefix}/MlpBlock_3/Dense_0/bias"]])
        block.mlp_dense2.set_weights([params[f"{prefix}/MlpBlock_3/Dense_1/kernel"], params[f"{prefix}/MlpBlock_3/Dense_1/bias"]])

    model.get_layer("encoder_norm").set_weights([params["Transformer/encoder_norm/scale"], params["Transformer/encoder_norm/bias"]])


def build_vit_s16(input_shape: tuple[int, int, int], *, weights: str | None) -> keras.Model:
    height, width, channels = input_shape
    patch_size = 16
    hidden_size = 384
    num_layers = 12
    num_heads = 6
    mlp_dim = 1536
    if channels != 3:
        raise ValueError(f"ViT-S/16 espera 3 canales, recibio {channels}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"ViT-S/16 requiere alto/ancho divisibles por {patch_size}; recibio {height}x{width}")

    grid_h, grid_w = height // patch_size, width // patch_size
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(hidden_size, patch_size, strides=patch_size, padding="valid", name="patch_embed")(inputs)
    x = layers.Reshape((grid_h * grid_w, hidden_size), name="flatten_patches")(x)
    x = ViTEmbeddings(name="embeddings")(x)
    for index in range(num_layers):
        x = TransformerBlock(hidden_size, num_heads, mlp_dim, name=f"encoderblock_{index}")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
    x = PatchTokensToGrid(grid_h, grid_w, name="patch_grid")(x)
    model = keras.Model(inputs, x, name="vits16")
    if weights == "imagenet":
        _load_augreg_weights(model, _download_vits16_weights(), grid_h=grid_h, grid_w=grid_w, num_layers=num_layers)
    elif weights is not None:
        raise ValueError("vits16 solo admite weights='imagenet' o None")
    return model


class ViTS16Backbone(Backbone):
    """ViT-S/16 (224) con pesos AugReg. Devuelve un mapa [B, 14, 14, 384] compatible con GAP/ABMIL."""

    key = "vits16"
    input_size = (224, 224)
    default_weights = "imagenet"
    preprocess_fn = staticmethod(vit_preprocess_input)
    # RESIZED a 672: atencion cuadratica (~1764 tokens). ABMIL tiles a 224.
    batch_memory_scale = "attention"
    batch_size = mode_batch_sizes(standard=128, resized=8, abmil=32)

    @override
    def preprocess_input(self, x):
        return vit_preprocess_input(x)

    @override
    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        if include_top:
            raise ValueError("vits16 no implementa include_top=True; se usa como extractor espacial")
        del kwargs
        shape = self.input_shape_or_default(input_shape)
        return build_vit_s16(shape, weights=self.coalesce_weights(weights))
