from __future__ import annotations

import json
import struct
from typing import override

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import Backbone, DEFAULT_WEIGHTS, mode_batch_sizes

_SWINT_WEIGHTS_URL = "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/model.safetensors"
_SWINT_WEIGHTS_MD5 = "06f626c9949735d52934bfdfe70d4e6c"
_SWINT_WEIGHTS_FNAME = "swin_tiny_patch4_window7_224.safetensors"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEPTHS = (2, 2, 6, 2)
_NUM_HEADS = (3, 6, 12, 24)
_EMBED_DIM = 96
_PATCH_SIZE = 4
_WINDOW_SIZE = 7
_MLP_RATIO = 4.0
_DROP_PATH_RATE = 0.1
_LN_EPS = 1e-5


def swin_preprocess_input(x):
    # ImageNet mean/std, igual que el processor de HuggingFace.
    x = tf.cast(x, tf.float32) / 255.0
    mean = tf.constant(_IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(_IMAGENET_STD, dtype=tf.float32)
    return (x - mean) / std


def _relative_position_index(window_size: int) -> np.ndarray:
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = np.transpose(relative_coords, (1, 2, 0))
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1).astype(np.int32)


def _shifted_window_mask(height: int, width: int, window_size: int, shift_size: int) -> np.ndarray | None:
    if shift_size <= 0:
        return None
    h_idx = np.arange(height)
    w_idx = np.arange(width)
    h_region = (h_idx >= height - window_size).astype(np.int32) + (h_idx >= height - shift_size).astype(np.int32)
    w_region = (w_idx >= width - window_size).astype(np.int32) + (w_idx >= width - shift_size).astype(np.int32)
    img_mask = (h_region[None, :, None, None] * 3 + w_region[None, None, :, None]).astype(np.float32)
    windows = img_mask.reshape(1, height // window_size, window_size, width // window_size, window_size, 1)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
    attn_mask = windows[:, None, :] - windows[:, :, None]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return attn_mask


def _window_partition(x, window_size: int):
    height, width, channels = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    x = tf.reshape(x, [-1, height // window_size, window_size, width // window_size, window_size, channels])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, window_size, window_size, channels])


def _window_reverse(windows, window_size: int, height: int, width: int):
    channels = int(windows.shape[-1])
    x = tf.reshape(windows, [-1, height // window_size, width // window_size, window_size, window_size, channels])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, height, width, channels])


def _drop_path(x, drop_prob, training):
    if (not training) or drop_prob == 0.0:
        return x
    keep = 1.0 - drop_prob
    shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
    random_tensor = keep + tf.random.uniform(shape, dtype=x.dtype)
    return x / keep * tf.floor(random_tensor)


def _linear(dense, tokens):
    """Dense como matmul 2D + bias broadcast.

    Keras Dense sobre (B, H, W, C) usa BiasAdd NHWC; en GPU el BiasAddGrad
    (kernel int32) pega CUDA_ERROR_ILLEGAL_ADDRESS al descongelar Swin.
    """
    in_dim = int(dense.kernel.shape[0])
    out_dim = int(dense.kernel.shape[1])
    lead = tf.shape(tokens)[:-1]
    flat = tf.reshape(tokens, [-1, in_dim])
    out = tf.matmul(flat, tf.cast(dense.kernel, tokens.dtype))
    if dense.bias is not None:
        out = out + tf.cast(dense.bias, tokens.dtype)
    return tf.reshape(out, tf.concat([lead, [out_dim]], axis=0))


class WindowAttention(layers.Layer):
    """W-MSA con relative position bias (tabla fija al window_size, no al canvas).

    Siempre float32. Las Dense no se llaman como capa: BiasAddGrad sobre (B,H,W,C)
    dispara CUDA_ERROR_ILLEGAL_ADDRESS al descongelar (etapa 2).
    """

    def __init__(self, dim, num_heads, window_size, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} no es divisible por num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = layers.Dense(dim, name="query", dtype="float32")
        self.key = layers.Dense(dim, name="key", dtype="float32")
        self.value = layers.Dense(dim, name="value", dtype="float32")
        self.proj = layers.Dense(dim, name="proj", dtype="float32")

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size - 1) ** 2, self.num_heads),
            initializer="zeros",
            trainable=True,
            dtype="float32",
            name="relative_position_bias_table",
        )
        # Numpy aca (no tf.constant en build): Keras construye las capas en un
        # scratch graph y ese Const queda "out of scope" al entrenar.
        self._relative_position_index = _relative_position_index(self.window_size).reshape(-1)
        in_shape = (None, self.dim)
        self.query.build(in_shape)
        self.key.build(in_shape)
        self.value.build(in_shape)
        self.proj.build(in_shape)
        super().build(input_shape)

    def _heads(self, x):
        seq = self.window_size * self.window_size
        x = tf.reshape(x, [-1, seq, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, attn_mask=None):
        x = tf.cast(x, tf.float32)
        seq = self.window_size * self.window_size
        query = self._heads(_linear(self.query, x))
        key = self._heads(_linear(self.key, x))
        value = self._heads(_linear(self.value, x))
        attn = tf.matmul(query, key, transpose_b=True) * tf.cast(self.scale, query.dtype)
        index = tf.constant(self._relative_position_index, dtype=tf.int32)
        bias = tf.gather(self.relative_position_bias_table, index)
        bias = tf.reshape(bias, [seq, seq, self.num_heads])
        bias = tf.transpose(bias, [2, 0, 1])
        attn = attn + bias
        if attn_mask is not None:
            num_windows = int(attn_mask.shape[0])
            attn = tf.reshape(attn, [-1, num_windows, self.num_heads, seq, seq])
            attn = attn + tf.cast(attn_mask[None, :, None, :, :], attn.dtype)
            attn = tf.reshape(attn, [-1, self.num_heads, seq, seq])
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, value)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, seq, self.dim])
        return _linear(self.proj, out)


class SwinBlock(layers.Layer):
    """Bloque pre-norm: W-MSA o SW-MSA + MLP, con drop-path."""

    def __init__(self, dim, num_heads, window_size, shift_size, drop_path_rate, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = int(shift_size)
        self.drop_path_rate = float(drop_path_rate)
        self.norm1 = layers.LayerNormalization(epsilon=_LN_EPS, name="norm1", dtype="float32")
        self.attn = WindowAttention(dim, num_heads, window_size, name="attn")
        self.norm2 = layers.LayerNormalization(epsilon=_LN_EPS, name="norm2", dtype="float32")
        self.mlp_fc1 = layers.Dense(int(dim * _MLP_RATIO), name="mlp_fc1", dtype="float32")
        self.mlp_fc2 = layers.Dense(dim, name="mlp_fc2", dtype="float32")

    def build(self, input_shape):
        height, width = int(input_shape[1]), int(input_shape[2])
        shift = 0 if min(height, width) <= self.window_size else self.shift_size
        self._shift = shift
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        self._pad_h = pad_h
        self._pad_w = pad_w
        self._height = height
        self._width = width
        padded_h, padded_w = height + pad_h, width + pad_w
        mask = _shifted_window_mask(padded_h, padded_w, self.window_size, shift)
        self._attn_mask = mask
        self.norm1.build((None, None, None, self.dim))
        self.attn.build((None, self.window_size * self.window_size, self.dim))
        self.norm2.build((None, None, None, self.dim))
        self.mlp_fc1.build((None, self.dim))
        self.mlp_fc2.build((None, int(self.dim * _MLP_RATIO)))
        super().build(input_shape)

    def _window_attention(self, x):
        if self._pad_h or self._pad_w:
            x = tf.pad(x, [[0, 0], [0, self._pad_h], [0, self._pad_w], [0, 0]])
        padded_h = self._height + self._pad_h
        padded_w = self._width + self._pad_w
        if self._shift:
            x = tf.roll(x, shift=[-self._shift, -self._shift], axis=[1, 2])
        windows = _window_partition(x, self.window_size)
        windows = tf.reshape(windows, [-1, self.window_size * self.window_size, self.dim])
        attn_mask = None if self._attn_mask is None else tf.constant(self._attn_mask, dtype=tf.float32)
        windows = self.attn(windows, attn_mask=attn_mask)
        windows = tf.reshape(windows, [-1, self.window_size, self.window_size, self.dim])
        x = _window_reverse(windows, self.window_size, padded_h, padded_w)
        if self._shift:
            x = tf.roll(x, shift=[self._shift, self._shift], axis=[1, 2])
        if self._pad_h or self._pad_w:
            x = x[:, : self._height, : self._width, :]
        return x

    def call(self, x, training=False):
        x = tf.cast(x, tf.float32)
        shortcut = x
        x = shortcut + _drop_path(self._window_attention(self.norm1(x)), self.drop_path_rate, training)
        tokens = tf.reshape(self.norm2(x), [-1, self.dim])
        tokens = keras.activations.gelu(_linear(self.mlp_fc1, tokens), approximate=False)
        mlp = tf.reshape(_linear(self.mlp_fc2, tokens), [-1, self._height, self._width, self.dim])
        return x + _drop_path(mlp, self.drop_path_rate, training)


class PatchMerging(layers.Layer):
    """Une parches 2x2 y proyecta 4C → 2C (downsample espacial)."""

    def __init__(self, dim, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=_LN_EPS, name="norm", dtype="float32")
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction", dtype="float32")

    def build(self, input_shape):
        self.norm.build((None, None, None, 4 * self.dim))
        self.reduction.build((None, 4 * self.dim))
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        merged = self.norm(tf.concat([x0, x1, x2, x3], axis=-1))
        return _linear(self.reduction, merged)


def _load_safetensors(path: str) -> dict[str, np.ndarray]:
    dtypes = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        payload = handle.read()
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = dtypes[meta["dtype"]]
        start, end = meta["data_offsets"]
        tensors[name] = np.frombuffer(payload[start:end], dtype=dtype).reshape(meta["shape"]).copy()
    return tensors


def _dense_weights(weight: np.ndarray, bias: np.ndarray | None = None) -> list[np.ndarray]:
    kernel = np.transpose(weight, (1, 0))
    return [kernel, bias] if bias is not None else [kernel]


def _conv_weights(weight: np.ndarray, bias: np.ndarray) -> list[np.ndarray]:
    return [np.transpose(weight, (2, 3, 1, 0)), bias]


def _download_swint_weights() -> str:
    return keras.utils.get_file(
        fname=_SWINT_WEIGHTS_FNAME,
        origin=_SWINT_WEIGHTS_URL,
        file_hash=_SWINT_WEIGHTS_MD5,
        cache_subdir="swin",
    )


def _load_hf_weights(model: keras.Model, path: str) -> None:
    tensors = _load_safetensors(path)
    model.get_layer("patch_embed").set_weights(
        _conv_weights(
            tensors["swin.embeddings.patch_embeddings.projection.weight"],
            tensors["swin.embeddings.patch_embeddings.projection.bias"],
        )
    )
    model.get_layer("patch_norm").set_weights([tensors["swin.embeddings.norm.weight"], tensors["swin.embeddings.norm.bias"]])
    for stage, depth in enumerate(_DEPTHS):
        for index in range(depth):
            block = model.get_layer(f"stage{stage}_block{index}")
            prefix = f"swin.encoder.layers.{stage}.blocks.{index}"
            block.norm1.set_weights([tensors[f"{prefix}.layernorm_before.weight"], tensors[f"{prefix}.layernorm_before.bias"]])
            block.norm2.set_weights([tensors[f"{prefix}.layernorm_after.weight"], tensors[f"{prefix}.layernorm_after.bias"]])
            attn_prefix = f"{prefix}.attention"
            block.attn.query.set_weights(_dense_weights(tensors[f"{attn_prefix}.self.query.weight"], tensors[f"{attn_prefix}.self.query.bias"]))
            block.attn.key.set_weights(_dense_weights(tensors[f"{attn_prefix}.self.key.weight"], tensors[f"{attn_prefix}.self.key.bias"]))
            block.attn.value.set_weights(_dense_weights(tensors[f"{attn_prefix}.self.value.weight"], tensors[f"{attn_prefix}.self.value.bias"]))
            block.attn.proj.set_weights(_dense_weights(tensors[f"{attn_prefix}.output.dense.weight"], tensors[f"{attn_prefix}.output.dense.bias"]))
            block.attn.relative_position_bias_table.assign(tensors[f"{attn_prefix}.self.relative_position_bias_table"])
            block.mlp_fc1.set_weights(_dense_weights(tensors[f"{prefix}.intermediate.dense.weight"], tensors[f"{prefix}.intermediate.dense.bias"]))
            block.mlp_fc2.set_weights(_dense_weights(tensors[f"{prefix}.output.dense.weight"], tensors[f"{prefix}.output.dense.bias"]))
        if stage < len(_DEPTHS) - 1:
            downsample = model.get_layer(f"stage{stage}_downsample")
            prefix = f"swin.encoder.layers.{stage}.downsample"
            downsample.norm.set_weights([tensors[f"{prefix}.norm.weight"], tensors[f"{prefix}.norm.bias"]])
            downsample.reduction.set_weights(_dense_weights(tensors[f"{prefix}.reduction.weight"]))
    model.get_layer("encoder_norm").set_weights([tensors["swin.layernorm.weight"], tensors["swin.layernorm.bias"]])


def build_swin_t(input_shape: tuple[int, int, int], *, weights: str | None) -> keras.Model:
    height, width, channels = input_shape
    if channels != 3:
        raise ValueError(f"Swin-T espera 3 canales, recibio {channels}")
    stride = _PATCH_SIZE * (2 ** (len(_DEPTHS) - 1))
    if height % stride != 0 or width % stride != 0:
        raise ValueError(f"Swin-T requiere alto/ancho divisibles por {stride}; recibio {height}x{width}")

    dpr = [_DROP_PATH_RATE * i / max(sum(_DEPTHS) - 1, 1) for i in range(sum(_DEPTHS))]
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(_EMBED_DIM, _PATCH_SIZE, strides=_PATCH_SIZE, padding="valid", name="patch_embed", dtype="float32")(inputs)
    x = layers.LayerNormalization(epsilon=_LN_EPS, name="patch_norm", dtype="float32")(x)
    block_id = 0
    dim = _EMBED_DIM
    for stage, (depth, num_heads) in enumerate(zip(_DEPTHS, _NUM_HEADS)):
        for index in range(depth):
            shift = 0 if index % 2 == 0 else _WINDOW_SIZE // 2
            x = SwinBlock(dim, num_heads, _WINDOW_SIZE, shift, dpr[block_id], name=f"stage{stage}_block{index}")(x)
            block_id += 1
        if stage < len(_DEPTHS) - 1:
            x = PatchMerging(dim, name=f"stage{stage}_downsample")(x)
            dim *= 2
    x = layers.LayerNormalization(epsilon=_LN_EPS, name="encoder_norm", dtype="float32")(x)
    model = keras.Model(inputs, x, name="swint")
    if weights == "imagenet":
        _load_hf_weights(model, _download_swint_weights())
    elif weights is not None:
        raise ValueError("swint solo admite weights='imagenet' o None")
    return model


class SwinTBackbone(Backbone):
    """Swin-T: atencion por ventanas, costo lineal en la resolucion. Salida [B, H/32, W/32, 768]."""

    key = "swint"
    input_size = (224, 224)
    default_weights = "imagenet"
    preprocess_fn = staticmethod(swin_preprocess_input)
    batch_size = mode_batch_sizes(standard=256, resized=64, abmil=64)
    supports_fused_train_steps = False

    @override
    def preprocess_input(self, x):
        return swin_preprocess_input(x)

    @override
    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        if include_top:
            raise ValueError("swint no implementa include_top=True; se usa como extractor espacial")
        del kwargs
        return build_swin_t(self.input_shape_or_default(input_shape), weights=self.coalesce_weights(weights))
