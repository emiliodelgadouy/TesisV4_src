import tensorflow as tf
from tensorflow.keras import layers


class GatedAttentionPooling(layers.Layer):
  # atencion gated de ABMIL: agrega instancias del bag con pesos aprendidos

    def __init__(self, attention_dim=128, gated=True, attention_initializer="glorot_uniform", **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.gated = gated
        self.attention_initializer = attention_initializer
        self.last_attention = None

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.V = self.add_weight(name="V", shape=(feature_dim, self.attention_dim), initializer="glorot_uniform", trainable=True, dtype="float32")
        if self.gated:
            self.U = self.add_weight(name="U", shape=(feature_dim, self.attention_dim), initializer="glorot_uniform", trainable=True, dtype="float32")
        self.w = self.add_weight(name="w", shape=(self.attention_dim, 1), initializer=self.attention_initializer, trainable=True, dtype="float32")
        super().build(input_shape)

    def attention_logits(self, inputs):
        h = tf.cast(inputs, tf.float32)
        vh = tf.tanh(tf.matmul(h, self.V))
        if self.gated:
            vh = vh * tf.sigmoid(tf.matmul(h, self.U))
        return tf.matmul(vh, self.w)

    def call(self, inputs):
        h = tf.cast(inputs, tf.float32)
        attention = tf.nn.softmax(self.attention_logits(h), axis=1)
        self.last_attention = attention
        return tf.reduce_sum(attention * h, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim, "gated": self.gated, "attention_initializer": self.attention_initializer})
        return config


class BagTiling(layers.Layer):
  # parte una imagen grande en K tiles (rows x cols) para MIL con tiling en keras

    def __init__(self, bag_grid, **kwargs):
        super().__init__(**kwargs)
        self.rows = int(bag_grid[0])
        self.cols = int(bag_grid[1])

    def call(self, x):
        rows, cols = self.rows, self.cols
        b = tf.shape(x)[0]
        static_h, static_w, static_c = x.shape[1], x.shape[2], x.shape[3]
        dyn = tf.shape(x)
        ph, pw, c = dyn[1] // rows, dyn[2] // cols, dyn[3]
        x = tf.reshape(x, [b, rows, ph, cols, pw, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [b, rows * cols, ph, pw, c])
        ph_s = static_h // rows if static_h is not None else None
        pw_s = static_w // cols if static_w is not None else None
        x.set_shape([None, rows * cols, ph_s, pw_s, static_c])
        return x

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        ph = h // self.rows if h is not None else None
        pw = w // self.cols if w is not None else None
        return (b, self.rows * self.cols, ph, pw, c)

    def get_config(self):
        config = super().get_config()
        config.update({"bag_grid": (self.rows, self.cols)})
        return config
