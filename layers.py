# TODO 1
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def scaled_dot_product_attention(self,q,k,v,mask):

        dk = tf.cast(tf.shape(k)[-1], dtype = tf.float32)

        attention_score = tf.matmul(q,k, transpose_b = True) / (tf.math.sqrt(dk))

        if mask is not None:
            attention_score += (mask* -1e10)
        weight_attention = tf.nn.softmax(attention_score, axis = -1)

        out = tf.matmul(weight_attention, v)

        return out

    def split_heads(self, x, batch_size):

        length = tf.shape(x)[1]

        x = tf.reshape(x,(batch_size, length, self.num_heads, self.depth))

        x = tf.transpose(x, [0,2,1,3])
        return x

    def call(self, v, k, q, mask):

        batch_size = tf.shape(q)[0]
        qw = self.wq(q)
        kw = self.wk(k)
        vw = self.wv(v)


        heads_qw = self.split_heads(qw, batch_size)
        heads_kw = self.split_heads(kw, batch_size)
        heads_vw = self.split_heads(vw, batch_size)



        out = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw, mask = mask)

        out = tf.transpose(out, [0,2,1,3])

        out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model))

        output = self.dense(out)

        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation = 'relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate = rate)
        self.dropout2 = tf.keras.layers.Dropout(rate = rate)

    def call(self, x, training, mask):
        mha_out = self.mha(x, x, x, mask=mask)

        x = self.layernorm1(x + self.dropout1(mha_out, training = training))

        ffn_out = self.ffn(x)

        out2 = self.layernorm2(x + self.dropout2(ffn_out, training=training))
        return out2


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.expand_dims (angle_rads , axis = 0)

    return tf.cast(pos_encoding, dtype=tf.float32)
