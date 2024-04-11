# TODO 2
import tensorflow as tf
import layers

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate = 0.1):
        super(TransformerClassifier, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim = input_vocab_size, output_dim = d_model)
        self.pos_encoding =
