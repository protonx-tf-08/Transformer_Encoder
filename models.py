# TODO 2
import tensorflow as tf
import layers as ls

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate):
        super(TransformerClassifier, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim = input_vocab_size, output_dim = d_model)
        self.pos_encoding = ls.positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [ls.EncoderLayer(d_model, num_heads, dff, rate = rate ) for _ in range(num_encoder_layers)]
        self.dropout = tf.keras.layers.Dropout(rate = rate)
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.final_layers = tf.keras.layers.Dense(1, 'sigmoid')

    def call(self, x, training):

        embedded_sequences = self.embedding(x)

        encoded_sequences = embedded_sequences + self.pos_encoding[:, :tf.shape(x)[1],:]

        for enc_layer in self.enc_layers:
          encoded_sequences = enc_layer(encoded_sequences, training = training, mask = None)

        encoded_sequences = self.dropout(encoded_sequences, training=training)

        encoded_sequences = self.global_average_pooling(encoded_sequences, training = training)

        logits = self.final_layers(encoded_sequences)

        return logits

