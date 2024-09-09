from keras.src.layers import Dense, Layer
import tensorflow as tf


class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__()
        # Define the dense layers used to compute attention scores
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_output, hidden_state):
        # encoder_output: (batch_size, sequence_length, hidden_size)
        # hidden_state: (batch_size, hidden_size)

        # Expand hidden state to match the shape of encoder_output
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)

        # Calculate attention scores (energy)
        score = self.V(tf.nn.tanh(self.W(encoder_output) + self.U(hidden_with_time_axis)))

        # Calculate attention weights using softmax
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector is the weighted sum of the encoder output
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = {
            "W": self.W,
            "U": self.U,
            "V": self.V,
            "units": self.units,
        }
        return config
