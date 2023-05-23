import tensorflow as tf


class FuzzyAttention(tf.keras.layers.Layer):
    """
    A layer of fuzzyAttention called from the BiGRU with fazzy attention
    """
    def __init__(self):
        """
        Constructs the FuzzyAttention layer
        """
        super(FuzzyAttention, self).__init__()

    def build(self, input_shape):
        """
        builds the FuzzyAttention layer
        :param input_shape: The shape of the input
        :return: gets the attention_weights to be used in the call method
        """
        self.attention_weights = self.add_weight(shape=(input_shape[-1], 1),
                                                 initializer='random_normal',
                                                 trainable=True)

    def call(self, inputs):
        """
        calls the
        :param inputs:
        :return:
        """

        # Compute attention scores
        attention_scores = tf.matmul(inputs, self.attention_weights)
        attention_scores = tf.squeeze(attention_scores, axis=-1)

        # Apply fuzzy logic to attention scores
        fuzzy_scores = tf.nn.softmax(attention_scores, axis=-1)

        # Apply element-wise multiplication with inputs to get the attended representation
        attended_inputs = tf.multiply(inputs, tf.expand_dims(fuzzy_scores, axis=-1))

        return attended_inputs
