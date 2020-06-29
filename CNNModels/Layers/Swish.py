from tensorflow.keras.layers import Layer
import tensorflow as tf

class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        relu6 = tf.nn.relu6(inputs)
        h_swish = relu6 * (inputs + 3) / 6

        return inputs * h_swish