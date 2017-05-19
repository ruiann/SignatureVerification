from RNNDecoder import Decoder
import tensorflow as tf


class Generator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with tf.variable_scope('generator'):
            self.decoder = Decoder('RNN_decoder', self.batch_size)

    def run(self, code):
        with tf.variable_scope('generator'):
            return self.decoder.run(code)