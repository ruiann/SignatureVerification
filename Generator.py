from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from RNNDecoder import Decoder
from BidirectionalRNN import BidirectionalRNN
import tensorflow as tf


class Generator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with tf.variable_scope('generator'):
            self.encoder = BidirectionalRNN('generator', rnn_size=[100, 500])
            self.decoder = Decoder('generator', self.batch_size)

    def run(self, data, reuse=False, time_major=False):
        with tf.variable_scope('generator'):
            code = self.encoder.run(data, reuse, time_major)
            return self.decoder.run(code)