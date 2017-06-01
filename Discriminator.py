from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalRNN import BidirectionalRNN


class Discriminator:

    def __init__(self):
        with tf.variable_scope('discriminator'):
            self.bidirectional_rnn = BidirectionalRNN('BidirectionalRNN', rnn_size=[100, 500])
            self.logistic_regression = LogisticRegression('LogisticRegression', [100, 1])

    # do classification
    def run(self, reference, target):
        reference_rnn_code = self.rnn(reference)
        target_rnn_code = self.rnn(target, reuse=True)
        return self.regression(reference_rnn_code - target_rnn_code)

    def rnn(self, data, reuse=False, time_major=False):
        with tf.variable_scope('discriminator'):
            return self.bidirectional_rnn.run(data, reuse, time_major)

    def regression(self, rnn_code):
        with tf.variable_scope('discriminator'):
            return self.logistic_regression.run(rnn_code)

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, rate, reference, target, labels):
        logits = self.run(reference, target)
        loss = self.loss(logits, labels)
        tf.summary.scalar('sigmoid loss', loss)
        return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
