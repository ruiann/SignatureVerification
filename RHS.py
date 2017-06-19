from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalRNN import BidirectionalRNN


class RHS:

    def __init__(self, rnn=[800], layer=[500, 350]):
        self.bidirectional_rnn = BidirectionalRNN('BidirectionalRNN', rnn)
        self.logistic_regression = LogisticRegression('LogisticRegression', rnn[-1], layer)

    # do classification
    def run(self, data, reuse=False):
        rnn_code = self.rnn(data, reuse=reuse)
        return self.regression(rnn_code, reuse=reuse)

    def rnn(self, data, reuse=False):
        return self.bidirectional_rnn.run(data, reuse=reuse, time_major=False)

    def regression(self, rnn_code, reuse=False):
        return self.logistic_regression.run(rnn_code, reuse=reuse)

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, data, labels):
        logits = self.run(data)
        classification = tf.to_int32(tf.arg_max(tf.nn.softmax(logits), dimension=1))
        differ = labels - classification
        tf.summary.histogram('classification difference', differ)
        loss = self.loss(logits, labels)
        tf.summary.scalar('classifier loss', loss)
        return loss
