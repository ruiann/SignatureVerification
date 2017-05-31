from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalRNN import BidirectionalRNN


class RHS:

    def __init__(self, lstm=[800], layer=[500, 350]):
        self.bidirectional_rnn = BidirectionalRNN('BidirectionalLSTM', lstm)
        self.logistic_regression = LogisticRegression('LogisticRegression', lstm[-1], layer)

    # do classification
    def run(self, data):
        lstm_code = self.lstm(data)
        return self.regression(lstm_code)

    def lstm(self, data):
        return self.bidirectional_rnn.run(data, reuse=False, time_major=False)

    def regression(self, lstm_code):
        return tf.nn.relu(self.logistic_regression.run(lstm_code)[-1])

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
