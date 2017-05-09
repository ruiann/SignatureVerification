# definition RHS model

import tensorflow as tf
from LogisticRegression import LogisticRegression
from BidirectionalLSTM import BidirectionalLSTM


class RHS:

    def __init__(self, lstm_size=800):
        self.bidirectional_LSTM = BidirectionalLSTM('BidirectionalLSTM', lstm_size)
        self.logistic_regression = LogisticRegression('LogisticRegression', lstm_size)

    # do classification
    def run(self, reference, target):
        reference_lstm_code = self.lstm(reference)
        target_lstm_code = self.lstm(target, reuse=True)
        return self.regression(reference_lstm_code - target_lstm_code)

    def lstm(self, data, reuse=False):
        return self.bidirectional_LSTM.run(data, reuse)

    def regression(self, lstm_code):
        return self.logistic_regression.run(lstm_code)

    # compute loss
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='RHS'))

    # return training operation, data should be a PlaceHolder
    def train(self, rate, reference, target, labels):
        logits = self.run(reference, target)
        loss = self.loss(logits, labels)
        tf.summary.scalar('sigmoid loss', loss)
        return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
