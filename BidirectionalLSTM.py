# define the bidirectional LSTM network for RHS feature extraction

import tensorflow as tf


class BidirectionalLSTM:

    def __init__(self, name, lstm_size, data_type=tf.float32):
        self.data_type = data_type
        self.name = name
        with tf.variable_scope(self.name):
            self.forward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.backward_lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    def run(self, data, reuse=False):
        with tf.variable_scope("ForwardLSTM", reuse=reuse):
            forward_output, state = tf.nn.dynamic_rnn(self.forward_lstm, data, dtype=self.data_type)
            forward_output = forward_output[:, -1, :]

        with tf.variable_scope("BackwardLSTM", reuse=reuse):
            backward_output, state = tf.nn.dynamic_rnn(self.backward_lstm, data, dtype=self.data_type)
            backward_output = backward_output[:, -1, :]

        tf.summary.histogram('forward_lstm_output', forward_output)
        tf.summary.histogram('backward_lstm_output', backward_output)

        return tf.add(forward_output, backward_output, 'feature')
