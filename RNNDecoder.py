from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from LogisticRegression import LogisticRegression


class DG:
    def __init__(self, name, input_dim, layer):
        self.name = name
        with tf.variable_scope(self.name):
            self.x = LogisticRegression('x_generator', input_dim, layer)
            self.y = LogisticRegression('y_generator', input_dim, layer)

    def run(self, data, reuse=False):
        with tf.variable_scope(self.name):
            x = self.x.run(data, reuse=reuse)
            y = self.y.run(data, reuse=reuse)
        return tf.concat([x, y], 1)


class Decoder:
    def __init__(self, name, batch_size, lstm_size=500, data_type=tf.float32):
        self.dtype = data_type
        self.name = name
        self.batch_size = batch_size
        with tf.variable_scope(self.name):
            self.gru_cell = tf.contrib.rnn.GRUCell(lstm_size)
            self.dg = DG('d_generator', lstm_size, [1])
            self.W = tf.Variable(tf.random_normal([lstm_size, 3], stddev=0.5, dtype=tf.float32))

    def run(self, encoding):
        shape = encoding.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            self.U = tf.Variable(tf.random_normal([2, shape[1]], stddev=0.5, dtype=tf.float32))
            self.V = tf.Variable(tf.random_normal([3, shape[1]], stddev=0.5, dtype=tf.float32))
            state = self.gru_cell.zero_state(self.batch_size, self.dtype)

            output_d = tf.constant(0, dtype=tf.float32, shape=[shape[0], 2])
            output_s = tf.constant(0, dtype=tf.float32, shape=[shape[0], 3])

            outputs_d = []
            outputs_s = []
            i = 0

            def cond(i, output_d, output_s, state):
                return tf.arg_max(output_s, 1) != 2

            def body(i, output_d, output_s, state):
                input = tf.matmul(output_d, self.U) + tf.matmul(output_s, self.V) + encoding
                output, state = self.gru_cell(input, state)

                def init_call():
                    output_d = self.dg.run(output, reuse=False)
                    output_s = tf.nn.softmax(tf.matmul(output, self.W))
                    return output_d, output_s

                def normal_call():
                    output_d = self.dg.run(output, reuse=True)
                    output_s = tf.nn.softmax(tf.matmul(output, self.W))
                    return output_d, output_s

                output_d, output_s = tf.cond(tf.equal(i, 0),
                                             lambda: init_call(),
                                             lambda: normal_call())
                outputs_d.append(output_d)
                outputs_s.append(output_s)
                i = i + 1

                return i, output_d, output_s, state

            tf.while_loop(cond, body, [i, output_d, output_s, state])

            return tf.concat((outputs_d, outputs_s), 2)
