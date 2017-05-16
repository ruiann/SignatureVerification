from tensorflow.python.ops import variable_scope
import tensorflow as tf
import pdb


class GMM:
    def __init__(self, name, k, input_dim):
        with tf.variable_scope(name or 'gmm_output'):
            self.W = tf.Variable(tf.random_normal([input_dim, 3 * k], stddev=0.5, dtype=tf.float32))
            self.b = tf.Variable(tf.random_normal([1, 3 * k], stddev=0.5, dtype=tf.float32))

    def run(self, input):
        output = tf.matmul(input, self.W) + self.b
        pi, mu_x, mu_y = tf.split(output, 3, 1)
        max_pi = tf.reduce_max(pi, 1, keep_dims=True)
        pi = pi - max_pi
        pi = tf.exp(pi)
        normalize_pi = tf.reciprocal(tf.reduce_sum(pi, 1, keep_dims=True))
        pi = tf.multiply(normalize_pi, pi)
        result_x = tf.reshape(tf.reduce_sum(tf.multiply(mu_x, pi), 1), (-1, 1))
        result_y = tf.reshape(tf.reduce_sum(tf.multiply(mu_y, pi), 1), (-1, 1))
        return tf.concat([result_x, result_y], 1)


class Decoder:
    def __init__(self, name, batch_size, lstm_size=500, data_type=tf.float32):
        self.dtype = data_type
        self.name = name
        self.batch_size = batch_size
        self.gmm_num = 20
        with tf.variable_scope(self.name):
            self.gru_cell = tf.contrib.rnn.GRUCell(lstm_size)
            self.gmm = GMM('gmm_prediction', self.gmm_num, lstm_size)
            self.W = tf.Variable(tf.random_normal([lstm_size, 3], stddev=0.5, dtype=tf.float32))

    def run(self, encoding):
        shape = encoding.get_shape().as_list()
        self.U = tf.Variable(tf.random_normal([2, shape[1]], stddev=0.5, dtype=tf.float32))
        self.V = tf.Variable(tf.random_normal([3, shape[1]], stddev=0.5, dtype=tf.float32))

        state = self.gru_cell.zero_state(self.batch_size, self.dtype)
        output_s = tf.constant([1, 0, 0])
        i = 0

        outputs_d = []
        outputs_s = []

        while tf.arg_max(output_s, 1) != 2:
            if i == 0:
                output, state = self.gru_cell(encoding, state)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
                input = tf.matmul(output_d, self.U) + tf.matmul(output_s, self.V) + encoding
                output, state = self.gru_cell(input, state)
            output_d = self.gmm.run(output)
            output_s = tf.nn.softmax(tf.matmul(output, self.W))
            outputs_d.append(output_d)
            outputs_s.append(output_s)
            i = i + 1
        return tf.concat([outputs_d, outputs_s], 2)
