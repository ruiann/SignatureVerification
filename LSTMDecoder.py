from tensorflow.python.ops import variable_scope
import tensorflow as tf


# copied from source code of seq2seq by TF
def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


class Decoder:
    def __init__(self, name, batch_size, lstm_size=3, data_type=tf.float32):
        self.dtype = data_type
        self.name = name
        self.batch_size = batch_size
        with tf.variable_scope(self.name):
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    def run(self, encoding):
        state = self.lstm_cell.zero_state(self.batch_size, self.dtype)
        # return rnn_decoder(encoding, state, self.lstm_cell)
        return tf.contrib.seq2seq.Decoder()
