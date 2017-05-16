import tensorflow as tf
from Discriminator import Discriminator
from LSTMDecoder import Decoder


class GAN:

    def __init__(self, batch_size):
        self.lstm_size = 800
        self.batch_size = batch_size

        with tf.variable_scope('generator'):
            self.decoder = Decoder('LSTMDecoder', self.batch_size)

        with tf.variable_scope('discriminator'):
            self.discriminator = Discriminator()

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    def generate(self, code):
        return self.decoder.run(code)

    def encode(self, sequence, reuse=False):
        return self.discriminator.lstm(sequence, reuse)

    def disciminate(self, lstm_code):
        return self.discriminator.regression(lstm_code)

    def run(self, reference, target):
        reference_code = self.encode(reference)
        target_code = self.encode(target, True)
        fake_target = self.generate(target_code)
        fake_code = self.encode(fake_target, True)
        target_output = self.disciminate(target_code - reference_code)
        fake_output = self.disciminate(fake_code - reference_code)
        tf.summary.histogram('target discrimination', target_output)
        tf.summary.histogram('fake discrimination', fake_output)
        return target_output, fake_output

    def train(self, rate, reference, target):
        target_output, fake_output = self.run(reference, target)
        target_label = tf.ones_like(target_output)
        fake_label = tf.zeros_like(fake_output)
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label, logits=target_output, name='target'))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_label, logits=fake_output, name='fake'))
        tf.summary.scalar('discriminator loss', d_loss)
        tf.summary.scalar('generator loss', g_loss)
        d_train = tf.train.AdamOptimizer(learning_rate=rate).minimize(d_loss, var_list=self.d_variables)
        g_train = tf.train.AdamOptimizer(learning_rate=rate).minimize(g_loss, var_list=self.g_variables)
        return d_train, g_train
