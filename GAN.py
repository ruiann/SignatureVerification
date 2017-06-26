from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Discriminator import Discriminator
from Generator import Generator


class GAN:

    def __init__(self, batch_size):
        self.lstm_size = 800
        self.batch_size = batch_size

        self.generator = Generator(self.batch_size)
        self.discriminator = Discriminator()

    def generate(self, code):
        return self.generator.run(code)

    def encode(self, sequence, reuse=False, time_major=False):
        return self.discriminator.rnn(sequence, reuse, time_major)

    def discriminate(self, lstm_code, reuse=False):
        return self.discriminator.regression(lstm_code, reuse)

    def run(self, reference, target):
        reference_code = self.encode(reference)
        target_code = self.encode(target, True)
        fake_target = self.generate(reference)
        fake_code = self.encode(fake_target, True, True)
        target_output = self.discriminate(target_code - reference_code)
        fake_output = self.discriminate(fake_code - reference_code, True)

        tf.summary.histogram('target discrimination', target_output)
        tf.summary.histogram('fake discrimination', fake_output)
        return target_output, fake_output

    def train(self, rate, reference, target):
        target_output, fake_output = self.run(reference, target)

        target_label = tf.ones_like(target_output)
        fake_label = tf.zeros_like(fake_output)
        d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_output, labels=target_label, name='d_loss_d'))
        d_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=fake_label, name='d_loss_g'))
        d_loss = (d_loss_d + d_loss_g) / 2
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=target_label, name='g_loss'))

        tf.summary.scalar('discriminator loss', d_loss)
        tf.summary.scalar('generator loss', g_loss)

        g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        g_train = tf.train.AdamOptimizer(learning_rate=rate).minimize(g_loss, var_list=g_variables)
        d_train = tf.train.AdamOptimizer(learning_rate=rate).minimize(d_loss, var_list=d_variables)
        return d_train, g_train
