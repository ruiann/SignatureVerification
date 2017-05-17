from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from RHS import RHS
from ATVS_reader import *
import random
import os

batch_size = 48
class_num = 350
rate = 0.0001
loop = 1000000

log_dir = './log'
model_dir = './model'

genuine_data = get_genuine_data()


def get_feed():
    d_feed = []
    s_feed = []
    label_feed = []
    max_length = 0
    for i in range(batch_size):
        label = random.randint(0, class_num - 1)
        index = random.randint(0, 24)
        (d, s) = genuine_data[label][index]
        max_length = max(max_length, len(d))
        d_feed.append(d)
        s_feed.append(s)
        label_feed.append(label)
    for i in range(batch_size):
        d_feed[i] = np.pad(d_feed[i], ((0, max_length - len(d_feed[i])), (0, 0)), 'constant', constant_values=0)
        if max_length - len(s_feed[i]):
            eos = np.array([[0, 0, 1]] * (max_length - len(s_feed[i])), np.float32)
            s_feed[i] = np.concatenate((s_feed[i], eos), axis=0)

    return d_feed, s_feed, label_feed


def train():
    rhs = RHS(lstm_size=800, class_num=class_num)
    d = tf.placeholder(tf.float32, shape=(batch_size, None, 2))
    s = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    x = tf.concat([d, s], 2)
    labels = tf.placeholder(tf.int32)
    train_op = rhs.train(rate, x, labels)

    sess = tf.Session()

    with sess.as_default():
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for global_step in xrange(loop):
            start_time = time.time()
            print('step: {}'.format(global_step))
            d_feed, s_feed, labels_feed = get_feed()
            summary_str, loss = sess.run([summary, train_op], feed_dict={d: d_feed, s: s_feed, labels: labels_feed})
            summary_writer.add_summary(summary_str, global_step)

            if global_step % 1000 == 0 and global_step != 0:
                checkpoint_file = os.path.join(model_dir, 'model.latest')
                saver.save(sess, checkpoint_file)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
            print("step cost: {0}".format(time.time() - start_time))

        summary_writer.close()

if __name__ == '__main__':
    train()
