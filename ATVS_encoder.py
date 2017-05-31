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

genuine_data = bucket_group()


def get_feed():
    s_feed = []
    label_feed = []
    bucket_index = random.randint(0, 9)
    bucket = genuine_data[bucket_index]
    for i in range(batch_size):
        index = random.randint(0, len(bucket) - 1)
        data = bucket[index]
        s_feed.append(data['signature'])
        label_feed.append(data['label'])

    return bucket_index, s_feed, label_feed


def train():
    sess = tf.Session()
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)

        rhs = RHS()
        x = tf.placeholder(tf.float32, shape=(batch_size, None, 7))
        labels = tf.placeholder(tf.int32)
        loss = rhs.train(x, labels)
        train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)

        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)

        step = global_step.eval()
        while step < loop:
            start_time = time.time()
            print('step: {}'.format(step))
            bucket_index, s_feed, labels_feed = get_feed()
            summary_str, step_loss, _ = sess.run([summary, loss, train_op], feed_dict={x: s_feed, labels: labels_feed})
            summary_writer.add_summary(summary_str, step)
            print('bucket: {} loss: {}'.format(bucket_index, step_loss))

            if step % 1000 == 999 and step != 0:
                checkpoint_file = os.path.join(model_dir, 'model')
                saver.save(sess, checkpoint_file, global_step)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            print("step cost: {0}".format(time.time() - start_time))
            sess.run(update_global_step)
            step = step + 1

        summary_writer.close()

if __name__ == '__main__':
    train()
