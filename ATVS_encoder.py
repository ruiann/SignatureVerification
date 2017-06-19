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


def get_feed(genuine_data):
    s_feed = []
    label_feed = []
    bucket_index = random.randint(0, len(genuine_data) - 1)
    bucket = genuine_data[bucket_index]
    for i in range(batch_size):
        index = random.randint(0, len(bucket) - 1)
        data = bucket[index]
        s_feed.append(data['signature'])
        label_feed.append(data['label'])

    return bucket_index, s_feed, label_feed


def train():
    genuine_data = bucket_group()
    sess = tf.Session()
    with sess.as_default():
        global_step = tf.Variable(0, name='global_step')
        update_global_step = tf.assign(global_step, global_step + 1)

        rhs = RHS()
        x = tf.placeholder(tf.float32, shape=(batch_size, None, 5))
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
            bucket_index, s_feed, labels_feed = get_feed(genuine_data)
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


def normalize(data, max_length=None):
    if (max_length and (len(data) > max_length)):
        data = data[0: max_length]
        data[-1][2:5] = [0, 0, 1]
    data = np.array(data, np.float32)
    data[:, 0], std_x = norm(data[:, 0])
    data[:, 1], _ = norm(data[:, 1], std_x)
    prev_x = 0
    prev_y = 0
    normalized_data = []
    for point in data:
        normalized_data.append([point[0] - prev_x, point[1] - prev_y, point[2], point[3], point[4]])
        prev_x = point[0]
        prev_y = point[1]
    return np.array(normalized_data, np.float32)


def infer(reference, target, max_length=None):
    reference = normalize(reference, max_length)
    target = normalize(target)
    sess = tf.Session()
    with sess.as_default():
        rhs = RHS()
        x_1 = tf.placeholder(tf.float32, shape=(1, None, 5))
        x_2 = tf.placeholder(tf.float32, shape=(1, None, 5))
        distance = tf.reduce_mean(tf.square(rhs.run(x_1) - rhs.run(x_2, reuse=True)))
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)

        dis = sess.run(distance, feed_dict={x_1: [reference], x_2: [target]})
        print(dis)

if __name__ == '__main__':
    train()
