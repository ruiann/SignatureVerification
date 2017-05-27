from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time
import resource
from GAN import GAN
from ATVS_reader import *
import random

rate = 0.001
loop = 2000 * 50
batch_size = 32
class_num = 350
scale = 100
sequence_limit = 500

log_dir = './gan_log'
model_dir = './gan_model'
genuine_data = bucket_writer_group()


def get_feed():
    r_feed = []
    t_feed = []
    bucket_index = random.randint(0, len(genuine_data))
    bucket = genuine_data[bucket_index]
    for i in range(batch_size):
        index = random.randint(0, len(bucket) - 1)
        writer_sample = bucket[index]
        reference_index = random.randint(0, len(writer_sample) - 1)
        signature = bucket[index][reference_index]
        r_feed.append(signature)
        target_index = random.randint(0, len(writer_sample) - 1)
        signature = bucket[index][target_index]
        t_feed.append(signature)

    return r_feed, t_feed


def train():
    gan = GAN(batch_size)
    reference = tf.placeholder(tf.float32, shape=(batch_size, None, 5))
    target = tf.placeholder(tf.float32, shape=(batch_size, None, 5))
    d_train_op, g_train_op = gan.train(rate, reference, target)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    with sess.as_default():
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary = tf.summary.merge_all()
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())

        print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        for step in range(loop):
            start_time = time.time()
            print('step: {}'.format(step))
            r, t = get_feed()
            _, _, summary_str = sess.run([d_train_op, g_train_op, summary], feed_dict={reference: r, target: t})
            summary_writer.add_summary(summary_str, step)

            if step % 1000 == 999:
                checkpoint_file = os.path.join(model_dir, 'model.latest')
                saver.save(sess, checkpoint_file)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            print("step cost: {0}".format(time.time() - start_time))
            print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()


if __name__ == '__main__':
    train()
