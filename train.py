import os
import tensorflow as tf
import time
import resource
from RHS import RHS
from SVC_reader import *
import random
import pdb

rate = 0.0001
loop = 2000 * 50
batch_size = 64
channel = 3

log_dir = './log'
model_dir = './model'

rhs = RHS(lstm_size=800)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(model_dir)
summary = tf.summary.merge_all()
run_metadata = tf.RunMetadata()

genuine_data = get_genuine_data()
fake_data = get_fake_data()
writer_list = get_writer_list()
genuine_range = genuine_data_range()
fake_range = fake_data_range()


def get_pair():
    writer = random.sample(writer_list, 1)[0] - 1
    reference = random.sample(genuine_range, 1)[0] - 1
    label = random.randint(0, 1)
    target = random.sample(genuine_range, 1)[0] - 1
    reference = genuine_data[writer][reference]
    target = genuine_data[writer][target] if label == 1 else fake_data[writer][target]
    return reference, target, label


def get_feed():
    reference = []
    target = []
    labels = []
    for i in range(batch_size):
        reference_data, target_data, label_data = get_pair()
        reference.append(reference_data)
        target.append(target_data)
        labels.append(label_data)
    return reference, target, labels

def train():

    reference_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    target_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    label_x = tf.placeholder(tf.float32, shape=(batch_size, 1))
    train_op = rhs.train(rate, reference_x, target_x, label_x)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    with sess.as_default():
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        for step in range(loop):
            start_time = time.time()
            print('step: {}'.format(step))
            reference_feed, target_feed, label = get_feed()
            result, summary_str = sess.run([train_op, summary], feed_dict={reference_x: reference_feed, target_x: target_feed})
            summary_writer.add_summary(summary_str, step)

            if step % 100 == 0 and step != 0:
                checkpoint_file = os.path.join(model_dir, 'model.latest')
                saver.save(sess, checkpoint_file)
                summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)

            print("step cost: {0}".format(time.time() - start_time))
            print('Memory usage: {0}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        summary_writer.close()


if __name__ == '__main__':
    train()
