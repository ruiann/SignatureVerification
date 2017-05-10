import os
import tensorflow as tf
import time
import resource
from GAN import GAN
from SVC_reader import *
import numpy as np

rate = 0.001
loop = 2000 * 50
batch_size = 64
channel = 3
scale = 100
sequence_limit = 500

log_dir = './log'
model_dir = './gan_model'
train_path = './SVC2004/Task1'

data = Data(train_path)


def get_feed():
    reference = []
    target = []
    labels = []
    max_reference = 0
    max_target = 0
    for i in range(batch_size):
        reference_data, reference_length, target_data, target_length, label_data = data.get_pair()
        reference.append(reference_data)
        target.append(target_data)
        labels.append(label_data)
        max_reference = max_reference if max_reference >= reference_length else reference_length
        max_target = max_target if max_target >= target_length else target_length
    for i in range(batch_size):
        reference[i] = np.pad(reference[i], ((0, max_reference - len(reference[i])), (0, 0)), 'constant', constant_values=0)
        target[i] = np.pad(target[i], ((0, max_target - len(target[i])), (0, 0)), 'constant', constant_values=0)

    return reference, target, labels


def train():
    gan = GAN(batch_size)
    reference_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    target_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    d_train_op, g_train_op = gan.train(rate, reference_x, target_x)

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
            reference_feed, target_feed, label_feed = get_feed()
            _, _, summary_str = sess.run([d_train_op, g_train_op, summary], feed_dict={reference_x: reference_feed, target_x: target_feed})
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
