import tensorflow as tf
from RHS import RHS
from SVC_reader import *
import numpy as np

test_loop = 50
batch_size = 1
channel = 3

test_path = './SVC2004/Task2'
model_dir = './model'

data = Data(test_path)


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


def test():
    rhs = RHS(lstm_size=800)
    reference_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    target_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    train_op = rhs.run(reference_x, target_x)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    with sess.as_default():
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())

        for step in range(test_loop):
            print('step: {}'.format(step))
            reference_feed, target_feed, label_feed = get_feed()
            prob = sess.run(train_op, feed_dict={reference_x: reference_feed, target_x: target_feed})

            print(prob)
            print(label_feed)


if __name__ == '__main__':
    test()
