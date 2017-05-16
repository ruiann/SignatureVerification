import tensorflow as tf
from DIscriminator import Discriminator
from SVC_reader import *
import numpy as np

test_loop = 10
batch_size = 3
channel = 3

test_path = './SVC2004/Task1'
model_dir = './model'

data = Data(test_path)


# get 1 target and multi reference to judge
def get_feed():
    max_reference = 0
    max_target = 0
    reference_data, target_data, label = data.get_multi_reference_pair(batch_size)

    for i in range(batch_size):
        max_reference = max(max_reference, len(reference_data[i]))
        max_target = max(max_target, len(target_data[i]))
    for i in range(batch_size):
        reference_data[i] = np.pad(reference_data[i], ((0, max_reference - len(reference_data[i])), (0, 0)), 'constant', constant_values=0)
        target_data[i] = np.pad(target_data[i], ((0, max_target - len(target_data[i])), (0, 0)), 'constant', constant_values=0)

    return reference_data, target_data, label


def test():
    discriminator = Discriminator(lstm_size=800)
    reference_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    target_x = tf.placeholder(tf.float32, shape=(batch_size, None, channel))
    train_op = discriminator.run(reference_x, target_x)

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
