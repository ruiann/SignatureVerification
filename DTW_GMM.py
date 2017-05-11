from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dtw import dtw
from tensorflow.contrib.factorization.python.ops import gmm
import tensorflow as tf
from reader_for_dtw_gmm import Data
import numpy as np


def my_custom_norm(x, y):
    return (x * x) + (y * y)


class DTW_GMM:
    def __init__(self, cluster_num=3):
        self.genuine_graph = tf.Graph()
        self.forgery_graph = tf.Graph()
        with self.genuine_graph.as_default():
            self.genuine_gmm = gmm.GMM(cluster_num)
        with self.forgery_graph.as_default():
            self.forgery_gmm = gmm.GMM(cluster_num)

    def compare(self, reference, target):
        channel_dtw = []
        for channel_index in range(len(reference)):
            dis, _, _, _ = dtw(reference[channel_index], target[channel_index], dist=my_custom_norm)
            channel_dtw.append(dis)
        return channel_dtw

    def input_fn(self, data):
        def fn():
            return data, None
        return fn

    def train_genuine(self, data, steps=10):
        with self.genuine_graph.as_default():
            self.genuine_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def train_forgery(self, data, steps=10):
        with self.forgery_graph.as_default():
            self.forgery_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def infer(self, data):
        genuine_score = self.genuine_gmm.score(input_fn=self.input_fn(data), steps=1)
        forgery_score = self.forgery_gmm.score(input_fn=self.input_fn(data), steps=1)
        return genuine_score >= forgery_score

batch_size = 64
loop = 100


def train():
    model = DTW_GMM()
    data = Data()

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for step in range(loop):
            print('step: {}'.format(step))
            genuine_data = []
            forgery_data = []
            for i in range(batch_size):
                reference, target = data.get_genuine_pair()
                genuine_data.append(model.compare(reference, target))
                reference, target = data.get_fake_pair()
                forgery_data.append(model.compare(reference, target))
            genuine_data = tf.constant(genuine_data, dtype=np.float32)
            forgery_data = tf.constant(forgery_data, dtype=np.float32)
            model.train_genuine(genuine_data)
            model.train_forgery(forgery_data)


if __name__ == '__main__':
    train()
