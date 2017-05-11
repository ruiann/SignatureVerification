from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dtw import dtw
from tensorflow.contrib.factorization.python.ops import gmm
import tensorflow as tf
from numpy.linalg import norm
from reader_for_dtw_gmm import Data
import pdb


class DTW_GMM:
    def __init__(self, cluster_num=3):
        with tf.variable_scope('separate_gmm'):
            self.genuine_gmm = gmm.GMM(cluster_num)
            self.forgery_gmm = gmm.GMM(cluster_num)

    def compare(self, reference, target):
        channel_dtw = []
        for channel_index in range(len(reference)):
            pdb.set_trace()
            dis, _, _, _ = dtw(reference[channel_index], target[channel_index], dist=lambda x, y: norm(x - y, ord=1))
            channel_dtw.append(dis)
        return channel_dtw

    def input_fn(self, data):
        def fn():
            return data, None
        return fn

    def train_genuine(self, data, steps=10):
        self.genuine_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def train_forgery(self, data, steps=10):
        self.genuine_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def infer(self, data):
        def fn():
            return data, None
        genuine_score = self.genuine_gmm.score(input_fn=self.input_fn(data), steps=1)
        forgery_score = self.forgery_gmm.score(input_fn=self.input_fn(data), steps=1)
        return genuine_score >= forgery_score

batch_size = 64
loop = 100


def train():
    model = DTW_GMM()
    data = Data()
    for step in range(loop):
        genuine_data = []
        forgery_data = []
        for i in range(batch_size):
            reference, target = data.get_genuine_pair()
            genuine_data.append(model.compare(reference, target))
            reference, target = data.get_fake_pair()
            forgery_data.append(model.compare(reference, target))

        model.train_genuine(genuine_data)
        model.train_forgery(forgery_data)


if __name__ == '__main__':
    train()
