from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dtw import dtw
from tensorflow.contrib.factorization.python.ops import gmm
import tensorflow as tf
import numpy as np
from reader_for_dtw_gmm import Data
import pdb


def my_custom_norm(x, y):
    return (x * x) + (y * y)

genuine_model_dir = './model_genuine'
forgery_model_dir = './model_forgery'


class DTW_GMM:
    def __init__(self, cluster_num=3):
        self.genuine_gmm = gmm.GMM(cluster_num, model_dir=genuine_model_dir)
        self.forgery_gmm = gmm.GMM(cluster_num, model_dir=forgery_model_dir)

    def compare(self, reference, target):
        channel_dtw = []
        for channel_index in range(len(reference)):
            dis, _, _, _ = dtw(reference[channel_index], target[channel_index], dist=my_custom_norm)
            channel_dtw.append(dis)
        return channel_dtw

    def input_fn(self, data):
        def fn():
            return tf.constant(data), None
        return fn

    def train_genuine(self, data, steps=1000):
        print(data)
        self.genuine_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def train_forgery(self, data, steps=1000):
        print(data)
        self.forgery_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def infer(self, data):
        genuine_result = self.genuine_gmm.predict(input_fn=self.input_fn(data))
        forgery_result = self.forgery_gmm.predict(input_fn=self.input_fn(data))
        genuine_score = []
        forgery_score = []
        for result in genuine_result:
            genuine_score.append(result['all_scores'][result['assignments']])
        for result in forgery_result:
            forgery_score.append(result['all_scores'][result['assignments']])
        return np.array(forgery_score) - np.array(genuine_score)

model = DTW_GMM()


def build_data(genuine_path, forgery_path, dir=None):
    data = Data(dir)
    genuine_pair = data.get_all_genuine_pair()
    forgery_pair = data.get_all_fake_pair()

    genuine_data = []
    forgery_data = []

    index = 0
    for pair in genuine_pair:
        print(index)
        index = index + 1
        (reference, target) = pair
        genuine_data.append(model.compare(reference, target))

    genuine_data = np.array(genuine_data, np.float32)
    np.savetxt(genuine_path, genuine_data)

    index = 0
    for pair in forgery_pair:
        print(index)
        index = index + 1
        (reference, target) = pair
        forgery_data.append(model.compare(reference, target))

    forgery_data = np.array(forgery_data, np.float32)
    np.savetxt(forgery_path, forgery_data)


def train():
    genuine_data = np.loadtxt('genuine_dtw.txt', dtype=np.float32)
    forgery_data = np.loadtxt('forgery_dtw.txt', dtype=np.float32)
    print('train genuine gmm')
    model.train_genuine(genuine_data, steps=1000)
    print('train forgery gmm')
    model.train_forgery(forgery_data, steps=1000)


def test():
    genuine_data = np.loadtxt('test_genuine_dtw.txt', dtype=np.float32)
    genuine_sample = genuine_data[0: 10000]
    result = model.infer(genuine_sample)
    print(np.sum(result > 0))


if __name__ == '__main__':
    # build_data('genuine_dtw.txt', 'forgery_dtw.txt')
    # build_data('test_genuine_dtw.txt', 'test_forgery_dtw.txt', './SVC2004/Task1')
    # train()
    test()
