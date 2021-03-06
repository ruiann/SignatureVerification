from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.factorization.python.ops import gmm
import tensorflow as tf
import numpy as np
from reader_for_dtw_gmm import Data
from mlpy import dtw

train_dir = './SVC2004/Task1'
test_dir = './SVC2004/Task2'
genuine_model_dir = './model_genuine'
forgery_model_dir = './model_forgery'
train_genuine_dtw_path = 'genuine_dtw.txt'
train_forgery_dtw_path = 'forgery_dtw.txt'
test_genuine_dtw_path = 'test_genuine_dtw.txt'
test_forgery_dtw_path = 'test_forgery_dtw.txt'


def input_fn(data):
    def fn():
        return tf.constant(data), None

    return fn


class DTW_GMM:
    def __init__(self, cluster_num=3):
        self.genuine_gmm = gmm.GMM(cluster_num, model_dir=genuine_model_dir)
        self.forgery_gmm = gmm.GMM(cluster_num, model_dir=forgery_model_dir)

    def compare(self, reference, target):
        channel_dtw = []
        for channel_index in range(len(reference)):
            dist = dtw.dtw_std(reference[channel_index], target[channel_index], dist_only=True)
            channel_dtw.append(dist)
        print(channel_dtw)
        return channel_dtw

    def train_genuine(self, data, steps=1000):
        self.genuine_gmm.fit(input_fn=input_fn(data), steps=steps)

    def train_forgery(self, data, steps=1000):
        self.forgery_gmm.fit(input_fn=input_fn(data), steps=steps)

    def infer(self, data):
        genuine_result = self.genuine_gmm.predict(input_fn=input_fn(data))
        forgery_result = self.forgery_gmm.predict(input_fn=input_fn(data))
        genuine_score = []
        forgery_score = []
        for result in genuine_result:
            genuine_score.append(result['all_scores'][result['assignments']])
        for result in forgery_result:
            forgery_score.append(result['all_scores'][result['assignments']])
        return np.array(forgery_score) - np.array(genuine_score)

model = DTW_GMM()


def build_data(genuine_path, forgery_path, dir):
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
    genuine_data = np.loadtxt(train_genuine_dtw_path, dtype=np.float32)
    forgery_data = np.loadtxt(train_forgery_dtw_path, dtype=np.float32)
    print('train genuine gmm')
    model.train_genuine(genuine_data, steps=10000)
    print('train forgery gmm')
    model.train_forgery(forgery_data, steps=10000)


def test():
    genuine_data = np.loadtxt(test_genuine_dtw_path, dtype=np.float32)
    result = model.infer(genuine_data)
    print(np.sum(result > 0) / len(genuine_data))

    forgery_data = np.loadtxt(test_forgery_dtw_path, dtype=np.float32)
    result = model.infer(forgery_data)
    print(np.sum(result < 0) / len(forgery_data))


if __name__ == '__main__':
    # build train data
    build_data(train_genuine_dtw_path, train_forgery_dtw_path, train_dir)
    # build test data
    build_data(test_genuine_dtw_path, test_forgery_dtw_path, test_dir)
    train()
    test()
