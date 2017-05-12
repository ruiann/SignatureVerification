from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dtw import dtw
from tensorflow.contrib.factorization.python.ops import gmm
import tensorflow as tf
from reader_for_dtw_gmm import Data


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

    def train_genuine(self, data, steps=100):
        print(data)
        self.genuine_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def train_forgery(self, data, steps=100):
        print(data)
        self.forgery_gmm.fit(input_fn=self.input_fn(data), steps=steps)

    def infer(self, data):
        genuine_score = self.genuine_gmm.score(input_fn=self.input_fn(data), steps=1)
        forgery_score = self.forgery_gmm.score(input_fn=self.input_fn(data), steps=1)
        return genuine_score - forgery_score

model = DTW_GMM()
data = Data()


def train():
    genuine_pair = data.get_all_genuine_pair()
    forgery_pair = data.get_all_fake_pair()

    genuine_data = []
    forgery_data = []

    for pair in genuine_pair:
        (reference, target) = pair
        genuine_data.append(model.compare(reference, target))

    print('train genuine gmm')
    model.train_genuine(genuine_data)

    for pair in forgery_pair:
        (reference, target) = pair
        forgery_data.append(model.compare(reference, target))

    print('train forgery gmm')
    model.train_forgery(forgery_data)


if __name__ == '__main__':
    train()
