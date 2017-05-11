import random
import numpy as np

base_path = './SVC2004/Task1'
useless_line = [0]


def get_writer_list():
    return range(1, 41)


def genuine_data_range():
    return range(1, 21)


def fake_data_range():
    return range(21, 41)


# data definition of BIT Handwriting
def read_file(path):
    try:
        file = open(path, 'r')
        lines = file.readlines()
        for line_index in useless_line:
            del lines[line_index]

        front_p = 0
        sample_x = []
        sample_y = []
        velocity_x = []
        velocity_y = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data = line.split()

            if int(data[3]) != 0:
                x = int(data[0])
                y = int(data[1])
                if front_p == 1:
                    sample_x.append(x)
                    sample_y.append(y)
                    velocity_x.append(x - front_x)
                    velocity_y.append(y - front_y)

                front_x = x
                front_y = y
                front_p = 1

            else:
                front_p = 0

    except Exception, e:
        print repr(e)
        return None

    return [sample_x, sample_y, velocity_x, velocity_y]


def get_genuine_data(dir_path=base_path):
    data = []
    for writer in get_writer_list():
        writer_sample = []
        for index in genuine_data_range():
            writer_sample.append(read_file('{}/U{}S{}.TXT'.format(dir_path, writer, index)))
        data.append(writer_sample)
    return data


def get_fake_data(dir_path=base_path):
    data = []
    for writer in get_writer_list():
        writer_sample = []
        for index in fake_data_range():
            writer_sample.append(read_file('{}/U{}S{}.TXT'.format(dir_path, writer, index)))
        data.append(writer_sample)
    return data


class Data:
    def __init__(self, dir_path=base_path, scale=100):
        self.genuine_data = get_genuine_data(dir_path)
        self.fake_data = get_fake_data(dir_path)
        self.writer_list = get_writer_list()
        self.genuine_range = genuine_data_range()
        self.fake_range = fake_data_range()
        self.scale = scale

    def norm(self, sequence):
        sequence = np.array(sequence, dtype=np.float32)
        max = sequence.max()
        min = sequence.min()
        sequence = self.scale * (sequence - min) / (max - min)
        return sequence

    def normalize(self, sample):
        for i in range(len(sample)):
            sample[i] = self.norm(sample[i])
        return np.array(sample, dtype=np.float32)

    def get_genuine_pair(self):
        writer = random.sample(self.writer_list, 1)[0] - 1
        reference_index = random.sample(self.genuine_range, 1)[0] - 1
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        reference = self.normalize(self.genuine_data[writer][reference_index])
        target = self.normalize(self.genuine_data[writer][target_index])
        return reference, target

    def get_fake_pair(self):
        writer = random.sample(self.writer_list, 1)[0] - 1
        reference_index = random.sample(self.genuine_range, 1)[0] - 1
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        reference = self.normalize(self.genuine_data[writer][reference_index])
        target = self.normalize(self.fake_data[writer][target_index])
        return reference, target


if __name__ == '__main__':
    data = Data()
    reference, target = data.get_genuine_pair()
    print(reference)
