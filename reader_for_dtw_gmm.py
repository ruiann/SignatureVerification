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
        before_front_p = 0
        sample_x = []
        sample_y = []
        velocity_x = []
        velocity_y = []
        acceleration_x = []
        acceleration_y = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data = line.split()

            if int(data[3]) != 0:
                x = int(data[0])
                y = int(data[1])
                sample_x.append(x)
                sample_y.append(y)
                if front_p == 1:
                    velocity_x.append(x - front_x)
                    velocity_y.append(y - front_y)
                    if before_front_p == 1:
                        acceleration_x.append(velocity_x[-1] - velocity_x[-2])
                        acceleration_y.append(velocity_y[-1] - velocity_y[-2])
                front_x = x
                front_y = y
                before_front_p = front_p
                front_p = 1

            else:
                before_front_p = front_p
                front_p = 0

    except:
        print(path)
        return None

    return [sample_x, sample_y, velocity_x, velocity_y, acceleration_x, acceleration_y]


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
    def __init__(self, dir_path=base_path):
        self.genuine_data = get_genuine_data(dir_path)
        self.fake_data = get_fake_data(dir_path)
        self.writer_list = get_writer_list()
        self.genuine_range = genuine_data_range()
        self.fake_range = fake_data_range()

    def norm(self, sequence):
        sequence = np.array(sequence, dtype=np.float32)
        mean = sequence.mean()
        std = sequence.std()
        sequence = (sequence - mean) / std
        return sequence

    def normalize(self, sample):
        for i in range(len(sample)):
            sample[i] = self.norm(sample[i])
        return sample

    def get_genuine_pair(self):
        writer = random.sample(self.writer_list, 1)[0] - 1
        reference_index = random.sample(self.genuine_range, 1)[0] - 1
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        reference = self.normalize(self.genuine_data[writer][reference_index])
        target = self.normalize(self.genuine_data[writer][target_index])
        return reference, target

    def get_all_genuine_pair(self):
        pair = []
        for writer in self.writer_list:
            writer = writer - 1
            for reference_index in self.genuine_range:
                for target_index in self.genuine_range:
                    if target_index != reference_index:
                        reference = self.normalize(self.genuine_data[writer][reference_index - 1])
                        target = self.normalize(self.genuine_data[writer][target_index - 1])
                        pair.append((reference, target))

        return pair

    def get_fake_pair(self):
        writer = random.sample(self.writer_list, 1)[0] - 1
        reference_index = random.sample(self.genuine_range, 1)[0] - 1
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        reference = self.normalize(self.genuine_data[writer][reference_index - 1])
        target = self.normalize(self.fake_data[writer][target_index - 1])
        return reference, target

    def get_all_fake_pair(self):
        pair = []
        for writer in self.writer_list:
            writer = writer - 1
            for reference_index in self.genuine_range:
                for target_index in self.genuine_range:
                    reference = self.normalize(self.genuine_data[writer][reference_index - 1])
                    target = self.normalize(self.fake_data[writer][target_index - 1])
                    pair.append((reference, target))

        return pair


if __name__ == '__main__':
    data = Data()
    reference, target = data.get_genuine_pair()
    print(reference)
