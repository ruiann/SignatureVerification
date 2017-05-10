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

        base_x = None
        base_y = None
        base_s = 0
        front_p = 0

        sample = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data = line.split()

            if int(data[3]) != 0:
                if base_x:
                    sample.append([int(data[0]) - base_x, int(data[1]) - base_y, base_s])
                base_x = int(data[0])
                base_y = int(data[1])
                front_p = 1
            else:
                if front_p == 1:
                    base_s = base_s + 1
                front_p = 0

    except Exception, e:
        print repr(e)
        return None

    return sample


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
    def __init__(self, dir_path=base_path, sequence_limit=500, scale=100):
        self.genuine_data = get_genuine_data(dir_path)
        self.fake_data = get_fake_data(dir_path)
        self.writer_list = get_writer_list()
        self.genuine_range = genuine_data_range()
        self.fake_range = fake_data_range()
        self.sequence_limit = sequence_limit
        self.scale = 100

    def normalize(self, sample):
        sample = np.array(sample, dtype=np.float32)
        if len(sample) > self.sequence_limit:
            sample = sample[0: self.sequence_limit]
        max_x = max(np.fabs(sample[:, 0]))
        max_y = max(np.fabs(sample[:, 1]))
        for line in sample:
            line[0] = self.scale * line[0] / max_x
            line[1] = self.scale * line[1] / max_y
        return sample

    def get_pair(self):
        writer = random.sample(self.writer_list, 1)[0] - 1
        reference_index = random.sample(self.genuine_range, 1)[0] - 1
        label = random.randint(0, 1)
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        reference = self.genuine_data[writer][reference_index]
        target = self.genuine_data[writer][target_index] if label == 1 else self.fake_data[writer][target_index]
        reference = self.normalize(reference)
        target = self.normalize(target)
        return reference, len(reference), target, len(target), [label]

    def get_multi_reference_pair(self, multi=3):
        writer = random.sample(self.writer_list, 1)[0] - 1
        label = random.randint(0, 1)
        target_index = random.sample(self.genuine_range, 1)[0] - 1
        target_sample = self.genuine_data[writer][target_index] if label == 1 else self.fake_data[writer][target_index]
        target_sample = self.normalize(target_sample)
        reference = []
        target = []
        for i in range(multi):
            reference_index = random.sample(self.genuine_range, 1)[0] - 1
            reference.append(self.normalize(self.genuine_data[writer][reference_index]))
            target.append(target_sample)
        return reference, target, label


if __name__ == '__main__':
    # data = read_file('{}/U{}S{}.TXT'.format(base_path, 1, 1))
    data = get_genuine_data()
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
    print(data[0][0])
