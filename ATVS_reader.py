import numpy as np
import pdb

base_path = './ATVS-SSig_DB/DS1_Modification_TimeFunctions'
useless_line = [-1]


def get_writer_list():
    return range(1001, 1351)


def genuine_data_range():
    return range(1, 26)


# data definition of BIT Handwriting
def read_file(path):
    try:
        file = open(path, 'r')
        lines = file.readlines()
        for line_index in useless_line:
            del lines[line_index]

        data = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            data.append(line.split())

        sample_x = []
        sample_y = []
        s = []
        before = 0
        for index in range(len(data)):
            p = data[index]
            down = int(p[3])
            if before:
                if down:
                    if index == len(data) - 1:
                        eos = 1
                        down = 0
                    else:
                        eos = 0
                else:
                    eos = 0
                x = int(p[0])
                y = int(p[1])
                sample_x.append(x)
                sample_y.append(y)
                s.append([down, 1 if not down and not eos else 0, eos])
            before = down

        sample_x, std_x = norm(sample_x)
        sample_y, _ = norm(sample_y, std_x)
        s = s[1: len(s)]
        signature = []
        for index in range(len(s)):
            p = [sample_x[index + 1] - sample_x[index], sample_y[index + 1] - sample_y[index]]
            signature.append(p)

    except Exception as e:
        print(repr(e))
        return None

    return signature, s


def norm(sequence, std=None):
    sequence = np.array(sequence, dtype=np.float32)
    mean = sequence.mean()
    std = std or sequence.std()
    sequence = 100 * (sequence - mean) / std
    return sequence, std


def get_genuine_data():
    data = []
    for writer in get_writer_list():
        writer_sample = []
        for index in genuine_data_range():
            sample = read_file('{0}/usuario{1}/u{1}_sg{2}.txt'.format(base_path, writer, index))
            writer_sample.append(sample)
        data.append(writer_sample)
    return data


if __name__ == '__main__':
    data = get_genuine_data()
    pdb.set_trace()
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))