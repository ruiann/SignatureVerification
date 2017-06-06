import numpy as np

base_path = './ATVS-SSig_DB/DS1_Modification_TimeFunctions'
useless_line = [-1]
bucket_gap = 50
bucket_size = 20


def get_writer_list():
    return range(0, 350)


def genuine_data_range():
    return range(1, 26)


# data definition of BIT Handwriting
def read_file(path, max_length=None):
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

        max_length = max_length or len(data)
        data_range = min(len(data), max_length)

        sample_x = []
        sample_y = []
        s = []
        before = 0
        for index in range(data_range):
            p = data[index]
            down = int(p[3])
            if before or down:
                if down:
                    if index == data_range - 1:
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
        signature = []
        for index in range(len(s) - 1):
            p = [sample_x[index], sample_y[index], sample_x[index + 1] - sample_x[index], sample_y[index + 1] - sample_y[index]]
            signature.append(p)
        signature.append([sample_x[-1], sample_y[-1], 0, 0])

    except Exception as e:
        print(repr(e))
        return None

    signature = np.concatenate((signature, s), axis=1)
    length = len(signature)

    return signature, length


def norm(sequence, std=None):
    sequence = np.array(sequence, dtype=np.float32)
    mean = sequence.mean()
    std = std or sequence.std()
    sequence = (sequence - mean) / std
    return sequence, std


def init_bucket():
    data = []
    for bucket_index in range(bucket_size):
        bucket_data = []
        for writer in get_writer_list():
            bucket_data.append([])
        data.append(bucket_data)
    return data


def get_genuine_data():
    data = init_bucket()
    for writer in get_writer_list():
        for index in genuine_data_range():
            sample, length = read_file('{0}/usuario{1}/u{1}_sg{2}.txt'.format(base_path, 1001 + writer, index))
            bucket_index = min(int(length / bucket_gap), bucket_size - 1)
            sample = pad(bucket_gap * bucket_index + bucket_gap, sample)
            bucket_data = data[bucket_index]
            bucket_data[writer].append(sample)
    return data


def bucket_group():
    data = get_genuine_data()
    buckets = []
    for bucket_data in data:
        bucket = []
        for writer in get_writer_list():
            writer_sample = bucket_data[writer]
            for sample in writer_sample:
                bucket.append({'label': writer, 'signature': sample})
        buckets.append(bucket)
    return buckets


def bucket_writer_group():
    data = get_genuine_data()
    buckets = []
    for bucket_data in data:
        bucket = []
        for writer in get_writer_list():
            writer_sample = bucket_data[writer]
            if len(writer_sample):
                bucket.append(writer_sample)
        buckets.append(bucket)
    return buckets


def pad(length, signature):
    pad = length - len(signature)
    if pad > 0:
        eos = np.array([[0, 0, 0, 0, 0, 0, 0]] * pad, np.float32)
        signature = np.concatenate((signature, eos), axis=0)
    else:
        signature = signature[0: length]
    return signature


if __name__ == '__main__':
    data = bucket_group()
    print(len(data))
    print(len(data[0]))
