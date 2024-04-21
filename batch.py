import os

import numpy as np

from load import get_trace_set, load_meta
from preprocess import pad_to_length
from common import *
import sklearn.preprocessing

filter_method = 'abs_nofilt'
datasets_root = './datasets/'
# input_size = 32768  # Test
input_size = 131072
# input_size = 4096
num_classes = len(op_to_int.keys())
noise_patch = None
use_augment_noise = False
use_newaugment = None  # Use argument


def normalize(data):
    # normalize the data to [-1, 1]
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def get_normalized_data(data):
    data_ = normalize(data)
    return data_[0:-1], data_[1:]


def label_to_title(label):
    label_class = np.argmax(label[0:num_classes])
    return int_to_op[label_class]


def get_batch(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, augment=False):
    trace_name = dataset_file.rpartition('_traces.npy')[0]
    meta_name = trace_name + '_meta.p'
    trace_path = os.path.join(dataset_path, dataset_file)
    meta_path = os.path.join(dataset_path, meta_name)

    # get traces
    trace_set = get_trace_set(trace_path, 'cw')

    meta_trace_set = load_meta(meta_path)

    for j, trace in enumerate(trace_set.traces):
        if "bad" in meta_trace_set[j]:  # Skip traces explicitly labeled as "bad"
            continue
        filtered_trace = pad_to_length(trace.signal, input_size + 1)  # Padding
        filtered_trace = filter_trace(filtered_trace, filter_method)  #

        filtered_trace = get_normalized_data(filtered_trace)

        wavenet_input, wavenet_target = get_normalized_data(filtered_trace)

        label = get_onehot(meta_trace_set[j]["op"])

        if np.isnan(wavenet_input).any() or np.isnan(wavenet_target).any() or np.isnan(label).any():  # Skip the data which contains nan.
            continue
        batch_c.append((wavenet_input, wavenet_target, label))

        if len(batch_c) == batch_size:

            batch = np.array(batch_c, dtype=list)  # must declear the dtype as list, or it will raise valueError
            np.random.shuffle(batch)
            yield batch
            batch_c.clear()


def get_validation_batch(dataset_name, with_bounds=False):
    dataset_path = os.path.join(datasets_root, dataset_name)
    dataset_files = list(os.listdir(dataset_path))

    batch_c = []
    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=64):
                return batch

# 
window_size = 131072 + 256  # 131'328

def normalize(data):
    # normalize the data to [-1, 1]
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def get_normalized_data_2d(trace):
    return sklearn.preprocessing.normalize(trace)

def transform_meta_1d_to_2d(meta, overlap=0.5):
    if "left_bound" in meta:
        meta["left_bound"] = (meta["left_bound"] / (512 * (1.0 - overlap)))
        meta["right_bound"] = (meta["right_bound"] / (512 * (1.0 - overlap)))
    return meta


def get_normalized_data(data):
    data_ = normalize(data)
    return data_[0:-1], data_[1:]  # 切片到最后一个【0，-1）步长为1


def label_to_title(label):
    label_class = np.argmax(label[0:num_classes])
    return int_to_op[label_class]


def get_batch(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, augment=False):
    trace_name = dataset_file.rpartition('_traces.npy')[0]
    meta_name = trace_name + '_meta.p'
    trace_path = os.path.join(dataset_path, dataset_file)
    meta_path = os.path.join(dataset_path, meta_name)  # 创建两个路径

    # get traces
    trace_set = get_trace_set(trace_path, 'cw')

    meta_trace_set = load_meta(meta_path)

    for j, trace in enumerate(trace_set.traces):
        if "bad" in meta_trace_set[j]:  # Skip traces explicitly labeled as "bad"
            continue
        filtered_trace = pad_to_length(trace.signal, input_size + 1)  # Padding
        filtered_trace = filter_trace(filtered_trace, filter_method)  #

        filtered_trace = get_normalized_data(filtered_trace)

        wavenet_input, wavenet_target = get_normalized_data(filtered_trace)

        label = get_onehot(meta_trace_set[j]["op"])
        # print(f"input={wavenet_input}, target={wavenet_target}, label={label}")
        batch_c.append((wavenet_input, wavenet_target, label))

        if len(batch_c) == batch_size:
            batch = np.array(batch_c, dtype=list)  # must declear the dtype as list, or it will raise valueError
            np.random.shuffle(batch)  # 打乱list排序
            yield batch
            batch_c.clear()


def get_validation_batch(dataset_name, with_bounds=False):
    dataset_path = os.path.join(datasets_root, dataset_name)
    dataset_files = list(os.listdir(dataset_path))

    batch_c = []
    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=64):
                return batch


def get_batch_2d(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, augment=False):
    trace_name = dataset_file.rpartition('_traces.npy')[0]
    meta_name = trace_name + '_meta.p'
    trace_path = os.path.join(dataset_path, dataset_file)
    meta_path = os.path.join(dataset_path, meta_name)  # 创建两个路径

    # get traces
    trace_set = get_trace_set(trace_path, 'cw')
    # get tracemetadata
    meta_trace_set = load_meta(meta_path)

    for j, trace in enumerate(trace_set.traces):
        if "bad" in meta_trace_set[j]:  # Skip traces explicitly labeled as "bad"
            continue
        filtered_trace = pad_to_length(trace.signal, window_size)  # Padding
        filtered_trace = get_stft(filtered_trace, overlap_rate=0.5, show_plot=False)  #
        meta_trace_set[j] = transform_meta_1d_to_2d(meta_trace_set[j], overlap=0.5)

        filtered_trace = get_normalized_data_2d(filtered_trace)
        nn_input = get_normalized_data_2d(filtered_trace)

        label = get_onehot(meta_trace_set[j]["op"])
        # print(f"input={wavenet_input}, target={wavenet_target}, label={label}")
        batch_c.append((nn_input, label))

        if len(batch_c) == batch_size:
            batch = np.array(batch_c, dtype=list)  # must declear the dtype as list, or it will raise valueError
            np.random.shuffle(batch)  # 打乱list排序
            yield batch
            batch_c.clear()


def get_validation_batch_2d(dataset_name, with_bounds=False):
    dataset_path = os.path.join(datasets_root, dataset_name)
    dataset_files = list(os.listdir(dataset_path))

    batch_c = []
    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            for batch in get_batch_2d(dataset_path, dataset_file, batch_c, batch_size=64):
                return batch