import os
from load import get_trace_set, load_meta

def get_batch(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, augment=False) :
    trace_name = dataset_file.rpartition('_traces.npy')[0]
    meta_name = trace_name + '_meta.p'
    trace_path = os.path.join(dataset_path, dataset_file)
    meta_path = os.path.join(dataset_path, meta_name)

    # get traces
    trace_set = get_trace_set(trace_path, 'cw')

    meta_trace_set = load_meta(meta_path)