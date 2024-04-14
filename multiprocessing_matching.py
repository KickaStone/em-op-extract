import multiprocessing as mp
import concurrent.futures
import time
import os
import sys
import regex
import numpy as np
import scipy
from matplotlib import pyplot as plt
from common import fast_xcorr, pad_to_length

debug = False
template_path = './arch'
test_path = './datasets/nodemcu-random-test2'
train_path = './datasets/nodemcu-random-train2'

def butter_filter(trace, order=1, cutoff=0.001, filter_type='low', fs=None):
    """
    Apply butter filter to trace
    """
    b, a = scipy.signal.butter(order, cutoff, btype=filter_type, fs=fs)
    trace_filtered = scipy.signal.filtfilt(b, a, trace)
    return trace_filtered


def plot_trace(trace, title='Trace', xlabel='Time', ylabel='Amplitude'):
    """
    Plot trace
    """
    plt.style.use('_mpl-gallery')
    x = np.linspace(0, len(trace), len(trace))
    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    ax.plot(x, trace, linewidth=2.0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()


def get_opname(filename):
    """
    Get operation name from filename
    """
    return regex.match(r'nodemcu-(\w+-?\w+)-5600', filename).group(1)

def filter(trace):
    """
    Apply butter filter to trace
    """
    trace_filtered = butter_filter(trace, order=1, cutoff=0.001, filter_type='high', fs=None)
    trace_filtered = np.abs(trace_filtered)
    trace_filtered = butter_filter(trace_filtered, order=1, cutoff=0.001, filter_type='high', fs=None)
    return trace_filtered

def get_corr(trace, temp):
    _, corr = fast_xcorr(trace, temp, normalized=True, prefilter=False, return_corr=True)
    return corr

# load templates
files = os.listdir('./arch')
arch_templaes = {}
for entry in files:
    if not entry.endswith('.npy'):
        continue
    template = np.load(os.path.join('./arch', entry))
    op = get_opname(entry).replace('-', '_')
    arch_templaes[op] = template

# load test datasets

test_traces = [entry for entry in os.listdir(test_path) if entry.endswith('_traces.npy')]
print('Test traces:', len(test_traces))

true_label = []
pred_label = []

for idx, entry in enumerate(test_traces):
    filename = entry.rpartition('_traces')[0]
    traces = np.load(os.path.join(test_path, entry), allow_pickle=True)
    metadata = np.load(os.path.join(test_path, filename+'_meta.p'), allow_pickle=True)
    for i, trace in enumerate(traces):
        trace_op = metadata[i]['op']
        if trace_op not in arch_templaes.keys() or len(trace) < 1000:
            continue
        trace = filter(pad_to_length(trace, 130000))
        
        max_corr = -1
        pred_op = None

        results = []
        tags = []
        corrs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for arch_template_op, arch_template in arch_templaes.items():
                result = executor.submit(get_corr, trace, arch_template)
                # _, corr = fast_xcorr(trace, arch_template, normalized=True, prefilter=False ,return_corr=True)
                results.append(result)
                tags.append(arch_template_op)
        
        for f in results:
            try:
                corrs.append(f.result())
            except Exception as e:
                print(e)
        best_idx = np.argmax(np.array(corrs))
        pred_op = tags[best_idx]        
                

        true_label.append(trace_op)
        pred_label.append(pred_op)

    precentage = int((idx+1)/len(test_traces)*100)
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*int(precentage), precentage))
    sys.stdout.flush()

print('Test finished\n')