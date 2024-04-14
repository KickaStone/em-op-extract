# 2020-02-17_11-24-50_980815

import numpy as np
import scipy
import matplotlib.pyplot as plt

def butter_filter(trace, order=1, cutoff=0.001, filter_type='low', fs=None):
    """
    Apply butter filter to trace
    """
    b, a = scipy.signal.butter(order, cutoff, btype=filter_type, fs=fs)
    print(b.shape, a.shape, trace.shape)
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
    filtered_trace = butter_filter(trace, 1, 0.001, 'high', None)  # Remove low freqs
    #debug_trace_specgram(filtered_trace, sample_rate)
    filtered_trace = np.abs(filtered_trace)
    filtered_trace = butter_filter(filtered_trace, 1, 0.001, 'high', None)  # Remove low freqs
    return filtered_trace

trace_path = r'datasets/nodemcu-random-train2/2020-02-17_11-25-04_601705_traces.npy'
traces = np.load(trace_path, allow_pickle=True)

for i, trace in enumerate(traces):
    meta_path = trace_path.rpartition("_")[0] + '_meta.p'
    meta = np.load(meta_path, allow_pickle=True)
    print(trace.shape, meta[i]['op'])
    trace = filter(trace)
    plot_trace(trace, title='Trace', xlabel='Time', ylabel='Amplitude')