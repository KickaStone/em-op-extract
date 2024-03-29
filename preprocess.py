import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt


def normalize(trace):
    """
    Z-score normalize trace
    """
    mean = np.mean(trace)
    std = np.std(trace)
    if std == 0:
        raise ValueError
    return (trace - mean) / std


def normalize_p2p(trace):
    ptp = trace.ptp(0)
    if ptp == 0:
        raise ValueError

    return (trace - trace.min(0)) / ptp


def ifreq(signal):
    instantaneous_phase = np.unwrap(np.angle(signal))
    instantaneous_frequency = np.diff(instantaneous_phase)
    return instantaneous_frequency


def butter_filter(trace, order=1, cutoff=0.01, filter_type='low', fs=None):
    """
    Apply butter filter to trace
    """
    b, a = scipy.signal.butter(order, cutoff, btype=filter_type, fs=fs)
    trace_filtered = scipy.signal.filtfilt(b, a, trace)
    return trace_filtered


def align(trace, reference, cutoff=0.01, order=1, prefilter=False):
    """
    Determine their offset using cross-correlation. This offset is then used to
    align the original signals.
    """
    # Preprocess
    try:
        trace = np.array(trace)
        reference = np.array(reference)
        if prefilter:
            processed_trace = butter_filter(trace, order=order, cutoff=cutoff)
            processed_reference = butter_filter(reference, order=order, cutoff=cutoff)
            processed_trace = normalize_p2p(processed_trace)  # normalize() seems to work pretty well too
            processed_reference = normalize_p2p(processed_reference)
        else:
            processed_trace = normalize_p2p(trace)  # normalize() seems to work pretty well too
            processed_reference = normalize_p2p(reference)
    except ValueError:  # Something is wrong with the signal
        return None

    # Correlated processed traces to determine lag
    result = scipy.signal.correlate(processed_trace, processed_reference, mode='valid')
    lag = np.argmax(result)

    # Align the original trace based on this calculation
    aligned_trace = trace[lag:]

    # Vertical align as well TODO add as new separate op?
    #bias = np.mean(aligned_trace)
    #aligned_trace -= bias
    #DEBUG = True

    # if DEBUG:
    #     plt.plot(range(0, len(processed_reference)), processed_reference, label="Normalized reference")
    #     plt.plot(range(0, len(processed_trace)), processed_trace, label="Normalized trace")
    #     plt.plot(range(0, len(result)), result, label="Correlation")
    #     #plt.plot(range(0, len(aligned_trace)), aligned_trace, label="Aligned trace")
    #     plt.legend()
    #     plt.show()

    return aligned_trace


def pad_to_length(trace, length):
    if len(trace) < length:
        trace = np.lib.pad(trace, (0, length - len(trace)), 'constant', constant_values=(0.0))
    return trace[0:length]


def decimate(signal, factor):
    pad_size = int(np.ceil(float(signal.size) / factor) * factor - signal.size)
    #signal_padded = np.append(signal, np.zeros(pad_size, dtype=signal.dtype) * np.NaN)
    signal_padded = np.lib.pad(signal, (0, pad_size), 'constant', constant_values=(np.NaN))
    return scipy.nanmean(signal_padded.reshape((-1, factor)), axis=1)


def fast_xcorr(trace, ref_trace, prefilter=True, required_corr=-1.0, normalized=True, debug=False, return_corr=False, return_corr_trace=False):
    """
    Fast normalized cross correlation between 1D signal and a reference signal.
    :param trace:
    :param ref_trace:
    :return:
    """
    comp_trace = trace
    comp_ref = ref_trace
    if prefilter:
        comp_trace = butter_filter(trace, 1, 0.02, 'low', None)
        comp_ref = butter_filter(ref_trace, 1, 0.02, 'low', None)

    # Not as precise as doing this on a per-window basis but much faster.
    comp_trace = comp_trace - np.mean(comp_trace)
    comp_ref = comp_ref - np.mean(comp_ref)

    # Correlate it with the ref signal and shift accordingly
    corr = scipy.signal.fftconvolve(comp_trace, comp_ref[::-1], mode="valid")
    if normalized:
        y = np.sqrt(np.sum(np.square(comp_ref)))
        x = np.sum(np.square(comp_trace[0:len(comp_ref)]))

        corr[0] /= np.sqrt(x) * y
        # Sliding update of sum
        for i in range(0, len(corr)-1):
            x -= comp_trace[i] ** 2
            x += comp_trace[i + len(comp_ref)] ** 2
            corr[i + 1] /= np.sqrt(x) * y

    # Get best correlation index and value
    best_index = np.argmax(corr)
    best_corr = max(corr)

    # Debug plot
    if debug:
        plt.plot(corr, label="corr")
        plt.plot(butter_filter(ref_trace, 1, 0.02, 'low', None), label="ref")
        plt.plot(butter_filter(trace[best_index:], 1, 0.02, 'low', None), label="aligned")
        plt.plot(butter_filter(trace, 1, 0.02, 'low', None), label="original")
        plt.legend()
        plt.show()

    if return_corr_trace:
        return corr

    if return_corr:
        if best_corr < required_corr:
            return None
        else:
            return best_index, best_corr
    else:
        if best_corr < required_corr:
            return None
        else:
            return best_index


def generate_impulse(high_duration, low_duration):
    """
    Generate square wave impulse to correlate with a signal
    :param high_duration:
    :param low_duration:
    :return:
    """

    # 5 small pulses (LED trigger)
    """
    pulse_all = int(self.trigger_samples * 0.131118881)
    nothing_all = self.trigger_samples - pulse_all
    pulse = pulse_all // 5
    nothing = nothing_all // 5
    trigger = [0.0] * nothing + [1.0] * pulse
    trigger = trigger * 5
    """

    # Square wave trigger
    """
    trigger = ([0.0] * self.trigger_low_samples + [1.0] * self.trigger_high_samples) * 4
    """

    # Coarse LED trigger
    trigger = [1.0] * high_duration
    pause = [0.0] * low_duration
    impulse = np.array(trigger + pause + trigger)

    return impulse


def generate_led_impulse(high_duration, low_duration):
    """
    Generate square wave impulse to correlate with a signal
    :param high_duration:
    :param low_duration:
    :return:
    """

    # 5 small pulses (LED trigger)
    pulse_len = high_duration // 10
    led_pulse = [0.0] * pulse_len + [1.0] * pulse_len
    trigger = led_pulse * 5
    pause = [0.0] * low_duration
    impulse = np.array(trigger + pause + trigger)

    return impulse

