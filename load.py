import numpy as np
import pickle
from TraceSet import Trace, TraceSet

def get_trace_set(trace_set_path, format, ignore_malformed=True):
    """
    Load traces in from absolute path trace_set_path into a TraceSet object depending on the format.
    """

    if format == "cw":
        name = trace_set_path.rpartition('_traces')[0]
        plaintext_set_path = name + '_textin.npy'
        ciphertext_set_path = name + '_textout.npy'
        key_set_path = name + '_knownkey.npy'

        existing_properties = []
        try:
            traces = np.load(trace_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(traces)
        except FileNotFoundError:
            traces = None

        try:
            plaintexts = np.load(plaintext_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(plaintexts)
        except FileNotFoundError:
            print("WARNING: No plaintext for trace %s" % name)
            plaintexts = None

        try:
            ciphertexts = np.load(ciphertext_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(ciphertexts)
        except FileNotFoundError:
            ciphertexts = None

        try:
            keys = np.load(key_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(keys)
        except FileNotFoundError:
            keys = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]*traces.shape[0])
            print("No key file found! Using 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
            #keys = None

        masks = None  # No masks for Arduino experiments

        if ignore_malformed:  # Discard malformed traces
            for property in existing_properties:
                if traces.shape[0] != property.shape[0]:
                    return None

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys, masks=masks)
        else:  # Just truncate malformed traces instead of discarding
            if not traces is None:
                traces = traces[0:len(plaintexts)]
            if not ciphertexts is None:
                ciphertexts = ciphertexts[0:len(plaintexts)]
            if not keys is None:
                keys = keys[0:len(plaintexts)]
            if not masks is None:
                masks = masks[0:len(plaintexts)]

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys, masks=masks)
    elif format == "sigmf":  # .meta
        raise NotImplementedError
    elif format == "gnuradio":  # .cfile
        raise NotImplementedError
    elif format == "ascad":
        # return get_ascad_trace_set(trace_set_path)
        raise NotImplementedError
    else:
        print("Unknown trace input format '%s'" % format)
        exit(1)


def load_meta(meta_path):
    with open(meta_path, 'rb') as f:
        meta_trace_set = pickle.load(f)
    return meta_trace_set