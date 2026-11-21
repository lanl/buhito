import numpy as np
import pandas as pd
from collections import defaultdict
import time
from buhito.utilities import evenly_distribute_jobs, uniquify_lol

def rank(a):
    return np.argsort(np.argsort(a))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)


def bits_by_size(bits, size):
    return {s: bits[size==s] for s in pd.Series(size).sort_values().unique()}


def count_fragments_by_size(fingerprint):
    counts_by_size = defaultdict(lambda: 0)
    for (size, _), count in fingerprint.items():
        counts_by_size[size] += count
    return dict(counts_by_size)


def group_bitinfo_by_size(bi):
    out = defaultdict(lambda: [])
    for bit_id, locations in bi.items():
        bit_size = bit_id[0]
        out[bit_size].append({bit_id: locations})
    return dict(out)


def counts_by_bit_by_size(bi):
    out = defaultdict(lambda: [])
    for bit_id, locations in bi.items():
        size = bit_id[0]
        out[size].append(len(locations))
    out = dict(out)
    for k, v in out.items():
        out[k] = tuple(sorted(v))
    return out


def get_graphs_with_bit(bit,
                      bit_ids,
                      X):
    """
    gives you the indices of the graphs with at least one instance of bit

    :param bit: a bit ID
    :param bit_ids: a list of length n containing bit ids in the order in which they appear in the columns of X
    :param X: a (m,n) scipy.sparse matrix with m graphs and n bits
    """
    i = bit_ids.index(bit)
    ii, jj = X.nonzero()
    rows_with_bit = ii[np.where(jj==i)]
    return rows_with_bit


def scan_magnitudes(lower, upper, base=10):
    """returns a numpy array of values scanning from lower to upper powers of base"""

    scan = [np.array([1., 2., 5.])*base**e for e in range(lower, upper)]
    scan.append(np.array([1.*base**upper]))
    scan = np.hstack(scan)
    return scan


def find_inds(bitid, biinfo):
    """
    Looks up the node indices associated with a given substructure fingerprint (hash).
    Returns only the *first* match encountered during iteration.

    Iterates over an graphlet bitInfo mapping and returns the first stored
    index/indices for the entry whose key's second element matches `bitid`.

    Parameters
    ----------
    bitid : int
        Fingerprint bit identifier to search for.
    biinfo : dict
        Bit information mapping (bit_info from graphlet featurizer output), where keys are tuples
        whose second element is a bit id, and values are node indices for that bit.

    Returns
    -------
    list or None
        List of node indices if found, else None.

    """
    for k,v in biinfo.items():
        if k[1]==bitid:
            return v[0]
    return None
