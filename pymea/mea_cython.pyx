import numpy as np
import pandas as pd
cimport numpy as np
from scipy import signal

__all__ = ['find_series_peaks', 'min_max_bin']

def find_series_peaks(series):
    cdef np.ndarray[float] input_data
    cdef np.ndarray[double] bf, af, data
    cdef double thresh, dt, t0, a, b, c, x
    cdef int n, maxn, min_sep
    cdef list peaks = []
    input_data = series.values

    # first perform high pass filter
    bf, af = signal.butter(1, 0.01, btype='highpass')
    data = signal.lfilter(bf, af, input_data)
    thresh = -5 * np.sqrt(np.mean(data**2))

    dt = series.index[1] - series.index[0]
    t0 = series.index[0]
    min_sep = int(0.0008/dt)

    # Find points which are smaller than neighboring points
    n = 0
    maxn = len(data) - 2
    while n < maxn:
        if data[n] < thresh and data[n] < data[n-1] and data[n] < data[n+1]:
            a, b, c = np.polyfit(np.arange(n-2, n+3), data[n-2:n+3], 2)
            x = -b/(2*a)
            peaks.append((x * dt + t0, np.polyval([a, b, c], x), thresh))
            n += min_sep
        n += 1

    if len(peaks) < 1:
        return pd.DataFrame(columns=['time', 'amplitude', 'threshold'])

    return pd.DataFrame(np.array(peaks, dtype=np.float32),
                        columns=['time', 'amplitude', 'threshold'])


def min_max(np.ndarray[float] d):
    if len(d) == 0:
        return 0, 0

    cdef int n
    cdef float minval = d[0]
    cdef float maxval = d[0]
    for n in range(len(d)):
        if d[n] > maxval:
            maxval = d[n]
        if d[n] < minval:
            minval = d[n]
    return minval, maxval


def min_max_bin(np.ndarray[float] series, int bin_size, int bin_count):
    cdef np.ndarray[float] sub
    cdef np.ndarray[long] edges = np.arange(0, bin_count * bin_size, bin_size)
    cdef np.ndarray[float] vals = np.zeros(len(edges)*2 - 2, np.float32)

    for n in range(len(edges) - 1):
        sub = series[edges[n]:edges[n+1]]
        minval, maxval = min_max(sub)
        vals[2*n] = minval
        vals[2*n+1] = maxval

    return vals
