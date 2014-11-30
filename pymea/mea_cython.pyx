import numpy as np
import pandas as pd
cimport numpy as np
from scipy import signal

def find_series_peaks(series):
    cdef np.ndarray[double] bf, af, data
    cdef double a, b, c, x, dt, t0, thresh
    cdef int n
    cdef list peaks = []
    data = series.values

    # first perform high pass filter
    bf, af = signal.butter(1, 0.01, btype='highpass')
    data = signal.lfilter(bf, af, data)
    thresh = -7 * np.sqrt(np.mean(data**2))

    dt = series.index[1] - series.index[0]
    t0 = series.index[0]

    # Find points which are smaller than neighboring points
    for n in range(2, len(data) - 2):
        if data[n] < thresh and data[n] < data[n-1] and data[n] < data[n+1]:
            a, b, c = np.polyfit(np.arange(n-2, n+3), data[n-2:n+3], 2)
            x = -b/(2*a)
            peaks.append((x * dt + t0, np.polyval([a, b, c], x), thresh))

    return pd.DataFrame(np.array(peaks),
                        columns=['time', 'amplitude', 'threshold'])


def min_max(np.ndarray[float] d):
    cdef int n
    cdef float minval = d[0]
    cdef float maxval = d[0]
    for n in range(len(d)):
        if d[n] > maxval:
            maxval = d[n]
        if d[n] < minval:
            minval = d[n]
    return minval, maxval


def min_max_bin(np.ndarray[float] series, bins=130):
    cdef np.ndarray[float] sub
    cdef int bin_size = int(len(series)/bins)
    cdef np.ndarray[long] edges = np.arange(0, len(series), bin_size)
    cdef np.ndarray[float] vals = np.zeros(len(edges)*2 - 2, np.float32)

    for n in range(len(edges) - 1):
        sub = series[edges[n]:edges[n+1]]
        minval, maxval = min_max(sub)
        vals[2*n] = minval
        vals[2*n+1] = maxval

    return vals
