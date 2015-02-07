#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from scipy import signal
import h5py

from . import util
from . import mea_cython

__all__ = ['MEARecording', 'coordinates_for_electrode', 'condense_spikes',
           'filter', 'export_peaks']

input_dir = os.path.expanduser(
    '~/Dropbox/Hansma/ncp/IA6787/2014_08_20_Baseline')


class MEARecording:
    def __init__(self, store):
        store = os.path.expanduser(store)
        if not os.path.exists(store):
            raise IOError('File not found.')
        self.store_path = store
        self.store = h5py.File(store)
        info = self.store[
            '/Data/Recording_0/AnalogStream/Stream_0/InfoChannel']
        self.sample_rate = 1e6/info[0][8]
        self.conv = 1e-3 * info[0][9]
        self.lookup = {}
        for i, l in enumerate([r[3].decode('ascii').lower() for r in info]):
            self.lookup[l] = i
        self.start_time = time.strftime(
            '%a, %d %b %Y %I:%M:%S %p',
            time.localtime(
                self.store['Data'].attrs['DateInTicks']/1e7 - 62135596800))
        self.mea = self.store['Data'].attrs['MeaName']
        self.electrode_data = self.store[
            'Data/Recording_0/AnalogStream/Stream_0/ChannelData']
        self.data_len = self.electrode_data.shape[1]
        self.duration = self.data_len / self.sample_rate
        self.peaks = []

    def get(self, channels, start_time=0, end_time=None):
        if channels == 'all':
            channels = list(self.lookup.keys())
        channels.sort(key=lambda s: self.lookup[s])
        if end_time is None:
            end_time = self.duration
        start_i = int(util.clip(start_time * self.sample_rate,
                                0, self.data_len - 1))
        end_i = int(util.clip(end_time * self.sample_rate, 0, self.data_len))
        rows = [self.lookup[channel] for channel in channels]
        data = (self.conv *
                self.electrode_data[rows, start_i:end_i]
                .astype(np.float32).transpose())
        return pd.DataFrame(data,
                            index=np.arange(start_i, end_i)/self.sample_rate,
                            columns=channels, dtype=np.float32)

    def find_peaks(self, amp=6.0):
        peaks = []
        df = self.get('all')
        for electrode in df.keys():
            p = mea_cython.find_series_peaks(df[electrode], amp)
            p.insert(0, 'electrode', electrode)
            peaks.append(p)
        self.peaks = pd.concat(peaks)
        return self.peaks

    def __del__(self):
        try:
            self.close()
        except ValueError:
            pass

    def __getitem__(self, key):
        item = self.conv * self.electrode_data[
            self.lookup[key]].astype(np.float32)
        return pd.Series(item, index=np.arange(len(item))/self.sample_rate)

    def __len__(self):
        return len(self.lookup)

    def __str__(self):
        return (
            'File:\t\t%s\nDate:\t\t%s\nMEA:\t\t%s\nSample Rate:\t%0.1f Hz\n'
            'Duration:\t%0.2f s' %
            (self.store_path,
             self.start_time,
             self.mea.decode('ascii'),
             self.sample_rate,
             self.duration))

    def close(self):
        self.store.close()


def export_peaks(fname, amp=6.0):
    fname = os.path.expanduser(fname)
    rec = MEARecording(fname)
    df = rec.find_peaks(amp)
    df.to_csv(fname[:-3] + '.csv', index=False)


def coordinates_for_electrode(tag):
    tag = tag.lower()
    cols = {'a':  0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
            'h': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11}
    return (cols[tag[0]], int(tag[1:]) - 1)


def condense_spikes(srcdir, fname):
    with open(fname, 'a') as dest:
        dest.write('electrode,time\n')
        for f in os.listdir(srcdir):
            with open(os.path.join(srcdir, f)) as src:
                label = f[:-4].split('_')[-1]
                for line in src:
                    if len(line) > 0 and line[0].isdigit():
                        dest.write('%s,%s' % (label, line))


def read_binary(fname, no_channels, columns, part=(0, -1),
                fs=20000, cal=0.0610):
    """
    Loads a binary data file. Data should be 16 bits wide.

    Parameters
    ----------
    fname : str
        File name to import.
    no_channels : int
        Number of channels represented in binary data.
    fs : float
        Sampling frequency in Hz.
    """
    d = np.fromfile(fname, np.uint16)
    d = d.reshape((-1, no_channels))
    d = pd.DataFrame((d - 32768.0) * cal,
                     columns=columns,
                     index=np.arange(0, len(d)/fs, 1/fs))
    return d


def filter(series):
    """
    Filters given series with a 2nd order bandpass filter with cutoff
    frequencies of 100Hz and 4kHz

    Parameters
    ----------
    series : pandas.Series
        The data series to filter

    Returns
    -------
    filtered_series : pandas.Series
    """
    dt = series.index[1] - series.index[0]
    fs_nyquist = (1.0/dt) / 2.0
    bf, af = signal.butter(2, (200.0/fs_nyquist, 4000.0/fs_nyquist),
                           btype='bandpass')
    return pd.Series(signal.filtfilt(bf, af, series).astype(np.float32),
                     index=series.index)
