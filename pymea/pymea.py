#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

input_dir = os.path.expanduser(
    '~/Dropbox/Hansma/ncp/IA6787/2014_08_20_Baseline')


def limit(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


class MEARecording:
    def __init__(self, store):
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

    def get(self, channels, start_time=0, end_time=None):
        if channels == 'all':
            channels = list(self.lookup.keys())
        channels.sort(key=lambda s: self.lookup[s])
        if end_time is None:
            end_time = self.duration
        start_i = int(limit(start_time * self.sample_rate,
                            0, self.data_len - 1))
        end_i = int(limit(end_time * self.sample_rate, 0, self.data_len))
        rows = [self.lookup[channel] for channel in channels]
        data = (self.conv *
                self.electrode_data[rows, start_i:end_i]
                .astype(np.float32).transpose())
        return pd.DataFrame(data,
                            index=np.arange(start_i, end_i)/self.sample_rate,
                            columns=channels, dtype=np.float32)

    def __getattr__(self, key):
        return self[key]

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


def coordinates_for_electrode(tag):
    tag = tag.lower()
    cols = {'a':  0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
            'h': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11}
    return (cols[tag[0]], 12 - int(tag[1:]))


def condense_spikes(srcdir, fname):
    with open(fname, 'a') as dest:
        dest.write('electrode,time\n')
        for f in os.listdir(srcdir):
            with open(os.path.join(srcdir, f)) as src:
                label = f[:-4].split('_')[-1]
                for line in src:
                    if len(line) > 0 and line[0].isdigit():
                        dest.write('%s,%s' % (label, line))


def raster_plot(df):
    """
    Dataframe has format (electrode, time):
        a9, 0.05
        h12, 0.09
    """
    plt.figure()
    plt.grid(False)
    plt.gca().tick_params(axis='y', which='major', labelsize=8)
    plt.gca().set_rasterized(True)

    ticks = []
    for i, e in enumerate(df.electrode.value_counts(ascending=True).keys()):
        plt.vlines(df[df.electrode == e].time, i - 0.5, i + 0.5)
        ticks.append(e)

    plt.yticks(np.arange(len(ticks)), ticks)
    plt.ylim(-0.5, len(ticks) - 0.5)


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
        Sampling frequency
    """
    d = np.fromfile(fname, np.uint16)
    d = d.reshape((-1, no_channels))
    d = pd.DataFrame((d - 32768.0) * cal,
                     columns=columns,
                     index=np.arange(0, len(d)/fs, 1/fs))
    return d
