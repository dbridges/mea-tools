#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
from scipy import signal
import h5py

from . import util
from . import mea_cython

__all__ = ['MEARecording', 'coordinates_for_electrode', 'tag_for_electrode',
           'condense_spikes', 'filter', 'export_peaks']

input_dir = os.path.expanduser(
    '~/Dropbox/Hansma/ncp/IA6787/2014_08_20_Baseline')


class MEARecording:
    """
    Main interface to an MEA recording.

    Parameters
    ----------
        store_path : str
            The path to the MEA recording, should be an HDF5 file.
    """
    def __init__(self, store_path):
        store_path = os.path.expanduser(store_path)
        if not os.path.exists(store_path):
            raise IOError('File not found.')
        self.store_path = store_path
        self.store = h5py.File(store_path)
        info = self.store[
            '/Data/Recording_0/AnalogStream/Stream_0/InfoChannel']
        self.sample_rate = 1e6/info['Tick'][0]
        self.conv = 1e-3 * info['ConversionFactor'][0]
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
        # Get 1st channel analog data if it exists:
        if 'Data/Recording_0/AnalogStream/Stream_1' in self.store:
            info = self.store[
                '/Data/Recording_0/AnalogStream/Stream_1/InfoChannel']
            self.analog_conv = 1e-3 * info['ConversionFactor'][0]
            self.analog_data = self.store[
                'Data/Recording_0/AnalogStream/Stream_1/ChannelData']
            self.analog_channels = [channel.decode().replace('A', 'analog')
                                    for channel in info['Label']]
        else:
            self.analog_data = None
            self.analog_channels = []

    def get(self, channels, start_time=0, end_time=None):
        """
        Returns a pandas DataFrame of the requested data.

        Parameters
        ----------
            channels : list of str or 'all'
                The requested channels, ['a4', 'b5'], or 'all' for all
                channels. Use 'analog1' to 'analog8' to access analog
                input channels.
            start_time : float
                The start time of the data requested in seconds. Defaults
                to 0.
            end_time : float
                The end time of the data requested in seconds. Defaults
                to the time of the last data point.
        """
        if channels == 'all':
            output_channels = list(self.lookup.keys())
        else:
            output_channels = [channel for channel in channels
                               if channel in list(self.lookup.keys())]
        output_channels.sort(key=lambda s: self.lookup[s])
        if end_time is None:
            end_time = self.duration

        # Find start and end indices.
        start_i = int(util.clip(start_time * self.sample_rate,
                                0, self.data_len - 1))
        end_i = int(util.clip(end_time * self.sample_rate, 0, self.data_len))
        rows = [self.lookup[channel] for channel in output_channels]
        data = (self.conv *
                self.electrode_data[rows, start_i:end_i]
                .astype(np.float32).transpose())

        # Load analog data if requested.
        if channels == 'all':
            analog_channels = self.analog_channels
        else:
            analog_channels = [channel for channel in channels
                               if channel in self.analog_channels]
        if self.analog_data is not None and len(analog_channels) > 0:
            analog_data = (self.analog_conv *
                           self.analog_data[
                               [int(channel[-1]) - 1
                                for channel in analog_channels],
                               start_i:end_i]
                           .astype(np.float32).transpose())
            data = np.concatenate((data, analog_data), axis=1)
            output_channels.extend(analog_channels)

        return pd.DataFrame(data,
                            index=np.arange(start_i, end_i)/self.sample_rate,
                            columns=output_channels, dtype=np.float32)

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
    """
    Returns MEA coordinates for electrode label.

    Parameters
    ----------
        tag : str
            The electrode label, i.e. A8 or C6

    Returns
    -------
        coordinates : tuple
            A tuple of length 2 giving the x and y coordinate of
            that electrode.
    """
    tag = tag.lower()
    if tag.startswith('analog'):
        if tag.endswith('1'):
            return (0, 0)
        elif tag.endswith('2'):
            return (1, 0)
        elif tag.endswith('3'):
            return (10, 0)
        elif tag.endswith('4'):
            return (11, 0)
        elif tag.endswith('5'):
            return (0, 11)
        elif tag.endswith('6'):
            return (1, 11)
        elif tag.endswith('7'):
            return (10, 11)
        elif tag.endswith('8'):
            return (11, 11)
    cols = {'a':  0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
            'h': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11}
    return (cols[tag[0]], int(tag[1:]) - 1)


def tag_for_electrode(coords):
    """
    Returns MEA tag for electrode coordinates.

    Parameters
    ----------
        coords : tuple
            The electrode coordinates.

    Returns
    -------
        tag : str
            The electrode name.
    """
    lookup = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
              8: 'j', 9: 'k', 10: 'l', 11: 'm'}
    tag = lookup[coords[0]] + str(coords[1])
    if tag == 'a1':
        return 'analog1'
    elif tag == 'b1':
        return 'analog2'
    elif tag == 'l1':
        return 'analog3'
    elif tag == 'm1':
        return 'analog4'
    elif tag == 'a12':
        return 'analog5'
    elif tag == 'b12':
        return 'analog6'
    elif tag == 'l12':
        return 'analog7'
    elif tag == 'm12':
        return 'analog8'
    else:
        return tag


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


def filter(series, low=200.0, high=4000.0):
    """
    Filters given series with a 2nd order bandpass filter with default
    cutoff frequencies of 200 Hz and 4 kHz.

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
    if low < 0.1:
        # Lowpass filter only.
        bf, af = signal.butter(2, high/fs_nyquist, btype='lowpass')
    elif high > 10000:
        # Highpass filter only.
        bf, af = signal.butter(2, low/fs_nyquist, btype='highpass')
    else:
        bf, af = signal.butter(2, (low/fs_nyquist, high/fs_nyquist),
                               btype='bandpass')
    return pd.Series(signal.filtfilt(bf, af, series).astype(np.float32),
                     index=series.index)
