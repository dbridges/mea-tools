#!/usr/bin/env python3

import os
import time
import itertools
import warnings

import numpy as np
import pandas as pd
import h5py

from scipy import signal
from scipy import stats
from scipy import interpolate

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import peak_local_max

from . import util
from . import optics
from . import mea_cython

__all__ = ['MEARecording', 'MEASpikeDict', 'coordinates_for_electrode',
           'tag_for_electrode', 'condense_spikes', 'bandpass_filter',
           'export_spikes', 'tag_conductance_spikes', 'cofiring_events',
           'choose_keep_electrode', 'extract_waveforms',
           'export_conduction_waveforms']


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
        self.spikes = []
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

    def __del__(self):
        try:
            self.close()
        except (ValueError, AttributeError):
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


class MEASpikeDict():
    """
    An immuatable data structure for manipulating spike data.

    Parameters
    ----------
    spike_table : DataFrame
        DataFrame as given by MEARecording.detect_spikes() or by reading
        a csv generated using pymea.export_spikes()


    MEASpikeDict acts as a dict in some situations, or a list in others. It
    maintains order and can be sorted by acting on its values.

    Examples:

    >>> spikes = MEASpikeDict(spike_table)

    >>> spikes['e8']

        electrode      time  amplitude  threshold
    158        e8   8.23285 -23.280405 -18.782986
    159        e8   8.53015 -20.926035 -18.782986
    160        e8   9.22945 -20.225355 -18.782986
    161        e8  10.17610 -19.328444 -18.782986
    162        e8  10.37980 -23.685604 -18.782986
    163        e8  10.91855 -19.250252 -18.782986
    164        e8  10.99360 -19.138470 -18.782986
    165        e8  11.18440 -20.593740 -18.782986
    166        e8  11.47700 -20.357075 -18.782986

    [9 rows x 4 columns]

    Iterate through keys:
    >>> for electrode in spikes:

    Iterate through keys and values:
    >>> for electrode, data in spikes.items()

    Sort by firing rate:
    >>> spikes.sort()
    """
    def __init__(self, spike_table):
        self.spike_table = spike_table
        self.spike_dict = {}
        self.spike_order = []
        for (tag, data) in self.spike_table.groupby('electrode'):
            self.spike_dict[tag] = data
            self.spike_order.append(tag)

    def __getitem__(self, key):
        try:
            if type(key) is int:
                return self.spike_dict[self.spike_order[key]]
            else:
                return self.spike_dict[key]
        except KeyError:
            return pd.DataFrame(columns=self.spike_table.columns)

    def __len__(self):
        return len(self.spike_order)

    def __iter__(self):
        for tag in self.spike_order:
            yield tag

    def __reversed__(self):
        for tag in reversed(self.spike_order):
            yield tag

    def items(self):
        for tag in self.spike_order:
            yield tag, self.spike_dict[tag]

    def keys(self):
        return self.spike_order

    def max_time(self):
        return self.spike_table['time'].max()

    def sort(self, key=None, reverse=True):
        """
        Sorts the electrodes by order given with key.

        Parameters
        ----------
        key : callable
            A callable which takes a single argument, the DataFrame of a
            single electrode, and returns a value to be used for sorting.

        reverse : boolean
            If True the order is reversed.
        """
        if key is None:
            self.spike_order.sort(key=lambda e: len(self.spike_dict[e]),
                                  reverse=reverse)
        else:
            self.spike_order.sort(key=lambda e: key(self.spike_dict[e]),
                                  reverse=reverse)


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


################################
# Spike detection, sorting and exporting
################################


def export_spikes(fname, amp=6.0, sort=True, conductance=True, neg_only=False):
    """
    Detect, sort, and export spikes to a csv file.

    Parameters
    ----------
    fname : str
        The source HDF5 file.
    amp : float
        This sets the amplitude threshold used to detect spikes, as a
        factor of the median rms noise level.
    sort : bool
        True to sort data after spike detection.
    conductance : bool
        True to find conductance signals. Sorting should be done first to
        reliably detect conduction signals.
    neg_only : bool
        True to only detect negative amplitude peaks.
    """
    fname = os.path.expanduser(fname)
    print('Loading analog data...', end='', flush=True)
    rec = MEARecording(fname)
    analog_data = rec.get('all')
    print('done.', flush=True)

    print('Detecting spikes...', end='', flush=True)
    spikes = detect_spikes(analog_data, amp, neg_only)
    print('done.', flush=True)

    if sort:
        print('Sorting spikes...', end='', flush=True)
        sort_spikes(spikes, analog_data)
        print('done.', flush=True)

    spikes['conductance'] = False

    if conductance:
        print('Detecting conduction signals...', end='', flush=True)
        tag_conductance_spikes(spikes)
        print('done.', flush=True)

    spikes.to_csv(fname[:-3] + '.csv', index=False)


def detect_spikes(analog_data, amp=6.0, neg_only=False):
    """
    Runs basic spike detection using threshold. Detects both positive and
    negative peaks. See mea_cython.find_series_peaks for more information
    on the detection algorithm.

    Parameters
    ----------
    analog_data : MEARecording
        The analog recording to detect spikes in.
    amp : float
        This sets the amplitude threshold used to detect spikes, as a
        factor of the median rms noise level.
    neg_only : bool
        True to only detect negative amplitude peaks.
    """
    peaks = []
    for electrode in analog_data.keys():
        p = mea_cython.find_series_peaks(analog_data[electrode], amp, neg_only)
        p.insert(0, 'electrode', electrode)
        peaks.append(p)
    peaks = pd.concat(peaks, ignore_index=True)
    return peaks.convert_objects(convert_numeric=True)


def extract_waveforms(series, times, window_len=0.003,
                      upsample=5, smoothing=0):
    """
    Extract waveform data from a series for the given times.

    Parameters
    ----------
    series : pandas.series
        The series containing the analog data.
    times : list
        A list of times to extract waveforms from.
    window_len : float
        The width of the window to extract in seconds, centered on each time
        given in times.
    upsample : int
        The factor to upsample the data if desired.
    smoothing : float
        A smoothing factor to be used during upsampling and performed by
        scipy.interolate.splrep

    Returns
    -------
    waveforms : np.array
        An array where element i is the waveform data for spike i.
    """
    dt = series.index[1] - series.index[0]
    span = int(window_len / 2 / dt)
    expected_length = 2*span
    extracted = []
    for t in times:
        i = int(t / dt)
        y = series.iloc[i - span:i + span].values
        if len(y) < expected_length:
            y = np.zeros(expected_length)
        x = np.arange(len(y))
        xnew = np.linspace(0, len(y), len(y) * upsample)
        tck = interpolate.splrep(x, y, s=smoothing)
        interped = interpolate.splev(xnew, tck, der=0)
        extracted.append(interped)
    return np.array(extracted)


def sort_spikes(dataframe, analog_data, standardize=False):
    """
    Sorts spikes in dataframe for the given analog_data in place. Spikes are
    sorted by the first two principal components after the waveforms have been
    smoothed and up-sampled. Cluster analysis is done using the OPTICS density
    based clustering algorithm. An appropriate epsilon is found by looking for
    significant peaks in the reachability plot.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame of spike data.
    analog_data : MEARecording
        The MEARecording for the spikes given in dataframe.
    standardize : bool
        If True, standardize data before cluster finding.

    """
    for (tag, sdf) in dataframe.groupby('electrode'):
        waveforms = extract_waveforms(
            bandpass_filter(analog_data[tag]), sdf.time.values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            pcs = PCA(n_components=2).fit_transform(waveforms)
            if standardize:
                pcs = StandardScaler().fit_transform(pcs)

        opt = optics.OPTICS(300, 5)
        opt.fit(pcs)

        reach = opt._reachability[opt._ordered_list]
        rprime = reach[np.isfinite(reach)]
        if len(rprime) < 2:
            continue
        thresh = 8.5*stats.tstd(rprime,
                                (np.percentile(rprime, 15),
                                 np.percentile(rprime, 85))) + np.median(rprime)  # noqa
        peaks = peak_local_max(reach, min_distance=4,
                               threshold_abs=thresh,
                               threshold_rel=0).flatten()
        # Find largest peak for close neighbors
        min_dist = 0.05 * len(reach)
        splits = np.where(np.diff(peaks) > min_dist)[0] + 1
        peak_vals = [np.max(x) for x in np.split(reach[peaks], splits)
                     if len(x) > 0]
        try:
            eps = 0.90*np.min(peak_vals)
        except:
            eps = 0.5*reach[-1]

        opt.extract(eps)

        dataframe.loc[sdf.index, 'electrode'] = \
            sdf.electrode.str.cat(opt.labels_.astype(str), sep='.')


def condense_spikes(srcdir, fname):
    """
    Condenses spikes generated with MC_Rack spike detection, 1
    channel per file, to be in the same format as detect_spikes
    gives.
    """
    with open(fname, 'a') as dest:
        dest.write('electrode,time\n')
        for f in os.listdir(srcdir):
            with open(os.path.join(srcdir, f)) as src:
                label = f[:-4].split('_')[-1]
                for line in src:
                    if len(line) > 0 and line[0].isdigit():
                        dest.write('%s,%s' % (label, line))


################################
# Conductance signal detection
################################


def cofiring_events(dataframe, min_sep=0.0005):
    """
    Parameters
    ----------
    dataframe : pandas DataFrame
        DataFrame of spike data.

    min_sep : float
        Minimum separation time between two events for
        them to be considered cofiring.

    Returns
    -------
    dataframe : pandas DataFrame
        A filtered version of dataframe which includes only events
        separated by less than min_sep.
    """
    electrode_count = len(dataframe.electrode.unique())
    sub_df = dataframe.sort('time')
    splits = np.concatenate([
        [0],
        np.where(np.diff(sub_df.time.values) > min_sep)[0] + 1,
        [len(sub_df)]])

    events = []
    for i in range(len(splits) - 1):
        if splits[i+1] - splits[i] == electrode_count:
            events.append(sub_df[splits[i]:splits[i]+electrode_count])
    return events


def choose_keep_electrode(dataframe):
    """
    Chooses the electrode with the highest average amplitude among
    cofiring events between electrodes. If multiple
    electrodes have amplitudes within 20% choose the first one alphabetically.

    Parameters
    ----------
    dataframe : pandas DataFrmae
        DataFrame of spike data, should only contain cofiring events.

    Returns
    -------
    keep_electrode : str
        The name of the electrode that should be kept when removing
        conductance signals.

    """
    amplitudes = dataframe.groupby('electrode').amplitude.mean().abs()
    big_amplitudes = amplitudes[amplitudes > 0.7 * amplitudes.max()]
    return sorted(list(big_amplitudes.index))[0]


def tag_conductance_spikes(df):
    """
    Tags conduction spikes in dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame of spike data.
    """
    spikes = MEASpikeDict(df)
    tags = [tag for tag in spikes if not tag.startswith('analog')]

    conductance_locs = []

    for e1, e2 in itertools.combinations(tags, 2):
        if len(spikes[e1]) < 15 or len(spikes[e2]) < 15:
            continue
        # Find events that cofire within 1.2ms
        sub_df = pd.concat([spikes[e1], spikes[e2]])
        try:
            cofiring = pd.concat(
                [event.sort('electrode') for event
                 in cofiring_events(sub_df, 0.0012)])
            diffs = cofiring.time.diff()
        except:
            continue
        if len(cofiring) < 24:
            continue
        cofiring_rate = len(cofiring)/2/min(len(spikes[e1]), len(spikes[e2]))
        cofiring_std = 1000*diffs[diffs < 0.0012].std()
        if cofiring_std < 0.25 or (cofiring_std < 0.5 and cofiring_rate > 0.6):
            # likely conductance sequence
            e_keep = choose_keep_electrode(cofiring)
            conductance_locs.extend(
                list(cofiring[cofiring.electrode != e_keep].index.values))

    df['conductance'] = False
    df.ix[conductance_locs, 'conductance'] = True


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


def bandpass_filter(series, low=200.0, high=4000.0):
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


def extract_conduction_windows(keys, spikes, rec, window=0.005):
    lead = keys[0]
    test = keys[1]

    try:
        if keys[2] == 'all':
            keys = [keys[0], keys[1]]
            rest = list(rec.lookup.keys())
            rest.remove(keys[0])
            rest.remove(keys[1])
            keys.extend(rest)
    except:
        pass

    times = []
    analog_data = rec.get(keys)
    waveforms = {}
    for t in spikes[lead].time:
        if len(spikes[test][(spikes[test]['time'] > t - 0.0007) &
                            (spikes[test]['time'] < t + 0.0007)]) > 0:
            times.append(t)
    for key in keys:
        waveforms[key] = extract_waveforms(analog_data[key],
                                           times,
                                           window_len=window,
                                           upsample=1)
    return waveforms


def export_waveforms(fname, waveforms):
    fname = fname.split('.')[0]
    for key, value in waveforms.items():
        np.savetxt(fname + '_' + key + '.csv', value, delimiter=',')


def export_conduction_waveforms(keys, spike_file, rec_file, window=0.005):
    rec = MEARecording(rec_file)
    basename = os.path.basename(rec_file)[:-3]
    output_dir = os.path.join(os.path.dirname(rec_file),
                              '%s_%s_%s_conduction' % (basename,
                                                       keys[0],
                                                       keys[1]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prespikes = pd.read_csv(spike_file)
    prespikes.electrode = prespikes.electrode.str.split('.').str.get(0)
    spikes = MEASpikeDict(prespikes)
    waveforms = extract_conduction_windows(keys, spikes, rec)
    export_waveforms(os.path.join(output_dir, basename + '_cond'),
                     waveforms)
