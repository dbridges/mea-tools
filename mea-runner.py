#!/usr/bin/env python3

import os
import sys
import argparse
import glob

import pandas as pd


def view(args):
    import pymea.ui.viewer

    if args.spikes is not None:
        if not os.path.exists(args.spikes):
            print('No such file or directory: %s.' % args.spikes)
            return

    if os.path.exists(args.FILE):
        spike_file = None
        analog_file = None
        if args.FILE.endswith('.csv'):
            spike_file = args.FILE
            if os.path.exists(args.FILE[:-4] + '.h5'):
                analog_file = args.FILE[:-4] + '.h5'
        elif args.FILE.endswith('.h5'):
            analog_file = args.FILE
            if args.spikes is not None:
                spike_file = args.spikes
            elif os.path.exists(args.FILE[:-3] + '.csv'):
                spike_file = args.FILE[:-3] + '.csv'
        else:
            raise IOError('Invalid input file, must be of type csv or h5.')

        if args.FILE.endswith('csv'):
            show = 'raster'
        else:
            show = 'analog'

        pymea.ui.viewer.run(analog_file, spike_file, show)
    else:
        print('No such file or directory.')


def info(args):
    import pymea as mea
    if args.FILE.endswith('.h5'):
        if os.path.exists(args.FILE):
            store = mea.MEARecording(args.FILE)
            print(store)


def detect_spikes(args):
    if len(args.FILES) == 1:
        files = [f for f in glob.glob(args.FILES[0])
                 if f.endswith('.h5') and os.path.exists(f)]
    else:
        files = [f for f in args.FILES
                 if f.endswith('.h5') and os.path.exists(f)]
    import pymea as mea
    for i, f in enumerate(files):
        mea.export_spikes(f, args.amplitude,
                          sort=args.sort,
                          conductance=args.sort,
                          neg_only=args.neg_only)
        print('%d of %d exported.' % (i + 1, len(files)))


def tag_cond(args):
    if len(args.FILES) < 2:
        print('Must specify src and dest files.')
        return

    src = args.FILES[-1]

    if len(args.FILES) == 2:
        files = [f for f in glob.glob(args.FILES[0])
                 if f.endswith('.csv') and os.path.exists(f)]
    else:
        files = [f for f in args.FILES[:-1] if f.endswith('.csv')]

    seqs = []
    with open(src, 'r') as f:
        for line in f:
            seqs.append([s.lower().strip() for s in line.split(',')])

    import pymea as mea

    for i, f in enumerate(files):
        conductance_locs = []
        df = pd.read_csv(f)
        spikes = mea.MEASpikeDict(df)
        for seq in seqs:
            e_keep = seq[0]
            sub_df = pd.concat([spikes[tag] for tag in seq])
            cofiring = pd.concat(mea.cofiring_events(sub_df, 0.0007))
            conductance_locs.extend(
                list(cofiring[cofiring.electrode != e_keep].index.values))
        df['conductance'] = False
        df.ix[conductance_locs, 'conductance'] = True
        df.to_csv(f, index=False)
        print('%d of %d exported.' % (i + 1, len(files)))


def main():
    parser = argparse.ArgumentParser(prog='mea')
    subparsers = parser.add_subparsers()

    parser_view = subparsers.add_parser('view',
                                        help='View a data file')
    parser_view.add_argument('FILE',
                             action='store',
                             help='File name or path.')
    parser_view.add_argument('--spikes',
                             type=str,
                             default=None,
                             help='File name or path for spike data.')
    parser_view.set_defaults(func=view)

    parser_info = subparsers.add_parser('info',
                                        help='Display file information.')
    parser_info.add_argument('FILE',
                             action='store',
                             help='File name or path.')
    parser_info.set_defaults(func=info)

    parser_detect_spikes = subparsers.add_parser(
        'detect', help='Detect spikes in h5 files.', aliases=['export_spikes'])
    parser_detect_spikes.add_argument('--amplitude',
                                      type=float,
                                      default=6.0,
                                      help='Amplitude threshold in std devs.')
    parser_detect_spikes.add_argument('--neg-only',
                                      dest='neg_only',
                                      action='store_true',
                                      help='Only detect negative amplitudes')  # noqa
    parser_detect_spikes.add_argument('--sort',
                                      dest='sort',
                                      action='store_true',
                                      help='Sort spikes after detection.')
    parser_detect_spikes.add_argument('--no-sort',
                                      dest='sort',
                                      action='store_false',
                                      help='Do not sort spikes after detection.')  # noqa
    parser_detect_spikes.add_argument('FILES',
                                      help='Files to convert.',
                                      nargs='+')
    parser_detect_spikes.set_defaults(sort=True, func=detect_spikes)

    parser_tag = subparsers.add_parser(
        'tag', help='Tag conductance traces using specified file.'
    )
    parser_tag.add_argument('FILES',
                            help='[src] [dest ...].',
                            nargs='+')
    parser_tag.set_defaults(sort=True, func=tag_cond)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args.func(args)

if __name__ == '__main__':
    main()
