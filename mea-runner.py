#!/usr/bin/env python3

import os
import sys
import argparse
import glob


def view(args):
    import pymea.ui.viewer
    pymea.ui.viewer.run(args.FILE)


def info(args):
    import pymea as mea
    if args.FILE.endswith('.h5'):
        if os.path.exists(args.FILE):
            store = mea.MEARecording(args.FILE)
            print(store)


def export_spikes(args):
    if len(args.FILES) == 1:
        files = [f for f in glob.glob(args.FILES[0])
                 if f.endswith('.h5') and os.path.exists(f)]
    else:
        files = [f for f in args.FILES
                 if f.endswith('.h5') and os.path.exists(f)]
    import pymea as mea
    for i, f in enumerate(files):
        mea.export_peaks(f)
        print('%d of %d exported.' % (i + 1, len(files)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mea')
    subparsers = parser.add_subparsers()

    parser_view = subparsers.add_parser('view',
                                        help='View a data file')
    parser_view.add_argument('FILE',
                             action='store',
                             help='File name or path.')
    parser_view.set_defaults(func=view)

    parser_info = subparsers.add_parser('info',
                                        help='Display file information.')
    parser_info.add_argument('FILE',
                             action='store',
                             help='File name or path.')
    parser_info.set_defaults(func=info)

    parser_export_spikes = subparsers.add_parser('export_spikes',
                                                 help='Detect spikes in h5 files.')
    parser_export_spikes.add_argument('FILES',
                                      help='Files to convert.',
                                      nargs='+')
    parser_export_spikes.set_defaults(func=export_spikes)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args.func(args)
