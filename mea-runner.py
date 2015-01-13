#!/usr/bin/env python3

import os
import sys

import pymea as mea
import pymea.ui.viewer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No option specified.')
        sys.exit()
    elif sys.argv[1] == 'view':
        pymea.ui.viewer.run(sys.argv[2])
    elif sys.argv[1] == 'export_spikes':
        files = [f for f in os.listdir() if f.endswith('h5')]
        for i, f in enumerate(files):
            mea.export_peaks(f)
            print('%d of %d exported.' % (i, len(files)))
