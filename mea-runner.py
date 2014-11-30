#!/usr/bin/env python3

import sys
import pymea.analog_display

if __name__ == '__main__':
    if sys.argv[1] == 'view':
        pymea.analog_display.run(sys.argv[2])
