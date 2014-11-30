#!/usr/bin/env python3

import sys
import pymea.interactive

if __name__ == '__main__':
    if sys.argv[1] == 'view':
        pymea.interactive.run(sys.argv[2])
