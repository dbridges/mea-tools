import pymea.ui.viewer
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.csv'):
            pymea.ui.viewer.run(None, sys.argv[1], 'raster')
        elif sys.argv[1].endswith('.h5'):
            pymea.ui.viewer.run(sys.argv[1], None, '')
        else:
            pymea.ui.viewer.run(None, None, '')
    else:
        pymea.ui.viewer.run(None, None, '')
