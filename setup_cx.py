# This script is used to build the Windows binary.
# You will manually have to copy libifcoremd.dll and libmmd.dll
# to the build directory.


import cx_Freeze


def load_h5py(finder, module):
    """h5py module has a number of implicit imports"""
    finder.IncludeModule('h5py.defs')
    finder.IncludeModule('h5py.utils')
    finder.IncludeModule('h5py._proxy')
    try:
        finder.IncludeModule('h5py._errors')
        finder.IncludeModule('h5py.h5ac')
    except:
        pass
    try:
        finder.IncludeModule('h5py.api_gen')
    except:
        pass

cx_Freeze.hooks.load_h5py = load_h5py

from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = {'packages': ['scipy',
                             'scipy.special._ufuncs_cxx',
                             'scipy.sparse.csgraph._validation'],
                'includes': ['sklearn.utils.lgamma',
                             'sklearn.utils.weight_vector',
                             'OpenGL.platform.win32',
                             'sklearn.neighbors.typedefs',
                             'sklearn.preprocessing',
                             'sklearn.decomposition',
                             'vispy.app.backends._pyqt4',
                             'sklearn.utils.sparsetools._graph_validation']}

import sys
base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('viewer-runner.py', base=base, targetName='MEAViewer.exe'),
    Executable('mea_tools_runner.py', base=base, targetName='MEATools.exe')
]

setup(name='MEA Viewer',
      version='1.0',
      description='MEA Viewer',
      options=dict(build_exe=buildOptions),
      executables=executables)
