from setuptools import setup
from Cython.Build import cythonize

import numpy

long_descr = """`Full Documentation
<http://github.com/dbridges/mea-tools>`_.
"""

setup(
    name='pymea',
    version='0.1.1',
    author='Dan Bridges',
    license='GPL3',
    description='Tools for analyzing MEA data.',
    long_description=long_descr,
    url='http://github.com/dbridges/mea-tools',
    ext_modules=cythonize("pymea/mea_cython.pyx"),
    include_dirs=[numpy.get_include()], requires=['pandas', 'h5py', 'scipy',
                                                  'sklearn', 'vispy', 'PyQt5']
)
