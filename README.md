# MEA Tools

Tools for viewing, analyzing, and processing multi-electrode array data.

MEA Tools consists of three main components: (1) a Python module (pymea) and command line script for interacting with multi-electrode recordings, (2) a Python GUI application for high performance visualization of raw analog recordings and spike raster data, and (3) a Mathematica library for manipulating and analyzing analog and spike data.

## Requirements

MEA Tools requires:

- Python 3.4
- [vispy](http://www.vispy.org) (best to install this from source due to its rapid development)
- PyQt4
- numpy
- scipy
- pandas
- PyOpenGL

Typically it is easiest to install Anaconda Python 3.4 to obtain these packages.

## Installation

Clone to a suitable directory:

```shell
cd ~
git clone https://github.com/dbridges/mea-tools.git
```

Add the following to your shell startup file (~/.bash_profile on Mac or ~/.bashrc on Linux):

```shell
alias mea='python3 ~/mea-tools/mea-runner.py'
export PYTHONPATH=$PYTHONPATH:$HOME/mea-tools
```

## PyMEA

The core of the package is a Python 3 module, PyMEA, which has many components for interacting with data acquired by MultiChannel Systems software. Data files must be converted to HDF5 files using [MultiChannel Systems Data Manager]( http://www.multichannelsystems.com/software/multi-channel-datamanager) before they can be viewed or analyzed with PyMEA.

### MEA Script Commands

#### view

Open a data file for viewing in the MEA Viewer application. MEA Viewer displays analog and spike data in an interactive application. Input data files should have a `*.h5` or `*.csv` file extension. All csv files should be built with the `detect` command listed below.

Interactively view an analog file:

```shell
mea view 2015-03-20_I9119.h5
```

or show an interactive raster plot of spike data:

```shell
mea view 2015-03-20_I9119.h5
```

#### info

Display information about an analog recording.

```shell
$ mea info 2014-10-30_I9119_Stimulate_D3.h5

File:           2014-10-30_I9119_Stimulate_D3.h5
Date:           Thu, 30 Oct 2014 02:36:35 PM
MEA:            120MEA200/30iR
Sample Rate:    20000.0 Hz
Duration:       19.00 s
```

#### detect
Find spikes in input files and export their timestamps to a csv file. Output files have the same filename as the input file, but with a `.csv` extension.

Export one file:

```shell
mea detect 2015-03-20_I9119.h5
```

Export all files in directory:

```shell
mea detect *.h5
```

Export a file using a threshold cutoff of 5 times the standard deviation of the input file noise:

```shell
mea detect --amplitude=5 2015-03-20_I9119.h5
```

## Mathematica Tools

A Mathematica library is also included to analyze analog and spike data, as well as to create useful static visualizations. See `mathematica/MEA_Examples.nb` for more information.
