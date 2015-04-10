# MEA Tools

Tools for viewing, analyzing, and processing multi-electrode array data.

The core of the package is a Python 3 module, pymea, which has many components for interacting with data acquired by MultiChannel Systems software. Data files must be converted to HDF5 files using [MultiChannel Systems Data Manager]( http://www.multichannelsystems.com/software/multi-channel-datamanager) before they can be viewed or analyzed with pymea.

This package also contains useful Mathematica functions for interacting with analog data and spike data.

## Command Line Usage

The MEA Tools package has a command line script `mea-runner.py` to interact with data files. It may be useful to add an alias to it in your bashrc file (i.e. add `alias mea='python3 ~/mea-tools/mea-runner.py'` to your appropriate shell startup file.).

### view

Open a data file for viewing in the MEA Viewer application. MEA Viewer displays analog and spike data in an interactive application. Input data files should have a `*.h5` or `*.csv` file extension. All csv files should be built with the export_spikes command below.

Interactively view an analog file:

```shell
mea view 2015-03-20_I9119.h5
```

or show an interactive raster plot of spike data:

```shell
mea view 2015-03-20_I9119.h5
```

### info

Display information about an analog recording.

```shell
$ mea info 2014-10-30_I9119_Stimulate_D3.h5

File:           2014-10-30_I9119_Stimulate_D3.h5
Date:           Thu, 30 Oct 2014 02:36:35 PM
MEA:            120MEA200/30iR
Sample Rate:    20000.0 Hz
Duration:       19.00 s
```

### export_spikes
Find spikes in input files and export their timestamps to a csv file. Output files have the same filename as the input file, but iwth a `.csv` extension.

Export one file:

```shell
mea export_spikes 2015-03-20_I9119.h5
```

Export all files in directory:

```shell
mea export_spikes *.h5
```

Export a file using a threshold cutoff of 5 times the standard deviation of the input file noise.:

```shell
mea export_spikes --amplitude=5 2015-03-20_I9119.h5
```
