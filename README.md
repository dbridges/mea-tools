# MEA Tools

Tools for viewing, analyzing, and processing multi-electrode array data.

MEA Tools consists of three main components: (1) a Python module (pymea) and command line script for interacting with multi-electrode recordings, (2) a Python GUI application for high performance visualization of raw analog recordings and spike raster data, and (3) a Mathematica library for manipulating and analyzing analog and spike data.

## Requirements

MEA Tools requires:

- Python 3.4
- [vispy](http://www.vispy.org) 0.4.0
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

The MEA Viewer application allows you to seemlesly move between spike data and analog data. As you switch views the time window of displayed data is maintained for easy comparison.

##### Raster

Show an interactive raster plot of spike data:

```shell
mea view 2015-03-20_I9119.csv
```

Use the mouse wheel to zoom in and out of time. Sort the raster rows by average firing rate or by latency from the left edge of the window. The latency sorting allows you to easily recognize repeating patterns of activity.

##### Flashing Spike

Once you have opened a spike file, you can change the view from the raster plot to a flashing spike display. This display slows down time, allowing you to view the patterns of activity. Each spike creates a flash at its electrode, which slowly fades out. First scroll the raster view so that the time of interest is on the far left edge. Then when you switch to the flashing spike view it will begin playing from this time.

Click inside the view window, then press Space to start and stop playing. The current time is displayed in the status bar.

##### Analog

Interactively view an analog file:

```shell
mea view 2015-03-20_I9119.h5
```

This brings up a grid showing the analog data for all of the channels. From here you can drag left and right to move through the data file. You can compare multiple electrodes by Shift-clicking on them, then press Enter to bring up the analog comparison view. From here you can right click and drag to measure time differences. You can also overlay spike data to verify the accuracy of spike detection and sorting.

##### Conduction

The conduction view detects conduction signals from two electrodes, then extracts the analog data in a user selectable time window for each electrode. This data is aggregated, aligned, and superimposed to give a view of the signals read by each electrode for this one neuron.

The easiest way to use the conduction view is by starting in the analog grid view, then select two channels and press 'C'. The first channel is used as the index, and the second is used as a test by seeing if a spike occurs within 0.7 ms for each spike in the first channel.

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

Full usage:

```shell
usage: mea detect [-h] [--amplitude AMPLITUDE] [--neg-only]
                  [--no-sort]
                  FILES [FILES ...]

positional arguments:
  FILES                 Files to convert.

optional arguments:
  -h, --help            show this help message and exit
  --amplitude AMPLITUDE
                        Amplitude threshold in std devs.
  --neg-only            Only detect negative amplitudes.
  --no-sort             Do not sort spikes after detection.
```

#### export_cond

Export conduction traces.

```shell
$ mea export_cond 2014-10-30_I9119_Stimulate_D3.h5 'h8,g8,g7'
```

Uses h8 as the source electrode and finds all signals in g8 that occur within +- 0.7 ms. Once signals are found, it exports the raw analog data for a 5 ms window for each electrode given in the list. The datafiles are automatically labeled in accordance to the source file and the electrode.

## Mathematica Tools

A Mathematica library is also included to analyze analog and spike data, as well as to create useful static visualizations. See `mathematica/MEA_Examples.nb` for more information.
