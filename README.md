# MEA Tools

Tools for viewing, analyzing, and processing multi-electrode array data.

MEA Tools consists of five main components including:

- `pymea`, a Python module for interacting with multi-electrode data within Python. 
- A command line script for spike detection/spike sorting.
- [MEA Viewer](#mea-viewer), a Python application for high performance visualization of raw analog recordings and spike raster data.
- [MEA Tools](#mea-tools-gui), a Python application which provides an easy to use interface to the spike detection/spike sorting routines provided by the command line script.
- A Mathematica library for manipulating and analyzing analog and spike data.

## Installation

#### Windows

Windows users can [download the Windows executable](http://mea-tools.s3.amazonaws.com/MEA_Tools.zip) containing pre-compiled versions of the GUI programs [MEA Viewer](#mea-viewer) and [MEA Tools](#mea-tools-gui). This executable comes with Python and all required libraries, so no additional dependencies are needed. After downloading, unzip the folder to a suitable location. `MEAViewer.exe` provides an interface to the MEA Viewer application, `MEATools.exe` provides an interface to the spike detection and sorting routines. If you need access to the pymea Python module please follow the instructions for a Full Installation. 

#### Full Installation

Follow these instructions if you want access to the pymea module for your own Python programming, or if you are on Mac or Linux.

MEA Tools requires:

- Python 3.6
- [vispy](http://www.vispy.org) 0.5.2
- PyQt5
- numpy
- scipy
- pandas
- PyOpenGL

Typically it is easiest to install Anaconda Python 3.6 to obtain these packages.

Clone the repository to a suitable directory:

```shell
cd ~
git clone https://github.com/dbridges/mea-tools.git
```

Add the following to your shell startup file (~/.bash_profile on Mac or ~/.bashrc on Linux):

```shell
alias mea='python3 ~/mea-tools/mea-runner.py'
export PYTHONPATH=$PYTHONPATH:$HOME/mea-tools
```

Be sure to build the Cython components:

```shell
cd ~/mea-tools
make
```

### pymea

The core of the package is a Python 3 module, pymea, which has many components for interacting with data acquired by MultiChannel Systems software. Data files must be converted to HDF5 files using [MultiChannel Systems Data Manager](http://www.multichannelsystems.com/software/multi-channel-datamanager) before they can be viewed or analyzed with pymea.

### MEA Viewer

![alt tag](http://mea-tools.s3.amazonaws.com/mea_viewer.png)

The MEA Viewer application allows you to seamlessly interact with both spike data and analog data. As you switch views the time window of displayed data is maintained for easy comparison. Interactive visualizations of raw analog data, spike time stamp data, and interfaces to averaging repeated events to reveal a neuron's propagation signal across an array are provided.

A [video demo](https://vimeo.com/143168058) of MEA Viewer is available.

#### Analog Grid View

The analog grid view displays an overview of analog data for all channels.

- `Drag` to pan through the dataset.
- `Double-Click` on a channel to view it in the analog comparison view.
- `Shift-Click` to select multiple electrodes, then press `Enter` to display them in the analog comparison view.
- `p` to switch to the propagation signal view after selecting two channels.  See the section on the [propagation signal](#propagation-signal-view) for more information on how these signals are defined and displayed.

#### Analog Comparison View

The analog comparison view displays a selected number (typically 1-5) of channels for comparison. A CSV file is required for display of detected spike markers.

- Detected spikes are color coded by sorting group. Spike markers in black indicate a failure in spike sorting for those events. 
- `Drag` to pan through the data record.
- `Scroll` to adjust the time scale.
- `Shift-Drag` to measure time differences between points in the data record.
- `Right Click` on a spike to display a menu allowing you to select a sorted spike group for use in the propagation signal view.
- `Esc` to exit back to the analog grid view.
- `Double-Click` to exit back to the analog grid view.
- `b` to toggle the background color between gray and white.

#### Propagation Signal View

The propagation signal view superimposes and averages multiple spiking events occurring from a single neuron. A single neuron is defined either as a sorted group from a specific electrode, or by selecting two electrodes in the analog grid view and finding coincident (delta t < 0.7 ms) spiking events. A CSV file is required for the propagation view to work correctly.

- `Scroll` to adjust the vertical scale.
- `Shift-Scroll` to adjust the temporal scale.
- `Shift-Drag` to measure time differences between waveforms. You can `Shift-Drag` from one point in one waveform to another point in a completely different waveform and the program will keep track of the correct time offsets.
- `Esc` to exit back to the view you came from.

#### Raster View

The raster view displays an interactive raster plot of the spike time stamp data. A CSV file is required for the raster view to work.

- `Drag` to pan data set.
- `Scroll` to adjust time scale.
- `Shift-Drag` to measure time differences.
- `Shift-Click` to select specific rows, then press `Enter` to display only those rows.

#### Flashing Spike View

The flashing spike view displays a spatial-temporal visualization of spiking activity in an array.

- `Space` to play/pause the animation. You may have to click inside the main window before these and other commands work.
- `Left-Arrow` to jump back in time briefly.

### MEA Tools GUI

![alt tag](http://mea-tools.s3.amazonaws.com/mea_tools.png)

The MEA Tools GUI provides an interface to basic spike detection, sorting, and tagging of redundant signals attributed to propagation along an axon. Select a directory, then within the application select the files you wish to run spike detection analysis on. CSV files will be generated and saved in the same directory as the source files.

### MEA Script Commands

#### view

Open a data file for viewing in the MEA Viewer application. MEA Viewer displays analog and spike data in an interactive application. Input data files should have a `*.h5` or `*.csv` file extension. All CSV files should be built with the `detect` command listed below or with the MEA Tools GUI interface.

Show an interactive raster plot of spike data:

```shell
mea view 2015-03-20_I9119.csv
```

Interactively view an analog file:

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

Uses h8 as the source electrode and finds all signals in g8 that occur within +- 0.7 ms. Once signals are found, it exports the raw analog data for a 5 ms window for each electrode given in the list. The data files are automatically labeled in accordance to the source file and the electrode.

## Mac Extras

Two applications in the `extras/mac` folder provide graphical user interfaces to the MEA Tools package. Copy them to your `/Applications` folder. To work correctly MEA Tools must be installed at `~/mea-tools`

## Mathematica Tools

A Mathematica library is also included to analyze analog and spike data, as well as to create useful static visualizations. See `mathematica/MEA_Examples.nb` for more information.

## Sample Data

Sample MEA data to verify correct operation of the various components is available [here](http://mea-tools.s3.amazonaws.com/MEAViewerSampleData.zip).
