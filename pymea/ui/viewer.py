import os
import sys
import platform

import pymea.pymea as mea
from pymea.ui.visualizations import (MEA120GridVisualization,
                                     RasterPlotVisualization,
                                     FlashingSpikeVisualization)

import pandas as pd
from vispy import app, gloo, visuals
import OpenGL.GL as gl
from PySide import QtGui, QtCore  # noqa
from .main_window import Ui_MainWindow


class VisualizationCanvas(app.Canvas):
    def __init__(self, controller):
        app.Canvas.__init__(self, keys='interactive', size=(1280, 768))
        self.controller = controller

        self.analog_visualization = None
        self.raster_visualization = None
        self.flashing_spike_visualization = None

        self.visualization = None

        self.tr_sys = visuals.transforms.TransformSystem(self)
        self._timer = app.Timer(1/30, connect=self.on_tick, start=True)

    def show_raster(self):
        if self.raster_visualization is None:
            if self.controller.spike_data is None:
                raise IOError('Spike data is unavailable.')
            else:
                self.raster_visualization = RasterPlotVisualization(
                    self, self.controller.spike_data)
        if self.visualization is not None:
            self.raster_visualization.t0 = self.visualization.t0
            self.raster_visualization.dt = self.visualization.dt
            self.visualization.on_hide()
        self.visualization = self.raster_visualization
        self.visualization.on_show()

    def show_flashing_spike(self):
        if self.flashing_spike_visualization is None:
            if self.controller.spike_data is None:
                raise IOError('Spike data is unavailable.')
            else:
                self.flashing_spike_visualization = FlashingSpikeVisualization(
                    self, self.controller.spike_data)
        if self.visualization is not None:
            self.flashing_spike_visualization.t0 = self.visualization.t0
            self.flashing_spike_visualization.dt = self.visualization.dt
            self.visualization.on_hide()
        self.visualization = self.flashing_spike_visualization
        self.visualization.on_show()

    def show_analog_grid(self):
        if self.analog_visualization is None:
            if self.controller.analog_data is None:
                raise IOError('Analog data is unavailable.')
            else:
                self.analog_visualization = MEA120GridVisualization(
                    self, self.controller.analog_data)
        if self.visualization is not None:
            self.analog_visualization.t0 = self.visualization.t0
            self.analog_visualization.dt = self.visualization.dt
            self.visualization.on_hide()
        self.analog_visualization.y_scale_index = \
            self.controller.analogGridScaleComboBox.currentIndex()
        self.visualization = self.analog_visualization
        self.visualization.on_show()

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x/(w/2.)-1., y/(h/2.)-1.

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, *event.size)
        self.tr_sys = visuals.transforms.TransformSystem(self)
        if self.visualization is not None:
            self.visualization.on_resize(event)

    def on_draw(self, event):
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        if self.visualization is not None:
            self.visualization.draw()

    def on_mouse_move(self, event):
        if self.visualization is not None:
            self.visualization.on_mouse_move(event)

    def on_mouse_wheel(self, event):
        if self.visualization is not None:
            self.visualization.on_mouse_wheel(event)

    def on_mouse_press(self, event):
        if self.visualization is not None:
            self.visualization.on_mouse_press(event)

    def on_mouse_release(self, event):
        if self.visualization is not None:
            self.visualization.on_mouse_release(event)

    def on_key_release(self, event):
        if self.visualization is not None:
            self.visualization.on_key_release(event)

    def on_tick(self, event):
        if self.visualization is not None:
            self.visualization.on_tick(event)
            self.controller.on_visualization_updated()
        self.update()


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    """
    Subclass of QMainWindow
    """
    def __init__(self, input_file, parent=None):
        super().__init__(parent)

        self.spike_data = None
        self.analog_data = None

        if input_file.endswith('.csv'):
            self.spike_file = input_file
            if os.path.exists(input_file[:-4] + '.h5'):
                self.analog_file = input_file[:-4] + '.h5'
            else:
                self.analog_file = None
            self.load_spike_data()
        elif input_file.endswith('.h5'):
            self.analog_file = input_file
            self.load_analog_data()
            if os.path.exists(input_file[:-3] + '.csv'):
                self.spike_file = input_file[:-3] + '.csv'
            else:
                self.spike_file = None
        else:
            raise IOError('Invalid input file, must be of type csv or h5.')

        # UI initialization
        self.setupUi(self)
        self.canvas = VisualizationCanvas(self)
        self.canvas.show_raster()

        self.canvas.native.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.mainLayout.addWidget(self.canvas.native)

        self.rasterRowCountSlider.setValue(
            self.canvas.raster_visualization.row_count)
        self.analogGridScaleComboBox.setCurrentIndex(4)

        self.flashingSpikeTimescaleComboBox.setCurrentIndex(4)

    def load_spike_data(self):
        self.spike_data = pd.read_csv(self.spike_file)

    def load_analog_data(self):
        store = mea.MEARecording(self.analog_file)
        self.analog_data = store.get('all')

    def on_visualization_updated(self):
        self.statusBar.t0 = self.canvas.visualization.t0
        self.statusBar.dt = self.canvas.visualization.dt
        self.statusBar.electrode = self.canvas.visualization.electrode

    @QtCore.Slot(int)
    def on_rasterRowCountSlider_valueChanged(self, val):
        self.canvas.raster_visualization.row_count = val

    @QtCore.Slot(str)
    def on_visualizationComboBox_currentIndexChanged(self, text):
        if text == 'Raster':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_raster()
        elif text == 'Flashing Spike':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_flashing_spike()
        elif text == 'Analog Grid':
            if self.analog_data is None:
                self.load_analog_data()
            self.canvas.show_analog_grid()

    @QtCore.Slot(int)
    def on_analogGridScaleComboBox_currentIndexChanged(self, index):
        if self.canvas.analog_visualization is not None:
            self.canvas.analog_visualization.y_scale_index = index

    @QtCore.Slot()
    def on_flashingSpikePlayButton_clicked(self):
        if (self.canvas.visualization is
                self.canvas.flashing_spike_visualization):
            if self.canvas.flashing_spike_visualization.paused:
                self.flashingSpikePlayButton.setText('Pause')
            else:
                self.flashingSpikePlayButton.setText('Play')
            self.canvas.flashing_spike_visualization.toggle_play()

    @QtCore.Slot(str)
    def on_flashingSpikeTimescaleComboBox_currentIndexChanged(self, text):
        if self.canvas.flashing_spike_visualization is None:
            return
        if text == '1x':
            self.canvas.flashing_spike_visualization.time_scale = 1
        elif text == '1/2x':
            self.canvas.flashing_spike_visualization.time_scale = 1/2
        elif text == '1/20x':
            self.canvas.flashing_spike_visualization.time_scale = 1/20
        elif text == '1/100x':
            self.canvas.flashing_spike_visualization.time_scale = 1/100
        elif text == '1/200x':
            self.canvas.flashing_spike_visualization.time_scale = 1/200
        elif text == '1/400x':
            self.canvas.flashing_spike_visualization.time_scale = 1/400
        elif text == '1/800x':
            self.canvas.flashing_spike_visualization.time_scale = 1/800
        elif text == '1/1600x':
            self.canvas.flashing_spike_visualization.time_scale = 1/1600

    @QtCore.Slot()
    def on_actionRaster_activated(self):
        self.visualizationComboBox.setCurrentIndex(0)

    @QtCore.Slot()
    def on_actionFlashingSpikes_activated(self):
        self.visualizationComboBox.setCurrentIndex(1)

    @QtCore.Slot()
    def on_actionAnalogGrid_activated(self):
        self.visualizationComboBox.setCurrentIndex(2)


def run(fname):
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow(fname)
    win.show()
    if platform.system() == 'Darwin':
        os.system('''osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')  # noqa
    appQt.exec_()
