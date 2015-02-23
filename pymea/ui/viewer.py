import os
import sys
import platform
import time

import pymea.pymea as mea
from pymea.ui.visualizations import (MEA120GridVisualization,
                                     MEAAnalogVisualization,
                                     RasterPlotVisualization,
                                     FlashingSpikeVisualization)

import pandas as pd
from vispy import app, gloo, visuals
import OpenGL.GL as gl
from PySide import QtGui, QtCore  # noqa
from .main_window import Ui_MainWindow


class VisualizationCanvas(app.Canvas):
    def __init__(self, controller):
        app.Canvas.__init__(self, vsync=True)
        self.controller = controller

        self.analog_grid_vis = None
        self.analog_vis = None
        self.raster_vis = None
        self.flashing_spike_vis = None

        self.visualization = None

        self.tr_sys = visuals.transforms.TransformSystem(self)
        self._timer = app.Timer(1/30, connect=self.on_tick, start=True)

        self.mouse_pos = (0, 0)
        self.prev_mouse_pos = (0, 0)

        self.last_mouse_release_t = time.time()

    def show_raster(self):
        if self.raster_vis is None:
            if self.controller.spike_data is None:
                raise IOError('Spike data is unavailable.')
            else:
                self.raster_vis = RasterPlotVisualization(
                    self, self.controller.spike_data)
        if self.visualization is not None:
            self.raster_vis.t0 = self.visualization.t0
            self.raster_vis.dt = self.visualization.dt
            self.visualization.on_hide()
        self.visualization = self.raster_vis
        self.visualization.on_show()

    def show_flashing_spike(self):
        if self.flashing_spike_vis is None:
            if self.controller.spike_data is None:
                raise IOError('Spike data is unavailable.')
            else:
                self.flashing_spike_vis = FlashingSpikeVisualization(
                    self, self.controller.spike_data)
        if self.visualization is not None:
            self.flashing_spike_vis.t0 = self.visualization.t0
            self.flashing_spike_vis.dt = self.visualization.dt
            self.visualization.on_hide()
        self.visualization = self.flashing_spike_vis
        self.visualization.on_show()

    def show_analog_grid(self):
        if self.analog_grid_vis is None:
            if self.controller.analog_data is None:
                raise IOError('Analog data is unavailable.')
            else:
                self.analog_grid_vis = MEA120GridVisualization(
                    self, self.controller.analog_data)
        if self.visualization is not None:
            self.analog_grid_vis.t0 = self.visualization.t0
            self.analog_grid_vis.dt = self.visualization.dt
            self.visualization.on_hide()
        self.analog_grid_vis.y_scale_index = \
            self.controller.analogGridScaleComboBox.currentIndex()
        self.visualization = self.analog_grid_vis
        self.visualization.on_show()

    def show_analog(self):
        if self.analog_vis is None:
            if self.controller.analog_data is None:
                raise IOError('Analog data is unavailable.')
            else:
                self.analog_vis = MEAAnalogVisualization(
                    self, self.controller.analog_data)
        if self.visualization is not None:
            self.analog_vis.t0 = self.visualization.t0
            self.analog_vis.dt = self.visualization.dt
            self.visualization.on_hide()
        self.visualization = self.analog_vis
        self.analog_vis.electrodes = [s.lower() for s in
                                      self.analog_grid_vis.selected_electrodes]
        self.analog_vis.y_scale = self.analog_grid_vis.y_scale
        self.visualization.on_show()

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x/(w/2.)-1., y/(h/2.)-1.

    def enable_antialiasing(self):
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_FASTEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def disable_antialiasing(self):
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glDisable(gl.GL_BLEND)

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, *event.size)
        self.tr_sys = visuals.transforms.TransformSystem(self)
        if self.visualization is not None:
            self.visualization.on_resize(event)

    def on_draw(self, event):
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
        event_time = time.time()
        if event_time - self.last_mouse_release_t < 0.3:
            self.on_mouse_double_click(event)
        else:
            if self.visualization is not None:
                self.visualization.on_mouse_release(event)
        self.last_mouse_release_t = event_time

    def on_mouse_double_click(self, event):
        if self.visualization is not None:
            self.visualization.on_mouse_double_click(event)

    def on_key_release(self, event):
        if self.visualization is not None:
            self.visualization.on_key_release(event)

    def on_tick(self, event):
        mouse_pos = self.native.mapFromGlobal(self.native.cursor().pos())
        self.prev_mouse_pos = self.mouse_pos
        self.mouse_pos = (mouse_pos.x(), mouse_pos.y())

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
        if input_file.endswith('.csv'):
            self.canvas.show_raster()
        else:
            self.canvas.show_analog_grid()

        self.canvas.native.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.mainLayout.removeWidget(self.widget)
        self.mainLayout.addWidget(self.canvas.native)
        self.mainLayout.setStretchFactor(self.canvas.native, 1)

        self.rasterRowCountSlider.setValue(120)
        self.analogGridScaleComboBox.setCurrentIndex(4)

        self.flashingSpikeTimescaleComboBox.setCurrentIndex(4)

    def load_spike_data(self):
        self.spike_data = pd.read_csv(self.spike_file)

    def load_analog_data(self):
        store = mea.MEARecording(self.analog_file)
        self.analog_data = store.get('all')

    def on_visualization_updated(self):
        self.statusBar.dt = self.canvas.visualization.dt
        self.statusBar.electrode = self.canvas.visualization.electrode
        self.statusBar.mouse_t = self.canvas.visualization.mouse_t

    @QtCore.Slot(int)
    def on_rasterRowCountSlider_valueChanged(self, val):
        try:
            self.canvas.raster_vis.row_count = val
        except AttributeError:
            pass

    @QtCore.Slot(str)
    def on_visualizationComboBox_currentIndexChanged(self, text):
        if text == 'Raster':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_raster()
            self.rasterRowCountSlider.setValue(
                self.canvas.raster_vis.row_count)
        elif text == 'Flashing Spike':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_flashing_spike()
        elif text == 'Analog Grid':
            if self.analog_data is None:
                self.load_analog_data()
            self.canvas.show_analog_grid()

    @QtCore.Slot(str)
    def on_analogGridScaleComboBox_currentIndexChanged(self, text):
        try:
            scale = float(text.split(' ')[0])
        except:
            scale = 150
        if self.canvas.analog_grid_vis is not None:
            self.canvas.analog_grid_vis.y_scale = scale
        if self.canvas.analog_vis is not None:
            self.canvas.analog_vis.y_scale = scale

    @QtCore.Slot()
    def on_flashingSpikePlayButton_clicked(self):
        if (self.canvas.visualization is
                self.canvas.flashing_spike_vis):
            if self.canvas.flashing_spike_vis.paused:
                self.flashingSpikePlayButton.setText('Pause')
            else:
                self.flashingSpikePlayButton.setText('Play')
            self.canvas.flashing_spike_vis.toggle_play()

    @QtCore.Slot(str)
    def on_flashingSpikeTimescaleComboBox_currentIndexChanged(self, text):
        if self.canvas.flashing_spike_vis is None:
            return
        if text == '1x':
            self.canvas.flashing_spike_vis.time_scale = 1
        elif text == '1/2x':
            self.canvas.flashing_spike_vis.time_scale = 1/2
        elif text == '1/20x':
            self.canvas.flashing_spike_vis.time_scale = 1/20
        elif text == '1/100x':
            self.canvas.flashing_spike_vis.time_scale = 1/100
        elif text == '1/200x':
            self.canvas.flashing_spike_vis.time_scale = 1/200
        elif text == '1/400x':
            self.canvas.flashing_spike_vis.time_scale = 1/400
        elif text == '1/800x':
            self.canvas.flashing_spike_vis.time_scale = 1/800
        elif text == '1/1600x':
            self.canvas.flashing_spike_vis.time_scale = 1/1600

    @QtCore.Slot()
    def on_actionRaster_activated(self):
        self.visualizationComboBox.setCurrentIndex(0)

    @QtCore.Slot()
    def on_actionFlashingSpikes_activated(self):
        self.visualizationComboBox.setCurrentIndex(1)

    @QtCore.Slot()
    def on_actionAnalogGrid_activated(self):
        self.visualizationComboBox.setCurrentIndex(2)

    def closeEvent(self, event):
        self.canvas.close()
        sys.exit()


def run(fname):
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow(fname)
    win.show()
    if platform.system() == 'Darwin':
        os.system('''osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')  # noqa
    appQt.exec_()
