import os
import sys
import platform

import pymea.pymea as mea
from pymea.ui.visualizations import (MEA120GridVisualization,
                                     MEAAnalogVisualization,
                                     RasterPlotVisualization,
                                     FlashingSpikeVisualization,
                                     MEA120ConductionVisualization)
import pymea.rsc  # noqa

import pandas as pd
from vispy import app, gloo, visuals
import OpenGL.GL as gl
from PyQt4 import QtGui, QtCore  # noqa
from .main_window import Ui_MainWindow


class VisualizationCanvas(app.Canvas):
    def __init__(self, controller):
        app.Canvas.__init__(self, vsync=True)
        self.controller = controller

        self.analog_grid_vis = None
        self.analog_vis = None
        self.raster_vis = None
        self.flashing_spike_vis = None
        self.conduction_vis = None

        self.previous_vis = None
        self.visualization = None

        self.tr_sys = visuals.transforms.TransformSystem(self)
        self._timer = app.Timer(1/30.0, connect=self.on_tick, start=True)

        self.mouse_pos = (0, 0)
        self.prev_mouse_pos = (0, 0)

    def show_raster(self, selected=None):
        if self.raster_vis is None:
            self.raster_vis = RasterPlotVisualization(
                self, self.controller.spike_data)
        if self.visualization is not None:
            self.raster_vis.t0 = self.visualization.t0
            self.raster_vis.dt = self.visualization.dt
            self.previous_vis = self.visualization
            self.visualization.on_hide()
        self.visualization = self.raster_vis
        if selected is not None:
            self.raster_vis.selected_electrodes = selected
        self.visualization.on_show()
        self.controller.on_show_raster()

    def show_flashing_spike(self):
        if self.flashing_spike_vis is None:
            self.flashing_spike_vis = FlashingSpikeVisualization(
                self, self.controller.spike_data)
        if self.visualization is not None:
            self.flashing_spike_vis.t0 = self.visualization.t0
            self.flashing_spike_vis.dt = self.visualization.dt
            self.previous_vis = self.visualization
            self.visualization.on_hide()
        self.visualization = self.flashing_spike_vis
        self.visualization.on_show()
        self.controller.on_show_flashing_spike()

    def show_analog_grid(self):
        if self.analog_grid_vis is None:
            self.analog_grid_vis = MEA120GridVisualization(
                self, self.controller.analog_data)
        if self.visualization is not None:
            self.analog_grid_vis.t0 = self.visualization.t0
            self.analog_grid_vis.dt = self.visualization.dt
            self.previous_vis = self.visualization
            self.visualization.on_hide()
        self.analog_grid_vis.y_scale = \
            self.controller.analogScaleSpinBox.value()
        self.visualization = self.analog_grid_vis
        self.visualization.on_show()
        self.controller.on_show_analog_grid()

    def show_conduction(self, selected_electrodes=None):
        if selected_electrodes is None:
            if (self.visualization is self.analog_grid_vis and
                    self.visualization is not None):
                selected_electrodes = \
                    self.analog_grid_vis.selected_electrodes
            elif (self.visualization is self.analog_vis and
                    self.visualization is not None):
                selected_electrodes = \
                    self.analog_vis.selected_electrodes
            else:
                selected_electrodes = []
        if self.conduction_vis is None:
            self.conduction_vis = MEA120ConductionVisualization(
                self, self.controller.analog_data,
                self.controller.spike_data)
        if self.visualization is not None:
            self.conduction_vis.t0 = self.visualization.t0
            self.conduction_vis.dt = self.visualization.dt
            self.previous_vis = self.visualization
            self.visualization.on_hide()
        self.visualization = self.conduction_vis
        self.visualization.on_show()
        self.visualization.selected_electrodes = selected_electrodes
        self.controller.on_show_conduction()

    def show_analog(self):
        if self.analog_vis is None:
            self.analog_vis = MEAAnalogVisualization(
                self, self.controller.analog_data, self.controller.spike_data)
            self.analog_vis.filtered = \
                self.controller.filterCheckBox.isChecked()
            self.analog_vis.show_spikes = \
                self.controller.showSpikesCheckBox.isChecked()
        if self.visualization is not None:
            self.analog_vis.t0 = self.visualization.t0
            self.analog_vis.dt = self.visualization.dt
            self.previous_vis = self.visualization
            self.visualization.on_hide()
        self.visualization = self.analog_vis
        self.analog_vis.selected_electrodes = [
            s.lower() for s in self.analog_grid_vis.selected_electrodes]
        self.analog_vis.y_scale = self.analog_grid_vis.y_scale
        self.visualization.on_show()
        self.controller.on_show_analog()

    def show_previous(self):
        if self.previous_vis is None:
            return
        self.previous_vis.selected_electrodes = \
            self.visualization.selected_electrodes
        self.previous_vis.t0 = self.visualization.t0
        self.previous_vis.dt = self.visualization.dt
        self.visualization.on_hide()
        self.visualization = self.previous_vis
        self.visualization.on_show()
        if isinstance(self.visualization, MEAAnalogVisualization):
            self.controller.on_show_analog()
        elif isinstance(self.visualization, MEA120GridVisualization):
            self.controller.on_show_analog_grid()
        elif isinstance(self.visualization, MEA120ConductionVisualization):
            self.controller.on_show_conduction()
        elif isinstance(self.visualization, FlashingSpikeVisualization):
            self.controller.on_show_flashing_spike()
        elif isinstance(self.visualization, RasterPlotVisualization):
            self.controller.on_show_raster()

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x / (w / 2.) - 1., y / (h / 2.) - 1.

    def enable_antialiasing(self):
        try:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        except:
            pass

    def disable_antialiasing(self):
        try:
            gl.glDisable(gl.GL_LINE_SMOOTH)
            gl.glDisable(gl.GL_BLEND)
        except:
            pass

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
        if self.visualization is not None:
            self.visualization.on_mouse_release(event)

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

    def __init__(self, analog_file, spike_file, start_vis, parent=None):
        super().__init__(parent)

        splash = QtGui.QSplashScreen(QtGui.QPixmap(':/splash@2x.png'))
        splash.show()
        self.analog_file = analog_file
        self.spike_file = spike_file

        # UI initialization
        self.setupUi(self)

        self._spike_data = None
        self._analog_data = None

        self.canvas = VisualizationCanvas(self)

        self.toolBar.addWidget(self.toolbarWidget)
        self.canvas.native.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.mainLayout.removeWidget(self.widget)
        self.mainLayout.addWidget(self.canvas.native)

        self.rasterRowCountSlider.setValue(120)

        self.flashingSpikeTimescaleComboBox.setCurrentIndex(4)

        if start_vis == 'raster':
            self.visualizationComboBox.setCurrentIndex(0)
            self.canvas.show_raster()
            self.rasterRowCountSlider.setMaximum(
                self.canvas.raster_vis.row_count)
            self.rasterRowCountSlider.setValue(
                self.canvas.raster_vis.row_count)
            filepath = spike_file
        else:
            self.visualizationComboBox.setCurrentIndex(2)
            filepath = analog_file

        self.load_settings()

        self.setWindowTitle('MEA Viewer - ' + os.path.basename(filepath))

        splash.finish(self)

    def load_settings(self):
        # Load gui settings and restore window geometery
        settings = QtCore.QSettings('UCSB', 'meaview')
        try:
            settings.beginGroup('MainWindow')
            self.restoreGeometry(settings.value('geometry'))
            self.analogScaleSpinBox.setValue(
                settings.value('analogScale', 100, type=float))
            self.filterCheckBox.setChecked(
                settings.value('filterCheckBox', False, type=bool)
            )
            self.showSpikesCheckBox.setChecked(
                settings.value('showSpikesCheckBox', False, type=bool)
            )
            settings.endGroup()
        except:
            pass

    def save_settings(self):
        settings = QtCore.QSettings('UCSB', 'meaview')
        settings.beginGroup('MainWindow')
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('analogScale', self.analogScaleSpinBox.value())
        settings.setValue('filterCheckBox', self.filterCheckBox.isChecked())
        settings.setValue('showSpikesCheckBox',
                          self.showSpikesCheckBox.isChecked())
        settings.endGroup()

    def load_spike_data(self):
        print('Loading spike data...', end='', flush=True)
        try:
            self._spike_data = pd.read_csv(self.spike_file)
        except:
            self._spike_data = pd.DataFrame({'electrode': [],
                                             'time': [],
                                             'amplitude': [],
                                             'threshold': []})
        print('done.')

    def load_analog_data(self):
        print('Loading analog data...', end='', flush=True)
        try:
            store = mea.MEARecording(self.analog_file)
            self._analog_data = store.get('all')
        except:
            self._analog_data = pd.DataFrame(index=[0, 1 / 20000.0])
        print('done.')

    def on_visualization_updated(self):
        self.statusBar.extra_text = self.canvas.visualization.extra_text
        self.statusBar.electrode = self.canvas.visualization.electrode
        self.statusBar.mouse_t = self.canvas.visualization.mouse_t
        self.statusBar.update()

    @property
    def spike_data(self):
        if self._spike_data is None:
            self.load_spike_data()
        return self._spike_data

    @spike_data.setter
    def spike_data(self, data):
        self._spike_data = data

    @property
    def analog_data(self):
        if self._analog_data is None:
            self.load_analog_data()
        return self._analog_data

    @analog_data.setter
    def analog_data(self, data):
        self._analog_data = data

    @QtCore.pyqtSlot(int)
    def on_rasterRowCountSlider_valueChanged(self, val):
        try:
            self.canvas.raster_vis.row_count = val
        except AttributeError:
            pass

    @QtCore.pyqtSlot(str)
    def on_visualizationComboBox_currentIndexChanged(self, text):
        if text == 'Raster':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_raster()
            self.rasterRowCountSlider.setMaximum(
                self.canvas.raster_vis._unselected_row_count)
            self.rasterRowCountSlider.setValue(
                self.canvas.raster_vis.row_count)
        elif text == 'Flashing Spike':
            if self.spike_data is None:
                self.load_spike_data()
            self.canvas.show_flashing_spike()
        elif text == 'Analog':
            self.canvas.show_analog_grid()
        elif text == 'Conduction':
            self.canvas.show_conduction()

    @QtCore.pyqtSlot(float)
    def on_analogScaleSpinBox_valueChanged(self, val):
        if self.canvas.analog_grid_vis is not None:
            self.canvas.analog_grid_vis.y_scale = val
        if self.canvas.analog_vis is not None:
            self.canvas.analog_vis.y_scale = val

    @QtCore.pyqtSlot(str)
    def on_flashingSpikeTimescaleComboBox_currentIndexChanged(self, text):
        if self.canvas.flashing_spike_vis is None:
            return
        if text == '1x':
            self.canvas.flashing_spike_vis.time_scale = 1
        elif text == '1/2x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 2
        elif text == '1/20x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 20
        elif text == '1/100x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 100
        elif text == '1/200x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 200
        elif text == '1/400x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 400
        elif text == '1/800x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 800
        elif text == '1/1600x':
            self.canvas.flashing_spike_vis.time_scale = 1 / 1600

    @QtCore.pyqtSlot(bool)
    def on_filterCheckBox_toggled(self, checked):
        if self.canvas.analog_vis is None:
            return
        self.canvas.analog_vis.filtered = checked

    @QtCore.pyqtSlot(bool)
    def on_showSpikesCheckBox_toggled(self, checked):
        if self.canvas.analog_vis is None:
            return
        self.canvas.analog_vis.show_spikes = checked

    @QtCore.pyqtSlot(bool)
    def on_dimConductanceCheckBox_toggled(self, checked):
        if self.canvas.raster_vis is None:
            return
        self.canvas.raster_vis.dim_conductance = checked

    @QtCore.pyqtSlot()
    def on_actionRaster_activated(self):
        self.visualizationComboBox.setCurrentIndex(0)

    @QtCore.pyqtSlot()
    def on_actionFlashingSpikes_activated(self):
        self.visualizationComboBox.setCurrentIndex(1)

    @QtCore.pyqtSlot()
    def on_actionAnalogGrid_activated(self):
        self.visualizationComboBox.setCurrentIndex(2)

    @QtCore.pyqtSlot(float)
    def on_filterLowSpinBox_valueChanged(self, val):
        if self.canvas.analog_vis is None:
            return
        self.canvas.analog_vis.filter_cutoff = [
            self.filterLowSpinBox.value(),
            self.filterHighSpinBox.value()
        ]

    @QtCore.pyqtSlot(float)
    def on_filterHighSpinBox_valueChanged(self, val):
        if self.canvas.analog_vis is None:
            return
        self.canvas.analog_vis.filter_cutoff = [
            self.filterLowSpinBox.value(),
            self.filterHighSpinBox.value()
        ]

    @QtCore.pyqtSlot(str)
    def on_sortRasterComboBox_activated(self, text):
        if self.canvas.raster_vis is None:
            return
        self.canvas.raster_vis.sort(text.lower())

    def on_show_analog(self):
        self.visualizationComboBox.blockSignals(True)
        self.visualizationComboBox.setCurrentIndex(2)
        self.stackedWidget.setCurrentIndex(2)
        self.visualizationComboBox.blockSignals(False)

    def on_show_raster(self):
        self.visualizationComboBox.blockSignals(True)
        self.visualizationComboBox.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(0)
        self.visualizationComboBox.blockSignals(False)

    def on_show_conduction(self):
        self.visualizationComboBox.blockSignals(True)
        self.visualizationComboBox.setCurrentIndex(3)
        self.stackedWidget.setCurrentIndex(3)
        self.visualizationComboBox.blockSignals(False)

    def on_show_analog_grid(self):
        self.visualizationComboBox.blockSignals(True)
        self.visualizationComboBox.setCurrentIndex(2)
        self.stackedWidget.setCurrentIndex(2)
        self.visualizationComboBox.blockSignals(False)

    def on_show_flashing_spike(self):
        self.visualizationComboBox.blockSignals(True)
        self.visualizationComboBox.setCurrentIndex(1)
        self.stackedWidget.setCurrentIndex(1)
        self.visualizationComboBox.blockSignals(False)

    def closeEvent(self, event):
        self.save_settings()
        self.canvas.close()
        sys.exit()


def get_file():
    fname = QtGui.QFileDialog.getOpenFileName(
        None,
        'Open datafile',
        os.path.expanduser('~/Desktop'),
        'HDF5 or CSV file (*.h5 *.csv)')
    if os.path.exists(fname):
        spike_file = None
        analog_file = None
        spike_file = None
        analog_file = None
        if fname.endswith('.csv'):
            spike_file = fname
            if os.path.exists(fname[:-4] + '.h5'):
                analog_file = fname[:-4] + '.h5'
        elif fname.endswith('.h5'):
            analog_file = fname
            if os.path.exists(fname[:-3] + '.csv'):
                spike_file = fname[:-3] + '.csv'
        else:
            raise IOError('Invalid input file, must be of type csv or h5.')

        if fname.endswith('csv'):
            show = 'raster'
        else:
            show = 'analog'

        return (analog_file, spike_file, show)
    else:
        raise IOError('Invalid input file, must be of type csv or h5.')


def run(analog_file, spike_file, start_vis):
    appQt = QtGui.QApplication(sys.argv)
    if analog_file is None and spike_file is None:
        analog_file, spike_file, start_vis = get_file()
    win = MainWindow(analog_file, spike_file, start_vis)
    win.show()
    if platform.system() == 'Darwin':
        try:
            os.system(
                '''osascript -e 'tell app "Finder" to set frontmost of process "python3" to true' ''')  # noqa
        except:
            pass
    appQt.exec_()
