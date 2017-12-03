import os
import sys
import platform
import glob

import pymea.pymea as mea

from PyQt5 import QtGui, QtCore, QtWidgets  # noqa
from .mea_tools_window import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Subclass of QMainWindow
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # UI initialization
        self.setupUi(self)

        self.setWindowTitle('MEA Tools')

        self.last_directory = os.path.expanduser('~')
        self.load_settings()
        self.directoryLabel.setText(self.last_directory)
        self.populateTable(self.last_directory)

        # Populate table view with last directory.

    @QtCore.pyqtSlot()
    def on_browseButton_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            'Select directory of experiment files.')
        if os.path.exists(directory):
            self.populateTable(directory)
            self.directoryLabel.setText(directory)
            self.last_directory = directory

    @QtCore.pyqtSlot()
    def on_convertButton_clicked(self):
        files = [i.text() for i in self.filenameListWidget.selectedItems()]
        exists = [os.path.exists(os.path.splitext(f)[0] + '.csv')
                  for f in files]

        if True in exists:
            reply = QtWidgets.QMessageBox.question(
                self, 'Overwrite Files?',
                'Some files will be overwritten. Proceed?',
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        else:
            reply = QtWidgets.QMessageBox.Yes

        if reply == QtWidgets.QMessageBox.Yes:
            self.logTextEdit.clear()
            self.tabWidget.setCurrentIndex(1)
            thread = WorkerThread(
                self,
                threshold=self.thresholdSpinBox.value(),
                sort=self.spikeSortCheckBox.isChecked(),
                tag=self.tagPropagationCheckBox.isChecked(),
                neg_only=self.negativePeakOnlyCheckBox.isChecked(),
                files=files)
            thread.event.connect(self.onThreadEvent,
                                 QtCore.Qt.QueuedConnection)

            thread.start()

    def onThreadEvent(self, text):
        currentText = self.logTextEdit.toHtml()
        self.logTextEdit.setHtml(currentText + text)

    @QtCore.pyqtSlot()
    def on_selectAllButton_clicked(self):
        self.filenameListWidget.selectAll()

    @QtCore.pyqtSlot()
    def on_selectNoneButton_clicked(self):
        self.filenameListWidget.clearSelection()

    def populateTable(self, directory):
        fnames = glob.glob(os.path.join(os.path.expanduser(directory), '*.h5'))
        self.filenameListWidget.clear()
        self.filenameListWidget.insertItems(0, fnames)

    def load_settings(self):
        # Load gui settings and restore window geometery
        settings = QtCore.QSettings('UCSB', 'meatools')
        try:
            settings.beginGroup('MainWindow')
            self.restoreGeometry(settings.value('geometry'))
            self.thresholdSpinBox.setValue(
                settings.value('threshold', 6, type=float))
            self.spikeSortCheckBox.setChecked(
                settings.value('spikeSortCheckBox', False, type=bool)
            )
            self.tagPropagationCheckBox.setChecked(
                settings.value('tagPropagationCheckBox', False, type=bool)
            )
            self.negativePeakOnlyCheckBox.setChecked(
                settings.value('negativePeakOnlyCheckBox', False, type=bool)
            )
            self.last_directory = settings.value(
                'directory',
                os.path.expanduser('~'), type=str)
            settings.endGroup()
        except:
            pass

    def save_settings(self):
        settings = QtCore.QSettings('UCSB', 'meatools')
        settings.beginGroup('MainWindow')
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('threshold', self.thresholdSpinBox.value())
        settings.setValue('spikeSortCheckBox',
                          self.spikeSortCheckBox.isChecked())
        settings.setValue('tagPropagationCheckBox',
                          self.tagPropagationCheckBox.isChecked())
        settings.setValue('negativePeakOnlyCheckBox',
                          self.negativePeakOnlyCheckBox.isChecked())
        settings.setValue('directory', self.last_directory)
        settings.endGroup()

    def closeEvent(self, event):
        self.save_settings()
        sys.exit()


class WorkerThread(QtCore.QThread):
    event = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, files=[], threshold=6.0,
                 sort=True, tag=True, neg_only=False):
        super().__init__(parent)
        self.exiting = False
        self.threshold = threshold
        self.sort = sort
        self.tag = tag
        self.neg_only = neg_only
        self.files = files

    def __del__(self):
        self.exiting = True
        # self.wait()

    def run(self):
        for i, f in enumerate(self.files):
            self.event.emit('Processing %s ...' % f)
            try:
                mea.export_spikes(
                    f,
                    self.threshold,
                    sort=self.sort,
                    conductance=self.tag,
                    neg_only=self.neg_only)
                self.event.emit(
                    '%d of %d exported.' % (i + 1, len(self.files)))
            except Exception:
                self.event.emit('<span style="color:red">Error processing: %s</span><BR>' % f)  # noqa
        self.event.emit('Done.')
        self.exit()


def run():
    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if platform.system() == 'Darwin':
        try:
            os.system(
                '''osascript -e 'tell app "Finder" to set frontmost of process "python3" to true' ''')  # noqa
        except:
            pass
    appQt.exec_()
