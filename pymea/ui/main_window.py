# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pymea/ui/PyMEAMainWindow.ui'
#
# Created: Tue Apr 28 15:06:30 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1146, 664)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.toolbarWidget = QtGui.QWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolbarWidget.sizePolicy().hasHeightForWidth())
        self.toolbarWidget.setSizePolicy(sizePolicy)
        self.toolbarWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.toolbarWidget.setObjectName(_fromUtf8("toolbarWidget"))
        self.horizontalLayout_8 = QtGui.QHBoxLayout(self.toolbarWidget)
        self.horizontalLayout_8.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout_8.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.visualizationComboBox = QtGui.QComboBox(self.toolbarWidget)
        self.visualizationComboBox.setObjectName(_fromUtf8("visualizationComboBox"))
        self.visualizationComboBox.addItem(_fromUtf8(""))
        self.visualizationComboBox.addItem(_fromUtf8(""))
        self.visualizationComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout_8.addWidget(self.visualizationComboBox)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.stackedWidget = QtGui.QStackedWidget(self.toolbarWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setObjectName(_fromUtf8("stackedWidget"))
        self.rasterPage = QtGui.QWidget()
        self.rasterPage.setObjectName(_fromUtf8("rasterPage"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.rasterPage)
        self.horizontalLayout_5.setMargin(0)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.dimConductanceCheckBox = QtGui.QCheckBox(self.rasterPage)
        self.dimConductanceCheckBox.setObjectName(_fromUtf8("dimConductanceCheckBox"))
        self.horizontalLayout_5.addWidget(self.dimConductanceCheckBox)
        self.rowCountLabel = QtGui.QLabel(self.rasterPage)
        self.rowCountLabel.setObjectName(_fromUtf8("rowCountLabel"))
        self.horizontalLayout_5.addWidget(self.rowCountLabel)
        self.rasterRowCountSlider = QtGui.QSlider(self.rasterPage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rasterRowCountSlider.sizePolicy().hasHeightForWidth())
        self.rasterRowCountSlider.setSizePolicy(sizePolicy)
        self.rasterRowCountSlider.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.rasterRowCountSlider.setFont(font)
        self.rasterRowCountSlider.setMinimum(2)
        self.rasterRowCountSlider.setMaximum(120)
        self.rasterRowCountSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rasterRowCountSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.rasterRowCountSlider.setTickInterval(0)
        self.rasterRowCountSlider.setObjectName(_fromUtf8("rasterRowCountSlider"))
        self.horizontalLayout_5.addWidget(self.rasterRowCountSlider)
        self.horizontalLayout_5.setStretch(2, 1)
        self.stackedWidget.addWidget(self.rasterPage)
        self.flashingSpikePage = QtGui.QWidget()
        self.flashingSpikePage.setObjectName(_fromUtf8("flashingSpikePage"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout(self.flashingSpikePage)
        self.horizontalLayout_6.setMargin(0)
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.speedLabel_2 = QtGui.QLabel(self.flashingSpikePage)
        self.speedLabel_2.setObjectName(_fromUtf8("speedLabel_2"))
        self.horizontalLayout_6.addWidget(self.speedLabel_2)
        self.flashingSpikeTimescaleComboBox = QtGui.QComboBox(self.flashingSpikePage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.flashingSpikeTimescaleComboBox.sizePolicy().hasHeightForWidth())
        self.flashingSpikeTimescaleComboBox.setSizePolicy(sizePolicy)
        self.flashingSpikeTimescaleComboBox.setObjectName(_fromUtf8("flashingSpikeTimescaleComboBox"))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.flashingSpikeTimescaleComboBox.addItem(_fromUtf8(""))
        self.horizontalLayout_6.addWidget(self.flashingSpikeTimescaleComboBox)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.horizontalLayout_6.setStretch(1, 1)
        self.stackedWidget.addWidget(self.flashingSpikePage)
        self.analogPage = QtGui.QWidget()
        self.analogPage.setObjectName(_fromUtf8("analogPage"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout(self.analogPage)
        self.horizontalLayout_7.setMargin(0)
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.scaleLabel = QtGui.QLabel(self.analogPage)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scaleLabel.sizePolicy().hasHeightForWidth())
        self.scaleLabel.setSizePolicy(sizePolicy)
        self.scaleLabel.setObjectName(_fromUtf8("scaleLabel"))
        self.horizontalLayout_7.addWidget(self.scaleLabel)
        self.analogScaleSpinBox = QtGui.QDoubleSpinBox(self.analogPage)
        self.analogScaleSpinBox.setDecimals(0)
        self.analogScaleSpinBox.setMinimum(1.0)
        self.analogScaleSpinBox.setMaximum(20000.0)
        self.analogScaleSpinBox.setSingleStep(20.0)
        self.analogScaleSpinBox.setProperty("value", 150.0)
        self.analogScaleSpinBox.setObjectName(_fromUtf8("analogScaleSpinBox"))
        self.horizontalLayout_7.addWidget(self.analogScaleSpinBox)
        self.showSpikesCheckBox = QtGui.QCheckBox(self.analogPage)
        self.showSpikesCheckBox.setObjectName(_fromUtf8("showSpikesCheckBox"))
        self.horizontalLayout_7.addWidget(self.showSpikesCheckBox)
        self.filterCheckBox = QtGui.QCheckBox(self.analogPage)
        self.filterCheckBox.setObjectName(_fromUtf8("filterCheckBox"))
        self.horizontalLayout_7.addWidget(self.filterCheckBox)
        self.filterLowSpinBox = QtGui.QDoubleSpinBox(self.analogPage)
        self.filterLowSpinBox.setKeyboardTracking(False)
        self.filterLowSpinBox.setMaximum(50000.0)
        self.filterLowSpinBox.setProperty("value", 200.0)
        self.filterLowSpinBox.setObjectName(_fromUtf8("filterLowSpinBox"))
        self.horizontalLayout_7.addWidget(self.filterLowSpinBox)
        self.label = QtGui.QLabel(self.analogPage)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_7.addWidget(self.label)
        self.filterHighSpinBox = QtGui.QDoubleSpinBox(self.analogPage)
        self.filterHighSpinBox.setKeyboardTracking(False)
        self.filterHighSpinBox.setMaximum(50000.0)
        self.filterHighSpinBox.setProperty("value", 4000.0)
        self.filterHighSpinBox.setObjectName(_fromUtf8("filterHighSpinBox"))
        self.horizontalLayout_7.addWidget(self.filterHighSpinBox)
        self.stackedWidget.addWidget(self.analogPage)
        self.horizontalLayout_8.addWidget(self.stackedWidget)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem2)
        self.horizontalLayout_8.setStretch(3, 1)
        self.verticalLayout_2.addWidget(self.toolbarWidget)
        self.mainLayout = QtGui.QVBoxLayout()
        self.mainLayout.setObjectName(_fromUtf8("mainLayout"))
        self.widget = QtGui.QWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 200))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.mainLayout.addWidget(self.widget)
        self.verticalLayout_2.addLayout(self.mainLayout)
        self.verticalLayout_2.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = MEAViewerStatusBar(MainWindow)
        self.statusBar.setObjectName(_fromUtf8("statusBar"))
        MainWindow.setStatusBar(self.statusBar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionSave_to_Spreadsheet = QtGui.QAction(MainWindow)
        self.actionSave_to_Spreadsheet.setEnabled(False)
        self.actionSave_to_Spreadsheet.setObjectName(_fromUtf8("actionSave_to_Spreadsheet"))
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionDefault = QtGui.QAction(MainWindow)
        self.actionDefault.setObjectName(_fromUtf8("actionDefault"))
        self.actionAnalog_Waveform = QtGui.QAction(MainWindow)
        self.actionAnalog_Waveform.setObjectName(_fromUtf8("actionAnalog_Waveform"))
        self.actionRaster = QtGui.QAction(MainWindow)
        self.actionRaster.setObjectName(_fromUtf8("actionRaster"))
        self.actionFlashingSpikes = QtGui.QAction(MainWindow)
        self.actionFlashingSpikes.setObjectName(_fromUtf8("actionFlashingSpikes"))
        self.actionAnalogGrid = QtGui.QAction(MainWindow)
        self.actionAnalogGrid.setObjectName(_fromUtf8("actionAnalogGrid"))

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.visualizationComboBox, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), self.stackedWidget.setCurrentIndex)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MEA Data Viewer", None))
        self.visualizationComboBox.setItemText(0, _translate("MainWindow", "Raster", None))
        self.visualizationComboBox.setItemText(1, _translate("MainWindow", "Flashing Spike", None))
        self.visualizationComboBox.setItemText(2, _translate("MainWindow", "Analog Grid", None))
        self.dimConductanceCheckBox.setText(_translate("MainWindow", "Dim Conductance", None))
        self.rowCountLabel.setText(_translate("MainWindow", "Row Count", None))
        self.speedLabel_2.setText(_translate("MainWindow", "Speed", None))
        self.flashingSpikeTimescaleComboBox.setItemText(0, _translate("MainWindow", "1x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(1, _translate("MainWindow", "1/2x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(2, _translate("MainWindow", "1/20x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(3, _translate("MainWindow", "1/100x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(4, _translate("MainWindow", "1/200x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(5, _translate("MainWindow", "1/400x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(6, _translate("MainWindow", "1/800x", None))
        self.flashingSpikeTimescaleComboBox.setItemText(7, _translate("MainWindow", "1/1600x", None))
        self.scaleLabel.setText(_translate("MainWindow", "Scale", None))
        self.analogScaleSpinBox.setSuffix(_translate("MainWindow", " uV", None))
        self.showSpikesCheckBox.setText(_translate("MainWindow", "Spikes", None))
        self.filterCheckBox.setText(_translate("MainWindow", "Bandpass Filter", None))
        self.filterLowSpinBox.setSuffix(_translate("MainWindow", " Hz", None))
        self.label.setText(_translate("MainWindow", "to", None))
        self.filterHighSpinBox.setSuffix(_translate("MainWindow", " Hz", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionSave_to_Spreadsheet.setText(_translate("MainWindow", "Save to Spreadsheet...", None))
        self.actionSave_to_Spreadsheet.setShortcut(_translate("MainWindow", "Ctrl+S", None))
        self.actionOpen.setText(_translate("MainWindow", "Open...", None))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O", None))
        self.actionDefault.setText(_translate("MainWindow", "Default", None))
        self.actionAnalog_Waveform.setText(_translate("MainWindow", "Analog Waveform", None))
        self.actionRaster.setText(_translate("MainWindow", "Raster", None))
        self.actionRaster.setShortcut(_translate("MainWindow", "Ctrl+R", None))
        self.actionFlashingSpikes.setText(_translate("MainWindow", "Flashing Spikes", None))
        self.actionFlashingSpikes.setShortcut(_translate("MainWindow", "Ctrl+F", None))
        self.actionAnalogGrid.setText(_translate("MainWindow", "Analog Grid", None))
        self.actionAnalogGrid.setShortcut(_translate("MainWindow", "Ctrl+A", None))

from pymea.ui.widgets import MEAViewerStatusBar
