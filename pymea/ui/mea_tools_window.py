# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pymea/ui/MEAToolsMainWindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
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
        MainWindow.resize(857, 428)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.tab)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.directoryLabel = QtGui.QLabel(self.tab)
        self.directoryLabel.setText(_fromUtf8(""))
        self.directoryLabel.setObjectName(_fromUtf8("directoryLabel"))
        self.horizontalLayout.addWidget(self.directoryLabel)
        self.browseButton = QtGui.QPushButton(self.tab)
        self.browseButton.setObjectName(_fromUtf8("browseButton"))
        self.horizontalLayout.addWidget(self.browseButton)
        self.horizontalLayout.setStretch(0, 1)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.filenameListWidget = QtGui.QListWidget(self.tab)
        self.filenameListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.filenameListWidget.setObjectName(_fromUtf8("filenameListWidget"))
        self.horizontalLayout_4.addWidget(self.filenameListWidget)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.selectAllButton = QtGui.QPushButton(self.tab)
        self.selectAllButton.setObjectName(_fromUtf8("selectAllButton"))
        self.verticalLayout.addWidget(self.selectAllButton)
        self.selectNoneButton = QtGui.QPushButton(self.tab)
        self.selectNoneButton.setObjectName(_fromUtf8("selectNoneButton"))
        self.verticalLayout.addWidget(self.selectNoneButton)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label = QtGui.QLabel(self.tab)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_3.addWidget(self.label)
        self.thresholdSpinBox = QtGui.QDoubleSpinBox(self.tab)
        self.thresholdSpinBox.setMinimum(1.0)
        self.thresholdSpinBox.setMaximum(100.0)
        self.thresholdSpinBox.setProperty("value", 6.0)
        self.thresholdSpinBox.setObjectName(_fromUtf8("thresholdSpinBox"))
        self.horizontalLayout_3.addWidget(self.thresholdSpinBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.spikeSortCheckBox = QtGui.QCheckBox(self.tab)
        self.spikeSortCheckBox.setChecked(True)
        self.spikeSortCheckBox.setObjectName(_fromUtf8("spikeSortCheckBox"))
        self.verticalLayout.addWidget(self.spikeSortCheckBox)
        self.tagPropagationCheckBox = QtGui.QCheckBox(self.tab)
        self.tagPropagationCheckBox.setObjectName(_fromUtf8("tagPropagationCheckBox"))
        self.verticalLayout.addWidget(self.tagPropagationCheckBox)
        self.negativePeakOnlyCheckBox = QtGui.QCheckBox(self.tab)
        self.negativePeakOnlyCheckBox.setObjectName(_fromUtf8("negativePeakOnlyCheckBox"))
        self.verticalLayout.addWidget(self.negativePeakOnlyCheckBox)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.convertButton = QtGui.QPushButton(self.tab)
        self.convertButton.setObjectName(_fromUtf8("convertButton"))
        self.verticalLayout.addWidget(self.convertButton)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.logTextEdit = QtGui.QTextEdit(self.tab_2)
        self.logTextEdit.setEnabled(False)
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setObjectName(_fromUtf8("logTextEdit"))
        self.verticalLayout_3.addWidget(self.logTextEdit)
        self.tabWidget.addTab(self.tab_2, _fromUtf8("Log"))
        self.verticalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.browseButton.setText(_translate("MainWindow", "Browse", None))
        self.selectAllButton.setText(_translate("MainWindow", "Select All", None))
        self.selectNoneButton.setText(_translate("MainWindow", "Select None", None))
        self.label.setText(_translate("MainWindow", "Threshold", None))
        self.spikeSortCheckBox.setText(_translate("MainWindow", "Sort Spikes", None))
        self.tagPropagationCheckBox.setText(_translate("MainWindow", "Tag Propagation Signals", None))
        self.negativePeakOnlyCheckBox.setText(_translate("MainWindow", " Negative Peaks Only", None))
        self.convertButton.setText(_translate("MainWindow", "Convert", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Files", None))

