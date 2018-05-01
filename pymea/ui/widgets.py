from PyQt5 import QtGui, QtCore, QtWidgets  # noqa


class MEAViewerStatusBar(QtWidgets.QStatusBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_fmt = 't: %1.4f    electrode: %s    %s'
        self.electrode = ''
        self.mouse_t = 0
        self.extra_text = ''

    def update(self):
        self.showMessage(self._text_fmt %
                         (self.mouse_t, self.electrode, self.extra_text))
