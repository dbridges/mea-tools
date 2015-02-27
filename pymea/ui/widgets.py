from PySide import QtGui, QtCore  # noqa


class MEAViewerStatusBar(QtGui.QStatusBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_fmt = 't: %1.4f    dt: %1.3f    electrode: %s    %s'
        self._dt = 0
        self._electrode = ''
        self._mouse_t = 0
        self.extra_text = ''

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val
        self._update()

    @property
    def electrode(self):
        return self._electrode

    @electrode.setter
    def electrode(self, val):
        self._electrode = val.upper()
        self._update()

    @property
    def mouse_t(self):
        return self._mouse_t

    @mouse_t.setter
    def mouse_t(self, val):
        self._mouse_t = val
        self._update()

    def _update(self):
        self.showMessage(self._text_fmt %
                         (self._mouse_t, self._dt,
                          self._electrode, self.extra_text))
