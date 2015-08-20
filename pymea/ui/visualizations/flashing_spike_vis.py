# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import numpy as np
from vispy import gloo, visuals
from vispy.visuals.shaders import ModularProgram

from .base import Visualization, Theme
import pymea as mea
import pymea.util as util


class FlashingSpikeElectrode:
    def __init__(self, tag, events):
        self.events = events
        self.events.sort()
        self.value = 0
        self.tag = tag

    def update(self, t, dt):
        try:
            len(self.events == 0)
        except:
            return
        self.value += 0.2*len(self.events[(self.events > t) &
                                          (self.events < t + dt)])
        self.value -= 0.01
        self.value = util.clip(self.value, 0.0, 1.0)


class FlashingSpikeVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec2 a_position;
    attribute float a_color;

    varying float v_color;

    void main(void)
    {
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
        v_color = a_color;
    }
    """

    FRAGMENT_SHADER = """
    varying float v_color;

    void main()
    {
        gl_FragColor = vec4(v_color, v_color, v_color, 1);
    }
    """

    mea_outline = np.array(
        [[3.0, 0.0], [9.0, 0.0], [9.0, 1.0],
         [10.0, 1.0], [10.0, 2.0], [11.0, 2.0],
         [11.0, 3.0], [12.0, 3.0], [12.0, 9.0],
         [11.0, 9.0], [11.0, 10.0], [10.0, 10.0],
         [10.0, 11.0], [9.0, 11.0], [9.0, 12.0],
         [3.0, 12.0], [3.0, 11.0], [2.0, 11.0],
         [2.0, 10.0], [1.0, 10.0], [1.0, 9.0],
         [0.0, 9.0], [0.0, 3.0], [1.0, 3.0],
         [1.0, 2.0], [2.0, 2.0], [2.0, 1.0],
         [3.0, 1.0], [3.0, 0.0]], dtype=np.float32)

    def __init__(self, canvas, spike_data):
        super().__init__()
        self.canvas = canvas
        self.program = ModularProgram(self.VERTEX_SHADER,
                                      self.FRAGMENT_SHADER)
        self._t0 = 0
        self._dt = 10
        self.mouse_t = self.t0
        self._interval = 1 / 30.0
        self.electrode = ''
        self.time_scale = 1 / 200
        self._vert = np.zeros((120 * 6, 2), dtype=np.float32)
        self._color = np.zeros(120 * 6, dtype=np.float32)
        self.electrodes = []
        self.spikes = spike_data.copy()
        self.spikes.electrode = self.spikes.electrode.str.extract('(\w+)\.*')
        for tag, df in self.spikes.groupby('electrode'):
            self.electrodes.append(FlashingSpikeElectrode(tag,
                                                          df['time'].values))
        self._create_vertex_data()
        self.paused = True
        self.program['a_position'] = self._vert
        self.program['a_color'] = self._color
        self.program.vert['transform'] = canvas.tr_sys.get_full_transform()
        self.outline = visuals.LineVisual(color=Theme.yellow)
        self.electrode_cols = [c for c in 'ABCDEFGHJKLM']
        self._rescale_outline()
        self.extra_text = ''

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val,
                             -self.spikes.time.max(),
                             self.spikes.time.max())
        self.mouse_t = self._t0

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, self.spikes.time.max())

    def _create_vertex_data(self):
        self._vert = np.zeros((120*6, 2), dtype=np.float32)
        for i, e in enumerate(self.electrodes):
            x, y = mea.coordinates_for_electrode(e.tag)
            size = self.canvas.size[1] / 14
            x = self.canvas.size[0] / 2 - 6 * size + x * size
            y = y * size + size
            self._vert[6*i] = [x, y]
            self._vert[6*i + 1] = [x + size, y]
            self._vert[6*i + 2] = [x + size, y + size]
            self._vert[6*i + 3] = [x, y]
            self._vert[6*i + 4] = [x + size, y + size]
            self._vert[6*i + 5] = [x, y + size]

    def _rescale_outline(self):
        size = self.canvas.size[1] / 14
        self.outline.set_data(self.mea_outline * size +
                              [self.canvas.size[0] / 2 - 6*size, size])

    def draw(self):
        gloo.clear((0.0, 0.0, 0.0, 1))
        self.outline.draw(self.canvas.tr_sys)
        self.program.draw('triangles')

    def pause(self):
        self.paused = True

    def run(self):
        self.paused = False

    def toggle_play(self):
        if self.paused:
            self.run()
        else:
            self.pause()

    def update(self):
        self.program['a_color'] = self._color

    def on_resize(self, event):
        self._create_vertex_data()
        self._rescale_outline()
        self.program['a_position'] = self._vert
        self.program.vert['transform'] = \
            self.canvas.tr_sys.get_full_transform()

    def on_tick(self, event):
        if self.paused:
            return
        for i, e in enumerate(self.electrodes):
            e.update(self.t0, self._interval * self.time_scale)
            self._color[6*i:6*i+6] = e.value
        self.t0 += self.time_scale * self._interval
        self.update()

    def on_key_release(self, event):
        if event.key == 'space':
            self.toggle_play()
        elif event.key == 'Left':
            self.t0 -= 4*self.time_scale  # Jump back in time

    def on_mouse_move(self, event):
        x, y = event.pos
        cell_size = self.canvas.size[1] / 14
        row = int((y - cell_size) / cell_size) + 1
        col = int((x - self.canvas.width/2) / cell_size + 6)
        if row < 1 or row > 12 or col < 0 or col > 11:
            self.electrode = ''
        else:
            self.electrode = '%s%d' % (self.electrode_cols[col], row)
