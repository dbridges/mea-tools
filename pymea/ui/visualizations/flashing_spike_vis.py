# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import numpy as np
from vispy import gloo, visuals
from vispy.visuals.shaders import ModularProgram

from .base import Visualization, Theme
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

        # Each quad is drawn as two triangles, so need 6 vertices per electrode
        self._vert = np.zeros((self.canvas.layout.count * 6, 2),
                              dtype=np.float32)
        self._color = np.zeros(self.canvas.layout.count * 6,
                               dtype=np.float32)
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
        self.program.vert['transform'] = canvas.tr_sys.get_transform()
        self.outline = visuals.LineVisual(color=Theme.yellow)
        self._rescale_outline()
        self.extra_text = ''
        self.configure_transforms()

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
        self._vert = np.zeros((self.canvas.layout.count*6, 2),
                              dtype=np.float32)
        half_cols = self.canvas.layout.columns / 2
        for i, e in enumerate(self.electrodes):
            x, y = self.canvas.layout.coordinates_for_electrode(e.tag)
            size = self.canvas.size[1] / (self.canvas.layout.rows + 2)
            x = self.canvas.size[0] / 2 - half_cols * size + x * size
            y = y * size + size
            self._vert[6*i] = [x, y]
            self._vert[6*i + 1] = [x + size, y]
            self._vert[6*i + 2] = [x + size, y + size]
            self._vert[6*i + 3] = [x, y]
            self._vert[6*i + 4] = [x + size, y + size]
            self._vert[6*i + 5] = [x, y + size]

    def _rescale_outline(self):
        size = self.canvas.size[1] / (self.canvas.layout.rows + 2)
        self.outline.set_data(self.canvas.layout.outline * size +
                              [self.canvas.size[0] / 2 -
                               self.canvas.layout.columns*size / 2, size])

    def draw(self):
        gloo.clear((0.0, 0.0, 0.0, 1))
        self.outline.draw()
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
            self.canvas.tr_sys.get_transform()

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
        size = self.canvas.size[1] / (self.canvas.layout.rows + 2)
        row = int((y - size) / size) + 1
        col = int((x - self.canvas.width/2) / size +
                  (self.canvas.layout.columns / 2))
        if (row < 1 or row > self.canvas.layout.rows or
                col < 0 or col > self.canvas.layout.columns):
            self.electrode = ''
        else:
            self.electrode = (
                '%s' %
                self.canvas.layout.electrode_for_coordinate((col, row)))

    def configure_transforms(self):
        vp = (0, 0, self.canvas.physical_size[0], self.canvas.physical_size[1])
        self.canvas.context.set_viewport(*vp)
        self.outline.transforms.configure(canvas=self.canvas, viewport=vp)
