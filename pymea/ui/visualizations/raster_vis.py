# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import math
from collections import Sequence

import numpy as np
from vispy import gloo, visuals

from .base import LineCollection, Visualization, Theme
import pymea.util as util


class ElectrodeData(Sequence):
    def __init__(self, tag, data=np.array([])):
        self.tag = tag
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class RasterPlotVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec2 a_position;
    attribute vec4 a_color;

    uniform float u_pan;
    uniform float u_y_scale;
    uniform float u_count;
    uniform float u_top_margin;

    varying vec4 v_color;

    void main(void)
    {
        float height = (2.0 - u_top_margin) / u_count;
        gl_Position = vec4((a_position.x - u_pan) / u_y_scale - 1,
                           1 - a_position.y * height - u_top_margin, 0.0, 1.0);
        v_color = a_color;
    }
    """

    FRAGMENT_SHADER = """
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    def __init__(self, canvas, spike_data):
        self.canvas = canvas
        self.spikes = spike_data
        self.program = gloo.Program(RasterPlotVisualization.VERTEX_SHADER,
                                    RasterPlotVisualization.FRAGMENT_SHADER)
        self._t0 = 0
        self._dt = self.spikes['time'].max()
        self.electrode = ''
        self.program['u_pan'] = self._t0
        self.program['u_y_scale'] = self._dt/2
        self.program['u_top_margin'] = 20.0 * 2.0 / canvas.size[1]
        self.spikes[['electrode', 'time']].values

        self._row_count = 120
        self._display_selected = False
        self._unselected_row_count = 120
        self.selected_electrodes = []

        # Load data
        self.data = []
        for e in self.spikes.groupby('electrode'):
            self.data.append(ElectrodeData(e[0], e[1]['time'].values))
        self.data.sort(key=lambda e: len(e), reverse=True)

        self.row_count = len(self.data)
        self.resample()
        self.margin = {}
        self.margin['top'] = 20

        self.velocity = 0
        self.tick_separtion = 50
        self.tick_labels = [visuals.TextVisual('', font_size=10, color='w')
                            for x in range(18)]
        self.tick_marks = LineCollection()
        self.mouse_t = 0
        self.extra_text = ''

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0 - self.dt/2,
                             self.spikes.time.max() - self.dt/2)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, self.spikes.time.max())

    @property
    def row_count(self):
        return self._row_count

    @row_count.setter
    def row_count(self, val):
        self.program['u_count'] = val
        self._row_count = val

    @property
    def display_selected(self):
        return self._display_selected

    @display_selected.setter
    def display_selected(self, val):
        self._display_selected = val
        if self._display_selected:
            self._unselected_row_count = self.row_count
            self.row_count = len(self.selected_electrodes)
        else:
            self.row_count = self._unselected_row_count

    def resample(self):
        verticies = []
        colors = []
        if self.display_selected:
            data = [d for d in self.data
                    if d.tag in self.selected_electrodes]
        else:
            data = self.data
        for i, e in enumerate(data):
            for t in e:
                verticies.append((t, i))
                verticies.append((t, i + 1))
                color = Theme.plot_colors[i % 3]
                colors.append(color)
                colors.append(color)
        self.program['a_position'] = verticies
        self.program['a_color'] = colors

    def create_labels(self):
        self.tick_marks.clear()
        self.tick_marks.append((0, self.margin['top']),
                               (self.canvas.size[0], self.margin['top']))

        log_val = math.log10(self.dt)

        if log_val > 0:
            self.tick_separtion = math.pow(10, int(log_val))
        else:
            self.tick_separtion = math.pow(10, int(log_val - 1))

        if self.dt / self.tick_separtion < 3:
            self.tick_separtion /= 4
        elif self.dt / self.tick_separtion < 7:
            self.tick_separtion /= 2

        tick_loc = int(self.t0 / self.tick_separtion) * self.tick_separtion
        xloc = 0
        i = 0
        while tick_loc < (self.t0 + self.dt + self.tick_separtion):
            xloc = (tick_loc - self.t0) / (self.dt / self.canvas.size[0])
            self.tick_labels[i].pos = (xloc, self.margin['top'] / 2 + 3)
            self.tick_labels[i].text = '%1.3f' % tick_loc
            tick_loc += self.tick_separtion
            self.tick_marks.append((xloc, self.margin['top']),
                                   (xloc, self.margin['top'] + 6))
            i += 1
        # Clear labels not being used.
        while i < len(self.tick_labels):
            self.tick_labels[i].pos = (0, self.margin['top'])
            self.tick_labels[i].text = ''
            i += 1

    def update(self):
        self.program['u_pan'] = self.t0
        self.program['u_y_scale'] = self.dt / 2
        self.create_labels()

    def draw(self):
        gloo.clear((0.5, 0.5, 0.5, 1))
        self.program.draw('lines')
        for label in self.tick_labels:
            label.draw(self.canvas.tr_sys)
        self.tick_marks.draw(self.canvas.tr_sys)

    def on_mouse_move(self, event):
        x, y = event.pos
        sec_per_pixel = self.dt / self.canvas.size[0]
        if event.is_dragging:
            x1, y1 = event.last_event.pos
            dx = x1 - x
            self.t0 += dx * sec_per_pixel
        row_height = ((self.canvas.height - self.margin['top']) /
                      self._row_count)
        row = util.clip(int((event.pos[1] - self.margin['top']) / row_height),
                        0, 119)
        try:
            if self.display_selected:
                self.electrode = self.selected_electrodes[row]
            else:
                self.electrode = self.data[row].tag
        except IndexError:
            self.electrode = ''
        self.mouse_t = self.t0 + sec_per_pixel * x

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / self.canvas.size[0]
        rel_x = event.pos[0]

        target_time = rel_x * sec_per_pixel + self.t0
        scale = math.exp(2.5 * -np.sign(event.delta[1]) * self.scroll_factor)
        self.dt *= scale
        self.velocity *= scale

        sec_per_pixel = self.dt / self.canvas.size[0]
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_key_release(self, event):
        if event.key == 'Enter' and len(self.selected_electrodes) > 0:
            self.display_selected = True
            self.resample()
        elif event.key == 'Escape':
            self.selected_electrodes = []
            self.display_selected = False
            self.resample()
        self.update_extra_text()

    def on_mouse_release(self, event):
        dx = self.canvas.mouse_pos[0] - self.canvas.prev_mouse_pos[0]
        self.velocity = self.dt * dx / self.canvas.size[0]

        if 'shift' in event.modifiers:
            if self.electrode in self.selected_electrodes:
                self.selected_electrodes.remove(self.electrode)
            else:
                self.selected_electrodes.append(self.electrode)
            self.update_extra_text()

    def on_mouse_press(self, event):
        self.velocity = 0

    def on_mouse_double_click(self, event):
        if self.display_selected:
            self.selected_electrodes = []
            self.display_selected = False
        else:
            self.selected_electrodes = [self.electrode]
            self.display_selected = True
        self.resample()
        self.update_extra_text()

    def on_resize(self, event):
        self.program['u_top_margin'] = 20.0 * 2.0 / self.canvas.size[1]

    def on_tick(self, event):
        self.velocity *= 0.98
        self.t0 -= self.velocity
        self.update()

    def on_show(self):
        self.canvas.enable_antialiasing()

    def on_hide(self):
        self.velocity = 0

    def update_extra_text(self):
        if len(self.selected_electrodes) > 0:
            self.extra_text = ('Selected: %s' %
                               ', '.join(self.selected_electrodes))
        else:
            self.extra_text = ''
