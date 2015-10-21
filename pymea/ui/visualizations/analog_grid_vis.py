# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import sys
import math

import numpy as np
from vispy import gloo

from .base import LineCollection, Visualization, Theme
import pymea as mea
import pymea.util as util


class MEA120GridVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec4 a_position;

    uniform float u_width;
    uniform vec2 u_pan;
    uniform float u_y_scale;

    varying vec2 v_index;

    void main (void)
    {
        float height = 2.0 / 12.0;
        float width = 2.0 / 12.0;
        float scale = height / (2 * u_y_scale);
        vec2 pan = vec2(-1, -1);

        vec2 position = vec2(a_position.x * width +
                            width * a_position.z / u_width,
                            a_position.y * height + height / 2 + scale *
                            clamp(a_position.w, -u_y_scale, u_y_scale));
        v_index = a_position.xy;
        gl_Position = vec4(position + pan, 0.0, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    uniform vec4 u_color;
    varying vec2 v_index;

    void main()
    {
        gl_FragColor = u_color;

        if (fract(v_index.x) > 0.0 || fract(v_index.y) > 0.0) {
            discard;
        }
    }
    """

    def __init__(self, canvas, data):
        super().__init__()
        self.canvas = canvas
        self.data = data
        self._t0 = 0
        self._dt = 20
        self.mouse_t = 0
        self.electrode = ''
        self._y_scale = 150
        self.program = gloo.Program(self.VERTEX_SHADER,
                                    self.FRAGMENT_SHADER)
        self.program['u_color'] = Theme.blue
        self.program['u_y_scale'] = self._y_scale
        self.grid = LineCollection()
        self.create_grid()
        self.electrode_cols = [c for c in 'ABCDEFGHJKLM']
        self.sample_rate = 1.0 / (self.data.index[1] - self.data.index[0])
        self.resample()
        self.selected_electrodes = []
        self.extra_text = ''
        self.needs_update = False

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0, self.data.index[-1] - self.dt/2)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, 30)
        self.mouse_t = self._t0

    @property
    def y_scale(self):
        return self._y_scale

    @y_scale.setter
    def y_scale(self, val):
        self.program['u_y_scale'] = val
        self._y_scale = val

    def create_grid(self):
        self.grid.clear()
        width = self.canvas.size[0]
        height = self.canvas.size[1]
        cell_width = width / 12
        cell_height = height / 12

        # vertical lines
        for x in np.arange(cell_width, width, cell_width):
            self.grid.append((x, 0), (x, height), Theme.grid_line)
        # horizontal lines
        for y in np.arange(cell_height, height, cell_height):
            self.grid.append((0, y), (width, y), Theme.grid_line)

    def resample(self, bin_count=100):
        start_i = int(self.t0 * self.sample_rate)
        end_i = util.clip(start_i + int(self.dt * self.sample_rate),
                          start_i, sys.maxsize)
        bin_size = (end_i - start_i) // bin_count
        if bin_size < 1:
            bin_size = 1
        bin_count = len(np.arange(start_i, end_i, bin_size))

        data = np.empty((self.data.shape[1], 2*bin_count, 4), dtype=np.float32)

        for i, column in enumerate(self.data):
            v = mea.min_max_bin(self.data[column].values[start_i:end_i],
                                bin_size, bin_count+1)
            col, row = mea.coordinates_for_electrode(column)
            row = 12 - row - 1
            x = np.full_like(v, col, dtype=np.float32)
            y = np.full_like(v, row, dtype=np.float32)
            t = np.arange(0, bin_count, 0.5, dtype=np.float32)
            data[i] = np.column_stack((x, y, t, v))

        # Update shader
        self.program['a_position'] = data.reshape(
            2*self.data.shape[1]*bin_count, 4)
        self.program['u_width'] = bin_count

    def update(self):
        self.resample()

    def draw(self):
        gloo.clear(Theme.background)
        self.program.draw('line_strip')
        self.grid.draw(self.canvas.tr_sys)

    def on_mouse_move(self, event):
        x, y = event.pos
        sec_per_pixel = self.dt / (self.canvas.width / 12.0)
        if event.is_dragging:
            x1, y1 = event.last_event.pos
            dx = x1 - x
            self.t0 += dx * sec_per_pixel
            self.needs_update = True

        x, y = event.pos
        cell_width = self.canvas.size[0] / 12.0
        cell_height = self.canvas.size[1] / 12.0
        col = int(x / cell_width)
        row = int(y / cell_height + 1)
        if row < 1 or row > 12 or col < 0 or col > 11:
            self.electrode = ''
        else:
            self.electrode = mea.tag_for_electrode((col, row))
        self.mouse_t = self.t0 + sec_per_pixel * (x % cell_width)

    def on_mouse_double_click(self, event):
        self.selected_electrodes = [self.electrode]
        self.update_extra_text()
        self.canvas.show_analog()

    def on_mouse_release(self, event):
        if 'shift' in event.modifiers:
            if self.electrode in self.selected_electrodes:
                self.selected_electrodes.remove(self.electrode)
            else:
                self.selected_electrodes.append(self.electrode)
            self.update_extra_text()

    def on_key_release(self, event):
        if event.key == 'Enter' and len(self.selected_electrodes) > 0:
            self.canvas.show_analog()
        elif event.key == 'Escape':
            self.selected_electrodes = []
            self.update_extra_text()
        elif event.key == 'c':
            self.canvas.show_conduction()
        elif event.key == 'r':
            self.canvas.show_raster(selected=self.selected_electrodes)

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        rel_x = event.pos[0] % (self.canvas.size[0] / 12)

        target_time = rel_x * sec_per_pixel + self.t0
        dx = -np.sign(event.delta[1]) * 2*self.scroll_factor
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        self.t0 = target_time - (rel_x * sec_per_pixel)
        self.needs_update = True

    def update_extra_text(self):
        if len(self.selected_electrodes) > 0:
            self.extra_text = ('Selected: %s' %
                               ', '.join(self.selected_electrodes))
        else:
            self.extra_text = ''

    def on_tick(self, event):
        if self.needs_update:
            self.update()
            self.needs_update = False

    def on_resize(self, event):
        self.create_grid()

    def on_show(self):
        self.selected_electrodes = []
        self.update_extra_text()
        self.canvas.disable_antialiasing()
