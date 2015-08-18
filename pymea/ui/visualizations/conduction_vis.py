# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import sys
import math

import numpy as np
from vispy import visuals, gloo

from .base import LineCollection, Visualization, Theme
import pymea as mea
import pymea.util as util


class MEA120ConductionVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec4 a_position;
    attribute vec4 a_color;

    uniform float u_width;
    uniform vec2 u_pan;
    uniform float u_y_scale;

    varying vec2 v_index;
    varying vec4 v_color;

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
        v_color = a_color;
        gl_Position = vec4(position + pan, 0.0, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    varying vec2 v_index;
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;

        if (fract(v_index.x) > 0.0 || fract(v_index.y) > 0.0) {
            discard;
        }
    }
    """

    def __init__(self, canvas, analog_data, spike_data):
        self.canvas = canvas
        self.analog_data = analog_data
        prespikes = spike_data
        prespikes.electrode = prespikes.electrode.str.split('.').str.get(0)
        self.spike_data = mea.MEASpikeDict(prespikes)
        self._t0 = 0
        self._dt = 20
        self.mouse_t = 0
        self.electrode = ''
        self._y_scale = 150

        self.extra_text = ''

        self._time_window = 5  # in milliseconds
        self._selected_electrodes = []

        # Create shaders
        self.program = gloo.Program(self.VERTEX_SHADER,
                                    self.FRAGMENT_SHADER)
        self.program['u_y_scale'] = self._y_scale
        self.program['a_position'] = [[0,0,0,0]]
        self.program['a_color'] = [[0,0,0,0]]
        self.program['u_width'] = 100
        self.grid = LineCollection()
        self.create_grid()
        self.electrode_cols = [c for c in 'ABCDEFGHJKLM']
        self.sample_rate = 1.0 / (
            self.analog_data.index[1] - self.analog_data.index[0])

        self.measuring = False
        self.measure_start = (0, 0)
        self.measure_line = visuals.LineVisual(np.array(((0, 0), (100, 100))),
                                               Theme.yellow,
                                               method='agg')


    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = val

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, 10)
        self.mouse_t = self._t0

    @property
    def time_window(self):
        return self._time_window

    @time_window.setter
    def time_window(self, val):
        self._time_window = val
        self.update()

    @property
    def y_scale(self):
        return self._y_scale

    @y_scale.setter
    def y_scale(self, val):
        self.program['u_y_scale'] = val
        self._y_scale = val

    @property
    def selected_electrodes(self):
        return self._selected_electrodes

    @selected_electrodes.setter
    def selected_electrodes(self, val):
        self._selected_electrodes = val
        self.update()

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

    def resample(self):
        if len(self.selected_electrodes) != 2:
            return

        keys = self.selected_electrodes
        keys = [keys[0], keys[1]]
        rest = list(self.analog_data.columns.values)
        rest.remove(keys[0])
        rest.remove(keys[1])
        keys.extend(rest)

        waveforms = mea.extract_conduction_windows(
            keys, self.spike_data, self.analog_data, self.time_window / 1000)

        dt = self.analog_data['a8'].index[1] - self.analog_data['a8'].index[0]
        count = int(self.time_window / dt / 1000)

        n = 0
        data = np.empty(
            (2 * len(waveforms) * count * len(waveforms['a8']), 4),
            dtype=np.float32)
        colors = np.empty_like(data)
        flip = False
        opacity = 0.03 - 0.0001 * len(waveforms['a8'])
        for electrode, waves in waveforms.items():
            col, row = mea.coordinates_for_electrode(electrode)
            row = 12 - row - 1
            for i, wave in enumerate(waves):
                flip = i % 2 == 1
                if flip:
                    wave = wave[::-1]
                data[n:n+count] = np.transpose([
                    [col]*count,
                    [row]*count,
                    np.arange(count, 0, -1) if flip else np.arange(count),
                    wave
                ])
                colors[n:n+count] = np.array((0.4, 0.4, 0.4, opacity))
                n += count
            if flip:
                avg_wave = waves.mean(0)
            else:
                avg_wave = waves.mean(0)[::-1]
            data[n:n+count] = np.transpose([
                [col]*count,
                [row]*count,
                np.arange(count, 0, -1) if not flip else np.arange(count),
                avg_wave
            ])
            colors[n:n+count] = np.array(Theme.black)
            n += count

        self.program['u_width'] = count
        self.program['a_position'] = data
        self.program['a_color'] = colors

    def update(self, resample=True):
        if resample:
            self.resample()

    def draw(self):
        gloo.clear(Theme.white)
        self.grid.draw(self.canvas.tr_sys)
        if self.measuring:
            self.measure_line.draw(self.canvas.tr_sys)
        self.program.draw('line_strip')

    def on_mouse_move(self, event):
        x, y = event.pos

        cell_width = self.canvas.size[0] / 12.0
        cell_height = self.canvas.size[1] / 12.0
        col = int(x / cell_width)
        row = int(y / cell_height + 1)
        if row < 1 or row > 12 or col < 0 or col > 11:
            self.electrode = ''
        else:
            self.electrode = mea.tag_for_electrode((col, row))

        sec_per_pixel = self.time_window / self.canvas.size[0] * 12

        if event.is_dragging and event.button == 2:
            self.measuring = True
            self.extra_text = 'dt: %1.1f ms' % (
                sec_per_pixel * ((x%cell_width - self.measure_start[0]%cell_width)))
            self.measure_line.set_data(np.array((self.measure_start,
                                                 event.pos)))

    def on_mouse_double_click(self, event):
        self.canvas.show_analog_grid()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.measure_start = event.pos

    def on_mouse_release(self, event):
        self.extra_text = ''
        self.measuring = False

    def on_key_release(self, event):
        if event.key == 'Escape':
            self.canvas.show_analog_grid()

    def on_mouse_wheel(self, event):
        scale = math.exp(2.5 * -np.sign(event.delta[1]) * self.scroll_factor)
        self.y_scale *= scale

    def on_tick(self, event):
        pass

    def on_resize(self, event):
        self.create_grid()

    def on_show(self):
        self.canvas.enable_antialiasing()
