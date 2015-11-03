# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import math

import numpy as np
from vispy import visuals, gloo

from .base import LineCollection, Visualization, Theme
import pymea as mea
import pymea.util as util


class MEAConductionVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec4 a_position;
    attribute vec4 a_color;

    uniform float u_width;
    uniform vec2 u_pan;
    uniform vec2 u_scale;
    uniform float u_rows;
    uniform float u_columns;

    varying vec2 v_index;
    varying vec4 v_color;
    varying float v_drop;

    void main (void)
    {
        float height = 2.0 / u_rows;
        float width = 2.0 / u_columns;
        float y_scale = height / (2 * u_scale.y);
        vec2 pan = vec2(-1, -1);

        vec2 position = vec2(a_position.x * width - (u_width / 2 - u_scale.x * 10) * width / (u_scale.x * 20) +
                             width * a_position.z / (u_scale.x * 20),
                             a_position.y * height + height / 2 + y_scale *
                             clamp(a_position.w, -u_scale.y, u_scale.y));
        v_index = a_position.xy;
        v_color = a_color;
        if (position.x < a_position.x * width || position.x > (a_position.x + 1) * width) {
            v_drop = 1;
        } else {
            v_drop = 0;
        }
        gl_Position = vec4(position + pan, 0.0, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    varying vec2 v_index;
    varying vec4 v_color;
    varying float v_drop;

    void main()
    {
        gl_FragColor = v_color;

        if (v_drop > 0.5 || fract(v_index.x) > 0.0 || fract(v_index.y) > 0.0) {
            discard;
        }
    }
    """

    def __init__(self, canvas, analog_data, spike_data):
        super().__init__()
        self.canvas = canvas
        self.analog_data = analog_data
        self.spike_data = mea.MEASpikeDict(spike_data)

        prespikes = spike_data.copy()
        prespikes.electrode = prespikes.electrode.str.split('.').str.get(0)
        self.condensed_spike_data = mea.MEASpikeDict(prespikes)

        self._t0 = 0
        self._dt = 20
        self.mouse_t = 0
        self.electrode = ''
        self._time_window = 20  # in milliseconds
        self._scale = (5, 150)
        self.extra_text = ''
        self._selected_electrodes = []
        self.program = gloo.Program(self.VERTEX_SHADER,
                                    self.FRAGMENT_SHADER)
        self.program['u_scale'] = self._scale
        self.program['a_position'] = [[0, 0, 0, 0]]
        self.program['a_color'] = [[0, 0, 0, 0]]
        self.program['u_width'] = 100
        self.program['u_rows'] = self.canvas.layout.rows
        self.program['u_columns'] = self.canvas.layout.columns
        self.grid = LineCollection()
        self.create_grid()
        self.sample_rate = 1.0 / (
            self.analog_data.index[1] - self.analog_data.index[0])
        self.measuring = False
        self.measure_start = (0, 0)
        self.measure_line = visuals.LineVisual(np.array(((0, 0), (0, 0))),
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

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self.program['u_scale'] = val
        self._scale = val

    @property
    def selected_electrodes(self):
        return [e.split('.')[0] for e in self._selected_electrodes]

    @selected_electrodes.setter
    def selected_electrodes(self, val):
        self._selected_electrodes = val
        self.update()

    def create_grid(self):
        self.grid.clear()
        width = self.canvas.size[0]
        height = self.canvas.size[1]
        cell_width = width / self.canvas.layout.columns
        cell_height = height / self.canvas.layout.rows

        # vertical lines
        for x in np.arange(cell_width, width, cell_width):
            self.grid.append((x, 0), (x, height), Theme.grid_line)
        # horizontal lines
        for y in np.arange(cell_height, height, cell_height):
            self.grid.append((0, y), (width, y), Theme.grid_line)

    def resample(self):
        if len(self._selected_electrodes) < 1:
            return
        elif len(self._selected_electrodes) == 2:
            keys = self._selected_electrodes
            keys = [keys[0], keys[1]]
            rest = list(self.analog_data.columns.values)
            rest.remove(keys[0])
            rest.remove(keys[1])
            keys.extend(rest)

            waveforms = mea.extract_conduction_windows(
                keys,
                self.condensed_spike_data,
                self.analog_data,
                self.time_window / 1000)
        else:
            # just use first electrode
            electrode = self._selected_electrodes[0]
            if '.' in electrode:
                times = self.spike_data[electrode][:200].time
            else:
                times = self.condensed_spike_data[electrode][:200].time
            waveforms = {}
            for key in self.analog_data.columns.values:
                waveforms[key] = mea.extract_waveforms(
                    self.analog_data[key],
                    times,
                    window_len=self.time_window / 1000,
                    upsample=1)

        pt_count = len(waveforms['a8'][0])

        # Only show first 50 waves, but average all of them
        n = 0
        channel_count = len(waveforms)

        waveform_count = len(waveforms['a8'])
        if waveform_count > 50:
            waveform_count = 50

        data = np.empty(
            (channel_count * pt_count * (waveform_count + 2), 4),
            dtype=np.float32)
        colors = np.empty_like(data)
        flip = False
        opacity = 0.16 - 0.0001 * waveform_count
        for electrode, waves in waveforms.items():
            col, row = self.canvas.layout.coordinates_for_electrode(electrode)
            row = self.canvas.layout.rows - row - 1
            for i, wave in enumerate(waves):
                flip = i % 2 == 1
                if flip:
                    wave = wave[::-1]
                data[n:n+pt_count] = np.transpose([
                    np.full((pt_count,), col),
                    np.full((pt_count,), row),
                    np.arange(pt_count, 0, -1) if flip else np.arange(pt_count),  # noqa
                    wave
                ])
                colors[n:n+pt_count] = np.array((0.4, 0.4, 0.4, opacity))
                n += pt_count
                if i >= waveform_count:
                    break
            if flip:
                avg_wave = waves.mean(0)
            else:
                avg_wave = waves.mean(0)[::-1]
            data[n:n+pt_count] = np.transpose([
                np.full((pt_count,), col),
                np.full((pt_count,), row),
                np.arange(pt_count, 0, -1) if not flip else np.arange(pt_count),  # noqa
                avg_wave
            ])
            colors[n:n+pt_count] = np.array(Theme.black)
            n += pt_count

        self.program['u_width'] = pt_count
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

        cell_width = self.canvas.size[0] / self.canvas.layout.columns
        cell_height = self.canvas.size[1] / self.canvas.layout.rows
        col = int(x / cell_width)
        row = int(y / cell_height + 1)
        if (row < 1 or row > self.canvas.layout.rows or
                col < 0 or col > self.canvas.layout.columns - 1):
            self.electrode = ''
        else:
            self.electrode = \
                self.canvas.layout.electrode_for_coordinate((col, row))

        sec_per_pixel = (self.scale[0] / self.canvas.size[0] *
                         self.layout.columns)

        if event.button != 1:
            return

        if event.is_dragging and 'shift' in event.modifiers:
            self.measuring = True
            self.extra_text = 'dt: %1.1f ms' % (
                sec_per_pixel * (x % cell_width -
                                 self.measure_start[0] % cell_width))
            self.measure_line.set_data(np.array((self.measure_start,
                                                 event.pos)))

    def on_mouse_double_click(self, event):
        self.canvas.show_previous()

    def on_mouse_press(self, event):
        if event.button == 1 and 'shift' in event.modifiers:
            self.measure_start = event.pos

    def on_mouse_release(self, event):
        self.extra_text = ''
        self.measuring = False

    def on_key_release(self, event):
        if event.key == 'Escape':
            self.canvas.show_analog_grid()

    def on_mouse_wheel(self, event):
        if 'shift' in event.modifiers:
            # Time window scaling.
            scale = math.exp(
                2.5 * -np.sign(event.delta[1]) * self.scroll_factor)
            self.scale = (util.clip(scale * self.scale[0],
                                    0.33 * 5,
                                    3*self.time_window),
                          self.scale[1])
        else:
            # Amplitude scaling
            scale = math.exp(
                2.5 * -np.sign(event.delta[1]) * self.scroll_factor)
            self.scale = (self.scale[0], scale * self.scale[1])

    def on_tick(self, event):
        pass

    def on_resize(self, event):
        self.create_grid()

    def on_show(self):
        self.canvas.enable_antialiasing()
        self.measure_line.draw(self.canvas.tr_sys)
