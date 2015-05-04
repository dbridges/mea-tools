# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import math

import numpy as np
from vispy import gloo, visuals

import pymea as mea
from .base import Visualization, Theme
import pymea.util as util


class MEAAnalogVisualization(Visualization):
    STRIP_VERTEX_SHADER = """
    // x, y is position, z is index to avoid connecting traces from
    // two electrodes.
    attribute vec3 a_position;

    uniform vec2 u_scale;
    uniform float u_pan;
    uniform float u_height;
    uniform float u_adj_y_scale;

    varying float v_index;

    void main(void)
    {
        v_index = a_position.z;
        float y_offset = u_height * (a_position.z + 0.5);
        gl_Position = vec4(u_scale.x * (a_position.x - u_pan) - 1,
                           u_adj_y_scale * a_position.y + 1 - y_offset,
                           0.0, 1.0);
    }
    """

    STRIP_FRAGMENT_SHADER = """
    uniform vec4 u_color;
    varying float v_index;

    void main()
    {
        gl_FragColor = u_color;

        if (fract(v_index) > 0.0) {
            discard;
        }

    }
    """

    POINT_VERTEX_SHADER = """
    // x, y is position, z is index to avoid connecting traces from
    // two electrodes.
    attribute vec3 a_position;

    uniform vec2 u_scale;
    uniform float u_pan;
    uniform float u_height;
    uniform float u_adj_y_scale;

    void main(void)
    {
        float y_offset = u_height * (a_position.z + 0.5);
        gl_Position = vec4(u_scale.x * (a_position.x - u_pan) - 1,
                           u_adj_y_scale * a_position.y + 1 - y_offset,
                           0.0, 1.0);
        gl_PointSize = 6.0;
    }
    """

    POINT_FRAGMENT_SHADER = """
    uniform vec4 u_color;

    void main()
    {
        gl_FragColor = u_color;
    }
    """

    def __init__(self, canvas, analog_data, spike_data):
        self.canvas = canvas
        self.analog_data = analog_data
        self.spike_data = mea.MEASpikeDict(spike_data)
        self.show_spikes = False
        self._t0 = 0
        self._dt = 20
        self._y_scale = 150
        self._pan = 0
        self._scale = 1
        self.mouse_t = 0
        self.electrode = ''
        self.electrodes = ['h11']  # l5, m5

        self.strip_program = gloo.Program(self.STRIP_VERTEX_SHADER,
                                          self.STRIP_FRAGMENT_SHADER)
        self.strip_program['u_color'] = Theme.blue

        self.point_program = gloo.Program(self.POINT_VERTEX_SHADER,
                                          self.POINT_FRAGMENT_SHADER)
        self.point_program['u_color'] = Theme.yellow

        self.pan = self._t0
        self.scale = (2.0/self._dt, 1/self._y_scale)

        self.velocity = 0

        self.measuring = False
        self.measure_start = (0, 0)
        self.measure_line = visuals.LineVisual(np.array(((0, 0), (100, 100))),
                                               Theme.grid_line)
        self.extra_text = ''
        self._filtered = False
        self._filter_cutoff = [200, 4000]
        self.resample()

        self.background_color = Theme.background

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0 - self.dt/2,
                             self.analog_data.index[-1] - self.dt/2)
        self.update()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, self.analog_data.index[-1])
        self.update()

    @property
    def pan(self):
        return self._pan

    @pan.setter
    def pan(self, val):
        self.strip_program['u_pan'] = val
        self.point_program['u_pan'] = val

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self.strip_program['u_scale'] = val
        self.point_program['u_scale'] = val

    @property
    def y_scale(self):
        return self._y_scale

    @y_scale.setter
    def y_scale(self, val):
        self._y_scale = val
        self.strip_program['u_adj_y_scale'] = 1 / (
            self._y_scale * len(self.electrodes))
        self.strip_program['u_height'] = 2.0 / len(self.electrodes)
        self.point_program['u_adj_y_scale'] = 1 / (
            self._y_scale * len(self.electrodes))
        self.point_program['u_height'] = 2.0 / len(self.electrodes)
        self.update()

    @property
    def filtered(self):
        return self._filtered

    @filtered.setter
    def filtered(self, val):
        self._filtered = val
        self.resample()

    @property
    def filter_cutoff(self):
        return self._filter_cutoff

    @filter_cutoff.setter
    def filter_cutoff(self, val):
        self._filter_cutoff = val
        self.resample()

    def draw(self):
        gloo.clear(self.background_color)
        if self.measuring:
            self.measure_line.draw(self.canvas.tr_sys)
        if self.show_spikes and len(self.spike_data) > 0:
            self.point_program.draw('points')
        self.strip_program.draw('line_strip')

    def resample(self):
        xs = []
        ys = []
        zs = []
        spike_data = []
        for i, e in enumerate(self.electrodes):
            x = self.analog_data[e].index.values.astype(np.float32)
            if self.filtered:
                y = mea.filter(self.analog_data[e],
                               *self._filter_cutoff).values
            else:
                y = self.analog_data[e].values
            z = np.full_like(x, i)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            spike_data.extend(
                [(r.time, r.amplitude, i) for j, r in
                 self.spike_data[e].iterrows()])

        self.strip_program['a_position'] = np.column_stack(
            (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)))

        try:
            self.point_program['a_position'] = spike_data
        except ValueError:
            pass

        if self.filtered:
            self.strip_program['u_color'] = Theme.pink
        else:
            self.strip_program['u_color'] = Theme.blue

    def update(self):
        self.pan = self.t0
        self.scale = (2.0 / self.dt, 1 / self._y_scale)

    def on_mouse_move(self, event):
        x, y = event.pos
        x1, y1 = event.last_event.pos
        sec_per_pixel = self.dt / self.canvas.size[0]
        if event.is_dragging:
            if event.button == 1:
                dx = x1 - x
                self.t0 += dx * sec_per_pixel
            elif event.button == 2:
                self.measuring = True
                self.extra_text = 'dt: %1.4f' % (
                    sec_per_pixel * (x - self.measure_start[0]))
                self.measure_line.set_data(np.array((self.measure_start,
                                                     event.pos)))
        self.mouse_t = self.t0 + sec_per_pixel * x
        try:
            self.electrode = self.electrodes[int(
                y // (self.canvas.size[1] / len(self.electrodes)))]
        except IndexError:
            self.electrode = ''

    def on_mouse_release(self, event):
        if event.button == 1:
            dx = self.canvas.mouse_pos[0] - self.canvas.prev_mouse_pos[0]
            self.velocity = self.dt * dx / self.canvas.size[0]
        self.measuring = False
        self.extra_text = ''

    def on_mouse_press(self, event):
        self.velocity = 0
        if event.button == 2:
            self.measure_start = event.pos

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / self.canvas.size[0]
        rel_x = event.pos[0]

        target_time = rel_x * sec_per_pixel + self.t0
        scale = math.exp(2.5 * -np.sign(event.delta[1]) * self.scroll_factor)
        self.dt *= scale
        self.velocity *= scale

        sec_per_pixel = self.dt / self.canvas.size[0]
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_mouse_double_click(self, event):
        self.canvas.show_analog_grid()

    def on_tick(self, event):
        self.velocity *= 0.98
        self.t0 -= self.velocity
        self.update()

    def on_key_release(self, event):
        if event.key == 'b':
            if self.background_color == Theme.background:
                self.background_color = Theme.white
            else:
                self.background_color = Theme.background

    def on_show(self):
        self.canvas.disable_antialiasing()
        self.resample()

    def on_hide(self):
        self.velocity = 0
