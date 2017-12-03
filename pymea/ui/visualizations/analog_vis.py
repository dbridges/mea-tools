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

from PyQt5 import QtGui, QtCore, QtWidgets  # noqa


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
    attribute vec4 a_color;

    uniform vec2 u_scale;
    uniform float u_pan;
    uniform float u_height;
    uniform float u_adj_y_scale;

    varying vec4 v_color;

    void main(void)
    {
        float y_offset = u_height * (a_position.z + 0.5);
        gl_Position = vec4(u_scale.x * (a_position.x - u_pan) - 1,
                           u_adj_y_scale * a_position.y + 1 - y_offset,
                           0.0, 1.0);
        gl_PointSize = 6.0;
        v_color = a_color;
    }
    """

    POINT_FRAGMENT_SHADER = """
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    def __init__(self, canvas, analog_data, spike_data):
        super().__init__()
        self.canvas = canvas
        self.analog_data = analog_data
        self.raw_data = spike_data
        self.spike_data = mea.MEASpikeDict(spike_data)
        self.show_spikes = False
        self._dim_conductance = False
        self._t0 = 0
        self._dt = 20
        self._y_scale = 150
        self._pan = 0
        self._scale = 1
        self.mouse_t = 0
        self.electrode = ''
        self.selected_electrodes = ['h11']  # l5, m5
        self.strip_program = gloo.Program(self.STRIP_VERTEX_SHADER,
                                          self.STRIP_FRAGMENT_SHADER)
        self.strip_program['u_color'] = Theme.blue
        self.point_program = gloo.Program(self.POINT_VERTEX_SHADER,
                                          self.POINT_FRAGMENT_SHADER)
        self.pan = self._t0
        self.scale = (2.0 / self._dt, 1 / self._y_scale)
        self.velocity = 0
        self.measuring = False
        self.measure_start = (0, 0)
        self.measure_line = visuals.LineVisual(np.array(((0, 0), (100, 100))),
                                               Theme.yellow)
        self.scale_bar = visuals.LineVisual(np.array(((10, 10), (200, 10))),
                                            Theme.black,
                                            width=10,
                                            method='agg')
        self.scale_label = visuals.TextVisual('', font_size=8)
        self.extra_text = ''
        self._filtered = False
        self._filter_cutoff = [200, 4000]
        self.all_spike_colors = []
        self.propagation_spike_colors = []
        self.resample()
        self.background_color = Theme.background

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0 - self.dt/2,
                             self.analog_data.index[-1] - self.dt/2)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, self.analog_data.index[-1])
        frac = util.nearest_decimal(self.dt / 6)
        multiplier = int(self.dt / 6 / frac)
        width = frac * multiplier * self.canvas.width / self.dt
        max_width = self.canvas.width / 6
        center_x = self.canvas.width - max_width / 2 - 20

        self.scale_bar.set_data(np.array(
            ((center_x - width / 2, 20),
             (center_x + width / 2, 20))))

        self.scale_label.pos = (center_x, 40)
        scale_val = multiplier * frac
        if scale_val < 0.1:
            self.scale_label.text = '%1.1f ms' % (1000 * multiplier * frac)
        else:
            self.scale_label.text = '%1.1f s' % (multiplier * frac)

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
            self._y_scale * len(self.selected_electrodes))
        self.strip_program['u_height'] = 2.0 / len(self.selected_electrodes)
        self.point_program['u_adj_y_scale'] = 1 / (
            self._y_scale * len(self.selected_electrodes))
        self.point_program['u_height'] = 2.0 / len(self.selected_electrodes)

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

    @property
    def dim_conductance(self):
        return self._dim_conductance

    @dim_conductance.setter
    def dim_conductance(self, val):
        self._dim_conductance = val
        try:
            if val:
                self.point_program['a_color'] = self.propagation_spike_colors
            else:
                self.point_program['a_color'] = self.all_spike_colors
        except ValueError:
            self.point_program['a_color'] = np.empty((1, 4), dtype=np.float32)

    def draw(self):
        gloo.clear(self.background_color)
        if self.measuring:
        if self.show_spikes and len(self.spike_data) > 0:
            self.point_program.draw('points')
        self.strip_program.draw('line_strip')
        self.scale_bar.draw(self.canvas.tr_sys)
        self.scale_label.draw(self.canvas.tr_sys)

    def resample(self):
        xs = []
        ys = []
        zs = []
        spike_data = []
        self.all_spike_colors = []
        self.propagation_spike_colors = []
        for i, e in enumerate(self.selected_electrodes):
            x = self.analog_data[e].index.values.astype(np.float32)
            if self.filtered:
                y = mea.bandpass_filter(self.analog_data[e],
                                        *self._filter_cutoff).values
            else:
                y = self.analog_data[e].values
            z = np.full_like(x, i)
            xs.append(x)
            ys.append(y)
            zs.append(z)

            # TODO vectorize this
            # find spikes for this electrode
            electrode_spikes = [k for k in self.spike_data.keys()
                                if k.split('.')[0] == e]
            electrode_spikes.sort()
            for k, esub in enumerate(electrode_spikes):
                try:
                    unit_number = int(esub.split('.')[1])
                except IndexError:
                    unit_number = 0
                for j, row in self.spike_data[esub].iterrows():
                    spike_data.append((row.time, row.amplitude, i))
                    try:
                        if unit_number == -1:
                            # Unsorted spikes in black.
                            self.all_spike_colors.append(Theme.black)
                            self.propagation_spike_colors.append(Theme.black)
                        elif row.conductance:
                            self.propagation_spike_colors.append(Theme.gray)
                            self.all_spike_colors.append(
                                Theme.indexed(unit_number))
                        else:
                            self.propagation_spike_colors.append(
                                Theme.indexed(unit_number))
                            self.all_spike_colors.append(
                                Theme.indexed(unit_number))
                    except AttributeError:
                        self.all_spike_colors.append(
                            Theme.indexed(unit_number))
                        self.propagation_spike_colors.append(
                            Theme.indexed(unit_number))

        self.strip_program['a_position'] = np.column_stack(
            (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)))
        if len(spike_data) > 0:
            self.point_program['a_position'] = spike_data
            if self.dim_conductance:
                self.point_program['a_color'] = self.propagation_spike_colors
            else:
                self.point_program['a_color'] = self.all_spike_colors
        else:
            self.point_program['a_position'] = np.empty((1, 3),
                                                        dtype=np.float32)
            self.point_program['a_color'] = np.empty((1, 4), dtype=np.float32)

        if self.filtered:
            self.strip_program['u_color'] = Theme.pink
        else:
            self.strip_program['u_color'] = Theme.blue

    def update(self):
        self.pan = self.t0
        self.scale = (2.0 / self.dt, 1 / self._y_scale)

    def selected_unit(self):
        """
        Returns the electrode + unit number (i.e. h11.0), of the closest spike
        to the mouse if it is within 8 px.
        """
        # Get label of electrode closest to mouse for selected
        # channel.
        t = self.mouse_t
        df = self.raw_data[
            self.raw_data.electrode.str.startswith(
                self.electrode + '.')]
        if len(df) == 0:
            # We probably don't have spike sorted data, so just return the
            # electrode.
            return self.electrode
        row = df.iloc[(df.time - t).abs().argsort()].iloc[0]
        tdiff = np.abs(row.time - t)
        px_per_sec = self.canvas.size[0] / self.dt
        if tdiff * px_per_sec < 8:
            return row.electrode
        else:
            return self.electrode

    def on_mouse_move(self, event):
        x, y = event.pos
        x1, y1 = event.last_event.pos
        sec_per_pixel = self.dt / self.canvas.size[0]
        if event.is_dragging:
            if event.button == 1 and 'shift' not in event.modifiers:
                dx = x1 - x
                self.t0 += dx * sec_per_pixel
            elif event.button == 1 and 'shift' in event.modifiers:
                self.measuring = True
                self.extra_text = 'dt: %1.4f' % (
                    sec_per_pixel * np.abs(x - self.measure_start[0]))
                self.measure_line.set_data(np.array((self.measure_start,
                                                     event.pos)))
        else:
            yscale = (2 * self.y_scale /
                      self.canvas.size[1] * len(self.selected_electrodes))
            yadj = y % (self.canvas.size[1] / len(self.selected_electrodes))
            self.extra_text = ('%1.1f uV' %
                               -(yadj * yscale - self.y_scale))
        self.mouse_t = self.t0 + sec_per_pixel * x
        try:
            self.electrode = self.selected_electrodes[int(
                y // (self.canvas.size[1] / len(self.selected_electrodes)))]
        except IndexError:
            self.electrode = ''

    def on_mouse_release(self, event):
        if event.button == 1:
            dx = self.canvas.mouse_pos[0] - self.canvas.prev_mouse_pos[0]
            self.velocity = self.dt * dx / self.canvas.size[0]
        elif event.button == 2:
            unit = self.selected_unit()
            menu = QtWidgets.QMenu(None)
            menu.addAction('Show Multi-electrode Signal (%s)' % unit.upper())
            try:
                action = menu.exec_(event.native.globalPos())
                if action.text().startswith('Show Multi-electrode Signal'):
                    self.canvas.show_conduction([unit])
            except RuntimeError:
                pass
            except AttributeError:
                pass
        self.measuring = False
        self.extra_text = ''

    def on_mouse_press(self, event):
        self.velocity = 0
        if event.button == 1 and 'shift' in event.modifiers:
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
        elif event.key == 'c':
            self.canvas.show_conduction()

    def on_show(self):
        self.canvas.disable_antialiasing()
        self.resample()

    def on_hide(self):
        self.velocity = 0
