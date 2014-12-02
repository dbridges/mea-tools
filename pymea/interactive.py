import os
import sys
import math

import numpy as np
from vispy import app, gloo, visuals
import OpenGL.GL as gl

import pymea.pymea as mea
import pymea.util as util
import pymea.mea_cython as meac


class Grid():
    VERTEX_SHADER = """
    attribute vec2 a_position;
    uniform vec4 u_color;
    varying vec4 v_color;

    void main (void)
    {
        v_color = u_color;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    def __init__(self, cols, rows):
        self.vertical_lines = np.column_stack(
            (np.repeat(2*np.arange(1, cols)/cols - 1, 2),
             np.tile([-1, 1], cols - 1)))
        self.horizontal_lines = np.column_stack(
            (np.tile([-1, 1], rows - 1),
             np.repeat(2*np.arange(1, rows)/rows - 1, 2)))
        self.program = gloo.Program(Grid.VERTEX_SHADER, Grid.FRAGMENT_SHADER)
        self.program['a_position'] = np.concatenate(
            (self.vertical_lines, self.horizontal_lines)).astype(np.float32)

        self.program['u_color'] = np.array([0.4, 0.4, 0.4, 1.0])

    def draw(self):
        self.program.draw('lines')


class MEA120GridVisualization():
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
    varying vec2 v_index;

    void main()
    {
        gl_FragColor = vec4(0.349, 0.5, 0.715, 1.0);

        if (fract(v_index.x) > 0.0 || fract(v_index.y) > 0.0) {
            discard;
        }
    }
    """

    def __init__(self, canvas, data):
        self.canvas = canvas
        self.data = data
        self._t0 = 0
        self._dt = 20
        self.scales = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 500.0,
                       1000.0]
        self.y_scale_i = 2

        # Create shaders
        self.program = gloo.Program(MEA120GridVisualization.VERTEX_SHADER,
                                    MEA120GridVisualization.FRAGMENT_SHADER)
        self.grid = Grid(12, 12)

        self.resample()

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0, self.data.index[-1])

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, 20)

    def resample(self, bin_count=250):
        sample_rate = 1 / (self.data.index[1] - self.data.index[0])
        start_i = int(self.t0 * sample_rate)
        end_i = util.clip(start_i + int(self.dt * sample_rate),
                          start_i, sys.maxsize)
        bin_size = (end_i - start_i) // bin_count
        if bin_size < 1:
            bin_size = 1
        bin_count = len(np.arange(start_i, end_i, bin_size))

        data = np.empty((120, 2*bin_count, 4), dtype=np.float32)

        for i, column in enumerate(self.data):
            v = meac.min_max_bin(self.data[column].values[start_i:end_i],
                                 bin_size, bin_count+1)
            col, row = mea.coordinates_for_electrode(column)
            x = np.full_like(v, col, dtype=np.float32)
            y = np.full_like(v, row, dtype=np.float32)
            t = np.arange(0, (2*bin_count) / 2, 0.5, dtype=np.float32)
            data[i] = np.column_stack((x, y, t, v))

        # Update shader
        self.program['a_position'] = data.reshape(120*(2*bin_count), 4)
        self.program['u_width'] = bin_count
        self.program['u_y_scale'] = self.scales[self.y_scale_i]

    def update(self):
        self.resample()

    def draw(self):
        self.program.draw('line_strip')
        self.grid.draw()

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = event.press_event.pos
            x1, y1 = event.last_event.pos
            x, y = event.pos
            dx = x1 - x
            sperpx = self.dt / (self.canvas.size[0] / 12)
            self.t0 = util.clip(self.t0 + dx * sperpx,
                                0, self.data.index[-1])
            self.update()

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        rel_x = event.pos[0] % (self.canvas.size[0] / 12)

        target_time = rel_x * sec_per_pixel + self.t0
        dx = np.sign(event.delta[1]) * 0.05
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        self.t0 = target_time - (rel_x * sec_per_pixel)

        self.update()

    def on_key_release(self, event):
        if event.key == 'Up':
            self.y_scale_i = util.clip(self.y_scale_i + 1,
                                       0, len(self.scales) - 1)
            self.program['u_y_scale'] = self.scales[self.y_scale_i]
        elif event.key == 'Down':
            self.y_scale_i = util.clip(self.y_scale_i - 1,
                                       0, len(self.scales) - 1)
            self.program['u_y_scale'] = self.scales[self.y_scale_i]


class Canvas(app.Canvas):
    def __init__(self, fname):
        app.Canvas.__init__(self, keys='interactive', size=(1280, 768))

        # Load data
        print('Loading data...')
        self.store = mea.MEARecording(fname)
        self.data = self.store.get('all')
        self.grid_visualization = MEA120GridVisualization(self, self.data)
        self.tr_sys = visuals.transforms.TransformSystem(self)
        self.visualization = self.grid_visualization

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x/(w/2.)-1., y/(h/2.)-1.

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, *event.size)

    def on_draw(self, event):
        gloo.clear((0.9, 0.91, 0.91, 1))
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.visualization.draw()

    def on_mouse_move(self, event):
        self.visualization.on_mouse_move(event)
        self.update()

    def on_mouse_wheel(self, event):
        self.visualization.on_mouse_wheel(event)
        self.update()

    def on_key_release(self, event):
        self.visualization.on_key_release(event)
        self.update()


def run(fname):
    c = Canvas(os.path.expanduser(fname))
    c.show()
    app.run()
