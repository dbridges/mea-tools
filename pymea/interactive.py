import os
import sys
import math

import numpy as np
import pandas as pd
from vispy import app, gloo, visuals
from vispy.visuals.shaders import ModularProgram
import OpenGL.GL as gl

import pymea.pymea as mea
import pymea.util as util
import pymea.mea_cython as meac


class Grid:
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


class LineCollection:
    VERTEX_SHADER = """
    attribute vec2 a_position;
    attribute vec4 a_color;

    varying vec4 v_color;

    void main (void)
    {
        v_color = a_color;
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    }
    """

    FRAGMENT_SHADER = """
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    def __init__(self):
        self._vert = []
        self._color = []
        self._program = ModularProgram(LineCollection.VERTEX_SHADER,
                                       LineCollection.FRAGMENT_SHADER)

    def clear(self):
        self._vert = []
        self._color = []

    def add(self, pt1, pt2, color=[1, 1, 1, 1]):
        self._vert.append(pt1)
        self._vert.append(pt2)
        self._color.append(color)
        self._color.append(color)
        self._program['a_position'] = np.array(self._vert, dtype=np.float32)
        self._program['a_color'] = np.array(self._color, dtype=np.float32)

    def draw(self, transforms):
        if len(self._vert) > 0:
            self._program.vert['transform'] = transforms.get_full_transform()
            self._program.draw('lines')


class Visualization:
    def __init__(self):
        pass

    def update(self):
        pass

    def draw(self):
        pass

    def on_mouse_move(self, event):
        pass

    def on_mouse_wheel(self, event):
        pass

    def on_key_release(self, event):
        pass

    def on_mouse_release(self, event):
        pass

    def on_mouse_press(self, event):
        pass

    def on_tick(self, event):
        pass

    def on_resize(self, event):
        pass


class RasterPlotVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec2 a_position;
    attribute vec3 a_color;

    uniform float u_pan;
    uniform float u_y_scale;
    uniform float u_count;
    uniform float u_top_margin;

    varying vec3 v_color;

    void main(void)
    {
        float height = (2.0 - u_top_margin) / u_count;
        gl_Position = vec4((a_position.x - u_pan) / u_y_scale - 1,
                           1 - a_position.y * height - u_top_margin, 0.0, 1.0);
        v_color = a_color;
    }
    """

    FRAGMENT_SHADER = """
    varying vec3 v_color;

    void main()
    {
        gl_FragColor = vec4(v_color, 1.0);
    }
    """

    def __init__(self, canvas, spike_data):
        self.canvas = canvas
        self.spikes = spike_data
        self.program = gloo.Program(RasterPlotVisualization.VERTEX_SHADER,
                                    RasterPlotVisualization.FRAGMENT_SHADER)
        self._t0 = 0
        self._dt = self.spikes['time'].max()
        self.program['u_pan'] = self._t0
        self.program['u_y_scale'] = self._dt/2
        self.program['u_top_margin'] = 20.0 * 2.0 / canvas.size[1]
        print(self.program['u_top_margin'])
        self.spikes[['electrode', 'time']].values
        self.electrode_row = {}
        d = self.spikes.groupby('electrode').size()
        d.sort(ascending=False)
        for i, tag in enumerate(d.index):
            self.electrode_row[tag] = i
        verticies = []
        colors = []
        for e, t in self.spikes[['electrode', 'time']].values:
            row = self.electrode_row[e]
            verticies.append((t, row))
            verticies.append((t, row + 1))
            if row % 3 == 0:
                color = (0.9254, 0.7411, 0.0588)
            elif row % 3 == 1:
                color = (0.396, 0.09, 0.62)
            else:
                color = (0.09, 0.365, 0.596)
            colors.append(color)
            colors.append(color)

        self.margin = {}
        self.margin['top'] = 20

        self.program['a_position'] = verticies
        self.program['a_color'] = colors
        self.program['u_count'] = len(self.electrode_row)
        self.velocity = 0
        self.last_dx = 0
        self.tick_separtion = 50
        self.tick_labels = [visuals.TextVisual('', font_size=10, color='w')
                            for x in range(18)]
        self.tick_marks = LineCollection()

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val,
                             -self.spikes.time.max(),
                             self.spikes.time.max())

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, self.spikes.time.max())

    def create_labels(self):
        self.tick_marks.clear()
        self.tick_marks.add((0, self.margin['top']),
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
            self.tick_labels[i].pos = (xloc, self.margin['top'] / 2 + 2)
            self.tick_labels[i].text = '%1.3f' % tick_loc
            tick_loc += self.tick_separtion
            self.tick_marks.add((xloc, self.margin['top']),
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
        self.program.draw('lines')
        for label in self.tick_labels:
            label.draw(self.canvas.tr_sys)
        self.tick_marks.draw(self.canvas.tr_sys)

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = event.press_event.pos
            x1, y1 = event.last_event.pos
            x, y = event.pos
            dx = x1 - x
            self.last_dx = dx
            sperpx = self.dt / self.canvas.size[0]
            self.t0 += dx * sperpx

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / self.canvas.size[0]
        rel_x = event.pos[0]

        target_time = rel_x * sec_per_pixel + self.t0
        dx = np.sign(event.delta[1]) * 0.05
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / self.canvas.size[0]
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_key_release(self, event):
        pass

    def on_mouse_release(self, event):
        self.velocity = -self.dt * self.last_dx / self.canvas.size[0]

    def on_mouse_press(self, event):
        self.velocity = 0
        self.last_dx = 0

    def on_resize(self, event):
        self.program['u_top_margin'] = 20.0 * 2.0 / self.canvas.size[1]

    def on_tick(self, event):
        self.velocity *= 0.98
        self.t0 -= self.velocity
        self.update()


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

    def on_tick(self, event):
        pass


class Canvas(app.Canvas):
    def __init__(self, fname):
        app.Canvas.__init__(self, keys='interactive', size=(1280, 768))

        # Load data
        if not os.path.exists(fname):
            raise IOError('File does not exist.')

        print('Loading data...')
        if fname.endswith('.csv'):
            self.spike_data = pd.read_csv(fname)
            if os.path.exists(fname[:-4] + '.h5'):
                self.store = mea.MEARecording(fname[:-4] + '.h5')
                self.data = self.store.get('all')
            else:
                self.store = None
                self.data = None
        elif fname.endswith('.h5'):
            self.store = mea.MEARecording(fname)
            self.data = self.store.get('all')
            if os.path.exists(fname[:-3] + '.csv'):
                self.spike_data = pd.read_csv(fname[:-3] + '.csv')
            else:
                self.spike_data = None

        if self.data is not None:
            self.grid_visualization = MEA120GridVisualization(self, self.data)
        else:
            self.grid_visualization = None
        if self.spike_data is not None:
            self.raster_visualization = RasterPlotVisualization(
                self, self.spike_data)
        else:
            self.raster_visualization = None

        if fname.endswith('.h5'):
            self.visualization = self.grid_visualization
        else:
            self.visualization = self.raster_visualization

        self.tr_sys = visuals.transforms.TransformSystem(self)
        self._timer = app.Timer(1/30, connect=self.on_tick, start=True)

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.width), float(self.height)
        return x/(w/2.)-1., y/(h/2.)-1.

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, *event.size)
        self.tr_sys = visuals.transforms.TransformSystem(self)
        self.visualization.on_resize(event)

    def on_draw(self, event):
        gloo.clear((0.5, 0.5, 0.5, 1))
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.visualization.draw()

    def on_mouse_move(self, event):
        self.visualization.on_mouse_move(event)

    def on_mouse_wheel(self, event):
        self.visualization.on_mouse_wheel(event)

    def on_mouse_press(self, event):
        self.visualization.on_mouse_press(event)

    def on_mouse_release(self, event):
        self.visualization.on_mouse_release(event)

    def on_key_release(self, event):
        self.visualization.on_key_release(event)

    def on_tick(self, event):
        self.visualization.on_tick(event)
        self.update()


def run(fname):
    c = Canvas(os.path.expanduser(fname))
    c.show()
    app.run()
