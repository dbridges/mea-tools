import sys
import math

import pymea.pymea as mea
import pymea.util as util
import pymea.mea_cython as meac

from vispy import gloo, visuals
from vispy.visuals.shaders import ModularProgram
import numpy as np


class Theme:
    yellow = (0.9254, 0.7411, 0.0588, 1.0)
    purple = (0.396, 0.09, 0.62, 1.0)
    blue = (0.09, 0.365, 0.596, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)
    grid_line = (0.7, 0.7, 0.7, 1.0)
    plot_colors = (yellow, purple, blue)


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

    def append(self, pt1, pt2, color=[1, 1, 1, 1]):
        """
        pt1 : 2 tuple
            The first point on the line, in screen coordinates.
        pt2 : 2 tuple
            The second point of the line, in screen coordinates.
        color : 4 tuple
            The color of the line in (r, g, b, alpha).
        """
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

    def on_mouse_double_click(self, event):
        pass

    def on_tick(self, event):
        pass

    def on_resize(self, event):
        pass

    def on_hide(self):
        pass

    def on_show(self):
        pass


class FlashingSpikeElectrode:
    def __init__(self, tag, events):
        self.events = events
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
        self.canvas = canvas
        self.program = ModularProgram(self.VERTEX_SHADER,
                                      self.FRAGMENT_SHADER)
        self._t0 = 0
        self._dt = 10
        self.mouse_t = self.t0
        self._interval = 1/30.0
        self.electrode = ''
        self.time_scale = 1/200
        self._vert = np.zeros((120*6, 2), dtype=np.float32)
        self._color = np.zeros(120*6, dtype=np.float32)
        self.electrodes = []
        self.spikes = spike_data
        for tag, df in spike_data.groupby('electrode'):
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
        self.electrode_row = {}
        d = self.spikes.groupby('electrode').size()
        d.sort(ascending=False)
        for i, tag in enumerate(d.index):
            self.electrode_row[tag] = i
        self._row_for_electrode = {v: k for
                                   k, v in self.electrode_row.items()}
        verticies = []
        colors = []
        for e, t in self.spikes[['electrode', 'time']].values:
            row = self.electrode_row[e]
            verticies.append((t, row))
            verticies.append((t, row + 1))
            color = Theme.plot_colors[row % 3]
            colors.append(color)
            colors.append(color)

        self.margin = {}
        self.margin['top'] = 20

        self.program['a_position'] = verticies
        self.program['a_color'] = colors
        self._row_count = len(self.electrode_row)
        self.program['u_count'] = self._row_count
        self.velocity = 0
        self.tick_separtion = 50
        self.tick_labels = [visuals.TextVisual('', font_size=10, color='w')
                            for x in range(18)]
        self.tick_marks = LineCollection()
        self.mouse_t = 0

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

    @property
    def row_count(self):
        return len(self.electrode_row)

    @row_count.setter
    def row_count(self, val):
        self.program['u_count'] = val
        self._row_count = val

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
        x1, y1 = event.last_event.pos
        x, y = event.pos
        sec_per_pixel = self.dt / self.canvas.size[0]
        if event.is_dragging:
            dx = x1 - x
            self.t0 += dx * sec_per_pixel
        row_height = ((self.canvas.height - self.margin['top']) /
                      self._row_count)
        row = util.clip(int((event.pos[1] - self.margin['top']) / row_height),
                        0, 119)
        self.electrode = self._row_for_electrode[row]
        self.mouse_t = self.t0 + sec_per_pixel * x

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / self.canvas.size[0]
        rel_x = event.pos[0]

        target_time = rel_x * sec_per_pixel + self.t0
        dx = -np.sign(event.delta[1]) * 0.10
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / self.canvas.size[0]
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_key_release(self, event):
        pass

    def on_mouse_release(self, event):
        dx = self.canvas.mouse_pos[0] - self.canvas.prev_mouse_pos[0]
        self.velocity = self.dt * dx / self.canvas.size[0]

    def on_mouse_press(self, event):
        self.velocity = 0

    def on_resize(self, event):
        self.program['u_top_margin'] = 20.0 * 2.0 / self.canvas.size[1]

    def on_tick(self, event):
        self.velocity *= 0.98
        self.t0 -= self.velocity
        self.update()

    def on_hide(self):
        self.velocity = 0


class MEAAnalogVisualization(Visualization):
    VERTEX_SHADER = """
    attribute vec2 a_position;

    uniform vec4 u_color;
    uniform vec2 u_scale;
    uniform float u_pan;
    uniform float u_count;
    uniform float u_top_margin;

    varying vec4 v_color;

    void main(void)
    {
        gl_Position = vec4(u_scale.x * (a_position.x - u_pan) - 1,
                           u_scale.y * a_position.y,
                           0.0, 1.0);
        v_color = u_color;
    }
    """

    FRAGMENT_SHADER = """
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
    """

    def __init__(self, canvas, data):
        self.canvas = canvas
        self.data = data
        self._t0 = 0
        self._dt = 20
        self.mouse_t = 0
        self.electrode = ''
        self.electrodes = ['h11']  # l5, m5

        self.program = gloo.Program(self.VERTEX_SHADER,
                                    self.FRAGMENT_SHADER)
        self.program['u_pan'] = self._t0
        self.program['u_scale'] = (2.0/self._dt, 1/200)
        self.program['u_top_margin'] = 20.0 * 2.0 / canvas.size[1]
        self.program['u_count'] = len(self.electrodes)
        self.program['u_color'] = Theme.blue

        self.margin = {}
        self.margin['top'] = 20

        self.velocity = 0

        self.resample()

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0 - self.dt/2,
                             self.data.index[-1] - self.dt/2)
        self.update()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, 20)
        self.update()

    def draw(self):
        gloo.clear((0.5, 0.5, 0.5, 1))
        self.program.draw('line_strip')

    def resample(self):
        # height = (2.0 - u_top_margin) / u_count
        e = self.electrodes[0]
        x = self.data[e].index.values.astype(np.float32)
        y = self.data[e].values
        self.program['a_position'] = np.column_stack((x, y))

    def update(self):
        self.program['u_pan'] = self.t0
        self.program['u_scale'] = (2.0 / self.dt, 1/200)

    def on_show(self):
        self.resample()

    def on_mouse_move(self, event):
        x, y = event.pos
        x1, y1 = event.last_event.pos
        sec_per_pixel = self.dt / self.canvas.size[0]
        if event.is_dragging:
            dx = x1 - x
            self.t0 += dx * sec_per_pixel
        self.mouse_t = self.t0 + sec_per_pixel * x

    def on_mouse_release(self, event):
        dx = self.canvas.mouse_pos[0] - self.canvas.prev_mouse_pos[0]
        self.velocity = self.dt * dx / self.canvas.size[0]

    def on_mouse_press(self, event):
        self.velocity = 0

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / self.canvas.size[0]
        rel_x = event.pos[0]

        target_time = rel_x * sec_per_pixel + self.t0
        dx = -np.sign(event.delta[1]) * 0.025
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / self.canvas.size[0]
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_mouse_double_click(self, event):
        self.canvas.show_analog_grid()

    def on_resize(self, event):
        self.program['u_top_margin'] = (self.margin['top'] * 2.0 /
                                        self.canvas.size[1])

    def on_tick(self, event):
        self.velocity *= 0.98
        self.t0 -= self.velocity
        self.update()

    def on_hide(self):
        self.velocity = 0


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
        self.canvas = canvas
        self.data = data
        self._t0 = 0
        self._dt = 20
        self.mouse_t = 0
        self.electrode = ''
        self.scales = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 500.0,
                       1000.0]
        self.y_scale_i = 2

        # Create shaders
        self.program = gloo.Program(self.VERTEX_SHADER,
                                    self.FRAGMENT_SHADER)
        self.program['u_color'] = Theme.blue
        self.grid = LineCollection()
        self.create_grid()
        self.electrode_cols = [c for c in 'ABCDEFGHJKLM']
        self.sample_rate = 1.0 / (self.data.index[1] - self.data.index[0])

        self.resample()

        self.selected_electrodes = []

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, val):
        self._t0 = util.clip(val, 0, self.data.index[-1])
        self.update()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = util.clip(val, 0.0025, 20)
        self.mouse_t = self._t0
        self.update()

    @property
    def y_scale_index(self):
        return self.y_scale_i

    @y_scale_index.setter
    def y_scale_index(self, val):
        self.y_scale_i = util.clip(val, 0, len(self.scales) - 1)
        self.program['u_y_scale'] = self.scales[self.y_scale_i]
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

    def resample(self, bin_count=200):
        start_i = int(self.t0 * self.sample_rate)
        end_i = util.clip(start_i + int(self.dt * self.sample_rate),
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
            row = 12 - row - 1
            x = np.full_like(v, col, dtype=np.float32)
            y = np.full_like(v, row, dtype=np.float32)
            t = np.arange(0, bin_count, 0.5, dtype=np.float32)
            data[i] = np.column_stack((x, y, t, v))

        # Update shader
        self.program['a_position'] = data.reshape(240*bin_count, 4)
        self.program['u_width'] = bin_count

    def update(self):
        self.resample()

    def draw(self):
        gloo.clear((0.5, 0.5, 0.5, 1))
        self.program.draw('line_strip')
        self.grid.draw(self.canvas.tr_sys)

    def on_mouse_move(self, event):
        x, y = event.pos
        sec_per_pixel = self.dt / (self.canvas.width / 12.0)
        if event.is_dragging:
            x1, y1 = event.last_event.pos
            dx = x1 - x
            self.t0 = util.clip(self.t0 + dx * sec_per_pixel,
                                0, self.data.index[-1])

        x, y = event.pos
        cell_width = self.canvas.size[0] / 12.0
        cell_height = self.canvas.size[1] / 12.0
        col = int(x / cell_width)
        row = int(y / cell_height + 1)
        if row < 1 or row > 12 or col < 0 or col > 11:
            self.electrode = ''
        else:
            self.electrode = '%s%d' % (self.electrode_cols[col], row)
        self.mouse_t = self.t0 + sec_per_pixel * (x % cell_width)

    def on_mouse_double_click(self, event):
        self.selected_electrodes = [self.electrode]
        self.canvas.show_analog()

    def on_mouse_wheel(self, event):
        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        rel_x = event.pos[0] % (self.canvas.size[0] / 12)

        target_time = rel_x * sec_per_pixel + self.t0
        dx = -np.sign(event.delta[1]) * 0.10
        self.dt *= math.exp(2.5 * dx)

        sec_per_pixel = self.dt / (self.canvas.size[0] / 12)
        self.t0 = target_time - (rel_x * sec_per_pixel)

    def on_tick(self, event):
        pass

    def on_resize(self, event):
        self.create_grid()
