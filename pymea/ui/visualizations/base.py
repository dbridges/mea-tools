# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import sys

import numpy as np
from vispy.gloo import VertexBuffer
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram


class Theme:
    yellow = (0.9254, 0.7411, 0.0588, 1.0)
    purple = (0.396, 0.09, 0.62, 1.0)
    blue = (0.09, 0.365, 0.596, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)
    pink = (0.529, 0.075, 0.349, 1.0)
    gray = (0.7, 0.7, 0.7, 1.0)
    light_gray = (0.9, 0.9, 0.9, 1.0)
    transparent_black = (0, 0, 0, 0.01)
    black = (0, 0, 0, 1.0)
    red = (0.85, 0.06, 0.12, 1.0)
    grid_line = (0.7, 0.7, 0.7, 1.0)
    plot_colors = (yellow, purple, blue)
    background = (0.5, 0.5, 0.5, 1.0)

    indexed_colors = [[1., 0.721569, 0.219608, 1.0],
                      [0.94902, 0.862745, 0.435294, 1.0],
                      [0.670588, 0.878431, 0.937255, 1.0],
                      [0.317647, 0.654902, 0.752941, 1.0],
                      [0.0901961, 0.337255, 0.494118, 1.0],
                      [0.705882, 0.494118, 0.545098, 1.0],
                      [0.894118, 0.709804, 0.74902, 1.0],
                      [0.921569, 0.494118, 0.431373, 1.0],
                      [1., 0.721569, 0.219608, 1.0]]

    @classmethod
    def indexed(cls, val):
        return cls.indexed_colors[val % len(cls.indexed_colors)]


class LineCollection(Visual):
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
        Visual.__init__(self, self.VERTEX_SHADER, self.FRAGMENT_SHADER)
        self._vert = []
        self._color = []
        self._draw_mode = 'lines'
        # self._program = ModularProgram(LineCollection.VERTEX_SHADER,
        #                                LineCollection.FRAGMENT_SHADER)

    def clear(self):
        self._vert = []
        self._color = []

    def append(self, pt1, pt2, color=(1, 1, 1, 1)):
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

    def _prepare_transforms(self, view=None):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        self.shared_program['a_position'] = VertexBuffer(self._vert)
        self.shared_program['a_color'] = VertexBuffer(self._color)


class Visualization:
    scroll_factor = 0.025 if sys.platform == 'darwin' else 0.125

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
