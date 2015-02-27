# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


import numpy as np
from vispy.visuals.shaders import ModularProgram


class Theme:
    yellow = (0.9254, 0.7411, 0.0588, 1.0)
    purple = (0.396, 0.09, 0.62, 1.0)
    blue = (0.09, 0.365, 0.596, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)
    grid_line = (0.7, 0.7, 0.7, 1.0)
    plot_colors = (yellow, purple, blue)
    background = (0.5, 0.5, 0.5, 1.0)


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
