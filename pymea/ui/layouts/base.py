# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


class Layout:
    # must reimplement these methods and member functions.

    def __init__(self):
        self.rows = 0
        self.columns = 0

    def coordinates_for_electrode(self, electrode):
        return (0, 0)
