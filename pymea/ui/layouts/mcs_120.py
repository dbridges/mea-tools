# -*- coding: utf-8 -*-
# Copyright (c) 2015, UC Santa Barbara
# Hansma Lab, Kosik Lab
# Originally written by Daniel Bridges


from .base import Layout


class MCS120Layout(Layout):
    """
    Layout for Multi Channel Systems 120 count MEA.

    An1            D1   E1   F1   G1   H1   J1

              C2   D2   E2   F2   G2   H2   J2   K2

         B3   C3   D3   E3   F3   G3   H3   J3   K3   L3

    A4   B4   C4   D4   E4   F4   G4   H4   J4   K4   L4   M4

    A5   B5   C5   D5   E5   F5   G5   H5   J5   K5   L5   M5

    A6   B6   C6   D6   E6   F6   G6   H6   J6   K6   L6   M6

    A7   B7   C7   D7   E7   F7   G7   H7   J7   K7   L7   M7

    A8   B8   C8   D8   E8   F8   G8   H8   J8   K8   L8   M8

    A9   B9   C9   D9   E9   F9   G9   H9   J9   K9   L9   M9

         B10  C10  D10  E10  F10  G10  H10  J10  K10  L10

              C11  D11  E11  F11  G11  H11  J11  K11

                   D12  E12  F12  G12  H12  J12
    """

    def __init__(self):
        super().__init__()
        self.rows = 12
        self.columns = 12

    def coordinates_for_electrode(self, electrode):
        """
        Returns MEA coordinates for electrode label.

        Parameters
        ----------
        electrode : str
            The electrode label, i.e. A8 or C6

        Returns
        -------
        coordinates : tuple
            A tuple of length 2 giving the x and y coordinate of
            that electrode.
        """
        electrode = electrode.lower().split('.')[0]
        if electrode.startswith('analog'):
            if electrode.endswith('1'):
                return (0, 0)
            elif electrode.endswith('2'):
                return (1, 0)
            elif electrode.endswith('3'):
                return (10, 0)
            elif electrode.endswith('4'):
                return (11, 0)
            elif electrode.endswith('5'):
                return (0, 11)
            elif electrode.endswith('6'):
                return (1, 11)
            elif electrode.endswith('7'):
                return (10, 11)
            elif electrode.endswith('8'):
                return (11, 11)
        cols = {'a':  0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
                'h': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11}
        return (cols[electrode[0]], int(electrode[1:]) - 1)

    def electrode_for_coordinate(self, pt):
        """
        Returns MEA tag for electrode coordinates.

        Parameters
        ----------
        pt : tuple
            The electrode coordinates.

        Returns
        -------
        electrode : str
            The electrode name.
        """
        lookup = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7:
                  'h', 8: 'j', 9: 'k', 10: 'l', 11: 'm'}
        electrode = lookup[pt[0]] + str(pt[1])
        if electrode == 'a1':
            return 'analog1'
        elif electrode == 'b1':
            return 'analog2'
        elif electrode == 'l1':
            return 'analog3'
        elif electrode == 'm1':
            return 'analog4'
        elif electrode == 'a12':
            return 'analog5'
        elif electrode == 'b12':
            return 'analog6'
        elif electrode == 'l12':
            return 'analog7'
        elif electrode == 'm12':
            return 'analog8'
        else:
            return electrode
