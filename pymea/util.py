#!/usr/bin/enb python3

import math

def clip(val, minval, maxval):
    """
    Returns val, not to exceed minval or maxval.

    Parameters
    ----------
    val : number
        The input value to clamp.
    minval : number
        Lower value to clamp to.
    maxval : number
        Uperr value to clamp to.

    Retunrs
    -------
    number
        The clamped value.
    """
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


def nearest_decimal(val):
    """
    Returns the nearest decimal fraction of val.

    Parameters
    ----------
    val : number
        The input value.
    """
    log_val = math.log10(val)
    if log_val > 0:
        return math.pow(10, int(log_val))
    else:
        return math.pow(10, int(log_val - 1))
