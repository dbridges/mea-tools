#!/usr/bin/enb python3


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

    Retruns
    -------
    number
        The clamped value.
    """
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val
