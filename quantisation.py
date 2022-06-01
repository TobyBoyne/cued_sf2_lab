import numpy as np

def quant(x, step, rise1=None):
    """
    Quantise the matrix x using steps of width step.

    The result is the quantised integers Q. If rise1 is defined,
    the first step rises at rise1, otherwise it rises at step/2 to
    give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        q = x.copy()
        return q
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Quantise abs(x) to integer values, and incorporate sign(x)..
    temp = np.ceil((np.abs(x) - rise)/step)
    q = temp*(temp > 0)*np.sign(x)
    return q


def inv_quant(q, step, rise1=None):
    """
    Reconstruct matrix Y from quantised values q using steps of width step.

    The result is the reconstructed values. If rise1 is defined, the first
    step rises at rise1, otherwise it rises at step/2 to give a uniform
    quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = q.copy()
        return y
    if rise1 is None:
        rise = step/2.0
        return q * step
    else:
        rise = rise1
        # Reconstruct quantised values and incorporate sign(q).
        y = q * step + np.sign(q) * (rise - step/2.0)
        return y