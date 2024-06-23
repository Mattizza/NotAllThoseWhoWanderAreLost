import numpy as np
# XXX: if using wandb, this is not needed.
from . import  MAX

# SLOPE SUPERLINEAR: 0.04, 0.05, 0.1, 0.2, 0.3

# SLOPE SUBLINEAR: 0.000000001, 0.00000005, 0.00000001, 0.0000001
# POW SUBLINEAR: 5

# SLOPE LINEAR: 0.5, 0.75, 1, 2

# SLOPE STAIR STEP: 2, 4, 6, 8
# STEP STAIR STEP: 2, 3, 4, 5


# XXX: MAX here is NOT NEEDED and can be removed. It is only used to allow for a dynamic assignment of the MAX when tuning.
def superlinear(x=None, MAX=MAX, slope=None):
    return (MAX - 1) * np.exp(-slope * x) + 1

def sublinear(x=None, MAX=MAX, slope=None):
    return (MAX - 1) * np.exp(-slope * x ** 5)

def linear(x=None, MAX=MAX, slope=None):
    return max(MAX - slope * x, 1)

def stair_step(x=None, MAX=MAX, step=None):
    return max(MAX - 3 * (x // step), 1)

# XXX: check this.
def sinusoidal(x=None, MAX=MAX, amplitude=None, frequency=None):
    # XXX: change 1.25 and 2 to frequency.
    return max(MAX - 1.5 * x - amplitude * np.sin(1.25*x), 1, amplitude * np.sin(2*x))