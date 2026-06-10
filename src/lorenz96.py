# Lorenz 96 model
# author: Quark
# ==================================================
import numpy as np
# ==================================================
def lorenz96(t, x, F=8.0):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F