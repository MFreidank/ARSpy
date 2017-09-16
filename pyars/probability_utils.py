import numpy as np
from numpy import exp


def exp_normalize(probs_array):
    xp = probs_array - np.max(probs_array)
    exp_x = exp(xp)
    return exp_x / sum(exp_x)
