from __future__ import print_function, division

from scipy.special import gammainc, gamma
import numpy as np

class EfficiencyFunction(object):
    """
    Subclasses must define __call__ method to take scalar
    or array-like argument and return a corresponding
    array of detection probabilities.
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class DetectionThreshold(EfficiencyFunction):
    """
    SNR threshold function
    """
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, snr):
        return np.atleast_1d(snr) > self.thresh

class DetectionRamp(EfficiencyFunction):
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, snr):
        s = np.atleast_1d(snr)
        x = (snr - self.lo)/(self.hi - self.lo)
        x[x < self.lo] = 0
        x[x > self.hi] = 1
        return x

class GammaCDF(EfficiencyFunction):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, snr):
        return  gammainc(self.a, snr/self.b)

