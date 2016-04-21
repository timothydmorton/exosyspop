from __future__ import division, print_function

import numpy as np

from .populations import BinaryPopulation

class ABCModel(object):

    params = ('fB', 'beta') # Names of parameters
    summary_stat_names = ('period_pdf','N') # names of summary statistics
    distance_functions = None # names of different distance function methods 
                              #  length same as summary_stat_names

    def __init__(self, population, eff=None,
                    theta_0=(0.14, -0.95), bounds=[(0,1), (-1.5, 0)],
                    priors=['uniform','uniform']):
        self.population = population
        self.eff = eff
        self.theta_0 = theta_0
        self.bounds = bounds
        self.priors = priors

    def draw_theta(self):
        for p,b in zip(self.priors, self.bounds):

        raise NotImplementedError

    def generate_data(self, theta):
        raise NotImplementedError

    def summary_stats(self, data):
        """Returns tuple containing summary statistics named in summary_stat_names
        """
        raise NotImplementedError

    def null_distance_test(self, theta=None, N=100):
        if theta is None:
            theta = self.theta_0
        data1 = [self.generate_data(theta) for i in range(N)]
        data2 = [self.generate_data(theta) for i in range(N)]
        
        ds = []
        for dfn in self.distance_functions:
            fn = getattr(self, dfn)
            ds.append([fn(self.summary_stats(data1[i]),
                         self.summary_stats(data2[i])) for i in range(N)])
        
        null_stds = np.array([np.std(d) for d in ds])
        self._distance_norms = null_stds / null_stds[0]

    @property
    def distance_norms(self):
        if not hasattr(self, '_distance_norms'):
            self.null_distance_test()
            
        return self._distance_norms 


    def distance(self, stats, stats_synth):

        ds = []
        for dfn in self.distance_functions:
            fn = getattr(self, dfn)
            ds.append(fn(stats, stats_synth))

        return np.sum([d / self.distance_norms[i] for i,d in enumerate(ds)])