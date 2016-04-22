from __future__ import division, print_function

import numpy as np
from scipy.stats import gaussian_kde, entropy
import logging

from .populations import BinaryPopulation

class ABCModel(object):

    params = ('fB', 'beta') # Names of parameters
    summary_stat_names = ('period_pdf','N') # names of summary statistics
    distance_functions = ('d_period', 'd_N') # names of different distance function methods 
                              #  length same as summary_stat_names
    theta_0 = (0.14, -0.95)
    bounds = ((0,1), (-1.5,0))
    priors = ('uniform', 'uniform')

    def __init__(self, population, eff=None):
        self.population = population
        self.eff = eff

    def draw_theta(self):
        theta = []
        for p,(lo,hi) in zip(self.priors, self.bounds):
            if p=='uniform':
                theta.append(np.random.random()*(hi-lo) + lo)
        return theta

    def generate_data(self, theta):
        param_dict = {p:v for p,v in zip(self.params, theta)}
        self.population.set_params(**param_dict)
        try:
            return self.population.observe(new=True, regr_trap=True).observe(self.eff)
        except:
            logging.warning('Error generating data: {}'.format(param_dict))

    @property
    def min_period(self):
        return self.population.params['period_min']
    
    @property
    def max_period(self):
        return self.population.params['period_max']

    def summary_stats(self, data):
        """Returns tuple containing summary statistics named in summary_stat_names
        """
        if data is None:
            return [None]*len(summary_stat_names)

        N = len(data)
        min_logP, max_logP = np.log(self.min_period), np.log(self.max_period)
        logP_grid = np.linspace(min_logP, max_logP, 1000)
        if N > 1:
            k = gaussian_kde(np.log(data.period.values))
            pdf = k(logP_grid)
        else:
            pdf = np.ones(len(logP_grid))*1./(max_logP - min_logP)

        return pdf, N

    def d_period(self, summary_stats, summary_stats_synth):
        p1 = summary_stats[0]
        p2 = summary_stats_synth[0]
        try:
            len(p1)
            len(p2)
        except:
            return np.inf
        kl_period = entropy(p1, p2)
        return kl_period
    
    def Ndist(self, N1, N2):
        if N1==0. or N2==0.:
            dist = 1
        else:
            dist = max(1 - 1.*N1/N2, 1-1*N2/N1)
        return dist
            
    def d_N(self, summary_stats, summary_stats_synth):
        N1 = summary_stats[1]
        N2 = summary_stats_synth[1]
        return self.Ndist(N1, N2) 
        

    def null_distance_test(self, theta=None, N=100):
        if theta is None:
            theta = self.theta_0

        logging.info('Performing null distance test (N={})'.format(N))
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


    def distance_function(self, stats, stats_synth):

        ds = []
        for dfn in self.distance_functions:
            fn = getattr(self, dfn)
            ds.append(fn(stats, stats_synth))

        return np.sum([d / self.distance_norms[i] for i,d in enumerate(ds)])