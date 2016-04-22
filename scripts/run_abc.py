#!/usr/bin/env python

from __future__ import print_function, division
import sys, os

ROOT = os.getenv('EXOSYSPOP','..')

sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT,'..'))


from exosyspop.populations import KeplerPowerLawBinaryPopulation
from exosyspop.survey import DetectionRamp
from exosyspop.abc import ABCModel

from simpleabc.simple_abc import pmc_abc

import numpy as np

import logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

import argparse

parser = argparse.ArgumentParser(description="Run ABC calculation for synthesized population.")
parser.add_argument('--pop', default=os.path.join(ROOT, 'plaw_pop'))
parser.add_argument('--epsilon_0', type=float, default=0.3)
parser.add_argument('--n_procs', type=int, default=12)
parser.add_argument('--min_samples', type=int, default=200)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--file', default='pmc_posterior')


args = parser.parse_args()

pop = KeplerPowerLawBinaryPopulation.load(args.pop)

pop.set_params(period_min=20, period_max=1200, beta=-0.95, fB=0.14)

eff = DetectionRamp(6,16)

data = pop.observe(new=True, regr_trap=True).observe(eff)

model = ABCModel(pop, eff)
model.null_distance_test()

#model._distance_norms = np.array([ 1.        ,  4.49241213,  2.60025772,  2.73734061])

pmc_posterior = pmc_abc(model, data, epsilon_0=args.epsilon_0, 
						min_samples=args.min_samples, steps=args.steps, verbose=True,
                       parallel=True, n_procs=args.n_procs)

try:
	np.save(args.file, pmc_posterior)
except:
	logging.warning('Posterior not saved to desired location ({}) because of problem!  Saved to recovered.npy'.format(args.file))
	np.save('recovered', pmc_posterior)

	