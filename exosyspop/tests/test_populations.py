from __future__ import print_function, division

from pkg_resources import resource_filename

import os, os.path
import tempfile
TMP = tempfile.gettempdir()

try:
    import cPickle as pickle
except ImportError:
    import pickle
import pandas as pd
import numpy as np

from exosyspop.populations import (KeplerBinaryPopulation,
                                   TRILEGAL_BGBinaryPopulation,
                                   PoissonPlanetPopulation)

target_file = resource_filename('exosyspop','tests/test_targets.h5')
bgstars_file = resource_filename('exosyspop','tests/test_bgstars.h5')

targets = pd.read_hdf(target_file,'df')
bgstars = pd.read_hdf(bgstars_file,'df')

def _do_the_things(pop):
    rB = pop.radius_B
    ok = ~np.isnan(rB)
    pop._train_trap(N=200)
    obs = pop.observe(regr_trap=True)
    fname = 'test_binpop'
    pop.save(os.path.join(TMP, fname), overwrite=True)
    pkl_file = os.path.join(TMP, '{}.pkl'.format(fname))
    pickle.dump(pop, open(pkl_file, 'wb'))
    pop = pickle.load(open(pkl_file, 'rb'))
    obs = pop.observe(regr_trap=True)
    assert np.all(rB[ok]==pop.radius_B[ok])

def get_binpop():
    return KeplerBinaryPopulation(targets, fB=1)

def test_binpop():
    pop = get_binpop()
    _do_the_things(pop)

def get_bgpop():
    return TRILEGAL_BGBinaryPopulation(targets, bgstars, fB=1)

def test_bgpop():
    pop = get_bgpop()
    _do_the_things(pop)

def get_planets():
    return PoissonPlanetPopulation(targets, Npl=1)

def test_planets():
    pop = get_planets()
    _do_the_things(pop)

