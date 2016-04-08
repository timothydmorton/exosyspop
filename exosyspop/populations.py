from __future__ import print_function, division

import numpy as np
import pandas as pd

import os, os.path, shutil
import logging

# This disables expensive garbage collection calls
# within pandas.  Took forever to figure this out.
pd.set_option('mode.chained_assignment', None)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from scipy.spatial import Delaunay

import cPickle as pickle

from astropy.coordinates import SkyCoord

from isochrones.dartmouth import Dartmouth_Isochrone
DAR = Dartmouth_Isochrone()
DAR.radius(1,9.5,0) #prime the isochrone object

# Still with vespa dependencies for now
from vespa.stars.utils import draw_eccs # this is a function that returns
                                        # empirically reasonable eccentricities
                                        # for given binary periods.
from vespa.transit_basic import (_quadratic_ld, eclipse_tt, 
                                 NoEclipseError, NoFitError)

from .catalog import SimulatedCatalog

from .utils import draw_powerlaw, semimajor, rochelobe
from .utils import G, MSUN, RSUN, AU, DAY, REARTH, MEARTH
from .utils import trap_mean_depth


class BinaryPopulation(object):
    """
    Base class for binary population simulations.

    Initialized with population of primary stars, which 
    is a DataFrame containing, at minimum, `mass_A`, `feh`,
    `dataspan`, `dutycycle` parameters.  `radius_A` and `age`
    are also appreciated, but will be generated if not provided.

    `prop_columns` attribute is a dictionary that maps
    column name to the name actually in the DataFrame.  For
    example, the Kepler stellar catalog has `mass` instead
    of `mass_A`, so the KeplerBinaryPopulation defines this.

    This object can very quickly generate different random 
    populations of binary companions according to the provided
    parameters, and "observe" the populations to generate a 
    synthetic catalog.  The machinery that allows this to happen efficiently is 
    training two different regression steps that enable bypassing
    more computationally intensive steps.  

    The first of these regressions is to predict the dmag between the
    secondary and primary, as well as the radius ratio, as a function
    of the other more readily simulated parameters.  First, `dmag` is
    predicted as a function of selected stellar features (by default
    these are `mass_A`, `radius_A`, `q`, `age`, and `feh`, but can be
    changed or set differently for subclasses.)  Then the radius ratio
    `qR` is predicted using all of the above, as well as `dmag`.
    These regressions are trained using quantities simulated directly
    according to the provided :class:`Isochrone` object, and this
    training takes just a few seconds.  Once trained, this regression
    (which is very accurate---R^2 ~ 0.999 with a stellar population of
    ~30,000) computes the secondary properties of a simulated
    population about a factor of 10x faster than direct calls to the
    :class:`Isochrone`.

    The second regression is more costly to train (~1 min) but saves 
    correspondingly much more computation time---this is a regression
    that predicts the exact trapezoidal shape parameters as a function of 
    the following easy-to-compute parameters: 

      * total duration `T14` (adjusted for exposure time),
      * ingress/egress duration `tau` (adjusted for exposure time),
      * radius ratio `k`,
      * impact parameter `b`,
      * log of the max depth `logd`---this is computed as the Mandel & Agol
        depth at the closest impact parameter, and
      * whether it is a primary eclipse or secondary occultation.

    After training, the :function:`BinaryPopulation.observe` function
    will generate an observed population of eclipsing systems, complete
    with trapezoidal shape parameters in < 1s, for a primary
    population of ~30,000 target stars.

    """
    #parameters for binary population (for period in years)
    param_names = ('fB', 'gamma', 'qmin', 'mu_logp', 'sig_logp', 
                   'beta_a', 'beta_b', 'period_min')
    #default_params = (0.4, 0.3, 0.1, np.log10(250), 2.3, 0.8, 2.0)

    default_params = {'fB':0.44, 'gamma':0.3, 'qmin':0.1, 'mu_logp':np.log10(250),
                      'sig_logp':2.3, 'beta_a':0.8, 'beta_b':2.0,
                      'period_min':5.}

    # Physical and orbital parameters that can be accessed.
    primary_props = ('mass_A', 'radius_A')

    secondary_props = ('mass_B', 'radius_B', 'flux_ratio')

    orbital_props = ('period', 'ecc', 'w', 'inc', 'a', 'aR',
                      'b_pri', 'b_sec', 'k', 'tra', 'occ',
                      'd_pri', 'd_sec', 'T14_pri', 'T14_sec',
                      'T23_pri', 'T23_sec')
                     
    obs_props = ('dataspan', 'dutycycle', 'b_target')

    binary_features = ('mass_A', 'radius_A', 'age', 'feh')

    # property dictionary mapping to DataFrame column
    prop_columns = {}
    default_name = 'EB'

    _attrs = ('name', 'band', 'texp', 'ecc_empirical',
             '_not_calculated', 'use_ic')
    _tables = ()

    def __init__(self, stars, name=None,
                 band='Kepler', texp=1626./86400,
                 ic=DAR, ecc_empirical=False, use_ic=False,
                 index=None, copy=True,
                 **kwargs):

        # Copy data, so as to avoid surprises.
        if copy:
            self._stars = stars.copy()
        else:
            self._stars = stars
        self._stars_cache = None

        if name is None:
            name = self.default_name
        self.name = name

        self._index = index
        self._ic = ic
        self.band = band
        self.texp = texp
        self.ecc_empirical = ecc_empirical
        self.use_ic = use_ic

        #Renames columns, sets self._not_calculated appropriately.
        self._initialize_stars()

        self._params = None
        self.set_params(**kwargs)

        # Regressions to be trained
        self._binary_trained = False
        self._dmag_pipeline = None
        self._qR_pipeline = None
        self._MR_tri = None

        self._trap_trained = False
        self._logd_pipeline = None
        self._dur_pipeline = None
        self._slope_pipeline = None

    @property
    def stars(self):
        if self._stars_cache is not None:
            return self._stars_cache
        elif self._index is not None:
            self._set_index(self._index)
            return self._stars_cache
        else:
            return self._stars

    def _set_index(self, ix):
        self._index = ix
        self._stars_cache = self._stars.loc[ix]

    def _record_stars_changes(self):
        if self._index is not None:
            self._stars.loc[self._index, :] = self._stars_cache

    @property
    def dilution_factor(self):
        return np.ones(self.N)

    def get_target(self, i):
        """Returns i-th target star
        """
        return self.stars.ix[i]

    def get_noise(self, i, T=3):
        """
        Returns noise in ppm (e.g. CDPP) for i-th target star, over timescale T (hrs)
        
        arbitrary default = 100ppm
        """
        return 100.
        

    def _initialize_stars(self):
        # Rename appropriate columns
        for k,v in self.prop_columns.items():
            self._stars.rename(columns={v:k}, inplace=True)

        # if ra, dec provided, but not b_target, then calculate it.
        if 'ra' in self._stars and 'b_target' not in self._stars:
            if 'b' in self._stars:
                self._stars.rename(columns={'b':'b_target'}, inplace=True)
            else:
                c = SkyCoord(self._stars.ra, self._stars.dec, unit='deg')
                self._stars.loc[:, 'b_target'] = c.galactic.b.deg

        # Create all the columns that will be filled later
        self._not_calculated = [c for c in self.primary_props + 
                                self.secondary_props + 
                                self.orbital_props + self.obs_props 
                                if c not in self._stars]

        for c in self._not_calculated:
            if c in ['tra','occ']:
                self._stars.loc[:, c] = False
            else:
                self._stars.loc[:, c] = np.nan        

    def _get_params(self, pars):
        return [self.params[p] for p in pars]

    def __getattr__(self, name):
        if name in self._not_calculated:
            if name in self.primary_props or name in self.secondary_props:
                logging.debug('Accessing {}, generating binaries.'.format(name))
                self._generate_binaries()
            elif name in self.orbital_props:
                logging.debug('Accessing {}, generating orbits.'.format(name))
                self._generate_orbits()
        try:
            vals = self.stars[name].values
            return vals
        except KeyError:
            raise AttributeError(name)

    def _mark_calculated(self, prop):
        try:
            i = self._not_calculated.index(prop)
            self._not_calculated.pop(i)
        except ValueError:
            pass
        
    def _remove_prop(self, prop):
        self.stars.loc[:, prop] = np.nan
        if prop not in self._not_calculated:
            self._not_calculated.append(prop)

    @property
    def params(self):
        if self._params is not None:
            return self._params
        else:
            return self.default_params.copy()

    def reset_params(self):
        self._params = self.default_params.copy()

    def set_params(self, **kwargs):
        """
        Set values of parameters. 

        Calling this sets all secondary & orbital properties to "unset"
        """
        if self._params is None:
            self._params = self.default_params.copy()
        for k,v in kwargs.items():
            self._params[k] = v
        for p in self.secondary_props + self.orbital_props:
            if p not in self._not_calculated:
                self._not_calculated.append(p)

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @property
    def N(self):
        return len(self.stars)

    def _ensure_age(self):
        # Stellar catalog doesn't have ages, so let's make them up.
        #  ascontiguousarray makes ic calls faster.
        if 'age' in self._stars:
            return

        ic = self.ic
        feh = np.ascontiguousarray(np.clip(self._stars.feh, ic.minfeh, ic.maxfeh))
        minage, maxage = ic.agerange(self._stars.mass_A, feh)
        maxage = np.clip(maxage, 0, ic.maxage)
        if 'age' not in self._stars:
            minage += 0.3 # stars are selected to not be active
            maxage -= 0.1
            age = np.random.random(size=len(feh)) * (maxage - minage) + minage
        else:
            age = np.clip(self._stars.age.values, minage, maxage)

        self._stars.loc[:,'age'] = age
        self._stars.loc[:,'feh'] = feh #reassigning feh

    def _ensure_radius(self):
        self._ensure_age()
        # Simulate primary radius (unless radius_A provided)
        if 'radius_A' in self._not_calculated:
            mass_A = np.ascontiguousarray(self._stars.mass_A)
            age = np.ascontiguousarray(self._stars.age)
            feh = np.ascontiguousarray(self._stars.feh)
            self._stars.loc[:, 'radius_A'] = self.ic.radius(mass_A, age, feh)
            self._mark_calculated('radius_A')



    def _simulate_binary_features(self):
        """
        Returns feature vector X, and binary mask b
        """
        N = self.N
        fB, gamma, qmin = self._get_params(['fB', 'gamma', 'qmin'])

        self._ensure_radius()

        # Simulate mass ratio
        minmass = self.ic.minmass
        qmin = np.maximum(qmin, minmass/self.mass_A)
        q = draw_powerlaw(gamma, (qmin, 1), N=N)

        b = np.random.random(N) < fB

        X = np.array([getattr(self, x) for x in self.binary_features]).T
        X = np.append(X, np.array([q]).T, axis=1)
        return X[b, :], b

    def _generate_binaries(self, use_ic=None):
        # Simulate directly from isochrones if desired; 
        # otherwise use regression.
        N = self.N
        logging.debug('Generating binary companions for {} stars...'.format(N))
        
        if use_ic is None:
            use_ic = self.use_ic

        if use_ic:
            fB, gamma, qmin = self._get_params(['fB', 'gamma', 'qmin'])
            b = np.random.random(N) < fB
        
            self._ensure_radius()
            
            # Simulate mass ratio
            minmass = self.ic.minmass
            qmin = np.maximum(qmin, minmass/self.mass_A)
            q = draw_powerlaw(gamma, (qmin, 1), N=N)
    
            ic = self.ic
            M1 = np.ascontiguousarray(self.mass_A[b])
            M2 = np.ascontiguousarray((q * self.mass_A)[b])
            feh = np.ascontiguousarray(self.feh[b])
            age = np.ascontiguousarray(self.age[b])
            R2 = ic.radius(M2, age, feh)

            dmag = ic.mag[self.band](M2, age, feh) - ic.mag[self.band](M1, age, feh)
            flux_ratio = 10**(-0.4 * dmag)

        else:
            X, b = self._simulate_binary_features()
            
            # Make sure there are some binaries; otherwise, skip regression.
            if  b.sum() > 0:                

                #q will always be last column, regardless of other features
                q = X[:, -1]  #already binary-masked
                M2 = q*self.mass_A[b]

                # Train pipelines if need be.
                if not self._binary_trained:
                    self._train_pipelines()

                # Calculate dmag->flux_ratio from trained regression
                dmag = self._dmag_pipeline.predict(X)
                flux_ratio = 10**(-0.4 * dmag)

                # Calculate qR->radius_B from trained regression
                X = np.append(X, np.array([dmag]).T, axis=1)
                qR = self._qR_pipeline.predict(X)
                R2 = qR * self.radius_A[b]

                # Make any out-of-bounds predictions -> nan
                #bad = self._check_MR(M2, R2)
                #R2[bad] = -100
            else:
                M2, R2, flux_ratio = [np.nan]*3

        # Create arrays of secondary properties
        mass_B = np.zeros(N)
        mass_B[b] = M2
        mass_B[~b] = np.nan

        radius_B = np.zeros(N)
        radius_B[b] = R2
        radius_B[~b] = np.nan

        fluxrat = np.zeros(N)
        fluxrat[b] = flux_ratio
        flux_ratio = fluxrat

        for c in self.secondary_props:
            self.stars.loc[:, c] = eval(c)
            self._mark_calculated(c)

        # Mark orbital params as not calculated.
        for c in self.orbital_props:
            self._remove_prop(c)

    def _sample_period(self, N):
        """
        Samples log-normal period distribution.
        """
        mu_logp, sig_logp, period_min = self._get_params(['mu_logp', 'sig_logp',
                                                          'period_min'])
        
        #  don't let anything shorter than minimum period
        period = 10**(np.random.normal(mu_logp, sig_logp, size=N)) * 365.25
        bad = period < period_min
        nbad = bad.sum()
        while nbad > 0:
            period[bad] = 10**(np.random.normal(mu_logp, 
                                                sig_logp, size=nbad)) * 365.25
            bad = period < period_min
            nbad = bad.sum()

        return period 
    
    def _sample_ecc(self, N):
        """
        Return N samples from eccentricity distribution
        """
        a, b = self._get_params(['beta_a', 'beta_b'])

        ecc = np.random.beta(a,b,N)
        return ecc

    def _generate_orbits(self, geom_only=False):

        # This order is important.  access secondary properties
        #  before primary, so they get created if need be.
        mass_B = self.mass_B
        radius_B = self.radius_B
        mass_A = self.mass_A
        radius_A = self.radius_A

        N = self.N
        logging.debug('Generating {} orbits...'.format(N))


        # draw orbital parameters
        period = self._sample_period(N)

        # if using empirical eccentricity distribution,
        # do so, otherwise sample from distribution.
        if self.ecc_empirical:
            ecc = draw_eccs(N, period)
        else:
            ecc = self._sample_ecc(N)
        a = semimajor(period, mass_A + mass_B) * AU

        # Here, crude hack for "circularization":
        # If orbit implies that periastron is within 3*roche radius,
        # then redraw eccentricity from a tight rayleigh distribution.
        # If still too close, assign e=0.
        q = mass_B/mass_A
        peri = a*(1-ecc)
        tooclose = (radius_A + radius_B)*RSUN > 3*rochelobe(1./q)*peri
        ecc[tooclose] = np.random.rayleigh(0.03)
        logging.debug('{} orbits assigned to ecc=rayleigh(0.03)'.format(tooclose.sum()))
        peri = a*(1-ecc)
        tooclose = (radius_A + radius_B)*RSUN > 3*rochelobe(1./q)*peri
        ecc[tooclose] = 0.
        logging.debug('{} orbits assigned to ecc=0'.format(tooclose.sum()))
        

        w = np.random.random(N) * 2 * np.pi
        inc = np.arccos(np.random.random(N))        
        aR = a / (radius_A * RSUN)
        if geom_only:
            # add the above properties
            for c in self.orbital_props[:6]:
                self.stars.loc[:, c] = eval(c)
                self._mark_calculated(c)
            # take away the others.
            for c in self.orbital_props[6:]:
                self._remove_prop(c)
            return


        # Determine closest approach
        b_pri = a*np.cos(inc)/(radius_A*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w))
        b_sec = a*np.cos(inc)/(radius_A*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w))

        R_tot = (radius_A + radius_B)/radius_A
        tra = (b_pri < R_tot)
        occ = (b_sec < R_tot)

        # Calculate eclipse depths, assuming Solar limb darkening for all
        d_pri = np.zeros(N)
        d_sec = np.zeros(N)
        k = radius_B / radius_A
        T14_pri = period/np.pi*np.arcsin(radius_A*RSUN/a * np.sqrt((1+k)**2 - b_pri**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1+ecc*np.sin(w))
        T14_sec = period/np.pi*np.arcsin(radius_A*RSUN/a * np.sqrt((1+k)**2 - b_sec**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1-ecc*np.sin(w))
        T23_pri = period/np.pi*np.arcsin(radius_A*RSUN/a * np.sqrt((1-k)**2 - b_pri**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1+ecc*np.sin(w))
        T23_sec = period/np.pi*np.arcsin(radius_A*RSUN/a * np.sqrt((1-k)**2 - b_sec**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1-ecc*np.sin(w))
    
        T14_pri[np.isnan(T14_pri)] = 0.
        T14_sec[np.isnan(T14_sec)] = 0.
        T23_pri[np.isnan(T23_pri)] = 0.
        T23_sec[np.isnan(T23_sec)] = 0.

        # Make sure no tra/occ where T14's were nans
        tra[T14_pri==0] = False
        occ[T14_sec==0] = False

        flux_ratio = self.flux_ratio
        for i in xrange(N):
            if tra[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_pri[i]]), k[i], 0.394, 0.296, 1)[0]
                F2 = flux_ratio[i]
                d_pri[i] = 1 - (F2 + f)/(1+F2)
            if occ[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_sec[i]/k[i]]), 1./k[i], 0.394, 0.296, 1)[0]
                F2 = flux_ratio[i]
                d_sec[i] = 1 - (1 + F2*f)/(1+F2)

        for c in self.orbital_props:
            self.stars.loc[:, c] = eval(c)
            self._mark_calculated(c)

    def _prepare_geom(self, new=False):
        if 'radius_B' in self._not_calculated or new:
            self._generate_binaries()
        if 'period' in self._not_calculated or new:
            self._generate_orbits(geom_only=True)

    def get_pgeom(self, query=None, new=False, sec=False):
        self._prepare_geom(new=new)
        if query is not None:
            df = self.stars.query(query)
        else:
            df = self.stars

        if sec:
            return ((df.radius_A + df.radius_B)*RSUN/(df.a) *
                    (1 - df.ecc*np.sin(df.w))/(1 - df.ecc**2))
        else:
            return ((df.radius_A + df.radius_B)*RSUN/(df.a) *
                    (1 + df.ecc*np.sin(df.w))/(1 - df.ecc**2))

    def get_necl(self, query=None, new=False):
        """
        Supposed to return expected number of geometrically eclipsing systems.

        *NOT CORRECT, DO NOT USE*

        Fun problem to take a stab at sometime, though...
        """
        self._prepare_geom(new=new)
        if query is not None:
            df = self.stars.query(query)
        else:
            df = self.stars

        pri = ((df.radius_A + df.radius_B)*RSUN/(df.a) *
               (1 + df.ecc*np.sin(df.w))/(1 - df.ecc**2))
        sec = ((df.radius_A + df.radius_B)*RSUN/(df.a) *
               (1 - df.ecc*np.sin(df.w))/(1 - df.ecc**2))

        # Trying to fudge the probability of pri | sec.
        #  Don't think I did it right.
        bad = np.isnan(df.radius_B)
        pri[bad] = 0
        sec[bad] = 0
        pri = np.clip(pri, 0, 1)
        sec = np.clip(pri, 0, 1)

        return np.maximum(pri, sec).sum()
        

    def observe(self, query=None, fit_trap=False, new=False,
                new_orbits=False, regr_trap=False, use_ic=None,
                dataspan=None, dutycycle=None):
        """
        Returns catalog of the following observable quantities:
          
          * n_pri
          * n_sec
          * d_pri
          * d_sec
          * T14_pri
          * T14_sec
          * T23_pri
          * T23_sec
          * phase_sec
          * trapezoidal fit params [either explicitly fit or regressed]
              * depth
              * duration
              * "slope" (T/tau)
          * SNR_pri [estimated from trapezoid fit]
          * SNR_sec

        Observations account for both geometry and duty cycle.  
        The latter is accounted for by drawing randomly from a binomial
        distribution B(n_exp, dutycycle), where n_exp is the number
        of eclipses that would be observed with 100% duty cycle.  This
        is done independently for primary and secondary eclipses.

        If `dataspan` and `dutycycle` are not provided, then they 
        must be part of the `stars` DataFrame.  If they weren't part
        before, they will be added by this function.

        TODO: incorporate pipeline detection efficiency.

        """
        #if fit_trap: #or use_ic ? took it out.
        #    new = True
        if new:
            self._generate_binaries(use_ic=use_ic)
            self._generate_orbits()
        elif new_orbits:
            self._generate_orbits()

        for v in ['dataspan', 'dutycycle']:
            var = eval(v)
            if v in self._not_calculated:
                if var is None:
                    raise ValueError('{0} must be provided'.format(v))
                else:
                    self.stars.loc[:, v] = var
                    self._mark_calculated(v)
            else:
                if var is not None:
                    self.stars.loc[:, v] = var

        # Select only systems with eclipsing (or occulting) geometry
        m = (self.tra | self.occ) & (self.dataspan > 0)
        cols = list(self.orbital_props + self.obs_props) + ['flux_ratio']
        df = self.stars.loc[m, cols].copy()

        df.loc[:, 'dilution'] = self.dilution_factor[m]


        # Phase of secondary (Hilditch (2001) p. 238, Kopal (1959))
        #  Primary is at phase=0
        N = len(df)
        X = np.pi + 2*np.arctan(df.ecc * np.cos(df.w) / np.sqrt(1-df.ecc**2))
        secondary_phase = (X - np.sin(X))/(2.*np.pi)

        # Assign each system a random phase at t=0;
        initial_phase = np.random.random(N)
        final_phase = initial_phase + df.dataspan/df.period

        # Determine number of primary & secondary eclipses, assuming perfect duty cycle
        n_pri_ideal = np.floor(final_phase) * df.tra
        n_sec_ideal = (np.floor(final_phase + secondary_phase) - 
                       np.floor(initial_phase + secondary_phase))*df.occ

        # Correct for duty cycle.  
        # Each event has probability (1-dutycycle) of landing in a gap.
        n_pri = np.zeros(N)
        n_sec = np.zeros(N)
        for i, (n1,n2,d) in enumerate(zip(n_pri_ideal,
                                          n_sec_ideal,
                                          df.dutycycle)):
            if n1 > 0:
                #n_pri[i] = binom(n1,d).rvs()
                n_pri[i] = np.random.binomial(n1, d)
            if n2 > 0:
                #n_sec[i] = binom(n2,d).rvs()
                n_sec[i] = np.random.binomial(n2, d)
        
        df.loc[:, 'n_pri'] = n_pri
        df.loc[:, 'n_sec'] = n_sec
        df.loc[:, 'phase_sec'] = secondary_phase

        m = (df.n_pri > 0) | (df.n_sec > 0)
        catalog = df[m].reset_index().rename(columns={'index':'host'})

        if fit_trap:
            N = len(catalog)
            catalog.loc[:, 'trap_dur_pri'] = np.zeros(N)
            catalog.loc[:, 'trap_depth_pri'] = np.zeros(N)
            catalog.loc[:, 'trap_slope_pri'] = np.zeros(N)
            catalog.loc[:, 'trap_dur_sec'] = np.zeros(N)
            catalog.loc[:, 'trap_depth_sec'] = np.zeros(N)
            catalog.loc[:, 'trap_slope_sec'] = np.zeros(N)

            period = catalog.period.values
            k = catalog.k.values
            b_pri = catalog.b_pri.values
            b_sec = catalog.b_sec.values
            aR = catalog.aR.values
            flux_ratio = catalog.flux_ratio.values
            ecc = catalog.ecc.values
            w = catalog.w.values
            tra = catalog.tra.values
            occ = catalog.occ.values
            d_pri = catalog.d_pri.values
            d_sec = catalog.d_sec.values

            trapfit_kwargs = dict(npts=50, width=3, cadence=self.texp)
            for i in xrange(N):
                # Primary
                if tra[i] and d_pri[i] > 0:
                    try:
                        trapfit = eclipse_tt(P=period[i], p0=k[i], b=b_pri[i],
                                         aR=aR[i], frac=1/(1 + flux_ratio[i]),
                                         u1=0.394, u2=0.296, 
                                         ecc=ecc[i], w=w[i]*180/np.pi,
                                         **trapfit_kwargs)
                        dur_pri, depth_pri, slope_pri = trapfit
                    except (NoEclipseError, NoFitError):
                        dur_pri, depth_pri, slope_pri = [np.nan]*3
                else:
                    dur_pri, depth_pri, slope_pri = [np.nan]*3
                # Secondary
                if occ[i] and d_sec[i] > 0:
                    try:
                        trapfit = eclipse_tt(P=period[i], p0=k[i], b=b_sec[i],
                                         aR=aR[i], 
                                         frac=flux_ratio[i]/(1 + flux_ratio[i]),
                                         u1=0.394, u2=0.296, 
                                         ecc=ecc[i], w=w[i]*180/np.pi,
                                         sec=True,
                                         **trapfit_kwargs)
                        dur_sec, depth_sec, slope_sec = trapfit
                    except (NoEclipseError, NoFitError):
                        dur_sec, depth_sec, slope_sec = [np.nan]*3
                else:
                    dur_sec, depth_sec, slope_sec = [np.nan]*3

                catalog.loc[i, 'trap_dur_pri'] = dur_pri
                catalog.loc[i, 'trap_depth_pri'] = depth_pri
                catalog.loc[i, 'trap_slope_pri'] = slope_pri
                catalog.loc[i, 'trap_dur_sec'] = dur_sec
                catalog.loc[i, 'trap_depth_sec'] = depth_sec
                catalog.loc[i, 'trap_slope_sec'] = slope_sec

                mean_depth_pri = trap_mean_depth(dur_pri, depth_pri, slope_pri) * catalog.dilution
                mean_depth_sec = trap_mean_depth(dur_sec, depth_sec, slope_sec) * catalog.dilution

        if regr_trap:
            if not self._trap_trained:
                self._train_trap()

            Xpri = self._get_trap_features(catalog, pri_only=True)
            Xsec = self._get_trap_features(catalog, sec_only=True)
            pri = (catalog.T14_pri.values > 0) & (catalog.d_pri.values > 0)
            sec = (catalog.T14_sec.values > 0) & (catalog.d_sec.values > 0)


            catalog.loc[pri, 'trap_dur_pri_regr'] = \
                self._dur_pipeline.predict(Xpri)
            catalog.loc[pri, 'trap_depth_pri_regr'] = \
                10**self._logd_pipeline.predict(Xpri)
            catalog.loc[pri, 'trap_slope_pri_regr'] = \
                self._slope_pipeline.predict(Xpri)
            if Xsec.shape[0] > 1:
                catalog.loc[sec, 'trap_dur_sec_regr'] = \
                    self._dur_pipeline.predict(Xsec)
                catalog.loc[sec, 'trap_depth_sec_regr'] = \
                    10**self._logd_pipeline.predict(Xsec)
                catalog.loc[sec, 'trap_slope_sec_regr'] = \
                    self._slope_pipeline.predict(Xsec)
            else:
                for c in ['trap_dur_sec_regr',
                          'trap_depth_sec_regr',
                          'trap_slope_sec_regr']:
                    catalog.loc[sec, c] = np.nan

            if not fit_trap:
                mean_depth_pri = catalog.dilution * trap_mean_depth(catalog.trap_dur_pri_regr, 
                                                 catalog.trap_depth_pri_regr, 
                                                 catalog.trap_slope_pri_regr)
                mean_depth_sec = catalog.dilution * trap_mean_depth(catalog.trap_dur_sec_regr, 
                                                 catalog.trap_depth_sec_regr, 
                                                 catalog.trap_slope_sec_regr)


        catalog['noise_pri'] = self.get_noise(catalog.host, catalog.T14_pri)
        catalog['noise_sec'] = self.get_noise(catalog.host, catalog.T14_sec)
        if fit_trap or regr_trap:
            catalog['snr_pri'] = mean_depth_pri / (catalog['noise_pri']*1e-6) * \
                np.sqrt(catalog['n_pri'])
            catalog['snr_sec'] = mean_depth_sec / (catalog['noise_sec']*1e-6) * \
                np.sqrt(catalog['n_sec'])
            catalog['mean_depth_pri'] = mean_depth_pri
            catalog['mean_depth_sec'] = mean_depth_sec

        if query is not None:
            return SimulatedCatalog(catalog.query(query))
        else:
            return SimulatedCatalog(catalog)

    def _check_MR(self, mass, radius):
        """
        Mass and radius in solar units; outputs boolean mask
        where True means M-R combination is out-of-bounds
        """
        if self._MR_tri is None:
            _ = self._get_binary_training_data()
        
        # Get simplex indices.  -1 means out-of-bounds
        s = self._MR_tri.find_simplex(np.ascontiguousarray([np.log10(mass),
                                                            np.log10(radius)]).T)
        
        return s==-1

    def _get_binary_training_data(self):
        """Returns features and target data for dmag/q training

        Also creates _MR_tri :class:`Delaunay` object, which 
        defines the physically allowed mass-radius region.
        """
        self._ensure_radius()

        X = np.array([getattr(self, x) for x in self.binary_features]).T

        gamma, qmin = self._get_params(['gamma','qmin'])

        M1 = np.ascontiguousarray(self.mass_A)
        minmass = self.ic.minmass
        qmin = np.maximum(qmin, minmass/M1)
        q = draw_powerlaw(gamma, (qmin, 1), N=X.shape[0])
        M2 = q*M1

        ic = self.ic
        feh = np.ascontiguousarray(self.feh)
        age = np.ascontiguousarray(self.age)
        R2 = ic.radius(M2, age, feh)
        R1 = self.radius_A
        qR = R2/R1        

        # defines physically allowed region in M-R space
        Ms = np.concatenate((M1, M2))
        Rs = np.concatenate((R1, R2))
        ok = np.isfinite(Ms) & np.isfinite(Rs)
        points = np.array([np.log10(Ms[ok]), np.log10(Rs[ok])]).T
        self._MR_tri = Delaunay(points)

        X = np.append(X, np.array([q]).T, axis=1)
        #X = np.array([M1,R1,age,feh,qR]).T
        dmag = ic.mag[self.band](M2, age, feh) - ic.mag[self.band](M1, age, feh)
        return X, dmag, qR


    def _train_pipelines(self, plot=False, **kwargs):
        Xorig, dmag, qR = self._get_binary_training_data()

        y = dmag.copy()
        ok = ~np.isnan(y)
        X = Xorig[ok, :]
        y = y[ok]

        # Separate train/test data
        u = np.random.random(X.shape[0])
        itest = u < 0.2
        itrain = u >= 0.2
        Xtest = X[itest, :]
        Xtrain = X[itrain, :]
        ytest = y[itest]
        ytrain = y[itrain]

        regr = RandomForestRegressor
        #regr = LinearRegression
        poly_kwargs = {'degree':3, 'interaction_only':False}
        dmag_pipeline = Pipeline([#('poly', PolynomialFeatures(**poly_kwargs)),
                                  ('scale', StandardScaler()), 
                                  ('regress', regr(**kwargs))])

        dmag_pipeline.fit(Xtrain,ytrain);
        yp = dmag_pipeline.predict(Xtest)
        if plot:
            fig, axes = plt.subplots(1,2, figsize=(10,4))
            axes[0].plot(ytest, yp, 'o', ms=1, mew=0.2, alpha=0.3)
            axes[0].plot(ytest, ytest, 'r-', lw=1, alpha=0.5)
            
        score = dmag_pipeline.score(Xtest, ytest)
        print('{0}: dmag regressor trained, R2={1}'.format(self.name, score))
        self._dmag_pipeline = dmag_pipeline
        self._dmag_pipeline_score = score

        Xtest_dmag = Xtest
        ytest_dmag = ytest
        yp_dmag = yp

        # Now train radius ratio qR, adding dmag to the training data.
        X = np.append(Xorig, np.array([dmag]).T, axis=1)
        y = qR
        X = X[ok, :]
        y = y[ok]

        # Separate train/test data
        Xtest = X[itest, :]
        Xtrain = X[itrain, :]
        ytest = y[itest]
        ytrain = y[itrain]

        qR_pipeline = Pipeline([#('poly', PolynomialFeatures(**poly_kwargs)),
                               ('scale', StandardScaler()), 
                               ('regress', regr(**kwargs))])

        qR_pipeline.fit(Xtrain, ytrain)
        yp = qR_pipeline.predict(Xtest)
        if plot:
            axes[1].loglog(ytest, yp, 'o', ms=1, mew=0.2, alpha=0.3)
            axes[1].plot(ytest, ytest, 'r-', lw=1, alpha=0.5)
        score = qR_pipeline.score(Xtest, ytest)
        print('{0}: qR regressor trained, R2={1}'.format(self.name, score))
        self._qR_pipeline = qR_pipeline
        self._qR_pipeline_score = score

        Xtest_qR = Xtest
        ytest_qR = ytest
        yp_qR = yp

        self._binary_trained = True

        return Xtest, (ytest_dmag, yp_dmag), (ytest_qR, yp_qR)
        
    def get_N_observed(self, query=None, N=10000, fit_trap=False,
                       regr_trap=True, new=False, new_orbits=True,
                       use_ic=None,
                       verbose=False, dataspan=None, dutycycle=None):
        df = pd.DataFrame()
        
        while len(df) < N:
            df = pd.concat([df, self.observe(query=query, new=new,
                                             new_orbits=new_orbits,
                                             fit_trap=fit_trap, use_ic=use_ic,
                                             regr_trap=regr_trap)])
            if verbose:
                print(len(df))
        return SimulatedCatalog(df.iloc[:N].reset_index())

    def _get_trap_features(self, df, sec_only=False, pri_only=False):
        #pri = ~np.isnan(df.trap_depth_pri.values) 
        #sec = ~np.isnan(df.trap_depth_sec.values)
        pri = (df.T14_pri.values > 0) & (df.d_pri.values > 0)
        sec = (df.T14_sec.values > 0) & (df.d_sec.values > 0)
        if sec_only:
            pri[:] = False
        if pri_only:
            sec[:] = False

        T14 = np.concatenate((df.T14_pri.values[pri], df.T14_sec.values[sec]))
        T23 = np.concatenate((df.T23_pri.values[pri], df.T23_sec.values[sec]))
        T14 += self.texp
        T23 = np.clip(T23 - self.texp, 0, T14)
        tau = (T14 - T23)/2.
        k = np.concatenate((df.k.values[pri], 1./df.k.values[sec]))
        b = np.concatenate((df.b_pri.values[pri], df.b_sec.values[sec]))
        logd = np.log10(np.concatenate((df.d_pri[pri], df.d_sec[sec])))
        secondary = np.concatenate((np.zeros(pri.sum()), np.ones(sec.sum())))

        X = np.array([T14, tau, k, b, logd, secondary]).T
        return X

    def _train_trap(self, query=None, N=10000,
                    plot=False, use_ic=None, **kwargs):
        """
        N is minimum number of simulated transits to train with.
        """
        # Deal with corner case where dataspan, dutycycle
        # not provided, and we have to invent them temporarily
        temp_obsdata = False
        if 'dataspan' in self._not_calculated:
            temp_obsdata = True
            self.stars.loc[:, 'dataspan'] = 1400
            self.stars.loc[:, 'dutycycle'] = 1.
            self._mark_calculated('dataspan')
            self._mark_calculated('dutycycle')

        df = self.get_N_observed(query=query, N=N, fit_trap=True, 
                                 new_orbits=True,
                                 regr_trap=False, use_ic=use_ic)

        if temp_obsdata:
            for c in ['dataspan', 'dutycycle']:
                self._remove_prop(c)

        X = self._get_trap_features(df)
        
        pri = (df.T14_pri.values > 0) & (df.d_pri.values > 0)
        sec = (df.T14_sec.values > 0) & (df.d_sec.values > 0)
        y1 = np.log10(np.concatenate((df.trap_depth_pri.values[pri],
                                  df.trap_depth_sec.values[sec])))
        y2 = np.concatenate((df.trap_dur_pri.values[pri],
                            df.trap_dur_sec.values[sec]))
        y3 = np.concatenate((df.trap_slope_pri.values[pri],
                            df.trap_slope_sec.values[sec]))
        ok = np.isfinite(X.sum(axis=1) + y1 + y2 + y3) 
        
        # Train/test split
        u = np.random.random(X.shape[0])
        itest = (u < 0.2) & ok
        itrain = (u >= 0.2) & ok
        Xtest = X[itest, :]
        Xtrain = X[itrain, :]

        regr = RandomForestRegressor

        # Train depth
        y = y1
        ytrain = y[itrain]
        ytest = y[itest]
        pipeline = Pipeline([('scale', StandardScaler()),
                                   ('regression', regr(**kwargs))])
        pipeline.fit(Xtrain, ytrain)
        score = pipeline.score(Xtrain, ytrain)
        if plot:
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            yp = pipeline.predict(Xtest)
            axes[0].plot(ytest, yp, '.', alpha=0.3)
            axes[0].plot(ytest, ytest, 'k-')
        print(('{}: Depth trained: R2={}'.format(self.name, score)))
        self._logd_pipeline = pipeline
        self._logd_score = score

        # Train duration
        y = y2
        ytrain = y[itrain]
        ytest = y[itest]
        pipeline = Pipeline([('scale', StandardScaler()),
                                   ('regression', regr(**kwargs))])
        pipeline.fit(Xtrain, ytrain)
        score = pipeline.score(Xtrain, ytrain)
        if plot:
            yp = pipeline.predict(Xtest)
            axes[1].plot(ytest, yp, '.', alpha=0.3)
            axes[1].plot(ytest, ytest, 'k-')
        print(('{}: Duration trained: R2={}'.format(self.name, score)))
        self._dur_pipeline = pipeline
        self._dur_score = score


        # Train slope
        y = y3
        ytrain = y[itrain]
        ytest = y[itest]
        pipeline = Pipeline([('scale', StandardScaler()),
                                   ('regression', regr(**kwargs))])
        pipeline.fit(Xtrain, ytrain)
        score = pipeline.score(Xtrain, ytrain)
        if plot:
            yp = pipeline.predict(Xtest)
            axes[2].plot(ytest, yp, '.', alpha=0.3)
            axes[2].plot(ytest, ytest, 'k-')
        print(('{}: Slope trained: R2={}'.format(self.name, score)))
        self._slope_pipeline = pipeline
        self._slope_score = score

        self._trap_trained = True
        
        return self._logd_pipeline, self._dur_pipeline, self._slope_pipeline

    def save(self, folder, overwrite=False):
        if os.path.exists(folder):
            if overwrite:
                shutil.rmtree(folder)
            else:
                raise IOError('{} exists.  Set overwrite if desired.'.format(folder))
        os.makedirs(folder)
        
        # Write stars table to HDF
        self._record_stars_changes()
        self._stars.to_hdf(os.path.join(folder, 'stars.h5'), 'df')
        store = pd.HDFStore(os.path.join(folder, 'stars.h5'))
        if self._index is not None:
            store['index'] = pd.Series(self._index)
        attrs = store.get_storer('df').attrs

        # Store attributes in HDF file
        for attr in self._attrs:
            val = getattr(self, attr)
            attrs[attr] = val
        store.close()

        # Write other tables that might be necessary
        for tbl in self._tables:
            df = getattr(self, tbl)
            df.to_hdf(os.path.join(folder, '{}.h5'.format(tbl)), 'df')
        
        # Save trained pipelines, if they exist
        pline_folder = os.path.join(folder,'pipelines')
        if self._binary_trained:
            if not os.path.exists(pline_folder):
                os.makedirs(pline_folder)
            joblib.dump(self._dmag_pipeline, 
                        os.path.join(pline_folder, 'dmag_pipeline.pkl'))
            joblib.dump(self._qR_pipeline, 
                        os.path.join(pline_folder, 'qR_pipeline.pkl'))

        if self._trap_trained:
            if not os.path.exists(pline_folder):
                os.makedirs(pline_folder)
            joblib.dump(self._logd_pipeline, 
                        os.path.join(pline_folder, 'logd_pipeline.pkl'))
            joblib.dump(self._dur_pipeline, 
                        os.path.join(pline_folder, 'dur_pipeline.pkl'))
            joblib.dump(self._slope_pipeline, 
                        os.path.join(pline_folder, 'slope_pipeline.pkl'))

        # Record the type of the object, in order to restore it correctly.
        tfile = os.path.join(folder,'type.pkl')
        pickle.dump(type(self), open(tfile, 'wb'))

    @classmethod
    def load(cls, folder):
        store = pd.HDFStore(os.path.join(folder, 'stars.h5'))
        stars = store['df']
        for c in ['tra','occ']:
            if c in stars:
                stars.loc[:,c] = stars.loc[:,c].astype(bool)
            
        attrs = store.get_storer('df').attrs

        if 'index' in store:
            index = pd.Index(store['index'])
        else:
            index = None

        # Read the proper type to load in
        t = pickle.load(open(os.path.join(folder,'type.pkl'),'rb'))

        #forward-hack for BGBinaryPopulation.  Should be a more
        # general way to do this (e.g. define the call signature of __init__)
        # w.r.t. tables
        if issubclass(t, BGBinaryPopulation):
            targets = pd.read_hdf(os.path.join(folder, 'targets.h5'),'df')
            new = t(targets, stars, index=index, copy=False)
        else:
            new = t(stars, index=index, copy=False)
        for attr in new._attrs:
            val = attrs[attr]
            setattr(new, attr, val)
        store.close()

        pline_folder = os.path.join(folder,'pipelines')
        if 'dmag_pipeline.pkl' in os.listdir(pline_folder):
            new._dmag_pipeline = joblib.load(os.path.join(pline_folder,
                                                          'dmag_pipeline.pkl'))
            new._qR_pipeline = joblib.load(os.path.join(pline_folder,
                                                        'qR_pipeline.pkl'))
            new._binary_trained = True

        if 'logd_pipeline.pkl' in os.listdir(pline_folder):
            new._logd_pipeline = joblib.load(os.path.join(pline_folder,
                                                          'logd_pipeline.pkl'))
            new._dur_pipeline = joblib.load(os.path.join(pline_folder,
                                                          'dur_pipeline.pkl'))
            new._slope_pipeline = joblib.load(os.path.join(pline_folder,
                                                          'slope_pipeline.pkl'))
            new._trap_trained = True

        return new

class PowerLawBinaryPopulation(BinaryPopulation):
    """
    This describes a population with power-law period distribution
 
    Appropriate for closer-in binary population probed with EBs.

    Default fB, beta tuned to roughly match with default lognormal
    period distribution from BinaryPopulation
    """
    param_names = ('fB', 'gamma', 'qmin', 'beta', 
                   'beta_a', 'beta_b', 'period_min', 'period_max')

    default_params = {'fB':0.15, 'gamma':0.3, 'qmin':0.1,
                      'beta_a':0.8, 'beta_b':2.0, 'beta':-0.75,
                      'period_min':5, 'period_max':20*365}

    def _sample_period(self, N):
        beta, lo, hi = self._get_params(['beta', 'period_min', 'period_max'])
        return draw_powerlaw(beta, (lo, hi), N=N)

    
class KeplerPopulation(BinaryPopulation):
    cdpp_durations = (1.5, 2.0, 2.5, 3.0, 3.5,
                      4.5, 5.0, 6.0, 7.5, 9.0,
                      10.5, 12.0, 12.5, 15.0)

    def get_noise(self, idx, T=3):
        s = self.get_target(idx)
        dur_bin = np.atleast_1d(np.digitize(T, self.cdpp_durations))
        off_grid = False

        lo = dur_bin==0
        hi = dur_bin >= len(self.cdpp_durations)

        i0 = dur_bin - 1
        x0 = np.array(self.cdpp_durations)[i0]
        i1 = dur_bin.copy()
        i1[hi] = -1
        x1 = np.array(self.cdpp_durations)[i1]

        x0[lo] = self.cdpp_durations[0]
        x1[hi] = self.cdpp_durations[-1]

        # Note: probably better way to do this?
        f0,i0 = np.modf(x0)
        col0 = ['rrmscdpp{:02.0f}p{:01.0f}'.format(i,f*10) for i,f in zip(i0, f0)]
        y0 = np.diag(s[col0])

        f1,i1 = np.modf(x1)
        col1 = ['rrmscdpp{:02.0f}p{:01.0f}'.format(i,f*10) for i,f in zip(i1, f1)]
        y1 = np.diag(s[col1])

        #print(x0,y0,s[col0])
        #print(x1,y1,s[col1])

        y = y0 + (y1 - y0)*(T - x0)/(x1 - x0)
        y[lo | hi] = y0[lo | hi]
        return y

class KeplerBinaryPopulation(KeplerPopulation):
    #  Don't use KIC radius here; recalc for consistency.
    prop_columns = {'mass_A':'mass'}

class KeplerPowerLawBinaryPopulation(PowerLawBinaryPopulation, KeplerPopulation):
    #  Don't use KIC radius here; recalc for consistency.
    prop_columns = {'mass_A':'mass'}


class BlendedBinaryPopulation(BinaryPopulation):
    """
    Class for diluted binary populations

    Implement `_get_dilution` method to dilute the depths
    """    
    default_name = 'blended EB'

    def _generate_orbits(self, *args, **kwargs):
        # First, proceed as before...
        super(BlendedBinaryPopulation, self)._generate_orbits(*args, **kwargs)
        
class TRILEGAL_BinaryPopulation(BinaryPopulation):
    prop_columns = {'age':'logAge', 'feh':'[M/H]', 
                    'mass_A':'m_ini'}

    binary_features = ('mass_A', 'feh', 'age', 'logL', 'logTe', 'logg')

    def _initialize_stars(self):
        # Proceed as before
        super(TRILEGAL_BinaryPopulation, self)._initialize_stars()
        self._set_radius()

    def _set_radius(self):
        # create radius_A column
        mass = self._stars.mass_A
        logg = self._stars.logg
        self._stars.loc[:, 'radius_A'] = np.sqrt(G * mass * MSUN / 10**logg)/RSUN
        self._mark_calculated('radius_A')

class BGBinaryPopulation(BlendedBinaryPopulation):
    """
    targets is DataFrame of target stars
    bgstars is DataFrame of background stars
    """
    param_names = ('fB', 'gamma', 'qmin', 'mu_logp', 'sig_logp', 
                   'beta_a', 'beta_b', 'rho_5', 'rho_20', 
                   'period_min')
    default_params = {'fB':0.44, 'gamma':0.3, 'qmin':0.1, 
                      'mu_logp':np.log10(250),
                      'sig_logp':2.3, 'beta_a':0.8, 'beta_b':2.0,
                      'rho_5':0.05, 'rho_20':0.005,
                      'period_min':5.}

    obs_props = BlendedBinaryPopulation.obs_props + ('target_mag',)

    default_name = 'BGEB'

    _attrs = BlendedBinaryPopulation._attrs + ('r_blend', 'target_band')
    _tables = BlendedBinaryPopulation._tables + ('targets',)

    def __init__(self, targets, bgstars, r_blend=4, 
                 target_band='kepmag', copy=True, **kwargs):
        # Copy data to avoid surprises
        if copy:
            self.targets = targets.copy()
        else:
            self.targets = targets

        self.r_blend = r_blend
        self.target_band = target_band

        super(BGBinaryPopulation, self).__init__(bgstars, copy=copy, **kwargs)
        if self._index is None:
            self._define_stars()
        #elif len(self._index)==0:
        #    self._define_stars()

    @property
    def dilution_factor(self):
        F_target = 10**(-0.4*self.stars.target_mag) #stars['target_mag'])
        F_A = 10**(-0.4*self.stars['{}_mag'.format(self.band)]) # primary mag
        # Force flux_ratio to be calculated by accessing as property
        F_B = self.flux_ratio*F_A 
        frac = (F_A + F_B)/(F_A + F_B + F_target)
        return np.array(frac)
        
    @property
    def b(self):
        """
        Galactic latitude of target stars
        """
        if 'b' not in self.targets:
            c = SkyCoord(self.targets.ra, self.targets.dec, unit='deg')
            self.targets.loc[:, 'b'] = c.galactic.b.deg
        return self.targets.loc[:, 'b'].values

    def rho_bg(self, b):
        """
        Density of BG stars at Galactic latitude, given parameters

        Given rho(5) = rho_5, rho(20) = rho_20,
        solve for A,B in rho(b) = A*exp(-b/B)
        """
        rho_5, rho_20 = self._get_params(['rho_5', 'rho_20'])
        B = 15/np.log(rho_5/rho_20)
        A = rho_5 / np.exp(-5./B)
        return A*np.exp(-b/B)

    def observe(self, *args, **kwargs):
        if 'new' in kwargs:
            if kwargs['new']:
                self._define_stars()

        return super(BGBinaryPopulation,self).observe(*args, **kwargs)

    def _define_stars(self):
        b = self.b
        Nexp = np.pi*self.r_blend**2 * self.rho_bg(b)

        # Calculate number of BG stars blended with each target
        N = np.random.poisson(Nexp, size=len(Nexp))
        Ntot = N.sum()

        logging.debug('Defining {} background stars.'.format(Ntot))

        # Choose bg stars
        i_bg = np.random.randint(0, len(self._stars), size=Ntot)
        i_bg = self._stars.index[i_bg]

        # Assign stars to target stars as appropriate.
        i_targ = np.zeros(Ntot, dtype=int)
        b_targ = np.zeros(Ntot)
        i = 0
        for j,n in enumerate(N):
            i_targ[i:i+n] = j
            b_targ[i:i+n] = b[j]
            i += n

        # Assign dataspan, dutycycle, target mag according to target stars
        dataspan = self.targets.dataspan.values[i_targ]
        dutycycle = self.targets.dutycycle.values[i_targ]
        mags = self.targets[self.target_band].values[i_targ]
        newvals = np.array([dataspan, dutycycle, mags, b_targ]).T
        cols = ['dataspan', 'dutycycle', 'target_mag', 'b_target']
        self._stars.loc[i_bg, cols] = newvals
        self._set_index(i_bg)

        # Reset binary/orbital properties (possible bug alert by ignoring obs_props?)
        self._not_calculated = [c for c in self.secondary_props + 
                                self.orbital_props]

    def _train_pipelines(self, **kwargs):
        # Train pipelines using entire bgstar catalog, not small selection.
        old_index = self._index
        self._index = self._stars.index

        super(BGBinaryPopulation, self)._train_pipelines(**kwargs)
        
        # Return as before
        self._index = old_index

class BGPowerLawBinaryPopulation(BGBinaryPopulation, PowerLawBinaryPopulation):
    param_names = ('fB', 'gamma', 'qmin', 'beta', 
                   'beta_a', 'beta_b', 'rho_5', 'rho_20', 
                   'period_min', 'period_max')

    default_params = {'fB':0.15, 'gamma':0.3, 'qmin':0.1, 
                      'beta':-0.75,
                      'beta_a':0.8, 'beta_b':2.0,
                      'rho_5':0.05, 'rho_20':0.005,
                      'period_min':5., 'period_max':20*365.}

    _sample_period = PowerLawBinaryPopulation._sample_period


class TRILEGAL_BGBinaryPopulation(TRILEGAL_BinaryPopulation, BGBinaryPopulation):
    
    # This from BGBinaryPopulation; otherwise want to inherit from 
    # TRILEGAL_BinaryPopulation
    __init__ = BGBinaryPopulation.__init__

class TRILEGAL_BGPowerLawBinaryPopulation(TRILEGAL_BinaryPopulation,
                                          BGPowerLawBinaryPopulation):
    __init__ = BGPowerLawBinaryPopulation.__init__

class PlanetPopulation(KeplerBinaryPopulation):
    """
    Need to implement _sample_Np, sample_Rp, _sample_period methods
    """

    default_name = 'Planet'

    def _sample_Np(self, N):
        raise NotImplementedError

    def _sample_Rp(self, N):
        raise NotImplementedError

    @property
    def host_stars(self):
        return self._stars

    def get_target(self, i):
        return self.host_stars.ix[i]

    def _generate_planets(self, **kwargs):
        N = len(self.host_stars)
        logging.debug('Generating planetary companions for {} stars...'.format(N))

        # need to create mass_B, radius_B, flux_ratio
        
        # First, determine how many planets to assign per star.
        Nplanets = self._sample_Np(N)
        Ntot = Nplanets.sum()

        i_host = np.zeros(Ntot, dtype=int)
        i = 0
        for n,ix in zip(Nplanets,self._stars.index):
            i_host[i:i+n] = ix
            i += n
        self._set_index(i_host)

        # Then, give them radii, masses, flux_ratio
        radius_B = self._sample_Rp(Ntot) #comes back in R_SUN
        mass_B = (radius_B/RSUN * REARTH)**2.06 * MEARTH/MSUN #in MSUN
        flux_ratio = 0. #could presumably give reflected/thermal emission here...
        
        for c in self.secondary_props:
            self.stars.loc[:, c] = eval(c)
            self._mark_calculated(c)
        
        logging.debug('{} planets generated.'.format(Ntot))


    def _generate_binaries(self, **kwargs):
        """
        Generating companion planets according to parameters

        This uses standard independent Poisson process model.
        """
        self._ensure_radius()
        self._generate_planets(**kwargs)

class PoissonPlanetPopulation(PlanetPopulation):
    """
    Poisson/isotropic model of planet occurrence.

    Parameters:
      * N_pl: average number of planets per system
      * beta: power-law period index
      * alpha: power-law radius index
      * beta_a, beta_b: eccentricity beta-distribution parameters
    """
    param_names = ('N_pl', 'beta', 'alpha', 'Rp_min', 'Rp_max',
                   'period_min', 'period_max', 'beta_a', 'beta_b')
    default_params = {'N_pl':1.0, 'beta':-0.75, 'alpha':-1.6,
                      'Rp_min':0.75, 'Rp_max':20, 
                      'period_min':5., 'period_max':10000.,
                      'beta_a':0.8, 'beta_b':2.0}

    
    def _sample_period(self, N):
        beta, lo, hi = self._get_params(['beta', 'period_min', 'period_max'])
        return draw_powerlaw(beta, (lo, hi), N=N)

    def _sample_Np(self, N):
        N_pl = self.params['N_pl']
        return np.random.poisson(N_pl, size=N)

    def _sample_Rp(self, N):
        alpha, lo, hi = self._get_params(['alpha', 'Rp_min', 'Rp_max'])
        return draw_powerlaw(alpha, (lo, hi), N=N) * REARTH/RSUN


class PopulationMixture(object):
    def __init__(self, poplist):
        self.poplist = poplist
        
    def __getitem__(self, name):
        for pop in self.poplist:
            if name==pop.name:
                return pop
        
    @property
    def param_names(self):
        p = []
        for pop in self.poplist:
            p += pop.param_names
        return list(set(p))
    
    @property
    def params(self):
        d = {}
        for pop in self.poplist:
            for k,v in pop.params.items():
                if k not in d:
                    d[k] = v
                else:
                    if d[k] != v:
                        raise ValueError('Parameter mismatch! ({})'.format(k))
        return d
    
    def set_params(self, **kwargs):
        for pop in self.poplist:
            pop.set_params(**kwargs)
                        
    def reset_params(self):
        for pop in self.poplist:
            pop.reset_params()
                
    def train_trap(self, **kwargs):
        return [p._train_trap(**kwargs) for p in self.poplist]
                
    def observe(self, **kwargs):
        obs = []
        for pop in self.poplist:
            o = pop.observe(**kwargs)
            if len(o)>0:
                o.loc[:, 'population'] = pop.name
            obs.append(o)
            
        return pd.concat(obs)

    def save(self, folder, overwrite=False):
        if os.path.exists(folder):
            if overwrite:
                shutil.rmtree(folder)
            else:
                raise IOError('{} exists.  Set overwrite=True to overwrite.'.format(folder))
        os.makedirs(folder)
        for pop in self.poplist:
            pop.save(os.path.join(folder,pop.name))
        
        
    @classmethod
    def load(cls, folder):
        names = os.listdir(folder)
        poplist = []
        for name in names:
            f = os.path.join(folder,name)
            poplist.append(BinaryPopulation.load(f))
        return cls(poplist)
