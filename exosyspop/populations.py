from __future__ import print_function, division

import numpy as np
import pandas as pd

# This disables expensive garbage collection calls
# within python.  Took forever to figure this out.
pd.set_option('mode.chained_assignment', None)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from isochrones.dartmouth import Dartmouth_Isochrone
DAR = Dartmouth_Isochrone()
DAR.radius(1,9.5,0) #prime the isochrone object

# Still with vespa dependencis for now
from vespa.stars.utils import draw_eccs # this is a function that returns
                                        # empirically reasonably eccentricities
                                        # for given binary periods.
from vespa.transit_basic import _quadratic_ld, eclipse_tt, NoEclipseError

from .utils import draw_powerlaw, semimajor
from .utils import G, MSUN, RSUN, AU, DAY

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

    The first of these regressions is to predict the flux ratio 
pd.set_option('mode.chained_assignment', None)    between the secondary and primary, as well as the mass ratio,
    as a function of the other more readily simulated parameters.
    Enabling this is that the population simulation is parameterized
    in order to directly infer the *radius ratio* (`qR`) distribution, 
    rather then the *mass ratio* distribution.  Then, `flux_ratio`
    is predicted as a function of `mass_A`, `radius_A`, `qR`, `age`,
    and `feh`.  And then the mass ratio `q` is predicted using all of
    the above, as well as `flux_ratio`.  These regressions are trained
    using quantities simulated directly according to the provided 
    :class:`Isochrone` object, and this training takes just a few seconds.
    Once trained, this regression (which is very
    accurate---R^2 ~ 0.999 with a stellar population of ~30,000) computes
    the secondary properties of a simulated population about a factor of 10x 
    faster than direct calls to the :class:`Isochrone`.

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
    param_names = ('fB', 'gamma', 'qRmin', 'mu_logp', 'sig_logp')
    default_params = (0.4, 0.3, 0.1, np.log10(250), 2.3)

    # Physical and orbital parameters that can be accessed.
    physical_props = ('mass_A', 'radius_A',
                      'mass_B', 'radius_B', 'flux_ratio')

    orbital_props = ('period', 'ecc', 'w', 'inc', 'a', 'aR',
                      'b_pri', 'b_sec', 'k', 'tra', 'occ',
                      'd_pri', 'd_sec', 'T14_pri', 'T14_sec',
                      'T23_pri', 'T23_sec')
                     
    # property dictionary mapping to DataFrame column
    prop_columns = {}

    # Minimum radius allowed, in Rsun.
    min_radius = 0.11

    # Minimum orbital period allowed.
    min_period = 1.

    # Band in which eclipses are observed,
    # and exposure integration time.
    band = 'Kepler'
    texp = 1626./86400

    def __init__(self, stars, params=None, 
                 ic=DAR, **kwargs):

        # Copy data, so as to avoid surprises.
        self.stars = stars.copy()
        self._ic = ic
        self._params = params

        # Put the appopriate renamed columns in
        for k,v in self.prop_columns.items():
            self.stars.loc[:, k] = self.stars[v]

        # Create all the columns that will be filled later
        self._not_calculated = [c for c in self.physical_props +
                                self.orbital_props if c not in self.stars]
        for c in self._not_calculated:
            self.stars.loc[:, c] = np.nan

        self.set_params(**kwargs)

        # Regressions to be trained
        self._fluxrat_pipeline = None
        self._q_pipeline = None
        self._logd_pipeline = None
        self._dur_pipeline = None
        self._slope_pipeline = None

    def __getattr__(self, name):
        if name in self._not_calculated:
            if name in self.physical_props:
                self._generate_binaries()
            elif name in self.orbital_props:
                self._generate_orbits()
        return self.stars[name].values


    def _mark_calculated(self, prop):
        try:
            i = self._not_calculated.index(prop)
            self._not_calculated.pop(i)
        except ValueError:
            pass
        
    def _remove_prop(self, prop):
        self.stars.loc[:, prop] = np.nan
        self._not_calculated.append(prop)

    @property
    def params(self):
        if self._params is not None:
            return self._params
        else:
            return self.default_params

    @params.setter
    def params(self, p):
        assert len(p)==len(self.param_names)
        self._params = p

    def set_params(self, **kwargs):
        if self._params is None:
            self._params = list(self.default_params)
        for k,v in kwargs.items():
            self._params[self.param_names.index(k)] = v

    @property
    def ic(self):
        if type(self._ic)==type:
            self._ic = self._ic()
        return self._ic

    @property
    def N(self):
        return len(self.stars)

    def _assign_ages(self):
        # Stellar catalog doesn't have ages, so let's make them up.
        #  ascontiguousarray makes ic calls faster.
        if 'age' in self.stars:
            return

        ic = self.ic
        feh = np.ascontiguousarray(np.clip(self.feh, ic.minfeh, ic.maxfeh))
        minage, maxage = ic.agerange(self.mass_A, feh)
        maxage = np.clip(maxage, 0, ic.maxage)
        if 'age' not in self.stars:
            minage += 0.3 # stars are selected to not be active
            maxage -= 0.1
            age = np.random.random(size=len(feh)) * (maxage - minage) + minage
        else:
            age = np.clip(self.stars.age.values, minage, maxage)

        self.stars.loc[:,'age'] = age
        self.stars.loc[:,'feh'] = feh #reassigning feh

    def _generate_binaries(self):
        N = self.N
        fB, gamma, qRmin, _, _ = self.params

        b = np.random.random(N) < fB

        self._assign_ages()

        # Simulate primary radius (unless radius_A provided)
        if 'radius_A' in self._not_calculated:
            self.stars.loc[:, 'radius_A'] = self.ic.radius(self.mass_A, 
                                                           self.age, 
                                                           self.feh)
            self._mark_calculated('radius_A')

        R1 = self.radius_A[b]

        # Simulate secondary radii (not masses!)
        minrad = self.min_radius
        qRmin = np.maximum(qRmin, minrad/self.radius_A)
        qR = draw_powerlaw(gamma, (qRmin, 1), N=N)
        R2 = (qR * self.radius_A)[b]

        # Calculate dmag->flux_ratio from trained regression
        if self._fluxrat_pipeline is None:
            self._train_pipelines()

        M1 = self.mass_A[b]
        age = self.age[b]
        feh = self.feh[b]
        X = np.array([M1, R1, qR[b], age, feh]).T
        flux_ratio = self._fluxrat_pipeline.predict(X)
        #dmag = self._dmag_pipeline.predict(X)
        #flux_ratio = 10**(-0.4 * dmag)

        # Calculate q->mass_B from trained regression
        X = np.array([M1, R1, qR[b], age, feh, flux_ratio]).T
        q = self._q_pipeline.predict(X)
        M2 = np.zeros(N)
        M2[b] = q*M1
        M2[~b] = np.nan

        # Recreate arrays to avoid pandas weirdness
        radius_B = np.zeros(N)
        radius_B[b] = R2
        radius_B[~b] = np.nan

        fluxrat = np.zeros(N)
        fluxrat[b] = flux_ratio
        
        self.stars.loc[:, 'mass_B'] = M2
        self.stars.loc[:, 'radius_B'] = radius_B
        self.stars.loc[:, 'flux_ratio'] = fluxrat
        for c in ['mass_B', 'radius_B', 'flux_ratio']:
            self._mark_calculated(c)

    def _generate_orbits(self, geom_only=False):
        _, _, _, mu_logp, sig_logp = self.params

        N = self.N

        
        # Generate periods, but 
        #  don't let anything shorter than minimum period
        period = 10**(np.random.normal(np.log10(mu_logp), sig_logp, size=N)) * 365.25
        bad = period < self.min_period
        nbad = bad.sum()
        while nbad > 0:
            period[bad] = 10**(np.random.normal(np.log10(mu_logp), 
                                                sig_logp, size=nbad)) * 365.25
            bad = period < self.min_period
            nbad = bad.sum()

        # Rest of the orbital parameters
        ecc = draw_eccs(N, period)
        w = np.random.random(N) * 2 * np.pi
        inc = np.arccos(np.random.random(N))        
        a = semimajor(period, self.mass_A + self.mass_B) * AU
        aR = a / (self.radius_A * RSUN)
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
        b_pri = a*np.cos(inc)/(self.radius_A*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w))
        b_sec = a*np.cos(inc)/(self.radius_A*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w))

        R_tot = (self.radius_A + self.radius_B)/self.radius_A
        tra = (b_pri < R_tot)
        occ = (b_sec < R_tot)

        # Calculate eclipse depths, assuming Solar limb darkening for all
        d_pri = np.zeros(N)
        d_sec = np.zeros(N)
        k = self.radius_B / self.radius_A
        T14_pri = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1+k)**2 - b_pri**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1+ecc*np.sin(w))
        T14_sec = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1+k)**2 - b_sec**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1-ecc*np.sin(w))
        T23_pri = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1-k)**2 - b_pri**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1+ecc*np.sin(w))
        T23_sec = period/np.pi*np.arcsin(self.radius_A*RSUN/a * np.sqrt((1-k)**2 - b_sec**2)/np.sin(inc)) *\
            np.sqrt(1-ecc**2)/(1-ecc*np.sin(w))
    
        T14_pri[np.isnan(T14_pri)] = 0.
        T14_sec[np.isnan(T14_sec)] = 0.
        T23_pri[np.isnan(T23_pri)] = 0.
        T23_sec[np.isnan(T23_sec)] = 0.

        for i in xrange(N):
            if tra[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_pri[i]]), k[i], 0.394, 0.296, 1)[0]
                F2 = self.flux_ratio[i]
                d_pri[i] = 1 - (F2 + f)/(1+F2)
            if occ[i]:
                f = _quadratic_ld._quadratic_ld(np.array([b_sec[i]/k[i]]), 1./k[i], 0.394, 0.296, 1)[0]
                F2 = self.flux_ratio[i]
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
                new_orbits=False, regr_trap=False):
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
          * trapezoidal fit params [either explicitly fit or regressed]
              * depth
              * duration
              * "slope" (T/tau)

        Observations account for both geometry and duty cycle.  
        The latter is accounted for by drawing randomly from a binomial
        distribution B(n_exp, dutycycle), where n_exp is the number
        of eclipses that would be observed with 100% duty cycle.  This
        is done independently for primary and secondary eclipses.

        TODO: incorporate pipeline detection efficiency.

        """
        if fit_trap:
            new = True
        if new:
            self._generate_binaries()
            self._generate_orbits()
        elif new_orbits:
            self._generate_orbits()

        # Select only systems with eclipsing (or occulting) geometry
        m = (self.tra | self.occ) & (self.stars.dataspan > 0)
        cols = list(self.orbital_props) + ['dataspan', 'dutycycle', 'flux_ratio']
        if query is not None:
            df = self.stars.loc[m, cols].query(query)
        else:
            df = self.stars.loc[m, cols].copy()

        # Phase of secondary (Hilditch (2001) , Kopal (1959))
        #  Primary is at phase=0
        X = np.pi + 2*np.arctan(df.ecc * np.cos(df.w) / np.sqrt(1-df.ecc**2))
        secondary_phase = X - np.sin(X)

        # Assign each system a random phase at t=0;
        N = len(df)
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

        m = (df.n_pri > 0) | (df.n_sec > 0)
        catalog = df[m].reset_index()
        
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

            trapfit_kwargs = dict(npts=50, width=3, cadence=self.texp)
            for i in xrange(N):
                # Primary
                if tra[i]:
                    try:
                        trapfit = eclipse_tt(P=period[i], p0=k[i], b=b_pri[i],
                                         aR=aR[i], frac=1/(1 + flux_ratio[i]),
                                         u1=0.394, u2=0.296, 
                                         ecc=ecc[i], w=w[i]*180/np.pi,
                                         **trapfit_kwargs)
                        dur_pri, depth_pri, slope_pri = trapfit
                    except NoEclipseError:
                        dur_pri, depth_pri, slope_pri = [np.nan]*3
                else:
                    dur_pri, depth_pri, slope_pri = [np.nan]*3
                # Secondary
                if occ[i]:
                    try:
                        trapfit = eclipse_tt(P=period[i], p0=k[i], b=b_sec[i],
                                         aR=aR[i], 
                                         frac=flux_ratio[i]/(1 + flux_ratio[i]),
                                         u1=0.394, u2=0.296, 
                                         ecc=ecc[i], w=w[i]*180/np.pi,
                                         sec=True,
                                         **trapfit_kwargs)
                        dur_sec, depth_sec, slope_sec = trapfit
                    except NoEclipseError:
                        dur_sec, depth_sec, slope_sec = [np.nan]*3
                else:
                    dur_sec, depth_sec, slope_sec = [np.nan]*3

                catalog.loc[i, 'trap_dur_pri'] = dur_pri
                catalog.loc[i, 'trap_depth_pri'] = depth_pri
                catalog.loc[i, 'trap_slope_pri'] = slope_pri
                catalog.loc[i, 'trap_dur_sec'] = dur_sec
                catalog.loc[i, 'trap_depth_sec'] = depth_sec
                catalog.loc[i, 'trap_slope_sec'] = slope_sec

        if regr_trap:
            if self._dur_pipeline is None:
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
            catalog.loc[sec, 'trap_dur_sec_regr'] = \
                self._dur_pipeline.predict(Xsec)
            catalog.loc[sec, 'trap_depth_sec_regr'] = \
                10**self._logd_pipeline.predict(Xsec)
            catalog.loc[sec, 'trap_slope_sec_regr'] = \
                self._slope_pipeline.predict(Xsec)


        return catalog

    def _train_pipelines(self, plot=False, **kwargs):
        self._assign_ages()
        M1 = np.ascontiguousarray(self.mass_A)
        
        # treat q now as mass-ratio powerlaw for training purposes
        # to generate toy secondary masses.
        fB, gamma, qmin, _, _ = self.params
        minmass = self.ic.minmass
        qmin = np.maximum(qmin, minmass/M1)
        q = draw_powerlaw(gamma, (qmin, 1), N=len(M1))
        M2 = q*M1

        ic = self.ic
        feh = np.ascontiguousarray(self.feh)
        age = np.ascontiguousarray(self.age)
        R1 = ic.radius(M1, age, feh)
        R2 = ic.radius(M2, age, feh)
        qR = R2/R1        

        # Train flux_ratio pipeline
        X = np.array([M1,R1,qR,age,feh]).T
        dmag = ic.mag['Kepler'](M2, age, feh) - ic.mag['Kepler'](M1, age, feh)
        fluxrat = 10**(-0.4*dmag)
        y = dmag
        y = fluxrat
        ok = ~np.isnan(y)
        X = X[ok, :]
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
        fluxrat_pipeline = Pipeline([#('poly', PolynomialFeatures(**poly_kwargs)),
                                  ('scale', StandardScaler()), 
                                  ('regress', regr(**kwargs))])

        fluxrat_pipeline.fit(Xtrain,ytrain);
        if plot:
            fig, axes = plt.subplots(1,2, figsize=(10,4))
            yp = fluxrat_pipeline.predict(Xtest)
            axes[0].loglog(ytest, yp, '.', alpha=0.3)
            axes[0].plot(ytest, ytest, 'k-')
            
        score = fluxrat_pipeline.score(Xtest, ytest)
        print('flux_ratio regressor trained, R2={0}'.format(score))
        self._fluxrat_pipeline = fluxrat_pipeline
        self._fluxrat_pipeline_score = score

        # Now train q pipeline
        X = np.array([M1, R1, qR, age, feh, fluxrat]).T
        y = q
        X = X[ok, :]
        y = y[ok]

        # Separate train/test data
        u = np.random.random(X.shape[0])
        itest = u < 0.2
        itrain = u >= 0.2
        Xtest = X[itest, :]
        Xtrain = X[itrain, :]
        ytest = y[itest]
        ytrain = y[itrain]

        q_pipeline = Pipeline([#('poly', PolynomialFeatures(**poly_kwargs)),
                               ('scale', StandardScaler()), 
                               ('regress', regr(**kwargs))])

        q_pipeline.fit(Xtrain, ytrain)
        if plot:
            yp = q_pipeline.predict(Xtest)
            axes[1].plot(ytest, yp, '.', alpha=0.3)
            axes[1].plot(ytest, ytest, 'k-')
        score = q_pipeline.score(Xtest, ytest)
        print('q regressor trained, R2={0}'.format(score))
        self._q_pipeline = q_pipeline
        self._q_pipeline_score = score
        
    def get_N_observed(self, query=None, N=10000, fit_trap=False,
                       regr_trap=True, new=False, new_orbits=True,
                       verbose=False):
        df = pd.DataFrame()
        while len(df) < N:
            df = pd.concat([df, self.observe(query=query, new=new,
                                             new_orbits=new_orbits,
                                             fit_trap=fit_trap,
                                             regr_trap=regr_trap)])
            if verbose:
                print(len(df))
        return df

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
                    plot=False, **kwargs):
        """
        N is minimum number of simulated transits to train with.
        """
        df = self.get_N_observed(query=query, N=N, fit_trap=True, regr_trap=False)
   
        X = self._get_trap_features(df)
        
        # Check for and remove infs/nans
        pri = ~np.isnan(df.trap_depth_pri.values)
        sec = ~np.isnan(df.trap_depth_sec.values)
        y1 = np.log10(np.concatenate((df.trap_depth_pri.values[pri],
                                  df.trap_depth_sec.values[sec])))
        y2 = np.concatenate((df.trap_dur_pri.values[pri],
                            df.trap_dur_sec.values[sec]))
        y3 = np.concatenate((df.trap_dur_pri.values[pri],
                            df.trap_dur_sec.values[sec]))
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
        print(('Depth trained: R2={}'.format(score)))
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
        print(('Duration trained: R2={}'.format(score)))
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
        print(('Slope trained: R2={}'.format(score)))
        self._slope_pipeline = pipeline
        self._slope_score = score

        

class KeplerBinaryPopulation(BinaryPopulation):
    #  Don't use KIC radius here; recalc for consistency.
    prop_columns = {'mass_A':'mass'}


class BG_BinaryPopulation(BinaryPopulation):
    """
    Assumes that 'mass_A', 'radius_A' are assigned appropriately.

    Also assumes that properties `kepmag_target`, `kepmag_A`
    have been defined.

    """
    def _generate_orbits(self, **kwargs):
        # First, proceed as before...
        super(BG_BinaryPopulation, self)._generate_orbits(**kwargs)
        
        # ...then, dilute the depths appropriately.
        F_target = 10**(-0.4*self.stars.kepmag_target)
        F_A = 10**(-0.4*self.stars.kepmag_A)
        F_B = self.stars.flux_ratio*F_A
        frac = (F_A + F_B)/(F_A + F_B + F_target)
        self.d_pri *= frac
        self.d_sec *= frac
        
