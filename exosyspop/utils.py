from __future__ import print_function, division
import numpy as np
from astropy import constants as const
from numba import jit

G = const.G.cgs.value
MSUN = const.M_sun.cgs.value
RSUN = const.R_sun.cgs.value
AU = const.au.cgs.value
DAY = 86400.
REARTH = const.R_earth.cgs.value
MEARTH = const.M_earth.cgs.value

def draw_powerlaw(alpha, rng, N=1):
    """
    Returns random variate according to x^alpha, between rng[0] and rng[1]
    """
    if alpha == -1:
        alpha = -1.0000001
    # Normalization factor
    x0, x1 = rng
    x0_alpha_p_1 = x0**(alpha + 1)
    C = (alpha + 1) / (x1**(alpha + 1) - x0_alpha_p_1)
    
    if N==1:
        u = np.random.random()
    else:
        u = np.random.random(N)
    x = ((u * (alpha + 1)) / C + x0_alpha_p_1)**(1./(alpha + 1))

    return x

def semimajor(P,mstar=1):
    """Returns semimajor axis in AU given P in days, mstar in solar masses.
    """
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU

def rochelobe(q):
    """returns r1/a; q = M1/M2"""
    return 0.49*q**(2./3)/(0.6*q**(2./3) + np.log(1+q**(1./3)))

def withinroche(semimajors,M1,R1,M2,R2,N=1):
    """
    Returns boolean array that is True where two stars are within Roche lobe
    """
    q = M1/M2
    return ((R1+R2)*RSUN) > (rochelobe(q)*semimajors*AU)

def Pbg_kepler(Kp, b, r=4):
    """Expected number of BG stars within r" in Kepler field within (Kp, Kp + 10)
    
    """
    if Kp < 11:
        Kp = 11
    if Kp > 16:
        Kp = 16
    pA = [-2.5038e-3, 0.12912, -2.4273, 19.980, -60.931]
    pB = [3.0668e-3, -0.15902, 3.0365, -25.320, 82.605]
    pC = [-1.5465e-5, 7.5396e-4, -1.2836e-2, 9.6434e-2, -0.27166]
    A = np.polyval(pA, Kp)
    B = np.polyval(pB, Kp)
    C = np.polyval(pC, Kp)
    print(A,B,C)
    return (r/2)**2*(np.polyval(pC, Kp) + 
                     np.polyval(pA, Kp)*np.exp(-b/np.polyval(pB, Kp)))
