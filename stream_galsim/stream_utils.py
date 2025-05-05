'''
Set of simulation tools for stellar stream generation. Include:
 - mock stream generation with gala 
 - perturbed stream generation with galpy
 - N-body simulated stellar stream with gala.Nbody() class
 - gri magnitude sampling
 - velocity dispersion for given potential
 - other
'''

import astropy.units as u
import astropy.constants as ac
import numpy as np


def Plummer_sigv(M, b, r): #returns sigv for a plummer potential
    return np.sqrt(ac.G * M.to(u.kg) / 6 * (r.to(u.m)**2+b.to(u.m)**2)**(-1/2)).to(u.km/u.s)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def normalize(lst):
    return [i/sum(lst) for i in lst]

def galcenframe_to_curvilignephi12(prog_orbit, x, y, z, t_disrupt):
    """
    Convert a position in the galcen frame into (Phi1,Phi2) stream frame. (0,0) define the current position of the progenitor.
    Along the progenitor orbit arc where the stream follow, Phi2 = 0
    - prog_orbit:  progenitor orbit instance as ["ra","dec","distance","pm_ra_cosdec","pm_dec","radial_velocity"], 
    to calculate an orbital arc from which it will be define a great-circle
    - x,y,z: coordinate of the star in galcen frame. Can be array 3D [(x,y,z),...]
    - 
    """
    return #todo

