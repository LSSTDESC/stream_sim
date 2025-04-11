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


# class IsochroneModel(ConfigurableModel):
#     """ Placeholder for isochrone model. """
#     def __init__(self, config, **kwargs):
#         super().__init__(config, **kwargs)
#     # 
#     def create_isochrone(self,config):
#         import ugali.isochrone
#         self.iso = ugali.isochrone.factory(**config)
#         self.iso.params['distance_modulus'].set_bounds([0,50])
#         if 'distance_modulus' in config:
#             warnings.warn('Please use the "distance_modulus" section of the configuration file, instead of the isochrone section, to define a distance module.')
#         self.iso.distance_modulus = 0
#         # 
#         # 
#     def sample(self, nstars,distance_modulus,**kwargs):
#         stellar_mass = nstars * self.iso.stellar_mass()
#         if np.isscalar(distance_modulus):
#             mag_g,mag_r = self.iso.simulate(stellar_mass,distance_modulus=self.iso.distance_modulus)
#             mag_g, mag_r = [mag + np.ones_like(mag)*distance_modulus  for mag in (mag_g, mag_r)]
#         else:
#             mag_g,mag_r = self.iso.simulate(stellar_mass,distance_modulus=self.iso.distance_modulus)
#             mag_g, mag_r = [mag + distance_modulus  for mag in (mag_g, mag_r)]
# # 
#         return mag_g, mag_r
#     # 
#     def _dist_to_modulus(self,distance):
#         """
#         Convert physical distances in pc into distance modulus
#         """
#         if distance is None:
#             return 0
#         elif np.all(distance == 0):
#             warnings.warn("Distances are equal to 0, distance modulus has been set to 0.")
#             return 0
#         else:
#             return 5*np.log10(distance)-5