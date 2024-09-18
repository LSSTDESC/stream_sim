#!/usr/bin/env python
"""
Models for simulating streams.
"""

import copy
import warnings

import numpy as np
import pandas as pd

from stream_sim.functions import function_factory
from stream_sim.samplers import sampler_factory


class ConfigurableModel(object):
    """ Baseclass for models built from configs. """

    def __init__(self, config, **kwargs):
        self._config = copy.deepcopy(config)
        self._config.update(**kwargs)
        self._create_model()

    def _create_model(self):
        pass

    def sample(self, size):
        pass


class StreamModel(ConfigurableModel):
    """ High-level object for the various components of the stream model. """

    def __init__(self, config, **kwargs):
        """ Create the stream from the config object.

        Parameters
        ----------
        config : configuration dictionary

        Returns
        -------
        self : stream model
        """
        super().__init__(config, **kwargs)

    def _create_model(self):
        self.density = self._create_density()
        self.track = self._create_track()
        self.distance_modulus = self._create_distance_modulus()
        self.isochrone = self._create_isochrone()
        self.velocity = self._create_velocity()

    def _create_density(self):
        config = self._config.get('density')
        return DensityModel(config)

    def _create_linear_density(self):
        """to be used with the cubic spline methods"""
        config = self._config.get('linear_density')
        return DensityModel(config)

    def _create_track(self):
        config = self._config.get('track')
        return TrackModel(config)

    def _create_distance_modulus(self):
        config = self._config.get('distance_modulus')
        if config:
            return TrackModel(config)
        else:
            return None

    def _create_isochrone(self):
        config = self._config.get('isochrone')
        if config:
            iso = IsochroneModel(config)
            iso.create_isochrone(config)
            return iso
        else:
            return None

    def _create_velocity(self):
        config = self._config.get('velocity')
        if config:
            return VelocityModel(config)
        else:
            return None

    def sample(self, size):
        """
        Sample the stream stellar distribution parameters.

        Parameters
        ----------
        size : number of stars to generate.

        Returns
        -------
        df : data frame of stream stars.
        """

        # Sample phi1 and phi2
        phi1 = self.density.sample(size)
        phi2 = self.track.sample(phi1)

        # Sample distances
        if self.distance_modulus:
            dist = self.distance_modulus.sample(phi1)
        else:
            dist = None

        # Sample magnitudes from isochrone
        if self.isochrone:
            
            mag_g, mag_r = self.isochrone.sample(size,dist)
        else:
            mag_g, mag_r = None, None

        # Sample kinematics
        if self.velocity:
            mu1, mu2, rv = self.velocity.sample(phi1)
        else:
            mu1, mu2, rv = None, None, None

        # Create the DataFrame of stream stars
        df = pd.DataFrame({'phi1': phi1, 'phi2': phi2, 'dist': dist,
                           'mu1': mu1, 'mu2': mu2, 'rv': rv,
                           'mag_g': mag_g, 'mag_r': mag_r})
        return df


class DensityModel(ConfigurableModel):

    def _create_model(self):
        kwargs = copy.deepcopy(self._config)
        type_ = kwargs.pop('type').lower()
        self.density = sampler_factory(type_, **kwargs)

    def sample(self, size):
        return self.density.sample(size)


class TrackModel(ConfigurableModel):

    def _create_model(self):
        kwargs = copy.deepcopy(self._config['center'])
        type_ = kwargs.pop('type').lower()
        self.center = function_factory(type_, **kwargs)

        kwargs = copy.deepcopy(self._config['spread'])
        type_ = kwargs.pop('type').lower()
        self.spread = function_factory(type_, **kwargs)

    def _create_sampler(self, x):
        type_ = self._config.get('sampler', 'Gaussian').lower()
        if type_ == 'gaussian':
            mu = self.center(x)
            sigma = self.spread(x)
            kwargs = dict(mu=mu, sigma=sigma)
        elif type_ == 'uniform':
            xmin = self.center(x) - self.spread(x)
            xmax = xmin + 2*self.spread(x)
            kwargs = dict(xmin=xmin, xmax=xmax)
        else:
            raise Exception(f"Unrecognized sampler: {type_}")

        self._sampler = sampler_factory(type_, **kwargs)

    def sample(self, x):
        size = len(x)
        self._create_sampler(x)
        return self._sampler.sample(size)

    
    
    
    

class DistanceModel(ConfigurableModel):
    pass
        
        

class IsochroneModel(ConfigurableModel):
    """ Placeholder for isochrone model. """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
    def create_isochrone(self,config):
        import ugali.isochrone
        self.iso = ugali.isochrone.factory(**config)
        self.iso.params['distance_modulus'].set_bounds([0,50])
        if 'distance_modulus' in config:
            warnings.warn('Please use the "distance_modulus" section of the configuration file, instead of the isochrone section, to define a distance module.')
        self.iso.distance_modulus = 0
        
        
    def sample(self, nstars,distance_modulus,**kwargs):
        stellar_mass = nstars * self.iso.stellar_mass()
        if np.isscalar(distance_modulus):
            mag_g,mag_r = self.iso.simulate(stellar_mass,distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + np.ones_like(mag)*distance_modulus  for mag in (mag_g, mag_r)]
        else:
            mag_g,mag_r = self.iso.simulate(stellar_mass,distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + distance_modulus  for mag in (mag_g, mag_r)]

        return mag_g, mag_r
    
    def _dist_to_modulus(self,distance):
        """
        Convert physical distances in pc into distance modulus
        """
        if distance is None:
            return 0
        elif np.all(distance == 0):
            warnings.warn("Distances are equal to 0, distance modulus has been set to 0.")
            return 0
        else:
            return 5*np.log10(distance)-5
        

class VelocityModel(ConfigurableModel):
    """ Placeholder for velocity model. """

    def sample(self, phi1):
        """ Placeholder """
        warnings.warn("VelocityModel not implemented!")
        
        if np.isscalar(phi1):
            mu1, mu2, rv = np.nan, np.nan, np.nan
        else:
            mu1, mu2, rv = np.nan*np.ones_like([phi1, phi1, phi1])
        
        return mu1, mu2, rv


class BackgroundModel(StreamModel):
    """ Background model. """
    pass


class SplineStreamModel(StreamModel):
    def __init__(self, config, **kwargs):
        """ Create the stream from the config object.

        Parameters
        ----------
        config : configuration dictionary

        Returns
        -------
        self : stream model
        """

        stream_name = None
        if config["stream_name"]:
            stream_name = config["stream_name"]

        config["linear_density"]["stream_name"] = stream_name
        config["track"]["center"]["stream_name"] = stream_name
        config["track"]["spread"]["stream_name"] = stream_name
        super().__init__(config, **kwargs)

    def _create_model(self):
        self.density = self._create_linear_density()
        self.track = self._create_track()
        self.distance_modulus = self._create_distance()
        self.isochrone = self._create_isochrone()
        self.velocity = self._create_velocity()
