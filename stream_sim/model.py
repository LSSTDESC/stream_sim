#!/usr/bin/env python
"""
Models for simulating streams.
"""
__author__ = "Alex Drlica-Wagner"

import copy
import warnings

import numpy as np
import pandas as pd

from importlib import reload
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
        self.distance = self._create_distance()
        self.isochrone = self._create_isochrone()
        self.velocity = self._create_velocity()

    def _create_density(self):
        config = self._config.get('density')
        return DensityModel(config)

    def _create_track(self):
        config = self._config.get('track')
        return TrackModel(config)

    def _create_distance(self):
        config = self._config.get('distance')
        if config:
            return DistanceModel()
        else:
            return None

    def _create_isochrone(self):
        config = self._config.get('isochrone')
        if config:
            return IsochroneModel(config)
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
        if self.distance:
            dist = self.distance.sample(phi1)
        else:
            dist = None

        # Sample magnitudes from isochrone
        if self.isochrone:
            mag1, mag2 = self.isochrone.sample(dist)
        else:
            mag1, mag2 = None, None

        # Sample kinematics
        if self.velocity:
            mu1, mu2, rv = self.velocity.sample(phi1)
        else:
            mu1, mu2, rv = None, None, None

        # Create the DataFrame of stream stars
        df = pd.DataFrame({'phi1': phi1, 'phi2': phi2, 'dist': dist,
                           'mu1': mu1, 'mu2': mu2, 'rv': rv,
                           'mag1': mag1, 'mag2': mag2})
        return df


class DensityModel(ConfigurableModel):

    def _create_model(self):
        kwargs = copy.deepcopy(self._config)
        type_ = kwargs.pop('type').lower()
        self.density = sampler_factory(type_, **kwargs)
        import pdb; pdb.set_trace()
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
        type_ = self._config.get('sampler','Gaussian').lower()
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

class DistanceModel(TrackModel): pass


class IsochroneModel(ConfigurableModel):
    """ Placeholder for isochrone model. """

    def sample(self, distance):
        """ Placeholder """
        warnings.warn("IsochroneModel not implemented!")

        if np.isscalar(distance):
            mag1, mag2 = np.nan, np.nan
        else:
            mag1, mag2 = np.nan*np.ones_like([distance, distance])

        return  mag1, mag2

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
        import pdb; pdb.set_trace()
        super().__init__(config, **kwargs)

    def _create_model(self):
        self.density = self._create_density_spline()
        self.track = self._create_track()
        self.distance = self._create_distance()
        self.isochrone = self._create_isochrone()
        self.velocity = self._create_velocity()
    def _create_density_spline(self):
        config = {}
        config["name"] = self._config.get("name")
        config["density"] = self._config.get('density')
        config["spread"] = self._config.get('spread')
        config["type"] = "peak_density"

        return SplineDensityModel(config)

class SplineDensityModel(DensityModel):
    def _create_model(self):
        kwargs = copy.deepcopy(self._config)
        type_ = kwargs.pop('type').lower()
        self.density = sampler_factory(type_, **kwargs)

    def sample(self, size):
        return self.density.sample(size)