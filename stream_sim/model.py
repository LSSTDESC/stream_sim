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
        if self._config is not None:
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

    def complete_catalog(self, catalog, columns_to_add=None, size=None, inplace=False, save_path=None, verbose=True):
        """Complete only the requested columns in a catalog.

        Notes
        - If catalog is None or empty (size required), a new empty frame of that size is created.
        - Existing values are preserved; only missing or absent columns are filled.
        - Dependencies: phi2/dist/velocities require phi1; magnitudes require dist and an isochrone model.
        """
        # Supported outputs and capability filtering
        all_cols = ('phi1', 'phi2', 'dist', 'mag_g', 'mag_r', 'mu1', 'mu2', 'rv') # columns this method can fill
        target_cols = list(all_cols) if columns_to_add is None else [c for c in columns_to_add if c in all_cols]
        unknown = [] if columns_to_add is None else sorted(set(columns_to_add) - set(all_cols))
        if unknown:
            warnings.warn(f"Ignoring unknown columns: {unknown}")

        if self.isochrone is None:
            removed = [c for c in target_cols if c in ('mag_g', 'mag_r')]
            target_cols = [c for c in target_cols if c not in ('mag_g', 'mag_r')]
            if removed:
                self._info(verbose, "Isochrone model not defined; skipping magnitudes.")
        if self.velocity is None:
            removed = [c for c in target_cols if c in ('mu1', 'mu2', 'rv')]
            target_cols = [c for c in target_cols if c not in ('mu1', 'mu2', 'rv')]
            if removed:
                self._info(verbose, "Velocity model not defined; skipping velocities.")
        if self.distance_modulus is None:
            removed = [c for c in target_cols if c == 'dist']
            target_cols = [c for c in target_cols if c != 'dist']
            if removed:
                self._info(verbose, "Distance modulus model not defined; skipping distances.")

        # Load/normalize input catalog
        df, src_path = self._open_catalog(catalog, size=size, inplace=inplace)
        N = len(df)

        # phi1
        if 'phi1' in target_cols:
            idx = self._missing_idx(df, 'phi1')
            if len(idx) > 0:
                if self.density is None:
                    raise ValueError("Density model is required to sample phi1")
                df.loc[idx, 'phi1'] = self.density.sample(len(idx))
                self._info(verbose, f"Filled {len(idx)} phi1 values.")

        # phi2 (needs phi1)
        if 'phi2' in target_cols:
            if 'phi1' not in df.columns or df['phi1'].isna().any():
                raise ValueError("phi1 required to sample phi2; include 'phi1' in columns_to_add or provide it in catalog")
            idx = self._missing_idx(df, 'phi2')
            if len(idx) > 0:
                if self.track is None:
                    raise ValueError("Track model is required to sample phi2")
                df.loc[idx, 'phi2'] = self.track.sample(df.loc[idx, 'phi1'].to_numpy())
                self._info(verbose, f"Filled {len(idx)} phi2 values.")

        # dist (needs phi1)
        if 'dist' in target_cols:
            if 'phi1' not in df.columns or df['phi1'].isna().any():
                raise ValueError("phi1 required to sample dist; include 'phi1' in columns_to_add or provide it in catalog")
            idx = self._missing_idx(df, 'dist')
            if len(idx) > 0:
                df.loc[idx, 'dist'] = self.distance_modulus.sample(df.loc[idx, 'phi1'].to_numpy())
                self._info(verbose, f"Filled {len(idx)} dist values.")


        if any(c in target_cols for c in ('mag_g', 'mag_r')):
            # Verify distance modulus availability
            if not 'dist' in df.columns:
                raise ValueError("dist is required to sample apparent magnitudes; include 'dist' in `columns_to_add` or provide it in catalog")
            dist_vals = df['dist'].to_numpy()
            mag_g, mag_r = self.isochrone.sample(N,dist_vals)
            # Add both magnitudes if missing to keep colors consistent
            if 'mag_g' in df.columns or 'mag_r' in df.columns:
                self._info(verbose, "Overwriting existing mag_g and/or mag_r to keep colors consistent.")    
            df['mag_g'] = mag_g
            df['mag_r'] = mag_r
            self._info(verbose, f"Filled magnitudes for {N} rows.")

        # velocities require velocity model and phi1
        if any(c in target_cols for c in ('mu1', 'mu2', 'rv')):
            if 'phi1' not in df.columns or df['phi1'].isna().any():
                raise ValueError("phi1 required to sample velocities")
            mu1, mu2, rv = self.velocity.sample(df['phi1'].to_numpy())
            if 'rv' in df.columns or 'mu1' in df.columns or 'mu2' in df.columns:
                self._info(verbose, "Overwriting existing velocity components to keep consistency.")
            df['mu1'] = mu1
            df['mu2'] = mu2
            df['rv'] = rv
            self._info(verbose, f"Filled velocities for {N} rows.")

        if save_path is not None:
            df.to_csv(save_path, index=False)
            self._info(verbose, f"Saved completed catalog to {save_path}.")
        elif isinstance(catalog, str) and inplace:
            df.to_csv(src_path, index=False)
            self._info(verbose, f"Overwrote original catalog at {src_path}.")

        return df

    def _missing_idx(self, df: pd.DataFrame, col: str):
        """Return index of rows missing column `col` (or all rows if absent)."""
        if col not in df.columns:
            return df.index
        return df.index[df[col].isna()]

    def _info(self, verbose: bool, msg: str):
        """Conditional verbose print."""
        if verbose:
            print(msg)

    def _open_catalog(self, catalog, size=None, inplace=False):
        # Load/normalize input catalog
        src_path = None
        if catalog is None:
            if size is None:
                raise ValueError("size must be provided when catalog is None")
            df = pd.DataFrame(index=np.arange(int(size)))
        elif isinstance(catalog, str):
            src_path = catalog
            df = pd.read_csv(catalog)
        elif isinstance(catalog, pd.DataFrame):
            df = catalog if inplace else catalog.copy()
        elif isinstance(catalog, dict):
            df = pd.DataFrame(catalog)
        else:
            raise TypeError("catalog must be None, path, DataFrame, or dict")

        if len(df) == 0:
            if size is None:
                raise ValueError("Empty catalog; provide size")
            df = pd.DataFrame(index=np.arange(int(size)))
        
        return df, src_path

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
