#!/usr/bin/env python
"""
Probabilistic samplers.
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import scipy.stats

from stream_sim.functions import (Interpolation,
                                  FileInterpolation,
                                  Sinusoid,
                                  CubicSplineInterpolation,
                                  FileCubicSplineInterpolation,
                                  LinearDensityCubicSplineInterpolation,
                                  FileLinearDensityCubicSplineInterpolation,
)

def sampler_factory(type_, **kwargs):
    """ Create a sampler with given kwargs.

    Parameters
    ----------
    type_ : sampler type
    kwargs: passed to sampler init

    Returns:
    -------
    sampler : the sampler
    """
    if type_ == 'uniform':
        sampler = UniformSampler(**kwargs)
    elif type_ == 'gaussian':
        sampler = GaussianSampler(**kwargs)
    elif type_ == 'sinusoid':
        sampler = SinusoidSampler(**kwargs)
    elif type_ in ('interpolation'):
        sampler = InterpolationSampler(**kwargs)
    elif type_ in ('file','fileinterpolation'):
        sampler = FileInterpolationSampler(**kwargs)
    elif type_ in ('file','FileLinearDensityCubicSplineInterpolation')
        sampler =
    else:
        raise Exception(f"Unrecognized sampler: {type_}")

    return sampler

def inverse_transform_sample(vals, pdf, size):
    """ Perform inverse transform sampling

    Parameters
    ----------
    vals: value at which pdf is measured
    pdf : pdf value
    size : number of stars to sample

    Returns
    -------
    samples : samples of vals
    """

    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    fn = scipy.interpolate.interp1d(cdf, list(range(0, len(cdf))),
                                    bounds_error=False,
                                    fill_value = 0.0)
    x_new = scipy.stats.uniform.rvs(size=np.rint(size).astype(int))
    index = np.rint(fn(x_new)).astype(int)
    return vals[index]

############################################################
# Probabilistic Samplers
############################################################

class Sampler(object):
    def __init__(self, *args, **kwargs):
        pass

    def pdf(self, values):
        return self._pdf(values)

    def sample(self, size):
        pass

class ScipySampler(Sampler):
    def __init__(self, rv):
        self._rv = rv

    def pdf(self, x):
        return self._rv.pdf(x)

    def sample(self, size, random_state=None):
        return self._rv.rvs(size=int(size), random_state=random_state)

class UniformSampler(ScipySampler):
    """ Sample from uniform distribution. """
    def __init__(self, xmin, xmax):
        """ Uniform distribution sampled between xmin and xmax.

        Parameters
        ----------
        xmin : minimum value of sampled range.
        xmax : maximum value of sampled range.

        Returns
        -------
        self
        """
        rv = scipy.stats.uniform(loc=xmin, scale=xmax-xmin)
        super().__init__(rv)

class GaussianSampler(ScipySampler):
    """ Sample from Gaussian. """
    def __init__(self, mu, sigma):
        """ Gaussian distribution described by mean and sigma.

        Parameters
        ----------
        mu : mean of Gaussian
        sigma : sigma of Gaussian

        Returns
        -------
        self
        """
        rv = scipy.stats.norm(loc=mu, scale=sigma)
        super().__init__(rv)

class InterpolationSampler(Sampler):
    """ Sample from interpolated function. """
    def __init__(self, xvals, yvals, **kwargs):
        """ Interpolation sampler.

        Parameters
        ----------
        xvals : x-values of interpolation
        yvals : y-values of interpolation
        kwargs : passed to baseclass

        Returns
        -------
        sampler
        """
        self.interp = Interpolation(xvals, yvals, **kwargs)

    def sample(self, size, nsteps=1e5):
        xvals = np.linspace(self.interp.xmin, self.interp.xmax, int(nsteps))
        pdf = self.interp(xvals)
        return inverse_transform_sample(xvals, pdf, size=size)

class FileInterpolationSampler(InterpolationSampler):

    def __init__(self, filename, columns=None):
        self.interp = FileInterpolation(filename, columns)

class SinusoidSampler(InterpolationSampler):

    def __init__(self, **kwargs):
        """ Sample from Sinusoid function.

        Parameters
        ----------
        Passed to Sinusoid function.

        Returns
        -------
        sampler
        """
        self.interp = Sinusoid(**kwargs)

class CubicSplineInterpolationSampler(InterpolationSampler):
    def __init__(self, filename, columns=None):
        self.interp = CubicSplineInterpolation(filename, columns)

class FileCubicSplineInterpolationSampler(InterpolationSampler):
    def __init__(self, filename, columns=None):
        self.interp = FileCubicSplineInterpolation(filename, columns)

class LinearDensityCubicSplineInterpolationSampler(InterpolationSampler):
    def __init__(self, filename):
        self.interp = LinearDensityCubicSplineInterpolation(filename)

class FileLinearDensityCubicSplineInterpolationSampler(InterpolationSampler):
    def __init__(self, filename):
        self.interp = FileLinearDensityCubicSplineInterpolation(filename)