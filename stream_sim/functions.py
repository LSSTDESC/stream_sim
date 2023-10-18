#!/usr/bin/env python
"""
Function definitions.
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import pandas as pd
import scipy.interpolate

def function_factory(type_, **kwargs):
    """ Create a function with given kwargs.

    Parameters
    ----------
    type_ : function type
    kwargs: passed to function init

    Returns:
    -------
    func : the function
    """
    if type_ == 'constant':
        func = Constant(**kwargs)
    elif type_ == 'line':
        func = Line(**kwargs)
    elif type_ == 'sinusoid':
        func = Sinusoid(**kwargs)
    elif type_ == 'interpolation':
        func = Interpolation(**kwargs)
    elif type_ in ('file', 'fileinterpolation'):
        func = FileInterpolation(**kwargs)
    else:
        raise Exception(f"Unrecognized function: {type_}")

    return func

class BoundedFunction(object):
    def __init__(self, xmin=-np.inf, xmax=np.inf):
        """ Function with bounds explicitly defined.

        Parameters
        ----------
        xmin : minimum of x-range
        xmax : maximum of x-range

        Returns
        -------
        function
        """

        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x, **kwargs):
        self.__dict__.update(kwargs)

        xvals = np.atleast_1d(x)
        yvals = self._evaluate(xvals)

        in_range = (xvals >= self.xmin) & (xvals <= self.xmax)
        ret = np.where(in_range, yvals, np.nan)

        if np.isscalar(x):
            return ret.item()
        else:
            return ret

    def _evaluate(self, xvals):
        pass

class Constant(BoundedFunction):

    def __init__(self, value=1.0, **kwargs):
        """ Constant value function.

        Parameters
        ----------
        value: value of the function
        kwargs: passed to baseclass

        Returns
        -------
        function
        """
        super().__init__(**kwargs)
        self.value = value

    def _evaluate(self, xvals):
        """ Evaluate the method """
        yvals = self.value*np.ones(len(xvals))
        return yvals


class Line(BoundedFunction):
    """Line function. Evaluated as:

      y(x) = slope*x + intercept
    """

    def __init__(self, slope=0.0, intercept=0.0, **kwargs):
        """ Line function. Evaluated as:

          y(x) = slope*x + intercept

        Parameters
        ----------
        slope : slope of the line
        intercept : y-intercept as x=0
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        super().__init__(**kwargs)
        self.slope = slope
        self.intercept = intercept

    def _evaluate(self, xvals):
        """ Evaluate the method """
        yvals = self.slope*xvals + self.intercept
        return yvals


class Sinusoid(BoundedFunction):
    """Sinusoid function. Evaluated as:

       y(x) = amplitude/2 * cos( 2pi*(x - phase)/period ) + amplitude/2 + offset

    """
    def __init__(self, amplitude=1.0, period=1.0, phase=0.0, offset=0.0, **kwargs):
        """ Sinusoid function

        Parameters
        ----------
        amplitude : peak to peak amplitude of the sinusoid
        period : period of the sinusoid
        phase  : x-offset of the sinusoid
        offset : y-offset of the sinusoid
        kwargs : passed to baseclass

        Returns
        -------
        instance
        """
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.offset = offset

    def _evaluate(self, xvals):
        """ Evaluate the sinusoid """
        norm = self.amplitude/2.0
        yvals = norm * np.cos(2*np.pi * (xvals - self.phase)/self.period) + norm + self.offset
        return yvals

class Interpolation(BoundedFunction):

    def __init__(self, xvals, yvals, **kwargs):
        """ Interpolation function.

        Parameters
        ----------
        xvals : x-values of interpolation
        yvals : y-values of interpolation
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        self.xvals = np.atleast_1d(xvals)
        self.yvals = np.atleast_1d(yvals)
        kwargs.setdefault('xmin', np.min(xvals))
        kwargs.setdefault('xmax', np.max(xvals))
        super().__init__(**kwargs)

    def _evaluate(self, xvals):
        """ Evaluate the method """
        interp = scipy.interpolate.interp1d(self.xvals, self.yvals,
                                            bounds_error=False,
                                            fill_value = np.nan)

        return interp(xvals)


class FileInterpolation(Interpolation):
    def __init__(self, filename, columns=None, **kwargs):
        """ Interpolation function from file.

        Parameters
        ----------
        filename : input filename
        columns  : list of column names
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        self._filename = filename
        self._data = pd.read_csv(self._filename)

        if columns:
            xname,yname = columns
        else:
            xname,yname = self._data.columns[[0,1]]

        xvals = self._data[xname].values
        yvals = self._data[yname].values

        super().__init__(xvals, yvals, **kwargs)

class CubicSplineInterpolation(BoundedFunction):

    def __init__(self, nodes, node_values, **kwargs):
        """
        Cubic spline interpolation given nodes and node values

        Parameters
        ----------
        nodes : array_like
            1-D array of nodes for the spline.
        node_values : array_like
            1-D array of node values corresponding to the nodes.

        Returns
        -------
        function
        """
        self.nodes = np.atleast_1d(nodes)
        self.node_values = np.atleast_1d(node_values)
        kwargs.setdefault('xmin', np.min(nodes))
        kwargs.setdefault('xmax', np.max(nodes))
        super().__init__(**kwargs)

    def _evaluate(self, xvals):
        """
        Evaluate the cubic spline interpolation at given x-values.
        for xvals outside the interpolation range nan is returned

        Parameters
        ----------
        xvals : array_like
            x-values at which the spline is to be evaluated.

        Returns
        -------
        array_like
            Interpolated values at the given x-values.

        """
        interp = scipy.interpolate.CubicSpline(self.nodes,
                                               self.node_values,
                                               bc_type='natural',
                                               extrapolate=False)
        return interp(xvals)


class FileCubicSplineInterpolation(CubicSplineInterpolation):
    def __init__(self, filename, type=None, stream_name=None,nodes_name="phi1",
                 node_vals_name="mean", **kwargs):
        """ Interpolation function from file.

        Parameters
        ----------
        filename : input filename
        columns  : list of column names
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        self._filename = filename
        self._data = pd.read_csv(self._filename)
        if type:
            self._data = self._data.loc[self._data["type"]==type,:]
        if stream_name:
            self._data = self._data.loc[self._data["stream"]==stream_name,:]

        nodes = self._data[nodes_name].values
        node_values = self._data[node_vals_name].values

        super().__init__(nodes, node_values, **kwargs)


class LinearDensityCubicSplineInterpolation():
    def __init__(self, intensity_nodes, intensity_node_values, spread_nodes,
                 spread_node_values, **kwargs):
        """ Interpolation function from file.

        Parameters
        ----------
        filename : input filename
        columns  : list of column names
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        self._peak_intensity = CubicSplineInterpolation(intensity_nodes,
                                                        intensity_node_values)
        self._spread = CubicSplineInterpolation(spread_nodes,
                                                spread_node_values)

    def __call__(self, x, **kwargs):
        self.__dict__.update(kwargs)

        xvals = np.atleast_1d(x)
        yvals = self._evaluate(xvals)
        return yvals

    def _evaluate(self,xvals):
        peak_intensity = self._peak_intensity(xvals)
        spread = self._spread(xvals)
        return np.sqrt(2 * np.pi) * peak_intensity * spread


class FileLinearDensityCubicSplineInterpolation(
        LinearDensityCubicSplineInterpolation):
    def __init__(self, filename, stream_name=None, **kwargs):
        """ Interpolation function from file.

        Parameters
        ----------
        filename : input filename
        columns  : list of column names
        kwargs : passed to baseclass

        Returns
        -------
        function
        """
        self._filename = filename
        self._data = pd.read_csv(self._filename)
        if stream_name:
            self._data = self._data.loc[self._data["stream"]==stream_name]
        sel_intensity = (self._data["type"] == "peak_intensity")
        intensity_nodes = self._data.loc[sel_intensity, "phi1"].values
        intensity_node_values = self._data.loc[sel_intensity, "mean"].values

        sel_spread = (self._data["type"] == "spread")
        spread_nodes = self._data.loc[sel_spread, "phi1"].values
        spread_node_values = self._data.loc[sel_spread, "mean"].values
        super().__init__(
            intensity_nodes=intensity_nodes,
            intensity_node_values=intensity_node_values,
            spread_nodes=spread_nodes,
            spread_node_values=spread_node_values,
            **kwargs)