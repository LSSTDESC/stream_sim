#!/usr/bin/env python
"""
Plotting functions.
"""

import numpy as np
import pylab as plt
import os
try:
    import healpy as hp
    import skyproj
except ImportError:
    healpy = None
    skyproj = None

def draw_stream(phi1, phi2):
    """ Create 2d histogram and draw stellar distribution.

    Parameters
    ----------
    phi1, phi2 : coordinates of stars (deg)

    Returns:
    --------
    num, xedges, yedges, im : output of imshow
    """

    phi1_bins = np.linspace(phi1.min(), phi1.max(), 100)
    phi2_bins = np.linspace(phi2.min(), phi2.max(), 100)

    hist, xedges, yedges = np.histogram2d(phi1, phi2,
                                           bins=[phi1_bins, phi2_bins],
    )

    ax = plt.gca()
    ax.set_title("Mock Stream")
    ax.set_xlabel("$\phi_1$ [deg]")
    ax.set_ylabel("$\phi_2$ [deg]")

    image = ax.imshow(
        hist.T,
        extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
        aspect="auto",
        vmin=np.percentile(hist, 0.1),
        vmax=np.percentile(hist, 99.9),
        origin="lower",
        cmap="gray_r",
    )
    return hist, xedges, yedges, image

def plot_stream(phi1, phi2):
    """Plot binned histogram of stream.

    Parameters
    ----------
    phi1, phi2 : coordinates of stars (deg)

    Returns
    -------
    fig, ax : the figure and axis
    """
    fig = plt.figure(dpi=175, figsize=(10, 2.5))
    gs = fig.add_gridspec(1, 1, wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.2)

    ax = fig.add_subplot(gs[0, 0])
    hist, xedges, yedges, image = draw_stream(phi1, phi2)
    return fig, ax


def plot_stream_in_mask(ra,dec,mask, nest=False, output_folder = None):
    """Plot stream in mask using healpy and skyproj.

    Parameters
    ----------
    ra, dec : coordinates of stars (deg)
    mask : boolean array, True for masked points

    Returns
    -------
    fig, ax : the figure and axis
    """
    if skyproj is None:
        raise ImportError("healpy or skyproj not installed, cannot plot stream in mask.")
    
    if mask is None:
        mask = np.ones(hp.nside2npix(32), dtype=bool)  # Default mask if none provided

    mask = mask.astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_hpxmap(mask, nest=nest)
    sp.ax.scatter(ra, dec, color='red', s=10, label='Stream', alpha=0.5)
    sp.ax.legend(loc='lower right')
    plt.colorbar()
    fig.tight_layout()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, 'stream_in_mask.png')
        if filename is not None:
            print(f"Writing figure: {filename}...")
            plt.savefig(filename)
        
    return fig, ax