'''
Set of simulation tools to ease stellar stream generation. Include:
 - mock stream generation with galpy
 - coordinate conversion
 - gri magnitude sampling
 - velocity dispersion for given potential
 - other
'''

import astropy.units as u
import astropy.constants as aconst
import astropy.coordinates as acoord
from astropy.table import Table
import numpy as np
import galpy.df as gd
import gala.coordinates as gcoord # only for great circle rotation
with acoord.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = acoord.Galactocentric()
import warnings
from ugali.isochrone import factory
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

def Plummer_sigv(M, b, r): #returns sigv for a plummer potential
    return np.sqrt(aconst.G * M.to(u.kg) / 6 * (r.to(u.m)**2+b.to(u.m)**2)**(-1/2)).to(u.km/u.s)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def normalize(lst):
    return [i/sum(lst) for i in lst]

def full_galpy_df(sigv, progenitor, pot, aA, ro, vo, vsun, tdisrupt, 
            timpact, impactb, subhalovel, subhalopot, impact_angle=1, perturbed = False, 
            nTrackIterations=1, deltaAngleTrack=None, nTrackChunks=26, higherorderTrack = False):
    '''
    Using galpy, generate the complete stream df (leading, trailing parts and perturbed part if specified)

    params:
    overall
     - sigv: Radial velocity dispersion of the progenitor (float or astropy quantity)
     - progenitor: progenitor orbit instance (galpy.orbit.Orbit(list) or list thereof)
     - pot: accreting host potential instance (galpy.potential)
     - aA: ActionAngle instance used to convert (x,v) to actions. (galpy.actionAngleIsochroneApprox instance)
     - leading: Choice of modelling the leading part (=True) or the trailing part (=False) of the stream (Bool)
     - nTrackChunks: Number of chunks to divide the progenitor track in (int)
     - deltaAngleTrack: Angle to estimate the stream track over in rad (float), usually computed from ntrachunk
     - nTrackIterations: Number of iterations to perform when establishing the track
     - ro: Distance scale for translation into internal units (float or astropy quantity)
     - vo: Velocity scale for translation into internal units (float or astropy quantity)
     - vsun: Galactocentric rectangular vsun for projection (3D list of float or astropy quantity)
     - tdisrupt: Time since start of disruption in Gyr (float or astropy quantity)

    perturbed
     - impactb: Impact parameter between halo and stream in kpc (float or astropy quantity)
     - subhalovel: 3D Velocity of the subhalo in km/s (list of float or astropy quantity) 
     - timpact: Time of impact in Gyr, between 0 and tdisrupt (float or astropy quantity)
     - impact_angle: Angle offset from progenitor at which the impact occurred in rad (float)
     - subhalopot: Gravitational potential of the subhalo (galpy.potential)
     - higherorderTrack: calculate the track using higher-order terms (int)
    
    output: 
     - s_lead: unperturbed stream df leading part (galpy.streamdf instance)
     - s_trail: unperturbed stream df trailing part (galpy.streamdf instance)
     - s_perturbed: perturbed stream df perturbed part (galpy.streamdf instance)

    '''
    #galpy generate stream lead and trail independently
    s_lead = gd.streamdf(sigv=sigv,
                progenitor=progenitor,
                pot=pot,
                aA=aA,
                leading=True,
                nTrackChunks=26,
                ro = ro,
                vo = vo,
                vsun=vsun,
                tdisrupt=tdisrupt)

    s_trail = gd.streamdf(sigv=sigv,
                progenitor=progenitor,
                pot=pot,
                aA=aA,
                leading=False,
                nTrackChunks=26,
                ro = ro,
                vo = vo,
                vsun=vsun,
                tdisrupt=tdisrupt)
    
    #if specified, return perturbed stream df as well
    if perturbed:
        if impact_angle <0:
            leading = False
        elif impact_angle >= 0:
            leading = True

        s_perturbed = gd.streamgapdf(sigv=sigv,
                                progenitor=progenitor,
                                pot = pot,
                                aA = aA,
                                tdisrupt = tdisrupt,
                                leading = True,
                                #deltaAngleTrack=3,
                                nTrackChunks = 26,
                                # nTrackIterations = 1,
                                vsun=vsun,
                                ro = ro,
                                vo = vo,
                                impactb = impactb,
                                subhalovel = subhalovel,
                                timpact = timpact,
                                impact_angle = impact_angle,
                                subhalopot = subhalopot,
                                higherorderTrack = higherorderTrack,
                                )
    else:
        s_perturbed = None
    
    return s_lead, s_trail, s_perturbed


def full_galpy_sampling(n_stars, s_lead, s_trail, s_perturbed, ro=8.122, vo=245.6, perturbed = 'not', rdseed = None):
    '''
    Using galpy, sample the complete stream from df (leading, trailing parts and perturbed part if specified) in galactocentric xyz,
    since galpy generates stream arms separately.
    params: 
     - n_stars: number of stars to generate along each arm (int)
     - s_lead: unperturbed stream df leading part (galpy.streamdf instance)
     - s_trail: unperturbed stream df trailing part (galpy.streamdf instance)
     - s_perturbed: perturbed stream df perturbed part (galpy.streamdf instance)
     - ro: Distance scale for translation into internal units (float or astropy quantity)
     - vo: Velocity scale for translation into internal units (float or astropy quantity)
     - perturbed: sample along s_perturbed (Bool)
     - rdseed: so the unperturbed and perturbed stream sample from the same random seed (int)
    '''

    if type(rdseed) == int:
        np.random.seed(rdseed)
    s_lead_stars = s_lead.sample(n_stars, xy=True)
    if type(rdseed) == int:
        np.random.seed(rdseed)
    s_trail_stars = s_trail.sample(n_stars, xy=True)
    s_lead_stars[:3] *= ro
    s_lead_stars[3:] *= vo
    s_trail_stars[:3] *= ro
    s_lead_stars[3:] *= vo


    s_full_stars = np.concatenate((np.array(s_lead_stars).T, np.array(s_trail_stars).T)).T #full unperturbed stream

    if perturbed == 'lead':
        if type(rdseed) == int:
            np.random.seed(rdseed)
        ps_stars = s_perturbed.sample(n_stars, xy=True)
        ps_stars[:3] *= ro
        ps_stars[3:] *= vo
        ps_full_stars = np.concatenate((np.array(ps_stars).T, np.array(s_trail_stars).T)).T #full perturbed stream

    elif perturbed == 'trail':
        if type(rdseed) == int:
            np.random.seed(rdseed)
        ps_stars = s_perturbed.sample(n_stars, xy=True)
        ps_stars[:3] *= ro
        ps_stars[3:] *= vo
        ps_full_stars = np.concatenate((np.array(s_lead_stars).T, np.array(ps_stars).T)).T #full perturbed stream

    else:
        ps_stars = None
        ps_full_stars = None
    
    return s_lead_stars, s_trail_stars, s_full_stars, ps_stars, ps_full_stars


def xyz_to_icrs(stream_stars, velocities = False):
    '''
    convert star position and velocities from galactocentrique rectangular to icrs or proper stream frame

    params:
    stream_stars: star phase-position from galpy in (x,y,z,vx,vy,vz)
    frame: to convert stars phase-space position in (ra, dec, d_mod, pm_ra, pm_dec, pm_r)
    velocities: convert velocities into wanted coordinate frame as well
    '''

    x, y, z = np.array(stream_stars[:3])* u.kpc
    x = -x #galpy use lefthanded frame, we use righthanded x-> -x

    if velocities == True:
        vx, vy, vz = np.array(stream_stars[3:])* u.km/u.s
        vx = -vx
        with acoord.galactocentric_frame_defaults.set("v4.0"):
            stream_stars = acoord.Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
        stream_stars_icrs = stream_stars.transform_to(acoord.ICRS())
        # stream_stars_icrs = stream_stars_icrs.T.ra, stream_stars_icrs.T.dec, stream_stars_icrs.T.distance, stream_stars_icrs.T.pm_ra_cosdec, stream_stars_icrs.T.pm_dec, stream_stars_icrs.T.radial_velocity
        #ra,dec,distance,pmra,pmdec,pmradial
    elif velocities == False:
        with acoord.galactocentric_frame_defaults.set("v4.0"):
            stream_stars = acoord.Galactocentric(x=x, y=y, z=z)
        stream_stars_icrs = stream_stars.transform_to(acoord.ICRS())
        # stream_stars_icrs = stream_stars_icrs.T.ra, stream_stars_icrs.T.dec, stream_stars_icrs.T.distance #ra,dec,distance
        
    return stream_stars_icrs #ra,dec,dist



def icrs_to_phi12(stream_stars, pole1, pole2, velocities = False):
    '''
    convert star position and velocities from icrs to proper stream frame

    params:
    stream_stars: star phase-position in (ra, dec, d_mod, pm_ra, pm_dec, pm_r)
    frame: to convert stars phase-space position in (phi1, phi2, d_mod, pm_phi1, pm_phi2, pm_r)
    velocities: convert velocities into wanted coordinate frame as well
    '''

    stream_frame = gcoord.GreatCircleICRSFrame.from_endpoints(pole1, pole2, origin=pole1)
    ra_s, dec_s, dist_s = stream_stars.ra, stream_stars.dec, stream_stars.distance

    if velocities == True:
        pm_ra_s, pm_dec_s, pm_r = stream_stars.pm_ra_cosdec, stream_stars.pm_dec, stream_stars.radial_velocity
        stream_icrs = acoord.SkyCoord(ra=ra_s, dec=dec_s, distance=dist_s, pm_ra_cosdec=pm_ra_s, pm_dec=pm_dec_s, radial_velocity=pm_r)
        stream_phi12 = stream_icrs.transform_to(stream_frame)
        # stream_phi12 = stream_phi12.phi1, stream_phi12.phi2, stream_phi12.distance, stream_phi12.pm_phi1_cosphi2, stream_phi12.pm_phi2, stream_phi12.radial_velocity
        
    elif velocities == False:
        stream_icrs = acoord.SkyCoord(ra=ra_s, dec=dec_s, distance=dist_s)
        stream_phi12 = stream_icrs.transform_to(stream_frame)
        # stream_phi12 = stream_phi12.phi1, stream_phi12.phi2, stream_phi12.distance
    return stream_phi12 #Phi1, Phi2, dist


def save_star_data(star_list, mag_g, mag_r, coord_system, filepath):
    """Save RA, Dec, Distance, mag_g, mag_r to a CSV file in 'data/' directory."""
    
    # Create Astropy table
    if coord_system == 'radec':
        table = Table(
            data=[
                star_list.ra.deg,         # RA in degrees
                star_list.dec.deg,        # Dec in degrees
                star_list.distance.kpc,   # Distance in kpc
                mag_g,                    # g-band magnitude
                mag_r                     # r-band magnitude
            ],
            names=["ra", "dec", "dist", "mag_g", "mag_r"]
        )
    elif coord_system == 'phi12':
        table = Table(
            data=[
                star_list.Phi1.deg,       # Phi1 in degrees
                star_list.Phi2.deg,       # Phi2 in degrees
                star_list.distance.kpc,   # Distance in kpc
                mag_g,                    # g-band magnitude
                mag_r                     # r-band magnitude
            ],
            names=["phi1", "phi2", "dist", "mag_g", "mag_r"]
        )
    else:
        raise NotImplementedError('use phi12 or radec')

    # Save to CSV
    table.write(filepath, format="csv", overwrite=True)
    print(f"Saved: {filepath}")



class IsochroneModel:
    """ Isochrone model for assigning magnitudes to stars. """
    
    def __init__(self, config):
        self.config = config
        self.create_isochrone(config)
    
    def create_isochrone(self, config):
        """Create the isochrone model from the configuration."""
        self.iso = factory(**config)
        self.iso.params['distance_modulus'].set_bounds([0, 50])

        if 'distance_modulus' in config:
            warnings.warn('Use the "distance_modulus" section of the config file instead.')
        self.iso.distance_modulus = 0  # Default value
        
    def _dist_to_modulus(self, distance):
        """Convert physical distances (kpc) into distance modulus."""
        if distance is None:
            return 0
        elif np.all(distance == 0):
            warnings.warn("Distances are 0; setting distance modulus to 0.")
            return 0
        else:
            return 5 * np.log10(distance * 1000) - 5  # Convert kpc to pc

    def sample_magnitudes(self, icrs_list):
        """Assign magnitudes (g, r) to stars based on their distance."""
        distances = icrs_list.distance.to(u.kpc).value  # Extract distances in kpc
        distance_moduli = self._dist_to_modulus(distances)  # Compute distance modulus

        # Sample magnitudes for each star
        mag_g, mag_r = self.sample(len(distances), distance_moduli)

        return mag_g, mag_r

    def sample(self, nstars, distance_modulus):
        """Sample magnitudes using the isochrone."""
        stellar_mass = nstars * self.iso.stellar_mass()

        if np.isscalar(distance_modulus):
            mag_g, mag_r = self.iso.simulate(stellar_mass, distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + distance_modulus for mag in (mag_g, mag_r)]
        else:
            mag_g, mag_r = self.iso.simulate(stellar_mass, distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + distance_modulus for mag in (mag_g, mag_r)]

        return mag_g, mag_r



class StreamInterpolateTrackDensity:
    def __init__(self, stars: acoord.SkyCoord):
        """
        Initialize the stream density analyzer.
        Parameters
        ----------
        stars : SkyCoord
            Coordinate object containing .phi1 and .phi2 (in a stream-aligned frame).
        """
        self.stars = stars
        self.phi1 = self.stars.phi1.wrap_at(180 * u.deg).deg
        self.phi2 = self.stars.phi2.deg
        self.density_table = None

    @staticmethod
    def _gaussian(x, A, mu, sigma):
        """Standard Gaussian function."""
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    def compute_density(self, delta_phi1=0.02, phi2_bins=50, max_fev=10000):
        """
        Compute the stream density track as a function of phi1.

        This function slides along phi1, bins stars in phi2, and fits a Gaussian
        to the phi2 distribution at each phi1 slice.

        Parameters
        ----------
        delta_phi1 : float
            Step size in phi1 for scanning the stream.
        phi1_width : float
            Half-width around each phi1 value to select stars.
        max_fev : int
            Maximum number of function evaluations in the optimizer.

        Returns
        -------
        density_table : astropy.table.Table
            Table containing phi1 positions, fitted phi2 center, amplitude, and width.
        """
        phi1_min = np.min(self.phi1) + delta_phi1
        phi1_max = np.max(self.phi1) - delta_phi1
        phi1_vals, phi2_vals, nstars_vals, width_vals = [], [], [], []

        for phi1_t in np.arange(phi1_min, phi1_max, delta_phi1):
            # Select stars within a small window around current phi1 value
            mask = np.abs(self.phi1 - phi1_t) < delta_phi1
            phi2_sel = self.phi2[mask]
            try:
                phi2_min = np.min(phi2_sel)
                phi2_max = np.max(phi2_sel)
            except ValueError:
                continue
            if len(phi2_sel) > 10:
                # Build histogram of phi2 values
                hist, bin_edges = np.histogram(phi2_sel, bins=phi2_bins)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # Initial guess for the fit
                A0 = np.max(hist)
                mu0 = bin_centers[int(len(bin_centers)/2)]
                sigma0 = 1.0
                p0 = [A0, mu0, sigma0]

                try:
                    popt, pcov = curve_fit(
                        self._gaussian,
                        bin_centers,
                        hist,
                        p0=p0,
                        bounds=([0, phi2_min, 1e-4], [1*A0, phi2_max, np.inf]),
                        maxfev=max_fev
                    )
                    A, mu, sigma = popt

                    if A < 1000:  # Simple threshold to reject junk fits
                        phi1_vals.append(phi1_t)
                        phi2_vals.append(mu)
                        nstars_vals.append(A)
                        width_vals.append(sigma)
                except RuntimeError:
                    # Fit did not converge
                    continue

        self.density_table = Table(
            [phi1_vals, phi2_vals, nstars_vals, width_vals],
            names=("phi1", "phi2", "nstars", "width")
        )
        return self.density_table
    
    def plot_phi2_slice(self, phi1_target, phi1_window=0.02, bins=30, normalized=True):
        """
        Plot the histogram of phi2 in a phi1 slice and the corresponding fitted Gaussian.
        Parameters
        ----------
        phi1_target : float
            Central phi1 value of the slice (in degrees).
        phi1_window : float
            Half-width of the slice in phi1 (default: 0.2 deg).
        bins : int
            Number of bins in the histogram.
        normalized : bool
            Whether to normalize the histogram and the Gaussian.
        """
        if self.density_table is None:
            raise RuntimeError("You need to run compute_density() before plotting.")

        # Find the closest fitted phi1
        idx = np.argmin(np.abs(self.density_table["phi1"] - phi1_target))
        phi1_fit = self.density_table["phi1"][idx]
        mu_fit = self.density_table["phi2"][idx]
        A_fit = self.density_table["nstars"][idx]
        sigma_fit = self.density_table["width"][idx]

        # Select stars in the phi1 window
        mask = np.abs(self.phi1 - phi1_fit) < phi1_window
        phi2_sel = self.phi2[mask]

        if len(phi2_sel) < 5:
            print("Not enough stars in selected slice.")
            return

        # Histogram
        phi2_min, phi2_max = np.min(phi2_sel), np.max(phi2_sel)
        hist, edges = np.histogram(phi2_sel, bins=bins, range=(phi2_min, phi2_max))
        bin_centers = 0.5 * (edges[1:] + edges[:-1])

        # Gaussian model
        phi2_fine = np.linspace(phi2_min, phi2_max, 500)
        gauss = self._gaussian(phi2_fine, A_fit, mu_fit, sigma_fit)

        if normalized:
            # Normalize both to unit area
            hist_norm = hist / trapezoid(hist, bin_centers)
            gauss_norm = gauss / trapezoid(gauss, phi2_fine)
        else:
            hist_norm = hist
            gauss_norm = gauss

        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist_norm, width=(edges[1] - edges[0]), alpha=0.5, label="Histogram")
        plt.plot(phi2_fine, gauss_norm, 'r-', linewidth=2, label="Fitted Gaussian")
        plt.axvline(mu_fit, color='k', linestyle='--', label="Gaussian Mean")
        plt.title(f"Phi1 ≈ {phi1_fit:.2f} deg")
        plt.xlabel("Phi2 [deg]")
        plt.ylabel("Normalized Star Count" if normalized else "Star Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()