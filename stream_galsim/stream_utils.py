'''
Set of simulation tools to ease stellar stream generation. Include:
 - mock stream generation with galpy
 - coordinate conversion
 - gri magnitude sampling
 - velocity dispersion for given potential
 - other
'''

#astropy
import astropy.units as u
import astropy.constants as aconst
import astropy.coordinates as acoord
from astropy.table import Table
#general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ugali.isochrone import factory
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import trapezoid
#galpy
import galpy.df as gd
import galpy.potential as gp
import gala.coordinates as gcoord # only for great circle rotation
import galpy.actionAngle as ga
from galpy.orbit import Orbit

with acoord.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = acoord.Galactocentric()
import warnings



def Plummer_sigv(M, b, r): #returns sigv for a plummer potential
    return np.sqrt(aconst.G * M.to(u.kg) / 6 * (r.to(u.m)**2+b.to(u.m)**2)**(-1/2)).to(u.km/u.s)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def normalize(lst):
    return [i/sum(lst) for i in lst]

def full_galpy_df(sigv, progenitor, pot, aA, ro, vo, vsun, tdisrupt, 
            timpact=None, impactb=None, subhalovel=None, subhalopot=None, impact_angle=None, perturbed = False, 
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

    #convert qunatities to correct astropy unit if needed.
    sigv=to_quantity(sigv, u.km/u.s)
    ro = to_quantity(ro, u.kpc).value
    vo = to_quantity(vo, u.km/u.s).value
    tdisrupt = to_quantity(tdisrupt, u.Gyr)
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
            print("Perturbing trailing arm")
        elif impact_angle >= 0:
            leading = True
            print("Perturbing leading arm")

        impactb =  to_quantity(impactb, u.kpc)
        timpact =  to_quantity(timpact, u.Gyr)
        impact_angle =  to_quantity(impact_angle, u.rad).value

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
    s_trail_stars[3:] *= vo

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



def icrs_to_phi12(stream_stars, pole1, pole2, velocities=False, panda=False):
    """
    Converts ICRS coordinates to the phi1/phi2 reference frame of a stellar stream.
    Can return a SkyCoord or enrich a DataFrame with Phi1, Phi2, etc. columns.

    Params:
    - stream_stars: SkyCoord or pandas.DataFrame with RA/DEC/etc. columns.
    - pole1, pole2: SkyCoord defining the stream plane.
    - velocities: If True, include proper velocities.
    - panda: If True, return an enriched DataFrame.
    """

    ra_s   = safe_getattr(stream_stars, 'ra', u.deg)
    dec_s  = safe_getattr(stream_stars, 'dec', u.deg)
    dist_s = safe_getattr(stream_stars, 'distance', u.kpc)

    if velocities:
        pm_ra_s = safe_getattr(stream_stars, 'pm_ra_cosdec', u.mas/u.yr)
        pm_dec_s = safe_getattr(stream_stars, 'pm_dec', u.mas/u.yr)
        pm_r = safe_getattr(stream_stars, 'radial_velocity', u.km/u.s)

        stream_icrs = acoord.SkyCoord(
            ra=ra_s, dec=dec_s, distance=dist_s,
            pm_ra_cosdec=pm_ra_s, pm_dec=pm_dec_s,
            radial_velocity=pm_r
        )
    else:
        stream_icrs = acoord.SkyCoord(ra=ra_s, dec=dec_s, distance=dist_s)

    stream_frame = gcoord.GreatCircleICRSFrame.from_endpoints(pole1, pole2, origin=pole1)
    stream_phi12 = stream_icrs.transform_to(stream_frame)

    if panda:
        try:
            df = pd.DataFrame(stream_stars) if not isinstance(stream_stars, pd.DataFrame) else stream_stars.copy()
        except Exception:
            raise ValueError("Impossible de convertir stream_stars en DataFrame.")

        df["phi1"] = stream_phi12.phi1.to(u.deg).value
        df["phi2"] = stream_phi12.phi2.to(u.deg).value
        df["distance"] = stream_phi12.distance.to(u.kpc).value

        if velocities:
            df["pm_phi1"] = stream_phi12.pm_phi1_cosphi2.to(u.mas/u.yr).value
            df["pm_phi2"] = stream_phi12.pm_phi2.to(u.mas/u.yr).value
            df["radial_velocity"] = stream_phi12.radial_velocity.to(u.km/u.s).value

        return df

    return stream_phi12


def to_quantity(val, unit):
    """
    Convert float or int to Quantity with assumed unit,
    or return the Quantity with correct unit.
    """
    if isinstance(val, u.Quantity):
        return val.to(unit)
    else:
        return val * unit


def safe_getattr(obj, attr, default_unit):
    """
    Essaie d'accéder à obj.attr. Si échoue ou sans unité, renvoie NaN * unité.
    Gère aussi les pandas Series en Quantity compatibles avec astropy.
    """
    try:
        value = getattr(obj, attr)
    except AttributeError:
        return 999 * default_unit

    # Si déjà une Quantity avec bonne unité
    if isinstance(value, u.Quantity):
        return value

    # Si c’est une pandas Series ou un numpy array
    if isinstance(value, (pd.Series, np.ndarray, list)):
        return u.Quantity(value, default_unit)

    # Sinon, on assume scalaire : float, etc.
    try:
        return value * default_unit
    except Exception:
        return 999 * default_unit



def safe_get_array(obj, attr_chain, length):
    """Safely get nested attribute (e.g., 'pm_phi1_cosphi2.value') or return NaNs."""
    try:
        value = obj
        for attr in attr_chain.split('.'):
            value = getattr(value, attr)
        return value
    except (AttributeError, TypeError, ValueError):
        return np.full(length, np.nan)


def save_star_data(star_list, mag_g, mag_r, coord_system, filepath=None, panda=True):
    """Save or return star data as Astropy Table or pandas DataFrame."""

    n_stars = len(star_list)

    if coord_system == 'radec':
        data = {
            "ra": star_list.ra.deg,
            "dec": star_list.dec.deg,
            "dist": star_list.distance.kpc,
            "muRA": safe_get_array(star_list, 'pm_ra_cosdec.value', n_stars),
            "muDec": safe_get_array(star_list, 'pm_dec.value', n_stars),
            "rv": safe_get_array(star_list, 'radial_velocity.value', n_stars),
            "mag_g": mag_g,
            "mag_r": mag_r
        }

    elif coord_system == 'phi12':
        data = {
            "phi1": star_list.phi1.deg,
            "phi2": star_list.phi2.deg,
            "dist": star_list.distance.kpc,
            "mu1": safe_get_array(star_list, 'pm_phi1_cosphi2.value', n_stars),
            "mu2": safe_get_array(star_list, 'pm_phi2.value', n_stars),
            "rv": safe_get_array(star_list, 'radial_velocity.value', n_stars),
            "mag_g": mag_g,
            "mag_r": mag_r
        }

    else:
        raise NotImplementedError('use phi12 or radec')

    if panda:
        df = pd.DataFrame(data)

        if filepath is None:
            os.makedirs("data", exist_ok=True)
            filepath = os.path.join("data", "star_data.csv")

        df.to_csv(filepath, index=False)
        print(f"Pandas DataFrame saved to: {filepath}")
        return df
    else:
        table = Table(data)

        if filepath is None:
            raise ValueError("filepath must be specified when panda=False.")

        table.write(filepath, format="csv", overwrite=True)
        print(f"Astropy Table saved to: {filepath}")



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

    def sample_magnitudes(self, icrs_list, sigma_mag=0, addnoise=False):
        """Assign magnitudes (g, r) to stars based on their distance."""
        distances = icrs_list.distance.to(u.kpc).value  # Extract distances in kpc
        distance_moduli = self._dist_to_modulus(distances)  # Compute distance modulus

        # Sample magnitudes for each star
        mag_g, mag_r = self.sample(len(distances), distance_moduli, sigma_mag, addnoise)

        return mag_g, mag_r

    def sample(self, nstars, distance_modulus, sigma_mag, addnoise):
        """Sample magnitudes using the isochrone."""
        stellar_mass = nstars * self.iso.stellar_mass()

        if np.isscalar(distance_modulus):
            mag_g, mag_r = self.iso.simulate(stellar_mass, distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + distance_modulus for mag in (mag_g, mag_r)]
            if addnoise:
                mag_g = mag_g+np.random.normal(loc=0, scale=sigma_mag, size=len(mag_g)), 
                mag_r = mag_r+np.random.normal(loc=0, scale=sigma_mag, size=len(mag_r))
        else:
            mag_g, mag_r = self.iso.simulate(stellar_mass, distance_modulus=self.iso.distance_modulus)
            mag_g, mag_r = [mag + distance_modulus for mag in (mag_g, mag_r)]
            if addnoise:
                mag_g = mag_g+np.random.normal(loc=0, scale=sigma_mag, size=len(mag_g))
                mag_r = mag_r+np.random.normal(loc=0, scale=sigma_mag, size=len(mag_r))

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
    
    def compute_density(self, delta_phi1=0.02, phi2_bins=50, max_fev=100000, plot=False):
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
                A0 = np.sum(hist)
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
        
        ###fit track
        lead_fit = np.polyfit(phi1_vals[phi1_vals>0], phi1_vals[phi2_vals>0], deg=3)
        trail_fit = np.polyfit(phi1_vals[phi1_vals<0], phi1_vals[phi2_vals<0], deg=3)
        

        self.density_table = Table(
            [phi1_vals, phi2_vals, nstars_vals, width_vals],
            names=("phi1", "phi2", "nstars", "width")
        )

        if plot:
            plt.figure(figsize=(5, 5))
            plt.scatter(phi1_vals, phi2_vals)

            # plt.title(f"Phi1 ≈ {phi1_fit:.2f} deg")
            plt.xlabel("Phi1 [deg]")
            plt.ylabel("Phi2 [deg]")
            plt.tight_layout()
            plt.show()

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

from numpy.polynomial.polynomial import Polynomial

class StreamDensityAnalyzer:
    def __init__(self, data, bin_width=0.1):
        """
        Initialize the analyzer with star data.

        Parameters:
        - data: SkyCoord, DataFrame, Table, or ndarray
        - bin_width: Bin width (deg) for density histogram
        """
        self.data = data
        self.bin_width = bin_width
        self.phi1 = self._extract_phi1(data)
        self.bin_centers = None
        self.densities = None

    def _extract_phi1(self, data):
        """Extract phi1 from supported data formats."""
        if isinstance(data, acoord.SkyCoord):
            try:
                return data.phi1.deg
            except AttributeError:
                raise ValueError("SkyCoord object must have .phi1 attribute.")
        elif isinstance(data, pd.DataFrame):
            if 'phi1' in data.columns:
                return data['phi1'].values
            elif 'Phi1' in data.columns:
                return data['Phi1'].values
        elif isinstance(data, Table):
            if 'phi1' in data.colnames:
                return np.array(data['phi1'])
            elif 'Phi1' in data.colnames:
                return np.array(data['Phi1'])
        elif isinstance(data, np.ndarray):
            if data.shape[1] >= 1:
                return data[:, 0]
        raise TypeError("Unsupported data format or missing 'phi1' column.")

    def compute_density(self, plot=False, normalize=True):
        """
        Compute the density histogram along phi1.

        Parameters:
        - plot: Whether to plot the density
        - normalize: Normalize density by its mean

        Returns:
        - bin_centers, densities (normalized if requested)
        """
        phi1_min, phi1_max = np.min(self.phi1), np.max(self.phi1)
        bins = np.arange(phi1_min, phi1_max + self.bin_width, self.bin_width)
        counts, edges = np.histogram(self.phi1, bins=bins)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        densities = counts / self.bin_width

        if normalize:
            densities /= np.mean(densities)

        self.bin_centers = bin_centers
        self.densities = densities

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(bin_centers, densities, drawstyle='steps-mid')
            # plt.xlabel("Phi1 (deg)")
            plt.ylabel("Normalized density (stars / deg)" if normalize else "Density (stars / deg)")
            # plt.title("Stellar stream density profile")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return bin_centers, densities

    def fit_polynomial(self, degree=3, plot=False, fontsize=12):
        """
        Fit a polynomial to the density profile.

        Parameters:
        - degree: Degree of the polynomial
        - plot: Whether to plot the fit result

        Returns:
        - coeffs: Polynomial coefficients
        - poly_func: Polynomial function
        """
        if self.bin_centers is None or self.densities is None:
            raise RuntimeError("Density must be computed before fitting.")

        coeffs = Polynomial.fit(self.bin_centers, self.densities, degree).convert().coef
        poly_func = Polynomial(coeffs)

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(self.bin_centers, self.densities, label="Density", drawstyle='steps-mid')
            plt.plot(self.bin_centers, poly_func(self.bin_centers),
                     label=f"Polynomial degree {degree}", color='red')
            # plt.xlabel(r'$\phi_1$ (°)', fontsize=fontsize)
            plt.ylabel(r"$\rho/<\rho> (\phi_1$)", fontsize=fontsize)
            # plt.title("Polynomial fit of stellar density")
            plt.grid(True)
            # plt.legend()
            plt.tight_layout()
            plt.show()

        return coeffs, poly_func


from scipy.ndimage import binary_dilation

class StreamDensityFiltering:
    def __init__(self, data, bins=(200, 200), threshold=1.5, buffer=1, upperthreshold=200):
        """
        Initialize the density selection.

        Parameters:
        ----------
        data: pandas.DataFrame
        Table with columns 'ra', 'dec'.
        bins: tuple
        Number of bins in (ra, dec).
        threshold: float
        Density threshold = threshold * average density.
        buffer: int
        Number of neighboring bins to include around dense areas.
        """
        self.data = data
        self.bins = bins
        self.threshold = threshold
        self.buffer = buffer

        # 2D histogrm
        self.H, self.ra_edges, self.dec_edges = np.histogram2d(data['ra'], data['dec'], bins=bins)
        # Threshold and mask
        self.background = np.mean(self.H)
        self.mask = (self.H > (threshold * self.background)) 
        self.mask_dilated = binary_dilation(self.mask, iterations=buffer) & (self.H < upperthreshold)

        # Identify selected stars
        self._compute_selection()
    
    def _compute_selection(self):
        """Select stars belonging to dense regions (and neighbor ones)"""
        ra_idx = np.digitize(self.data['ra'], self.ra_edges) - 1
        dec_idx = np.digitize(self.data['dec'], self.dec_edges) - 1

        valid = (
            (ra_idx >= 0) & (ra_idx < self.bins[0]) &
            (dec_idx >= 0) & (dec_idx < self.bins[1])
        )

        selected_mask = np.zeros(len(self.data), dtype=bool)
        selected_mask[valid] = self.mask_dilated[ra_idx[valid], dec_idx[valid]]
        self.selected_mask = selected_mask
        self.filtered_data = self.data[selected_mask].copy()

        # Filtered Histogram
        self.H_filtered = np.zeros_like(self.H)
        self.H_filtered[self.mask_dilated] = self.H[self.mask_dilated]
    
    def plot(self, filtered=True, cmap='viridis'):
        """Plot the filtered histogram"""
        H_to_plot = self.H_filtered if filtered else self.H
        extent = [
            self.ra_edges[0], self.ra_edges[-1],
            self.dec_edges[0], self.dec_edges[-1]
        ]
        plt.figure(figsize=(6, 6))
        plt.imshow(
            H_to_plot.T,
            origin='lower',
            extent=extent,
            aspect='auto',
            cmap=cmap
        )
        plt.xlabel("ra (°)", fontsize=14)
        plt.ylabel("dec (°)", fontsize=14)
        # plt.title("Surfacic density" + (" (filtered)" if filtered else ""))
        # plt.colorbar(label="Stars per bins")
        plt.tight_layout()
        plt.show()

    def values(self, return_hist=False):
        """Returns the computed filtered table"""
        if return_hist:
            return self.filtered_data, self.H_filtered
        return self.filtered_data


##### Only works with stream_generator.py #####
class StreamArgumentParser:
    @staticmethod
    def load_config_file(path):
        config = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                line = line.split('#', 1)[0].strip()  # Remove inline comments
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        config[key] = eval(val)
                    except Exception:
                        config[key] = val
        return config

    @staticmethod
    def build_parser():
        import argparse
        from stream_galsim.generator_keywords import PARAMETERS

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='config.txt', help='Path to config file')

        for key, meta in PARAMETERS.items():
            parser.add_argument(f'--{key}', type=meta['type'], default=None)

        return parser

    @staticmethod
    def merge_configs(file_config, cli_args):
        from stream_galsim.generator_keywords import PARAMETERS

        merged = {}
        for key, meta in PARAMETERS.items():
            cli_val = getattr(cli_args, key)
            if cli_val is not None:
                merged[key] = cli_val
            elif key in file_config:
                merged[key] = file_config[key]
            else:
                print(f"{key} not specified - using default: {meta['default']}")
                merged[key] = meta['default']

        for key in file_config:
            if key not in PARAMETERS:
                raise ValueError(f"Unknown keyword in config file: {key}")
        
        return merged

class GeneratorInit:
    def progenitor(config):
        import os
        import yaml
        import pandas as pd

        print("Initializing progenitor...")

        column_name = config['column_name']
        name_o = config['name_o']
        prog_config_path = config.get('prog_config_path')

        data = {
            'column_name': column_name,
            'name_o': name_o,
            'prog_config_path': prog_config_path,
        }

        if config.get('init_method') == 'galstream':
            if prog_config_path is None:
                base_dir = os.path.abspath(os.getcwd())
                config_file = f'galstreams_{name_o}_config.yaml'
                prog_config_path = os.path.join(base_dir, 'config', config_file)

            if not os.path.exists(prog_config_path):
                print(f"No config file found at {prog_config_path}. Generating {name_o} config file from galstream")
                from stream_galsim.create_config_file import generate
                generate(name_o, column_name)
                base_dir = os.path.abspath(os.getcwd())
                config_file = f'galstreams_{name_o}_config.yaml'
                prog_config_path = os.path.join(base_dir, 'config', config_file)

            with open(prog_config_path, 'r') as file:
                print(f"Loading config file from {prog_config_path}")
                galstream_data = yaml.safe_load(file)

            galstream_df = pd.DataFrame(galstream_data)
            if galstream_df.shape[0] > 1:
                raise ValueError("Multiple entries in galstream config.")

            data.update(galstream_df.iloc[0].to_dict())

        for key in ['ra_o', 'dec_o', 'distance_o', 'pm_ra_cosdec_o', 'pm_dec_o', 'vrad_o', 'frame', 'prog_mass', 'prog_sigv']:
            data[key] = config[key]

        return pd.DataFrame([data])

    def acc_host(config):
        import pandas as pd
        import numpy as np
        import galpy.potential as gp
        import galpy.actionAngle as ga

        print("Initializing accreting host...")

        V0 = config['V0']
        R0 = config['R0']
        vsun = config['vsun']
        pot = eval(config['mw_pot']) if isinstance(config['mw_pot'], str) else config['mw_pot'] or gp.MWPotential2014
        b_scale = config['b_mw']

        aaisochrone = ga.actionAngleIsochroneApprox(pot=pot, b=b_scale)

        acc_host = pd.DataFrame([{
            'V0': V0, 'V0_unit': 'km/s',
            'R0': R0, 'R0_unit': 'kpc',
            'vsun': vsun, 'vsun_unit': 'km/s',
        }])

        return acc_host, pot, aaisochrone

    def halo_impact(config):
        import pandas as pd
        import numpy as np
        import galpy.potential as gp

        print("Initializing halo and impact properties...")

        halo_mass = config['halo_mass']
        halo_a = config['halo_a']
        v_halo = config['v_halo']
        impact_b = config['b_impact']
        t_impact = config['t_impact']
        angle_impact = config['angle_impact']

        halo_pot = gp.NFWPotential(amp=halo_mass, a=halo_a)

        halo_df = pd.DataFrame([{
            'halo_mass': halo_mass,
            'halo_a': halo_a,
            'v_halo': np.array(v_halo).tolist(),
            'impact_b': impact_b,
            't_impact': t_impact,
            'angle_impact': angle_impact,
        }])

        return halo_df, halo_pot


###old


    # stream_frame = gcoord.GreatCircleICRSFrame.from_endpoints(pole1, pole2, origin=pole1)
    # ra_s, dec_s, dist_s = stream_stars.ra, stream_stars.dec, stream_stars.distance

    # if velocities == True:
    #     pm_ra_s, pm_dec_s, pm_r = stream_stars.pm_ra_cosdec, stream_stars.pm_dec, stream_stars.radial_velocity
    #     stream_icrs = acoord.SkyCoord(ra=ra_s, dec=dec_s, distance=dist_s, pm_ra_cosdec=pm_ra_s, pm_dec=pm_dec_s, radial_velocity=pm_r)
    #     stream_phi12 = stream_icrs.transform_to(stream_frame)
    #     # stream_phi12 = stream_phi12.phi1, stream_phi12.phi2, stream_phi12.distance, stream_phi12.pm_phi1_cosphi2, stream_phi12.pm_phi2, stream_phi12.radial_velocity
        
    # elif velocities == False:
    #     stream_icrs = acoord.SkyCoord(ra=ra_s, dec=dec_s, distance=dist_s)
    #     stream_phi12 = stream_icrs.transform_to(stream_frame)
    #     # stream_phi12 = stream_phi12.phi1, stream_phi12.phi2, stream_phi12.distance
    # return stream_phi12 #Phi1, Phi2, dist