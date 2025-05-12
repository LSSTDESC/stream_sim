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
import astropy.constants as aconst
import astropy.coordinates as acoord
import numpy as np
import galpy.df as gd
import gala.coordinates as gcoord # only for great circle rotation
with acoord.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = acoord.Galactocentric()
import warnings
from ugali.isochrone import factory


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
