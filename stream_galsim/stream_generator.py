##### Load libraries #####

#Utils
from importlib import reload
import argparse
import pandas as pd
import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Galpy
import galpy.potential as gp
import galpy.df as gd #for streams PDF generation
import galpy.actionAngle as ga
from galpy.orbit import Orbit
import galpy.util.coords as gucoord
import galpy.util.conversion as guconv

#Coordinate tools and astropy
import astropy.units as u
import astropy.coordinates as ac
from astropy.table import Table
# import galpy.util.conversion as guconv #beurk
import gala.coordinates as gc #for great circle rotation
with ac.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = ac.Galactocentric()

#Define user path
sys.path.append(os.path.abspath('../'))

#Custom scripts
from stream_galsim.create_config_file import generate
import stream_galsim.stream_utils as sutils


##### Read keywords #####
base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
#Load keyword file
init_method = 'galstream' #custom
column_name = 'TrackName' #Name
name_o = 'Pal5-I21'
impact = False
##### Initialize progenitor #####

def init_progenitor(init_method, column_name, name_o,
                    ra_o=None, dec_o=None, distance_o=None,
                    pm_ra_cosdec_o=None, pm_dec_o=None, vrad_o=None, frame=None,
                    mass_o=None, sigv_o=None):
    import os
    import yaml
    import pandas as pd

    print("Initializing progenitor...")

    DEFAULT_PARAMS = {
        'ra_o': (229, 'deg'),
        'dec_o': (-0.124, 'deg'),
        'distance_o': (22.9, 'kpc'),
        'pm_ra_cosdec_o': (-2.296, 'mas/yr'),
        'pm_dec_o': (-2.257, 'mas/yr'),
        'vrad_o': (-58.7, 'km/s'),
        'frame': ('icrs', None),
    }

    DEFAULT_PHYS = {
        'mass_o': (2e4, 'Msun'),
        'sigv_o': (0.365, 'km/s'),
    }

    user_params = {
        'ra_o': ra_o,
        'dec_o': dec_o,
        'distance_o': distance_o,
        'pm_ra_cosdec_o': pm_ra_cosdec_o,
        'pm_dec_o': pm_dec_o,
        'vrad_o': vrad_o,
        'frame': frame,
    }

    user_phys = {
        'mass_o': mass_o,
        'sigv_o': sigv_o,
    }

    data = {'name_o': name_o}

    # Load galstream config file
    if init_method == 'galstream':
        base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        config_file = f'galstreams_{name_o}_config.yaml'
        config_path = os.path.join(base_dir, 'stream_sim/config', config_file)

        if not os.path.exists(config_path):
            print(f"No config file found. Generating {name_o} config file from galstream")
            generate(name_o, column_name)

        with open(config_path, 'r') as file:
            print("Loading config file...")
            config_data = yaml.safe_load(file)

        # Transform into DataFrame and extract the dict
        galstream_df = pd.DataFrame(config_data)
        if galstream_df.shape[0] > 1:
            raise ValueError(f"Expected a single entry in config, found {galstream_df.shape[0]}")

        data.update(galstream_df.iloc[0].to_dict())

    # Complete missing values with user values, ordefault values if missing as well
    for key, (default_val, unit) in DEFAULT_PARAMS.items():
        val = user_params.get(key)
        if val is not None:
            data[key] = val
        elif key not in data or data[key] is None:
            print(f"{key} not specified - using default value: {default_val}")
            data[key] = default_val
        data[f"{key}_unit"] = unit

    for key, (default_val, unit) in DEFAULT_PHYS.items():
        val = user_phys.get(key)
        if val is not None:
            data[key] = val
        elif key not in data or data[key] is None:
            print(f"{key} not specified - using default phys value: {default_val}")
            data[key] = default_val
        data[f"{key}_unit"] = unit

    # Rturn as a dataframe
    progenitor_coord = pd.DataFrame([data])
    return progenitor_coord


progenitor = init_progenitor('custom', column_name, name_o )
keys = ['ra_o', 'dec_o', 'distance_o', 'pm_ra_cosdec_o', 'pm_dec_o', 'vrad_o']
prog_orbit = Orbit([progenitor.at[0, key] for key in keys], radec=True) 

##### Initialize accreting host #####

def init_acc_host(V0=None, R0=None, vsun=None, pot=None, b_scale=None):
    print("Initializing accreting host...")
    DEFAULTS = {
        'V0': 245.6,
        'R0': 8.122,
        'vsun': [-12.9, 245.6, 7.78],
        'pot': gp.MWPotential2014,
        'b_scale': 1.5
    }

    def get_or_default(val, key):
        if val is None:
            print(f"{key} not specified - using default: {DEFAULTS[key]}")
            return DEFAULTS[key]
        return val

    V0 = get_or_default(V0, 'V0')
    R0 = get_or_default(R0, 'R0')
    vsun = get_or_default(vsun, 'vsun')
    pot = get_or_default(pot, 'pot')
    b_scale = get_or_default(b_scale, 'b_scale')

    # ActionAngle isochrone approx
    aaisochrone = ga.actionAngleIsochroneApprox(pot=pot, b=b_scale)

    acc_host = pd.DataFrame([{
        'V0': V0,
        'V0_unit': 'km/s',
        'R0': R0,
        'R0_unit': 'kpc',
        'vsun': vsun,
        'vsun_unit': 'km/s'
    }])

    return acc_host, pot, aaisochrone

acc_host, mw_pot, aaisochrone = init_acc_host()
##### Initialize impact #####if specified

def init_halo_impact(halo_mass=None, halo_a=None, v_halo=None,
                     impact_b=None, t_impact=None, angle_impact=None):
    print("Initializing halo and impact properties...")
    DEFAULTS = {
        'halo_mass': 1e7,
        'halo_a': 0.01,
        'v_halo': np.array([6.82200571, 132.7700529, 149.4174464]),
        'impact_b': 0,
        't_impact': 2,
        'angle_impact': 0.12
    }

    def get_or_default(param, name):
        if param is None:
            print(f"{name} not specified - using default: {DEFAULTS[name]}")
            return DEFAULTS[name]
        return param

    # Retrieving or assigning values
    halo_mass = get_or_default(halo_mass, 'halo_mass')
    halo_a = get_or_default(halo_a, 'halo_a')
    v_halo = get_or_default(v_halo, 'v_halo')
    impact_b = get_or_default(impact_b, 'impact_b')
    t_impact = get_or_default(t_impact, 't_impact')
    angle_impact = get_or_default(angle_impact, 'angle_impact')

    halo_pot = gp.NFWPotential(amp=halo_mass, a=halo_a)

    halo_impact = pd.DataFrame([{
        'halo_mass': halo_mass,
        'halo_a': halo_a,
        'v_halo': v_halo.tolist(),
        'impact_b': impact_b,
        't_impact': t_impact,
        'angle_impact': angle_impact
        }])

    return halo_impact, halo_pot


if impact == True:
    halo_impact, halo_pot = init_halo_impact()


##### Generating df #####
t_disrupt = 4*u.Gyr
perturbed = 'not'
print("Generating stream df...")
if (perturbed == 'lead') | (perturbed == 'trail'):
    s_lead, s_trail, s_perturbed = sutils.full_galpy_df(
                        sigv=progenitor.at[0, 'sigv_o'], #Radial velocity dispersion of the progenitor
                        progenitor=prog_orbit, #Progenitor orbit instance
                        pot = mw_pot, #accreting host potential
                        aA = aaisochrone, #ActionAngle instance used to convert (x,v) to actions. Generally a actionAngleIsochroneApprox instance.                        vsun=vsun,
                        ro = acc_host.at[0, 'R0'], #Distance scale for translation into internal units
                        vo = acc_host.at[0, 'V0'], #Velocity scale for translation into internal units
                        vsun = acc_host.at[0, 'vsun'],                         
                        tdisrupt = t_disrupt,#  Time since start of disruption
                        perturbed = True, #Including or not a DMS perturbation
                        timpact = halo_impact.at[0, 't_impact'], # Time of impact
                        impactb = halo_impact.at[0, 'impact_b'], #Impact parameter between halo and streams
                        subhalovel = halo_impact.at[0, 'v_halo'], # Velocity of the subhalo
                        subhalopot = halo_pot,# Gravitational potential of the subhalo
                        impact_angle = halo_impact.at[0, 'angle_impact'], #Angle offset from progenitor at which the impact occurred
                        )
elif perturbed == 'not':
    s_lead, s_trail, s_perturbed = sutils.full_galpy_df(
                        sigv=progenitor.at[0, 'sigv_o'], #Radial velocity dispersion of the progenitor
                        progenitor=prog_orbit, #Progenitor orbit instance
                        pot = mw_pot, #accreting host potential
                        aA = aaisochrone, #ActionAngle instance used to convert (x,v) to actions. Generally a actionAngleIsochroneApprox instance.                        vsun=vsun,
                        ro = acc_host.at[0, 'R0'], #Distance scale for translation into internal units
                        vo = acc_host.at[0, 'V0'], #Velocity scale for translation into internal units
                        vsun = acc_host.at[0, 'vsun'],                         
                        tdisrupt = t_disrupt,#  Time since start of disruption
                        perturbed = False, #Including or not a DMS perturbation
                        )
else:
    raise NotImplementedError("Unrecognized argument. Choose between 'lead', 'trail' or 'not'.")


#ToDo: add pickle instance saving

##### Sampling stars #####
rdseed = 56
n_stars = 10000
print("Sampling stars along the df...")
s_lead_stars, s_trail_stars, s_full_stars, ps_stars, ps_full_stars = sutils.full_galpy_sampling(n_stars=n_stars, 
                                                                s_lead=s_lead, s_trail=s_trail, s_perturbed=s_perturbed,
                                                                ro=acc_host.at[0, 'R0'], vo=acc_host.at[0, 'V0'], 
                                                                perturbed = perturbed, rdseed = rdseed)
##### Conversion #####
coord = 'phi12'
print(f"Convert to {coord} frame...")

if perturbed == 'not':
    stream_stars = s_full_stars

elif (perturbed == 'lead') | (perturbed == 'trail'):
    stream_stars = ps_full_stars

if coord == 'xyz':
    stream_stars = stream_stars
elif coord == 'icrs':
    stream_stars = sutils.xyz_to_icrs(stream_stars,  velocities=True)
elif coord == 'phi12':
    pole = (237.5, 5)
    pole1 = ac.SkyCoord(ra=progenitor.at[0, 'ra_o']*u.deg, dec=progenitor.at[0, 'dec_o']*u.deg)
    pole2 = ac.SkyCoord(ra=pole[0]*u.deg, dec=pole[1]*u.deg)

    stream_stars = sutils.xyz_to_icrs(stream_stars,  velocities=True)
    stream_stars = sutils.icrs_to_phi12(stream_stars, pole1, pole2, velocities=True) 
else:
    raise NotImplementedError("Unrecognized argument. Choose between 'xyz', 'icrs' or 'phi12'.")


##### Generate ouput file #####
output_method = 'stars'#choose ouput type (stars or density map)

if output_method == 'stars':
    config = {
    "name": "Bressan2012",
    "survey": "des",
    "age": 12.0,
    "z": 0.0006,
    "band_1": "g",
    "band_2": "r",
    "band_1_detection": True}
    print("Generating magnitudes...")
    isochrone_model = sutils.IsochroneModel(config)

    mag_g_s, mag_r_s = isochrone_model.sample_magnitudes(stream_stars)

    # Save stars table with sampled magnitudes
    sutils.save_star_data(stream_stars, mag_g_s, mag_r_s, 'phi12', filepath=f"{base_dir}/data/{name_o}_stars.csv")

elif output_method == 'density':
    print("Generating density map parameters...")
    stream_track_density = sutils.StreamInterpolateTrackDensity(s_stars_phi12[mask])
    density_table = stream_track_density.compute_density(delta_phi1=0.05, phi2_bins=15, plot=True, max_fev=1000000)
    density_table.write(f'{base_dir}/data/hall_2024_{name_o}_input.csv', format='csv', overwrite=True)
    print(f"density map parameters saved at {base_dir}/data/hall_2024_{name_o}_input.csv")
#ps_coord
#mag choice



#maybe in another script
####stream-sim
#choose


