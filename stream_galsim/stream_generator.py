#!/usr/bin/env python


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

#Galpy
# import galpy.potential as gp
# import galpy.df as gd #for streams PDF generation
# import galpy.actionAngle as ga
from galpy.orbit import Orbit


#Coordinate tools and astropy
import astropy.units as u
import astropy.coordinates as ac
from astropy.table import Table
import gala.coordinates as gc #for great circle rotation
with ac.galactocentric_frame_defaults.set("v4.0"):
    galcen_frame = ac.Galactocentric()

#Define user path
base_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(base_dir)

#Custom scripts/functions
import stream_galsim.stream_utils as sutils
from stream_galsim.stream_utils import GeneratorInit as gen_init
from stream_galsim.stream_utils import StreamArgumentParser as parser
from stream_galsim.generator_keywords import PARAMETERS



##### read keywords from config file and command #####
#init parser
parser_obj = parser.build_parser()
args = parser_obj.parse_args()
file_config = parser.load_config_file(args.config)
merged_config = parser.merge_configs(file_config, args)

#read

##### Initialize simulation components #####
### Initialize progenitor
progenitor = gen_init.progenitor(merged_config)
header_keys = ['ra_o', 'dec_o', 'distance_o', 'pm_ra_cosdec_o', 'pm_dec_o', 'vrad_o']
prog_orbit = Orbit([progenitor.at[0, key] for key in header_keys], radec=True) 

### Initialize accreting host
acc_host, acc_pot, aaisochrone = gen_init.acc_host(merged_config)

##### Generating df #####
print("Generating stream df...")

perturbed = merged_config['perturbed']
if (perturbed == 'lead') | (perturbed == 'trail'):
    ### Initialize impact if specified
    halo_impact, halo_pot = gen_init.halo_impact(merged_config)

    s_lead, s_trail, s_perturbed = sutils.full_galpy_df(
                        sigv=progenitor.at[0, 'prog_sigv'], #Radial velocity dispersion of the progenitor
                        progenitor=prog_orbit, #Progenitor orbit instance
                        pot = acc_pot, #accreting host potential
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
    print(merged_config['t_disrupt'])
    s_lead, s_trail, s_perturbed = sutils.full_galpy_df(
                        sigv=progenitor.at[0, 'prog_sigv'], #Radial velocity dispersion of the progenitor
                        progenitor=prog_orbit, #Progenitor orbit instance
                        pot = acc_pot, #accreting host potential
                        aA = aaisochrone, #ActionAngle instance used to convert (x,v) to actions. Generally a actionAngleIsochroneApprox instance.                        vsun=vsun,
                        ro = acc_host.at[0, 'R0'], #Distance scale for translation into internal units
                        vo = acc_host.at[0, 'V0'], #Velocity scale for translation into internal units
                        vsun = acc_host.at[0, 'vsun'],                         
                        tdisrupt = merged_config['t_disrupt'],#  Time since start of disruption
                        perturbed = False, #Including or not a DMS perturbation
                        )
else:
    raise NotImplementedError("Unrecognized argument. Choose between 'lead', 'trail' or 'not'.")

#ToDo: add pickle instance saving

##### Sampling stars #####
print("Sampling stars along the df...")
s_lead_stars, s_trail_stars, s_full_stars, ps_stars, ps_full_stars = sutils.full_galpy_sampling(n_stars=merged_config['n_stars'], 
                                                                s_lead=s_lead, s_trail=s_trail, s_perturbed=s_perturbed,
                                                                ro=acc_host.at[0, 'R0'], vo=acc_host.at[0, 'V0'], 
                                                                perturbed = perturbed, rdseed = merged_config['rdseed'])
##### Conversion #####
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
    pole = merged_config['phi12_pole']
    pole1 = ac.SkyCoord(ra=progenitor.at[0, 'ra_o']*u.deg, dec=progenitor.at[0, 'dec_o']*u.deg)
    pole2 = ac.SkyCoord(ra=pole[0]*u.deg, dec=pole[1]*u.deg)

    stream_stars = sutils.xyz_to_icrs(stream_stars,  velocities=True)
    stream_stars = sutils.icrs_to_phi12(stream_stars, pole1, pole2, velocities=True) 
else:
    raise NotImplementedError("Unrecognized argument. Choose between 'xyz', 'icrs' or 'phi12'.")


##### Generate ouput file #####
if output_method == 'stars':
    iso_config = {
    "name": merged_config['iso_name'],
    "survey": merged_config['survey'],
    "age": merged_config['age'],
    "z": merged_config['z'],
    "band_1": merged_config['band_1'],
    "band_2": merged_config['band_2'],
    "band_1_detection": merged_config['band_1_detection']}
    print("Generating magnitudes...")
    isochrone_model = sutils.IsochroneModel(iso_config)

    mag_g_s, mag_r_s = isochrone_model.sample_magnitudes(stream_stars)

    # Save stars table with sampled magnitudes
    sutils.save_star_data(stream_stars, mag_g_s, mag_r_s, 'phi12', filepath=f"{base_dir}/data/{name_o}_stars.csv")

### To improve
elif output_method == 'density':
    print("Generating density map parameters...")
    stream_track_density = sutils.StreamInterpolateTrackDensity(s_stars_phi12[mask])
    density_table = stream_track_density.compute_density(delta_phi1=0.05, phi2_bins=15, plot=True, max_fev=1000000)
    density_table.write(f'{base_dir}/data/hall_2024_{name_o}_input.csv', format='csv', overwrite=True)
    print(f"density map parameters saved at {base_dir}/data/hall_2024_{name_o}_input.csv")

