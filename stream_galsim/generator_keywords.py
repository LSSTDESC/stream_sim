PARAMETERS = {
    #init method
    'init_method':     {'type': str,    'default': 'galstream'},
    'column_name':     {'type': str,    'default': 'TrackName'},
    'name_o':          {'type': str,    'default': 'Pal5-I21'},
    'prog_config_path':{'type': str,    'default': None},

    #progenitor
    'frame':           {'type': str,    'default': 'icrs'},
    'ra_o':            {'type': float,  'default': 229},
    'dec_o':           {'type': float,  'default': -0.124},
    'distance_o':      {'type': float,  'default': 22.9},
    'pm_ra_cosdec_o':  {'type': float,  'default': -2.296},
    'pm_dec_o':        {'type': float,  'default': -2.257},
    'vrad_o':          {'type': float,  'default': -58.7},
    'prog_mass':       {'type': float,  'default': 2e4},
    'prog_sigv':       {'type': float,  'default': 0.365},

    #acc_host
    'V0':              {'type': float,  'default': 245.6},
    'R0':              {'type': float,  'default': 8.122},
    'vsun':            {'type': eval,   'default': [-12.9, 245.6, 7.78]},
    'mw_pot':          {'type': str,    'default': 'gp.MWPotential2014'},

    #impact
    'perturbed':       {'type': str,    'default': 'not'},
    'b_mw':            {'type': float,  'default': 1.5},
    'b_impact':        {'type': float,  'default': 0},
    't_impact':        {'type': float,  'default': 2},
    'angle_impact':    {'type': float,  'default': 0.12},

    #stream
    'n_stars':         {'type': int,    'default': 10000},
    't_disrupt':       {'type': float,  'default': 4.0}, #* u.Gyr
    'rdseed':          {'type': int,    'default': 4},
    'phi12_pole':      {'type': tuple,  'default': (237.5, 5)},

    #magnitudes
    'iso_name':        {'type': str,    'default':'Bressan2012'},
    'survey':          {'type': str,    'default':'des'},
    'age':             {'type': float,  'default':12.0},
    'z':               {'type': float,  'default':0.0006},
    'band_1':          {'type': str,    'default': 'g'},
    'band_2':          {'type': str,    'default': 'r'},
    'band_1_detection':{'type': bool,   'default': True},


    ##### Output options #####
    'output_method':   {'type': float,  'default': 'stars'},          # or density
    'output_coord':    {'type': float,  'default': 'phi12'},          # icrs, xyz, phi12, etc.
}
