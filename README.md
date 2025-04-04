# Simple Stellar Stream Simulator (about stream_sim)

This package provides some tools for simple stellar stream simulation. The goal is to provide a modular codebase that can be easily configured and run. The user should be able to generate a variety of stream morphologies by changing the configuration files. The output will be the "true" properties of stream stars (e.g., location in stream coordinates, heliocentric distance, magnitude, and velocity) that can be used for image-level or catalog-level injection.

Note, this package is not intended to generate dynamic models or N-body simulations, but rather to study statistical realizations of parametrized stream geometries.

## Installation

Installation consists of git cloning `stream_sim`, adding the python module to your `PYTHONPATH`, and adding the `bin` directory to your `PATH`:

```bash
git clone https://github.com/LSSTDESC/stream_sim.git
export PYTHONPATH=${PWD}/stream_sim:${PYTHONPATH}
export PATH=${PWD}/stream_sim/bin:${PATH}
```

The code has some common dependencies (numpy, scipy, pandas, matplotlib) that should be present in your environment.

Eventually stream_sim will be installable through common package managers (i.e., pip and/or conda).

## Examples

Several simple configuration files are provided as examples in the `config` directory. To generate streams from these config files, use the `generate_streams.py` executable found in the `bin` directory. From the `stream_sim` directory, this code can be executed as shown below.

```bash
# Simple linear stream
./bin/generate_stream.py config/toy1_config.yaml -o toy1.csv --plot
# More complex sinusoidal stream
./bin/generate_stream.py config/toy2_config.yaml -o toy2.csv --plot
# Stream generated from Pal 5 interpolation
./bin/generate_stream.py config/pal5_config.yaml -o pal5.csv --plot
# Stream generated from Spline interpolation
./bin/generate_spline_stream.py config/atlas_spline_config.yaml -o atlas_spline.csv --plot
```

# stream_sim-hall - Stellar stream simulation using galpy (or N-body simulation)

Stream_sim package functions gives a simple way to simulate a stream (model.py, samplers.py), but also tools to add survey properties to sampled stars and use is it for stream searching using match filtering (observed.py). It starts from a config_file including stream, background and properties for simulation, and a path to a file (erkal_.csv) in the format [Phi1 Phi2 nstars width], in order to generate a stream in a Phi12 frame.  

With galpy (or gala) and ugali we can sample stars within a stellar stream in desired coordinate frames (galactocentric, radec, Phi12,...) and give them magnitudes in wanted filters. It also starts with a config file created from create_config_file.py (which use galstreams catalog to create automatically a galstream_clustername_config.yaml). User can specify custom data for simulation, since create_config_file.py is curently in development.

There's an example of stream simulation is available in stream_sim_galpy.py for palomar_5 cluster. Steps are specified in the notebook. The output is a file (clustername.csv) with sampled stars in the icrs coordinate frames with sampled magnitudes (at least, it's easy to add sampled velocities or corresponding to each star, and/or a file in the same format as the erkal for stream_sim package.

Using this sampled stream, stream_search either starts from full_clustername.csv by applying survey properties to star data, or from erkal-like file by sampling positions and magnitudes, then applying survey properties (following injection_lsst_example.ipynb)

Finally, the user can use isochrone matchfiltering to isolate stream stars from those from the background.