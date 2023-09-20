# Simple Stream Simulation

This package provides some tools for simple stellar stream simulation. The goal is to provide a modular codebase that can be easily configured and run. The user should be able to generate a variety of stream morphologies by changing the configuration files. The output will be the "true" properties of stream stars (e.g., location in stream coordinates, heliocentric distance, magnitude, and velocity) that can be used for image-level or catalog-level injection. 

## Installation

TBD (probably via pip and conda)

## Examples

> ./bin/generate_stream.py config/toy1_config.yaml -o toy1.csv --plot
> ./bin/generate_stream.py config/toy2_config.yaml -o toy2.csv --plot
> ./bin/generate_stream.py config/pal5_config.yaml -o pal5.csv --plot