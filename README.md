# Compy: The CoMPASS Python Companion

## Introduction

Compy is a Python package for easily accessing and analyzing data acquired using CAEN Multi-PArameter Spectroscopy Software (CoMPASS). Compy takes as input the CoMPASS directories where experimental data is stored and processes the settings, spectra, and data files, storing them in a CompassRun object. Compy currently can only process data saved as either .csv or .bin files. Compy enables user filtering of data by energy and/or PSD cut and easy plotting of un/filtered energy/TOF/PSD spectra.

## Scripts

Included with the compy package are several scripts for commonly desired processes, such as plotting CoMPASS spectra, manually calculating TOF from signal and reference channels, and enabling user cuts.

### compy-run

The compy-run script provides a quick way of reading in CoMPASS data, merging related runs, plotting spectra, and saving the CoMPASS data to a pickle file for easy access later in Python.
