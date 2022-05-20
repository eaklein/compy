#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:55:02 2022

@author: diesel
"""

import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

from compy import compy_script, utilities


def main():
    """Process runs and print transmission."""
    # change directory to experiments folder
    os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")

    # load selected runs
    runs = compy_script.main()

    # specify keys to plot
    keys = list(runs.keys())
    print(f'Processed keys are {keys}.')
    key_ob = input('Which key would you like to use for open beam?\n')

    # add transmission
    [runs[key].add_trans(runs, key_ob, t_offset=0)
     for key in keys if key != key_ob]

    # plot transmission for target runs
    [utilities.plot_trans(runs, key, key_ob, n_bins=600, t_offset=5.56)
     for key in keys if key != key_ob]
    return runs


runs = main()
