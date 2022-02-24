#!/usr/bin/env python3

"""
A command line script for processing CoMPASS data
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import click

from compy import compassrun, utilities


def main():
    """Process user-selected runs and plot filtered TOF spectra."""
    args = sys.argv[1:]
    argc = len(args)
    if argc > 0:
        folders = [str(Path(arg).resolve()) for arg in args]
        print(folders)
    else:
        folders = None

    # process data
    pkl_flag = click.confirm('\nWould you like to load data from pickle?',
                             default=False)
    if pkl_flag:
        runs = utilities.load_pickle()
    else:
        folder, key_tuples, VERBOSE = compassrun.initialize(folders=folders)
        runs = compassrun.process_runs(key_tuples)
        utilities.merge_related_runs(runs, quiet=True)

    # plot filtered TOF spectra for all keys
    print_flag = click.confirm('\nWould you like to plot the spectra?',
                             default=True)
    if print_flag:
        plt.figure(figsize=(16, 9))
        for key in runs.keys():
            print(key)
            if 'vals' in runs[key].spectra['filtered']['TOF']:
                vals_raw = np.array(runs[key].spectra['filtered']['TOF']['vals'])
                bins = np.array(runs[key].spectra['filtered']['TOF']['bins'])
                t = runs[key].t_meas
                print('plotting key: ', key, t, sum([i for i in vals_raw]))
                vals_err = np.sqrt(vals_raw) / t
                vals = vals_raw / t
                plt.errorbar(x=bins, y=vals, yerr=vals_err,
                              marker='s', linestyle='None', drawstyle='steps-mid',
                              label=key.replace('_', '-'))
        if len(runs.keys()) > 0:
            plt.xlim(25, 185)
            plt.xlabel(r'TIME [$\mu$s]')
            plt.ylabel('COUNTS/MINUTE')
            plt.ylim(0, 3.5)
            plt.legend()
            plt.tight_layout()
        else:
            print('No spectra found to plot!')

    # save data to pickle
    save_flag = click.confirm('\nWould you like to save the runs as a pickle?',
                              default=True)
    if save_flag:
        utilities.save_pickle(runs)
    print('\nThank you for using compy, the CoMPASS Python Companion!')
