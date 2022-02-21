#!/usr/bin/env python3

"""
A command line script for processing CoMPASS data
"""

import numpy as np
import matplotlib.pyplot as plt 

from compy import compassrun

def main():
    folders = ['/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/'
		+ 'Experiments/du-studies/DAQ/']
    folder, key_tuples, VERBOSE = compassrun.initialize(folders=folders)
    runs = compassrun.process_runs(key_tuples)
    compassrun.merge_related_runs(runs, quiet=True)

    # plot filtered TOF spectra for all keys
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
    plt.xlim(25, 185)
    plt.xlabel(r'TIME [$\mu$s]')
    plt.ylabel('COUNTS/MINUTE')
    plt.ylim(0, 3.5)
    plt.legend()
    plt.tight_layout()
