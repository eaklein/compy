#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:16:06 2022

@author: diesel
"""

import matplotlib.pyplot as plt
import numpy as np

# import compy_script

# runs = compy_script.main()
# runs_old = compy_script.main()

# runs['w-single-empty'] = runs_old['w-single-empty']

colors = ['red', 'blue', 'black']
labels = ['HDPE', 'HDPE/Pb', 'HDPE (old)']

plt.figure(figsize=(16,9))
for i, run in enumerate(runs.values()):
    # run.add_tof()
    data = run.data['filtered']['CH0'].TOF
    plt.hist(data, bins=512, range=[0, 192],
             color=colors[i], histtype='step', label=labels[i], lw=2,
             weights=np.ones_like(data)/run.t_meas)
plt.xlabel(r'TOF [$\mu$s]')
plt.ylabel('COUNTS')
plt.xlim(20, 192)
plt.ylim(0, 8)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/Desktop/TOF-linear.svg')

plt.figure(figsize=(16,9))
for i, run in enumerate(runs.values()):
    # run.add_tof()
    data = run.data['filtered']['CH0'].TOF
    plt.hist(data, bins=512, range=[0, 192],
             color=colors[i], histtype='step', label=labels[i], lw=2,
             weights=np.ones_like(data)/run.t_meas)
plt.xlabel(r'TOF [$\mu$s]')
plt.ylabel('COUNTS')
plt.yscale('log')
plt.xlim(0, 192)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/Desktop/TOF-log.svg')

plt.figure(figsize=(12,9))
for i, run in enumerate(runs.values()):
    # run.add_tof()
    data = run.data['unfiltered']['CH0'].ENERGY
    plt.hist(data, bins=200, range=[0, 200],
             color=colors[i], histtype='step', label=labels[i], lw=2,
             weights=np.ones_like(data)/run.t_meas)
plt.xlim(0, 200)
plt.yscale('log')
plt.xlabel('ENERGY [ADC]')
plt.ylabel('COUNTS')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/Desktop/energy.svg')
