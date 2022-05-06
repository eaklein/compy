#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:20:22 2022

@author: diesel
"""

import os
import matplotlib.pyplot as plt
from compy import compy_script


os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")
runs= compy_script.main()
keys = ['20220504-W-3_5-actual']

plt.figure()
for key in keys:
    data = runs[key].data['unfiltered']['CH0']
    # plt.hist(data.TOF, range=[0, 192], bins=512, histtype='step')
    plt.hist(data.ENERGY, range=[0, 500], bins=250, histtype='step')
