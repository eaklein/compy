#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:20:22 2022

@author: diesel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from compy import compy_script
from scipy.stats import gaussian_kde

os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")
runs = compy_script.main()
# keys = ['20220506-55uA-130kV-3dc-DU-5-actual']
# keys = ['20220510-55uA-130kV-3dc-DU-10-actual-merged']
keys = ['20220518-DU-1-merged',
        '20220518-DU-3',
        '20220518-DU-5',
        '20220518-DU-7',
        '20220518-DU-9',
        '20220518-DU-11']

plt.figure()
for i, key in enumerate(keys):
    data = runs[key].data['filtered']['CH0']
    if key.endswith('merged'):
        label = key[:-7].rsplit('-')[-1] + ' mm'
    else:
        label = key.rsplit('-')[-1] + ' mm'
    hist_raw, bin_edges = np.histogram(data.TOF, range=[0, 192], bins=512)
    yerr = np.sqrt(hist_raw) / runs[key].t_meas 
    hist = hist_raw / runs[key].t_meas 
    print(label, sum(hist[170:190]))
    bin_centers = bin_edges[1:] + (bin_edges[1] - bin_edges[0])/2
    plt.errorbar(bin_centers, hist + 5*(len(runs.keys()) - i), yerr=yerr,
             drawstyle='steps-mid', label=label)
    # plt.hist(data.ENERGY, range=[0, 500], bins=250, histtype='step')
plt.xlim(20, 180)
plt.xscale('log')
plt.ylim(0, 50)
plt.xlabel(r'TIME-OF-FLIGHT [$\mu$s]')
plt.ylabel('COUNTS PER MINUTE')
plt.legend(loc='upper right')

J_eV = 6.242E18
eV_J = 1/J_eV
m_n = 1.675E-27  # kg/mol


def t_to_E(t_lin, l_tof=2.59, t_0=0.00):
    """Convert time of flight (us) to neutron energy (eV)."""
    E_TOF = np.zeros_like(t_lin)
    for i, t in enumerate(t_lin):
        if t > 0:
            E_TOF[i] = (J_eV*(m_n/2)*(l_tof/((t-t_0)/1e6))**2)
        else:
            E_TOF[i] = 1e9
    return E_TOF


def E_to_t(E_lin, l_tof=2.59, t_0=0.00):
    """Convert neutron energy (eV) to time of flight (us)."""
    return np.sqrt(m_n/(2*E_lin*eV_J))*l_tof*1E6 + t_0


# key = '20220506-55uA-130kV-3dc-DU-5-actual'
key = '20220518-DU-9'

t_lo = 20
t_hi = 192
t_res = 0.10
n_bins = int((t_hi - t_lo)/t_res) + 1
l_tof = 2.23
t_0 = 5.50
t_lin = np.linspace(t_lo, t_hi, int((t_hi - t_lo)/t_res) + 1)

x_rebin = []
y_rebin = []

t_bins = t_edges[:-1] + (t_edges[1] - t_edges[0])/2
e_edges = t_to_E(t_edges, l_tof=l_tof, t_0=t_0)
e_widths = [i-j for i, j in zip(e_edges, e_edges[1:])]
e_bins = t_to_E(t_bins, l_tof=l_tof, t_0=t_0)
e_lo = e_bins[-1]
e_hi = e_bins[0]
e_res = 0.1
e_lin = np.linspace(e_lo, e_hi, int((e_hi - e_lo)/e_res) + 1)
t_e_lin = E_to_t(e_lin, l_tof=l_tof)

from cycler import cycler
cmap=plt.cm.magma

mode = 't'
keys = ['20220518-DU-1-merged',
        '20220518-DU-3',
        '20220518-DU-5',
        '20220518-DU-7',
        '20220518-DU-9',
        '20220518-DU-11']
c = cycler('color', cmap(np.linspace(0.3, 0.9, len(keys))[::-1]))
plt.rcParams["axes.prop_cycle"] = c
plt.figure(figsize=(16, 9))
for i, key in enumerate(keys):
    if key.endswith('merged'):
        label = key[:-7].rsplit('-')[-1] + ' mm DU'
    else:
        label = key.rsplit('-')[-1] + ' mm DU'
    data = runs[key].data['filtered']['CH0'].copy(deep=True)
    kde = gaussian_kde(data.TOF, bw_method=0.0005)
    hist_tof, t_edges = np.histogram(data.TOF, range=[t_lo, t_hi], bins=n_bins)
    y = hist_tof
    if mode == 'e':
        plt.xlim(e_lo, e_hi)
        plt.xlabel(r'ENERGY [eV]', labelpad=10)
        plt.ylabel(r'COUNTS [NORM.]', labelpad=10)
        x = e_bins
        yerr = np.sqrt(y)/e_widths
        y = y/e_widths
        plt.plot(x, y, 
                  drawstyle='steps-mid', lw=1, color='black', label='Experiment')
        plt.plot(e_lin, kde.evaluate(t_e_lin)*sum(data.TOF)/(t_hi - t_lo)/t_res, 
                 lw=2, color='blue', label='kde')
    elif mode == 't':
        plt.xlim(t_lo, t_hi)
        plt.xlabel(r'TIME [$\mu$s]', labelpad=10)
        plt.ylabel(r'COUNTS PER MINUTE [NORM.]', labelpad=10)
        x = t_bins
        yerr = np.sqrt(y)/t_res
        y = y/t_res/runs[key].t_meas  # normalize counts per microsecond
        # plt.plot(x, y, 
        #           drawstyle='steps-mid', lw=1, color='black', label='Experiment')
        plt.plot(t_lin, 
                 kde.evaluate(t_lin)*sum(data.TOF)/(t_hi - t_lo)/2.84/t_res/runs[key].t_meas, 
                 lw=2,
                 alpha= 0.7 + 0.3*i/len(keys),
                 # color='blue', 
                 label=label)
                 # label='Experiment (KDE)')

for i, E_res in enumerate([6.67, 20.87, 36.68, 66.03, 80.75]):
    t_reso = E_to_t(E_res, l_tof=l_tof, t_0=t_0)
    plt.axvline(t_reso, color='black', lw=2)
    t = plt.text(t_reso - 2.0, 4.5 + 1*i, f'{E_res} eV', color='black')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='black'))
t_reso = E_to_t(102.56, l_tof=l_tof, t_0=t_0)
plt.axvline(t_reso, color='black', lw=2, label=r'U-238 resonances')
t = plt.text(t_reso - 1.0, 10, '102.56 eV', color='black')
t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='black'))
for i, E_res in enumerate([27.57, 66.78, 84.91, 89.55]):
    t_reso = E_to_t(E_res, l_tof=l_tof, t_0=t_0)
    plt.axvline(t_reso, lw=2, color='gray')
t_reso = E_to_t(99.491, l_tof=l_tof, t_0=t_0)
plt.axvline(t_reso, color='gray', lw=2, label=r'Cd resonances')
plt.ylim(4, 50)
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/Desktop/DU-10mm.svg')

"""
data_new = runs_new['du-all'].data['filtered']['CH0']
hist_tof_new, t_edges = np.histogram(data_new.TOF, 
                                     range=[t_lo, t_hi], 
                                     bins=n_bins)
plt.xlim(t_lo, t_hi)
plt.xlabel(r'TIME [$\mu$s]', labelpad=10)
plt.ylabel(r'COUNTS [$\mu s^{-1}$]', labelpad=10)
x = t_bins
yerr = np.sqrt(y)/t_res
y = y/t_res
plt.plot(x, y, 
          drawstyle='steps-mid', lw=1, color='green', label='Expt. (hist)')

plt.figure(figsize=(16, 9))
for key in keys:
    plt.hist(data.TOF, range=[20, 192], bins=172*4, histtype='step',
             weights=np.ones_like(data.TOF)/10, color='blue', label='P383')
    plt.hist(data_new.TOF, range=[20, 192], bins=172*4, histtype='step',
             weights=np.ones_like(data_new.TOF)/60, color='red', label='A325')
plt.xlim(20, 192)
# plt.yscale('log')
plt.xlabel(r'TIME [$\mu$s]', labelpad=10)
plt.ylabel(r'COUNTS [$\mu s^{-1}$]', labelpad=10)
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/Desktop/flux-comparison-direct.svg')

plt.figure()
plt.hist(data.TIMETAG, histtype='step', bins=4200, color='red')
plt.hist(data_new.TIMETAG, histtype='step', bins=3600, color='blue')
"""
