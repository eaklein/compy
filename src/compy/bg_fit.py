#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:44:58 2022

@author: diesel
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from ImagingReso.resonance import Resonance
import matplotlib.pyplot as plt

from compy import compy_script

# physical constants
J_eV = 6.242e18
eV_J = 1 / J_eV
m_n = 1.675e-27  # kg/mol


def f_gauss(x, a, x0, sigma):
    """Return gaussian function."""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def f_poly(x, a_0, a_1, b):
    """Return polynomial function."""
    return a_0 + a_1 * x**b


def f_exponential(t, b, tau):
    """Return exponential function."""
    return b * np.exp(-t / tau)


def E_to_t(E_lin, l_tof=2.59):
    """Convert neutron energy (eV) to time of flight (us)."""
    return np.sqrt(m_n / (2 * E_lin * eV_J)) * l_tof * 1e6


t_lo = 0.0
t_hi = 200.0
n_bins = 400
t_res = 0.5
t_lin = np.linspace(t_lo, t_hi, int((t_hi - t_lo) / t_res) + 1)
t_lin_mid = t_lin[:-1] + t_res/2

# bg fit
popt_bg, pcov_bg = curve_fit(
    f_exponential,
    [43.7, 67.7, 84.2],
    [5.20*4, 4.56*4, 4.18*4],
    p0=[14, 100],
    bounds=([0.0, 1], [100.0, 1000]),
)
bg_lin = f_exponential(t_lin_mid, *popt_bg)

E_min = 0.5
E_max = 1000.0
E_res = 0.1

l_tof = 2.21
t_offset = 5.56

o_reso = Resonance(energy_min=E_min, energy_max=E_max, energy_step=E_res)
o_reso.add_layer(formula="Cd", thickness=3.5)
o_reso.add_layer(formula="U", thickness=11)
DU_ratio = [0, 0, 0, 0, 0.000005, 0.0025, 0, 0, 0.997495, 0, 0, 0]
o_reso.set_isotopic_ratio(compound="U", element="U", list_ratio=DU_ratio)
trans = o_reso.export(
    output_type="df",
    x_axis="time",
    t_start_us=0,
    time_resolution_us=t_res,
    source_to_detector_m=l_tof,
    y_axis="transmission",
    mixed=True,
)["Total_transmission"].to_numpy()

E_lin = np.linspace(E_min, E_max, int((E_max - E_min) / E_res) + 1)
t_E_lin = E_to_t(E_lin, l_tof) + t_offset

os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")
runs = compy_script.main()

key_ob = "20220517-DU-0mm-merged"

colors = ['blue', 'red']
labels = ['W', 'U']
plt.figure(figsize=(16, 9))
for i, key in enumerate(runs.keys()):
    runs[key].user_filter(e_lo=90, e_hi=135)
    # if key != key_ob:
    #     runs[key].add_trans(runs, key_ob, filtered='user')
    # runs[key].add_kde(filtered="user")
    runs[key].plot_tof(filtered="user", plot_kde=True, add=True,
                       color=colors[i], label=labels[i])
plt.yscale('linear')
plt.tight_layout()
plt.xlim(20, 200)
plt.ylim(0, 45)
plt.plot(t_lin_mid, bg_lin, color='black', label='background fit')
plt.legend()
plt.savefig('/mnt/c/Users/Avram/background-fit.svg')

data_kde = runs[key_ob].data[runs[key_ob].kde_filtered]["CH0"].TOF
norm = len(data_kde) * (t_hi - t_lo) / n_bins / runs[key_ob].t_meas
kde_lin = runs[key_ob].kde.evaluate(t_lin) * norm - bg_lin

popt_ob, pcov_ob = curve_fit(
    f_poly,
    t_lin[100:1800],
    kde_lin[100:1800],
    p0=[2, 20, -0.1],
    bounds=([0.0, 0.0, -10.0], [5.0, 1000, 10.0]),
)

# histogram the neutron pulse
key_pulse = "EJ309-50uA-130kV-5kHz-3_1dc"
n_bins_pulse = 500
t_pulse_lo = 0
t_pulse_hi = 10
data_pulse = runs[key_pulse].data["filtered"]["CH0"].TOF
t_pulse_res = (t_pulse_hi - t_pulse_lo) / n_bins_pulse
hist_pulse, bin_edges_pulse = np.histogram(
    data_pulse,
    range=[t_pulse_lo, t_pulse_hi],
    bins=n_bins_pulse,
)

bin_centers_pulse = bin_edges_pulse[:-1] + t_pulse_res / 2
t_pulse_mid = data_pulse.median()

f_pulse = interp1d(
    bin_centers_pulse, hist_pulse, fill_value=0.0, bounds_error=False
)

popt_pulse, pcov_pulse = curve_fit(f_gauss, t_lin[27:127], kde_lin[27:127])

# t_pulse_lo = -10.0
# t_pulse_hi = 10.0
# t_pulse_res = t_res
# t_pulse_lin = np.linspace(
#     t_pulse_lo, t_pulse_hi, int((t_pulse_hi - t_pulse_lo) / t_res) + 1
# )

f_interp = interp1d(
    t_E_lin,
    f_poly(t_E_lin, *popt_ob) * trans,
    fill_value=0.0,
    bounds_error=False,
)

stdev = 0.938
# f_pulse = f_gauss(t_pulse_lin, 1, 0, stdev)
plt.figure()
plt.plot(t_lin, f_interp(t_lin))
plt.plot(
    t_lin,
    np.convolve(f_interp(t_lin), f_pulse(t_lin), mode="same")
    / sum(f_pulse(t_lin)),
    lw=3,
)

keys = list(runs.keys())[::-1]
labels = ["HDPE", "HDPE/Pb"]
colors = ["blue", "red"]
# multiplier / moderator comparison
plt.figure(figsize=(16, 9))
# for i, key in enumerate(keys):
    # runs[key].user_filter(e_lo=90, e_hi=135)
    # runs[key].plot_tof(
    #     filtered="user",
    #     plot_kde=False,
    #     add=True,
    #     label=labels[i],
    #     color=colors[i],
    # )
plt.plot(t_lin_mid, (hist_mod)/runs["20220519-HDPE-W"].t_meas/t_res - bg_lin,
         lw=2, drawstyle='steps-mid', color='blue', label='HDPE')
plt.plot(t_lin_mid, hist_mult/runs["20220523-PbHDPE-frontcap-1cmback"].t_meas/t_res - bg_lin,
         lw=2, drawstyle='steps-mid', color='red', label='HDPE/Pb')
plt.xlim(15, 200)
plt.yscale("linear")
plt.ylim(0, 45)
plt.xlabel(r"TIME [$\mu$s]", labelpad=10)
plt.ylabel("COUNTS/MINUTE [NORM.]", labelpad=10)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/multiplier-comp-W-flux-bg.svg')

# get mult/mod flux ratio
flux_mod = runs["20220519-HDPE-W"].data["user"]["CH0"].TOF
flux_mult = runs["20220523-PbHDPE-frontcap-1cmback"].data["user"]["CH0"].TOF
hist_mod, bins_ratio = np.histogram(
    flux_mod, range=[t_lo, t_hi], bins=int(n_bins/2)
)
hist_mod = hist_mod/runs["20220519-HDPE-W"].t_meas/(t_res/2) - bg_lin
hist_mult, __ = np.histogram(
    flux_mult, range=[t_lo, t_hi], bins=int(n_bins/2)
)
hist_mult = hist_mult/runs["20220523-PbHDPE-frontcap-1cmback"].t_meas/(t_res/2) - bg_lin
wgt = (
    runs["20220519-HDPE-W"].t_meas
    / runs["20220523-PbHDPE-frontcap-1cmback"].t_meas
)
ratio = [x / y if y != 0 else 0 for x, y in zip(hist_mult, hist_mod)]

# plot flux ratio
plt.figure(figsize=(12, 9))
plt.plot(bins_ratio[:-1], ratio, linestyle="None", marker="s", color="black")
plt.axhline(
    sum(hist_mult[100:180]) / sum(hist_mod[100:180]), lw=2, color="red"
)
plt.xlabel(r"TIME [$\mu$s]", labelpad=10)
plt.ylabel("FLUX RATIO", labelpad=10)
plt.xlim(0, 200)
plt.ylim(0, 2.5)
plt.tight_layout()
plt.savefig('/mnt/c/Users/Avram/multiplier-comp-W-ratio.svg')

# plot kV comparison for all DU
plt.figure(figsize=(16, 9))
for i, key in enumerate(runs.keys()):
    runs[key].user_filter(e_lo=90, e_hi=135)
    runs[key].add_kde(filtered="user")
    runs[key].plot_tof(
        filtered="user",
        plot_hist=True,
        plot_kde=True,
        add=True)
plt.xlim(25, 200)
plt.yscale('linear')
plt.ylim(0, 18)
