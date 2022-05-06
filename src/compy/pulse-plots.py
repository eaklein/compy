#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:15:45 2022
@author: E. Avram Klein
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from bisect import bisect_left
import re
from scipy.signal import savgol_filter

from compy import compy_script
from compy.utilities import merge_runs

# change to NRTA experiments directory
os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")

# set color scheme for plotting
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.inferno(np.linspace(0, 0.8, 6))
)


# fit gaussian to pulses
def gauss(x, A, mu, sigma):
    """Gaussian function."""
    return (A / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def log(x, A, B, C, D):
    """Return logarithmic function."""
    return A + B * np.log(C * (x - D))


def linear(x, m, b):
    """Return linear function."""
    return m * x + b


def calc_TOF(t_pulse, t_signal):
    """Calculate TOF from pulse and signal time arrays."""
    tof = []
    idxs = [-1]
    dbls = []
    for t in t_signal:
        idx = bisect_left(t_pulse, t)
        if idx == len(t_pulse):
            t_0 = t_pulse[-1]
        else:
            t_0 = t_pulse[idx - 1]
        if idx == idxs[-1]:
            dbls[-1] = 1
            dbls.append(1)
        else:
            dbls.append(0)
        idxs.append(idx)
        tof.append((t - t_0) / 1e6)  # convert to ps to us
    return dbls, idxs, tof


# load in experimental data from pickle
runs = compy_script.main()


# fix dead time issue with 50uA 130kV 5% dc run
runs['EJ309-50uA-130kV-5kHz-5_0dc'].t_meas = 51.5/60
runs['EJ309-50uA-120kV-5kHz-3_1dc'].t_meas = 51.5/60


# merge half-runs and remove old keys from dictionary
flag_save = False

key_list = [key for key in runs.keys()]
for key in key_list:
    if key.endswith("actual2"):
        key_new = key[:-1]
        merge_runs(
            [key_new, key], runs, merge_key=key_new, filts=["unfiltered"]
        )
        runs.pop(key)


"""
# print pulse timing for different beam currents at 80 kV and 3.1% DC
plt.figure(figsize=(16, 9))
cts_3 = {}
for key, run in runs.items():
    if key.endswith("80-5-3_1-actual"):
        uA = key.split("LaBr-")[1][:2]
        run.plot_tof(
            t_lo=3,
            t_hi=13,
            label=rf"{uA} $\mu A$, $80$ kV, $3.1\%$ DC",
            add=True,
        )
        cts_3[key] = len(run.data["unfiltered"]["CH0"])
plt.xlim(4, 10)
plt.yscale("linear")
# plt.xlabel(r'TIME [$\mu s$]')
# plt.ylabel(r'COUNTS $\cdot \: s^{-1}$')
if flag_save:
    plt.savefig("P383-training/figures/pulse-80kV-3dc.svg")

# print neutron flux for different beam currents at 80 kV and 3.1% DC
uA = []
y = []
yerr = []
t_meas = 100
for key, cnt in cts_3.items():
    uA.append(int(key.split("LaBr-")[1][:2]))
    y.append(cnt / t_meas)
    yerr.append(np.sqrt(cnt) / t_meas)

plt.figure(figsize=(16, 9))
plt.errorbar(
    x=uA,
    y=y,
    yerr=yerr,
    marker="s",
    ms=8,
    elinewidth=2,
    capsize=2,
    linestyle="None",
    color="black",
)
plt.xlim(18, 57)
plt.ylim(400, 1000)
plt.xlabel(r"BEAM CURRENT [$\mu A$]")
plt.ylabel(r"COUNTS $\cdot \: s^{-1}$")
if flag_save:
    plt.savefig("P383-training/figures/flux-80kV-5dc.svg")

# print pulse timing for different beam currents at 80 kV and 5.0% DC
plt.figure(figsize=(16, 9))
cts_5 = {}
for key, run in runs.items():
    if key.endswith("80-5-5_actual"):
        uA = key.split("LaBr-")[1][:2]
        run.plot_tof(
            t_lo=3, t_hi=13, label=rf"{uA} $\mu$A, $80$ kV, $5\%$ DC", add=True
        )
        cts_5[key] = len(run.data["unfiltered"]["CH0"])
plt.xlim(4, 13)
plt.yscale("linear")
plt.tight_layout()
# plt.xlabel(r'TIME [$\mu s$]')
# plt.ylabel(r'COUNTS $\cdot \: s^{-1}$')
if flag_save:
    plt.savefig("P383-training/figures/pulse-80kV-5dc.svg")

# print pulse timing for different voltages at 35 uA and 3.1% DC
keys = [
    "LaBr-35-80-5-3_1-actual",
    "LaBr-40-80-5-3_1-actual",
    "LaBr-35-100-5-3_1-actual",
    "LaBr-40-100-5-3_1-actual",
]
labels = [
    r"35 $\mu$A, $80$ kV, $3.1\%$ DC",
    r"40 $\mu$A, $80$ kV, $3.1\%$ DC",
    r"35 $\mu$A, $100$ kV, $3.1\%$ DC",
    r"40 $\mu$A, $100$ kV, $3.1\%$ DC",
]

plt.figure(figsize=(12, 9))
for key, label in zip(keys, labels):
    runs[key].plot_tof(t_lo=3, t_hi=13, add=True, label=label)
plt.xlim(4, 8)
plt.yscale("linear")
plt.tight_layout()
plt.legend(loc="upper left")
# plt.xlabel(r'TIME [$\mu s$]')
# plt.ylabel(r'COUNTS $\cdot \: s^{-1}$')
if flag_save:
    plt.savefig("P383-training/figures/pulse-80kV-100kV-3dc.svg")

# t_lo = 0
# t_hi = 200
# n_bins = 500
# plt.figure(figsize=(12, 9))
# for key, label in zip(keys, labels):
#     x = runs[key].data['unfiltered']['CH0'].TOF
#     weights = [1/runs[key].t_meas]*len(x)
#     print(key, len(x), weights[0])
#     plt.hist(x, range=[t_lo, t_hi], bins=n_bins, weights=weights,
#              histtype='step', lw=2,  #color=color,
#              label=label)

# # used to discover many NaN values in TOF for 40 uA, 100kv run
# test = copy.deepcopy(runs[keys[3]].data['unfiltered']['CH0'])
# test_filt = test.loc[~test.TOF.isnull()]


t_lo = 3
t_hi = 13
n_bins = 400
mus_pred = [10.3, 8.8, 7.2, 6.0]
amps_pred = [200, 350, 400, 500]
ranges_pred = [(9.4, 10.9), (7.5, 9.2), (6.0, 7.6), (5.2, 6.4)]

runs['LaBr-40-80-5-5_actual'].data['unfiltered']['CH0'] = runs[
    'LaBr-40-80-5-5_actual'].data['unfiltered']['CH0'].dropna()

i = 0
plt.figure(figsize=(16, 9))
for key, run in runs.items():
    if key.endswith("80-5-5_actual"):
        uA = key.split("LaBr-")[1][:2]
        x = runs[key].data["unfiltered"]["CH0"].TOF.to_numpy()
        hist, bin_edges = np.histogram(
            x,
            range=[t_lo, t_hi],
            bins=n_bins,
        )
        weights = [1 / runs[key].t_meas] * len(x)
        p0 = [
            mus_pred[i],
            (ranges_pred[i][1] - ranges_pred[i][0]) / 3,
            amps_pred[i],
        ]
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        idx_lo = bisect_left(bin_centres, ranges_pred[i][0])
        idx_hi = bisect_left(bin_centres, ranges_pred[i][1])
        weights_hist = [1 / runs[key].t_meas] * len(hist[idx_lo:idx_hi])
        popt, pcov = curve_fit(
            gauss,
            bin_centres[idx_lo:idx_hi],
            hist[idx_lo:idx_hi] * weights_hist,
            p0,
        )
        pcov = np.diag(pcov)
        plt.hist(
            x,
            range=[t_lo, t_hi],
            bins=n_bins,
            weights=weights,
            histtype="step",
            lw=2,
            color=colors[i],
            label=rf"{uA} $\mu$A, $80$ kV, $5\%$ DC",
        )
        plt.plot(
            bin_centres,
            gauss(bin_centres, *popt),
            color=colors[i],
            lw=2,
            label=rf"$\mu = {popt[1]:.2f}$, $\sigma = {popt[2]:.2f}$",
        )
        gauss_lo = bisect_left(bin_centres, popt[1] - 2 * popt[2])
        gauss_hi = bisect_left(bin_centres, popt[1] + 2 * popt[2])
        counts_gauss = sum(hist[gauss_lo:gauss_hi])
        print("\n" f"=========={key}============")
        print(p0)
        print(
            f"Counts: {counts_gauss}"
            "\n"
            f"Mean: {popt[1]:.2f} +/- {np.sqrt(pcov[1]):.2f}"
            "\n"
            f"Sigma: {popt[2]:.2f} +/- {np.sqrt(pcov[2]):.2f}"
        )
        print(key, len(x), weights[0])
        i += 1
plt.xlim(t_lo, t_hi)
plt.xlabel(r"TIME-OF-FLIGHT [$\mu s$]")
plt.ylabel(r"COUNTS / MINUTE")
plt.legend()
plt.tight_layout()
if flag_save:
    plt.savefig("P383-training/figures/pulse_fit-80kV-5dc.svg")
"""

# EJ309 PLOTS #################################################################
plots = {
    "accel-voltage": {
        "selects": [("current", 50), ("duty", 3.1)],
        "filt": "filtered",
        "n_bins": 150,
        "t_lo": 3.5,
        "t_hi": 7.5,
        "x": "voltage",
        "x_lo": 95,
        "x_hi": 135,
        "x_label": r"ACCEL. VOLTAGE [kV]",
        "plot_name": "EJ309-pulse_fit-50uA-3_1dc.svg",
        "plot_name2": "EJ309-flux-50uA-3_1dc.svg",
    },
    "beam-current-3": {
        "selects": [("voltage", 100), ("duty", 3.1)],
        "filt": "filtered",
        "n_bins": 125,
        "t_lo": 3.5,
        "t_hi": 7.5,
        "x": "current",
        "x_lo": 25,
        "x_hi": 60,
        "x_label": r"BEAM CURRENT [$\mu$A]",
        "plot_name": "EJ309-pulse_fit-100kV-3_1dc.svg",
        "plot_name2": "EJ309-flux-100kV-3_1dc.svg",
    },
    "beam-current-5": {
        "selects": [("voltage", 130), ("duty", 5.0)],
        "filt": "filtered",
        "n_bins": 200,
        "t_lo": 3.5,
        "t_hi": 11.5,
        "x": "current",
        "x_lo": 25,
        "x_hi": 70,
        "x_label": r"BEAM CURRENT [$\mu$A]",
        "plot_name": "EJ309-pulse_fit-130kV-5dc.svg",
        "plot_name2": "EJ309-flux-130kV-5dc.svg",
    },
    "duty-cycle-100": {
        "selects": [("current", 55), ("voltage", 100)],
        "filt": "filtered",
        "n_bins": 200,
        "t_lo": 3.5,
        "t_hi": 11.5,
        "x": "duty",
        "x_lo": 3.0,
        "x_hi": 5.1,
        "x_label": r"DUTY CYCLE [\%]",
        "plot_name": "EJ309-pulse_fit-55uA-100kV.svg",
        "plot_name2": "EJ309-flux-55uA-100kV.svg",
    },
    "duty-cycle-130": {
        "selects": [("current", 55), ("voltage", 130)],
        "filt": "filtered",
        "n_bins": 200,
        "t_lo": 3.5,
        "t_hi": 11.5,
        "x": "duty",
        "x_lo": 3.0,
        "x_hi": 5.1,
        "x_label": r"DUTY CYCLE [\%]",
        "plot_name": "EJ309-pulse_fit-55uA-130kV.svg",
        "plot_name2": "EJ309-flux-55uA-130kV.svg",
    },
}


def make_plots(runs, plots):
    """Make plots from CompassRun objects and plot parameters."""
    summary = {}
    for plot, params in plots.items():
        print(f"Plotting {plot}...")
        summary[plot] = {}
        run_params = {}
        counts = {}
        y = []
        yerr = []

        # find matching runs in dictionary
        keys_selected = []
        for key in runs.keys():
            run_params[key] = {}
            (
                run_params[key]["det"],
                run_params[key]["current"],
                run_params[key]["voltage"],
                run_params[key]["frequency"],
                run_params[key]["duty"],
            ) = [
                float(re.sub("[^0-9.]", "", x.replace("_", ".")))
                if x[0].isdigit()
                else x
                for x in key.split("-")
            ]
            if all(
                [
                    run_params[key][param] == val
                    for param, val in params["selects"]
                ]
            ):
                keys_selected.append(key)
        runs_selected = {k: runs[k] for k in keys_selected}
        print(f"selected keys: {keys_selected}")

        # fit and plot matching runs
        colors = plt.cm.inferno(np.linspace(0, 0.8, len(keys_selected)))
        plt.figure(figsize=(16, 9))
        for i, (key, run) in enumerate(runs_selected.items()):
            run_param = run_params[key]
            t_lo = params["t_lo"]
            t_hi = params["t_hi"]
            n_bins = params["n_bins"]
            val = run_param[params["x"]]
            run.plot_tof(
                filtered=params["filt"],
                t_lo=t_lo,
                t_hi=t_hi,
                n_bins=n_bins,
                color=colors[i],
                label=rf'{run_params[key]["current"]:.0f} $\mu$A, '
                rf'{run_params[key]["voltage"]:.0f} kV, '
                rf'{run_params[key]["duty"]:.1f}\% DC',
                add=True,
            )
            xvals = run.data[params["filt"]]["CH0"].TOF.to_numpy()
            counts[val] = len(xvals)
            y.append(counts[val] / run.t_meas)
            yerr.append(np.sqrt(counts[val]) / run.t_meas)
            # fit Gaussian
            hist, bin_edges = np.histogram(
                xvals,
                range=[t_lo, t_hi],
                bins=n_bins,
            )
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            argmax = np.argmax(hist)
            valmax = bin_centres[argmax]
            idx_lo = (np.abs(0.3 * valmax - hist[:argmax])).argmin()
            idx_hi = (np.abs(0.7 * valmax - hist[argmax:])).argmin() + argmax
            # print(argmax, valmax, idx_lo, idx_hi)
            p0 = [
                valmax,
                1,
                np.max(hist),
            ]
            weights_hist = [1 / runs[key].t_meas] * len(hist[idx_lo:idx_hi])
            popt, pcov = curve_fit(
                gauss,
                bin_centres[idx_lo:idx_hi],
                hist[idx_lo:idx_hi] * weights_hist,
                p0,
            )
            pcov = np.diag(pcov)
            plt.plot(
                bin_centres,
                gauss(bin_centres, *popt),
                lw=2,
                color=colors[i],
                label=rf"$\mu = {popt[1]:.2f}$, $\sigma = {popt[2]:.2f}$",
            )
        plt.xlim(params["t_lo"], params["t_hi"])
        plt.yscale("linear")
        plt.legend()
        plt.tight_layout()
        if flag_save:
            plt.savefig(
                "P383-training/figures/" + params["plot_name"], dpi=800
            )

        # plot neutron flux
        x = list([float(key) for key in counts.keys()])
        plt.figure(figsize=(16, 9))
        plt.errorbar(
            x=x,
            y=y,
            yerr=yerr,
            marker="s",
            ms=8,
            elinewidth=2,
            capsize=2,
            linestyle="None",
            color="black",
        )
        x_lin = np.linspace(params["x_lo"], params["x_hi"], 1001)
        popt_lin, pcov_lin = curve_fit(
            linear,
            x,
            y,
            p0=[y[0], (y[1]-y[0])/(x[1]-x[0])]
        )
        plt.plot(
            x_lin,
            linear(x_lin, *popt_lin),
            color='red',
            label=rf'{popt_lin[0]:1.2e}*x {popt_lin[1]:+1.2g}'
        )
        try:
            popt_log, pcov_log = curve_fit(
                log,
                x,
                y,
                p0=[popt_lin[1], popt_lin[0], 1, 1])
            pcov_log = np.diag(pcov_log)
            plt.plot(
                x_lin,
                log(x_lin, *popt_log),
                color='blue',
                label=rf'{popt_log[0]:1.2e} {popt_log[1]:+1.2g}'
                rf'*log({popt_log[2]:1.2e}*(x {-1*popt_log[3]:+1.2g}))'
            )
        except:
            print('Could not fit log function!')
        plt.xlim(params["x_lo"], params["x_hi"])
        # plt.ylim(400, 1000)
        plt.xlabel(params["x_label"])
        plt.ylabel(r"COUNTS / MINUTE")
        plt.legend()
        plt.tight_layout()
        if flag_save:
            plt.savefig(
                "P383-training/figures/" + params["plot_name2"], dpi=800
            )
        # save info to summary dict
        summary[plot]["x"] = list(counts.keys())
        summary[plot]["y"] = y
        summary[plot]["yerr"] = yerr
    return summary


plt.close("all")
summary = make_plots(runs, plots)

# colors = ['k', 'b', 'r', 'g', 'm']
# plt.figure(figsize=(16, 9))
# for i, (key, val) in enumerate(summary.items()):
#     x = np.arange(0, len(val['x']))
#     plt.errorbar(x=x, y=val['y'], yerr=val['yerr'], color=colors[i], label=key)
# plt.legend()

from scipy.interpolate import interp1d
from scipy.signal import convolve

t_lo = 0
t_hi = 200
t_res = 0.1
t_lin = np.linspace(t_lo, t_hi, int((t_hi-t_lo)/t_res)+1)
f = interp1d(np.histogram(runs['']))
exp_pulse = f(t_lin)
convolved = convolve(trans(t_lin), exp_pulse, mode='full')[:len(t_lin)] \
    / sum(exp_pulse)
