# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:21:38 2022
@author: Avram
"""

from bisect import bisect_left
from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import click
import pickle
from scipy.stats import gaussian_kde


def merge_copy(d1, d2):
    """Merge nested dicts."""
    return {
        k: merge_copy(d1[k], d2[k])
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict)
        else merge_vals(d1[k], d2[k])
        for k in d2
    }


def merge_vals(x, y):
    """Combine keys in dict."""
    if x == y:
        return x
    if isinstance(x, list) and isinstance(y, list):
        return [*x, *y]
    return [x, y]


def merge_runs(
    keys, runs, filts=["unfiltered", "filtered"], merge_key="", quiet=False
):
    """Merge data from CoMPASS runs.

    Parameters
    ----------
        keys : list[str]
            list of keys to check for merging
        runs : dict[CompassRun]
            dictionary of runs from which to search for keys
        merge_key : str
            run key to use for the merged CompassRun
        quiet : bool
            flag for whether to prompt user in case of diff settings (False)
            or to use defaults (True)
    """
    # choose key for merged run
    if merge_key == "":
        merge_key = click.prompt(
            "Which key should be used for the merged run?"
        )
    # initialize merged run with first run provided
    run_merged = deepcopy(runs[keys[0]])
    run_merged.key = merge_key
    # iterate over additional keys to merge
    for key in keys[1:]:
        keep_settings = True
        run = deepcopy(runs[key])
        # check settings
        if run.settings != run_merged.settings:
            [
                print(
                    f"Different settings found for {x[0]} ({x[1]} vs. {y[1]})"
                )
                for x, y in zip(
                    run.settings.items(), run_merged.settings.items()
                )
                if x != y
            ]
            if not quiet:
                keep_settings = click.confirm(
                    "Would you like to keep the settings for the merged key?",
                    default="Y",
                )
            else:
                keep_settings = True
            if not keep_settings:
                run_merged.settings = deepcopy(run.settings)
        # check folder
        run_merged.folder = [
            [run.folder, run_merged.folder]
            if run.folder != run_merged.folder
            else run.folder
        ]
        # check if spectral parameters are equal
        if not all(
            [
                run.params["E"] == run_merged.params["E"],
                run.params["TOF"] == run_merged.params["TOF"],
            ]
        ):
            print(
                "Spectra parameters are not the same for "
                f"{run.key} and {run_merged.key}.\n"
                f"Will keep spectra parameters from {run_merged.key}"
                " but will not store any spectra."
            )
        else:
            # if yes, merge spectra
            print(f"Merging spectra for {run.key} and {run_merged.key}.")
            run_merged.spectra = {}
            for filtered in filts:
                run_merged.spectra[filtered] = {}
                for key in (
                    run.spectra[filtered].keys()
                    and run_merged.spectra[filtered].keys()
                ):
                    run_merged.spectra[filtered][key] = [
                        xi + yi
                        for xi, yi in zip(
                            run.spectra[filtered][key],
                            run_merged.spectra[filtered][key],
                        )
                    ]
        # merge params
        print(f"Merging parameters for {run.key} and {run_merged.key}.")
        t_meas = run_merged.t_meas
        if not keep_settings:
            run_merged.params = deepcopy(run.params)
        # run_merged.params = merge_copy(run.params, run_merged.params)
        run_merged.t_meas = t_meas
        run_merged.t_meas += run.t_meas
        # merge data
        for filtered in filts:
            for r in [run, run_merged]:
                if "TOF" not in r.data[filtered]["CH0"]:
                    r.add_tof(filtered=[filtered])
            run_merged.data[filtered]["CH0"] = pd.concat(
                [run.data[filtered]["CH0"], run_merged.data[filtered]["CH0"]],
                axis=0,
            )
            print(
                key,
                len(run_merged.data[filtered]["CH0"].index),
                len(run.data[filtered]["CH0"].index),
            )
    run_merged.add_spectra(filtered=filts)
    runs[run_merged.key] = run_merged
    return run_merged


def merge_related_runs(runs, quiet=False):
    """Get list of total run basenames.

    Parameters
    ----------
        runs : dict{CompassRun}
            dictionary of compass runs to search for related runs
        quiet : bool
            flag for whether to prompt user in case of diff settings (False)
            or to use defaults (True)
    -------
    """
    # merge all runs that only differ by end underscore
    stem_keys = list(runs.keys())
    # generate dictionary with stems as keys and run names as values
    key_stems = {}
    for stem_key in stem_keys:
        key_split = stem_key.rsplit("_", maxsplit=1)
        if (len(key_split) > 1) and (key_split[-1].isdigit()):
            stem = key_split[0]
            if key_split[0] not in key_stems:
                key_stems[stem] = []
            key_stems[stem].append(stem_key)
        else:
            key_stems[stem_key] = [stem_key]
    for stem, keys in key_stems.items():
        if len(keys) > 1:
            print("\nMerging runs:", ", ".join(str(i) for i in keys))
            merge_runs(keys, runs, merge_key=stem + "-merged", quiet=quiet)
            [runs.pop(key) for key in keys]


def calc_TOF(t_pulse, t_signal):
    """Calculate TOF from pulse and signal time arrays."""
    tof = []
    idx = 1
    for t in t_signal:
        idx = bisect_left(t_pulse, t)
        if idx == len(t_pulse):
            t_0 = t_pulse.iloc[-1]
        elif idx == 0:
            t_0 = 0
            print("\nWARNING: count detected before first pulse!")
        else:
            t_0 = t_pulse[idx - 1]
        tof.append((t - t_0) / 1e6)  # convert to ps to us
    return tof


def calc_trans(counts_in, counts_out, t_meas_in, t_meas_out):
    """Calculate transmission and propagate error."""
    vals_trans = [
        (x / t_meas_in) / (y / t_meas_out) if y != 0 else 0
        for x, y in zip(counts_in, counts_out)
    ]
    err_trans = [
        (x / t_meas_in)
        / (y / t_meas_out)
        * np.sqrt((1 / np.sqrt(x)) ** 2 + (1 / np.sqrt(y)) ** 2)
        if y != 0
        else 0
        for x, y in zip(counts_in, counts_out)
    ]
    return np.array(vals_trans), np.array(err_trans)


def calc_atten(
    data,
    thick,
    err_thick=None,
    keys=None,
    key_ref="target_out",
    bin_lo=90,
    bin_hi=135,
):
    """Calc transuation in number of counts."""
    if err_thick is None:
        err_thick = {}
    if keys is None:
        keys = list(data.keys())
        keys.remove(key_ref)
    if err_thick == {}:
        err_thick = {key: 0.0 for key in keys}
    trans = {}
    err_trans = {}
    mu = {}
    err_mu = {}
    c_in = sum(data[key_ref][bin_lo:bin_hi])
    for key in keys:
        c_out = sum(data[key][bin_lo:bin_hi])
        trans[key] = c_out / c_in
        err_trans[key] = trans[key] * np.sqrt(1 / c_out + 1 / c_in)
        mu[key] = -1 * np.log(trans[key]) / thick[key]
        err_mu[key] = np.sqrt(
            (err_trans[key] / trans[key]) ** 2
            + (err_thick[key] / thick[key]) ** 2
        )
        print(
            f"{key}:".ljust(20) + f"mu = {mu[key]*10:.2f} "
            f"+/- {err_mu[key]*10:.2f}" + " [cm-1]"
        )
    return (trans, mu), (err_trans, err_mu)


def plot_2d(runs, key, var1, var2, filtered="unfiltered"):
    """Make a 2D plot for two variables."""
    run = runs[key]
    data = run.data[filtered]["CH0"]
    data["PSD"] = 1 - (data["ENERGYSHORT"] / data["ENERGY"])
    plt.figure(figsize=(16, 9))
    x = data[var1]
    y = data[var2]
    plt.scatter(x, y, s=0.1, c="b", label=key)
    plt.xlim(left=0)
    plt.ylim([0.0, 1.0])
    plt.xlabel(var1, labelpad=10)
    plt.ylabel(var2, labelpad=10)
    plt.legend()


def plot_e_psd(runs, key, filter_params=None, w=1.0):
    """Plot energy vs PSD."""
    if filter_params is None:
        filter_params = {}
    data = runs[key].data["unfiltered"]["CH0"]
    data["PSD"] = 1 - (data["ENERGYSHORT"] / data["ENERGY"])
    data = df_filter(data, filter_params)
    heatmap, xedges, yedges = np.histogram2d(
        data["ENERGY"] * w, data["PSD"], bins=400, range=[[0, 4000], [0, 1]]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cmap = mpl.cm.get_cmap("plasma").copy()
    cmap.set_under(color="white")
    plt.figure(figsize=(12, 12))
    plt.imshow(
        heatmap.T,
        extent=extent,
        origin="lower",
        vmin=0.5,
        vmax=25,
        cmap=cmap,
        aspect=2000,
    )
    plt.xlabel("Energy [ADC]")
    plt.ylabel("PSD")
    plt.tight_layout()


def df_filter(df, filter_params):
    """Filter parameter by values."""
    df = df.copy()
    for key, value in filter_params.items():
        if value != ("0.0", "0.0"):
            df = df.loc[
                (df[key] >= float(value[0])) and (df[key] <= float(value[1]))
            ]
    return df


def plot_trans(
    runs,
    key_target,
    key_open,
    t_lo=0.0,
    t_hi=200.0,
    n_bins=400,
    t_offset=10.0,
    color="black",
    plot_kde=False,
    color_kde="blue",
    kde_bw=0.005,
    add_plot=False,
):
    """Calculate transmission and plot."""
    target_in = runs[key_target].data["filtered"]["CH0"]["TOF"]
    target_out = runs[key_open].data["filtered"]["CH0"]["TOF"]
    t_meas_in = runs[key_target].t_meas
    t_meas_out = runs[key_open].t_meas
    counts_in, __ = np.histogram(target_in, bins=n_bins, range=[t_lo, t_hi])
    counts_out, __ = np.histogram(target_out, bins=n_bins, range=[t_lo, t_hi])
    bins = np.linspace(t_lo, t_hi, n_bins + 1)[:-1] + (
        (t_hi - t_lo) / n_bins / 2 - t_offset
    )
    vals_trans, vals_errs = calc_trans(
        counts_in, counts_out, t_meas_in, t_meas_out
    )
    lw = 2
    if not add_plot:
        plt.figure(figsize=(16, 9))
    if plot_kde:
        kde_in = gaussian_kde(target_in, bw_method=kde_bw)
        kde_out = gaussian_kde(target_out, bw_method=kde_bw)
        t_lin = np.linspace(t_lo, t_hi, int((t_hi - t_lo) / 0.1) + 1)
        trans_kde, __ = calc_trans(
            kde_in.evaluate(t_lin) * len(target_in),
            kde_out.evaluate(t_lin) * len(target_out),
            t_meas_in,
            t_meas_out,
        )
        plt.plot(
            t_lin - t_offset,
            trans_kde,
            lw=2,
            color=color_kde,
            label=key_target + " transmission (KDE)",
            zorder=10,
        )
        lw = 1
    plt.errorbar(
        x=bins,
        y=vals_trans,
        yerr=vals_errs,
        lw=lw,
        drawstyle="steps-mid",
        elinewidth=0.5,
        capsize=1,
        color=color,
        label=key_target + " transmission",
    )
    plt.xlim([max(t_offset, max(0, t_lo)), t_hi - t_offset])
    plt.ylim(0, 2)
    plt.xlabel(r"TIME [$\mu$s]", labelpad=10)
    plt.ylabel(r"TRANSMISSION", labelpad=10)
    plt.legend()
    plt.tight_layout()
    return vals_trans, vals_errs, bins


def load_pickle(fname=None):
    """Save dictionary of CoMPASS runs as pickle."""
    print(f"The current directory is: {os.path.dirname(__file__)}")
    print(
        "Available pickle files are:",
    )
    if fname is None:
        fname = input("\nWhat is the name of the pickle file to load?:\n")
    if not fname.endswith(".pkl"):
        fname += ".pkl"
    with open(fname, "rb") as file:
        data = pickle.load(file)
    return data


def save_pickle(runs, fname=None):
    """Save dictionary of CoMPASS runs as pickle."""
    if fname is None:
        fname = input("What filename would you like to store pickle?:\n")
    if not fname.endswith(".pkl"):
        fname += ".pkl"
    if not runs.keys():
        print("No runs found!")
        return
    folder_default = Path(runs[list(runs.keys())[0]].folder).parent
    folder_save = click.confirm(
        f"Would you like to save to {folder_default}?", default="Y"
    )
    if folder_save:
        fname = folder_default / fname
    else:
        fname = (
            input("What folder would you like to save to?:\n") + "/" + fname
        )
    with open(fname, "wb") as file:
        pickle.dump(runs, file, protocol=pickle.HIGHEST_PROTOCOL)
