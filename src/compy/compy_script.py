#!/usr/bin/env python3

"""
A command line script for processing CoMPASS data
"""
import os
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
        print(f"Folders specified: {folders}")
    else:
        folders = None

    # process data
    pkl_flag = click.confirm(
        "\nWould you like to load data from pickle?", default=False
    )
    if pkl_flag:
        runs = utilities.load_pickle()
    else:
        key_tuples, VERBOSE = compassrun.initialize(folders=folders)
        runs = compassrun.process_runs(key_tuples)
        merge_flag = click.confirm(
            "\nWould you like to merge runs?", default=True
        )
        if merge_flag:
            utilities.merge_related_runs(runs, quiet=True)

    # plot filtered TOF spectra for all keys
    print_flag = click.confirm(
        "\nWould you like to plot the (filtered) spectra?", default=False
    )
    if print_flag:
        plt.figure(figsize=(16, 9))
        for key in runs.keys():
            print(key)
            if ("TOF" in runs[key].spectra["filtered"]) and (
                "vals" in runs[key].spectra["filtered"]["TOF"]
            ):
                vals_raw = np.array(
                    runs[key].spectra["filtered"]["TOF"]["vals"]
                )
                bins = np.array(runs[key].spectra["filtered"]["TOF"]["bins"])
                t = runs[key].t_meas
                print("plotting key: ", key, t, sum([i for i in vals_raw]))
                vals_err = np.sqrt(vals_raw) / t
                vals = vals_raw / t
                plt.errorbar(
                    x=bins,
                    y=vals,
                    yerr=vals_err,
                    marker="s",
                    linestyle="None",
                    drawstyle="steps-mid",
                    label=key.replace("_", "-"),
                )
            else:
                print(f'Did not find TOF data for key: {key}.')
        if len(runs.keys()) > 0:
            plt.xlim(25, 185)
            plt.xlabel(r"TIME [$\mu$s]")
            plt.ylabel("COUNTS/MINUTE")
            # plt.ylim(0, 3.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No spectra found to plot!")

    trans_flag = click.confirm(
        "\nWould you like to calculate transmission?", default=False
    )
    if trans_flag:
        keys = list(runs.keys())
        print("\nProcessed keys are", f"{keys}")
        key_ob = input("Which key would you like to use for open beam?\n")

        # add transmission
        [
            runs[key].add_trans(runs, key_ob, t_offset=0)
            for key in keys
            if key != key_ob
        ]

        # plot transmission for target runs
        trans_plot_flag = click.confirm(
            "\nWould you like to plot transmission?", default=False
        )
        if trans_plot_flag:
            [
                runs[key].plot_trans(t_offset=5.56)
                for key in keys
                if key != key_ob
            ]

    # save data to pickle
    save_flag = click.confirm(
        "\nWould you like to save the runs as a pickle?", default=False
    )
    if save_flag:
        utilities.save_pickle(runs)
    print("\nThank you for using compy, the CoMPASS Python Companion!")

    return runs


if __name__ == "__main__":
    os.chdir("/mnt/c/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/experiments/")
    runs = main()
