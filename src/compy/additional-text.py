# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:43:56 2022

@author: Avram
"""


if __name__ == '__main__':
    folder, keys, verbose = initialize(folder=folder)
    # set verbose mode
    VERBOSE = True
    if VERBOSE:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None # do-nothing function
    # verboseprint = print(*a, **k) if verbose else lambda *a, **k: None
    # process CoMPASS files, reqs .CSV saving
    runs = process_runs(keys, folder=folder, verbose=verbose)

    bool_ex = False

    """ EXAMPLES """

    if bool_ex:
        # choose key
        key = 'U1Pb9'
        run = runs[key]

        # Manual filter on neutron peak in energy spectrum
        run.user_filter()

        # process another run and add to dictionary
        keys_new = ['W_nozzle', 'W_part']
        runs = process_runs(keys_new, runs=runs)

        # merge two runs
        # include runs dictionary to add merged run
        run_merged = merge_runs(keys=keys_new, runs=runs)

        runs[keys_new[0]].plot_tof('unfiltered', add=False)
        runs[keys_new[1]].plot_tof('unfiltered', add=True)
        runs['merged'].plot_tof('unfiltered', add=True)

        # Manual adition of TOF if not done during initial processing
        run.add_tof()   # manually add TOF for run

        # plot pulse area for all counts
        run.plot_spectrum(mode='E', filtered='unfiltered')

        # compare manually calculated TOF spectrum for all / neutron counts
        # use add to plot on existing figure
        run.plot_tof(filtered='user', add=False)
        run.plot_tof(filtered='filtered', add=True)

        # plot TOF for all keys overlaid
        plt.figure(figsize=(16, 9))
        for key in [
            # 'w-double', 'w-double-empty', 'w-double-nolead',
            # 'w-double-nolead_1', 'w-single', 'w-single-empty', 'w-single_1']:
            'w-single', 'w-double'
        ]:
            # use 'norm' to normalize to meas. time in min.
            runs[key].plot_tof(norm=True, filtered='filtered', add=True)

        for i, key in enumerate(runs):
            run = runs[key]
            if i != 0:
                # use add to plot on existing figure
                run.plot_spectrum(filtered='filtered', add=True)
            else:
                # use add to plot on existing figure
                run.plot_spectrum(filtered='filtered', add=False)

"""
filtered = 'filtered'
key = 'U_5_B4C-backfull_all'
y = runs[key].data['filtered']['CH0']['TOF']
yvals, __ = np.histogram(y, bins=800, range=[0, 200])
yvals = yvals.astype(float)
yerr = np.sqrt(yvals)
yvals /= runs[key].params['t_meas']
yerr /= runs[key].params['t_meas']
plt.figure(figsize=(16, 9))
weights = [1/runs[key].params['t_meas']]*len(y)
plt.ylabel('COUNTS/MINUTE')
plt.errorbar(np.linspace(0, 200, 801)[:-1], yvals, yerr=yerr,
             color='black', capsize=1, elinewidth=0.5, drawstyle='steps-mid',
             label='5 mm DU', lw=2)
plt.xlim(0, 200)
plt.yscale('log')
plt.legend(loc='upper right')
plt.xlabel('TIME (us)')
plt.tight_layout()
"""

# merge all runs that only differ by end underscore
key_stems = list({key.rsplit('_', maxsplit=1)[0] for key in keys})
key_stems.sort(key=len)
keys_all = []

if 'w-double-nolead' in key_stems:
    key_stems.remove('w-double-nolead')  # fix issue with double-nolead
    for key in keys:
        if key.startswith('w-double-nolead'):
            keys.remove(key)
stemkeys = list(keys)
for stem in key_stems[::-1]:
    keys_stem = [key for key in stemkeys
                 if key.startswith(stem) and not key.endswith('all')]
    if len(keys_stem) == 0:
        continue
    if len(keys_stem) > 1:
        merge_runs(keys_stem, runs, merge_key=stem+'-merged')
        keys_all.append(stem+'-merged')
    else:
        keys_all.append(keys_stem[0])
    for key in keys_stem:
        stemkeys.remove(key)
keys = list(runs.keys())

# plot filtered TOF spectra for all keys
plt.figure(figsize=(16, 9))
for key in keys:
    vals_raw = np.array(runs[key].spectra['filtered']['TOF'])
    t = runs[key].params['t_meas']
    vals_err = np.sqrt(vals_raw) / t
    vals = vals_raw / t
    plt.errorbar(x=np.linspace(0, 192, 513)[1:], y=vals, yerr=vals_err,
                 marker='s', linestyle='None', drawstyle='steps-mid',
                 label=key.replace('_', '-'))
plt.xlim(25, 185)
plt.xlabel(r'TIME [$\mu$s]')
plt.ylabel('COUNTS/MINUTE')
plt.ylim(0, 3.5)
plt.legend()
plt.tight_layout()

# merge selected runs
run_merged = merge_runs([key for key in keys if key.startswith('du-1')], runs,
                        merge_key='1 mm DU')
run_merged = merge_runs([key for key in keys if key.startswith('du-all')], runs,
                        merge_key='13 mm DU')

keys_selected = [key for key in keys_all if key.startswith('w-single')]

# plot TOF histogram for selected keys
plt.figure(figsize=(16, 9))
for key in keys_selected:
    if not key.endswith('background'):
        vals_raw, bins = np.histogram(runs[key].data['filtered']['CH0']['TOF'],
                                      bins=512, range=[0, 192])
        t = runs[key].params['t_meas']
        vals_err = np.sqrt(vals_raw) / t
        vals = vals_raw / t
        plt.errorbar(x=np.linspace(0, 192, 513)[1:], y=vals, yerr=vals_err,
                     marker='s', linestyle='None', drawstyle='steps-mid',
                     label=key.replace('_', '-'))
plt.xlim(25, 185)
plt.xlabel(r'TIME [$\mu$s]')
plt.ylabel('COUNTS/MINUTE')
plt.ylim(0, 3.5)
plt.legend()
plt.tight_layout()

plot_trans(runs, key_target='1 mm DU', key_open='no-target', n_bins=800,
           color='blue')
plot_trans(runs, key_target='13 mm DU', key_open='no-target', n_bins=800,
           color='red')

"""
# merge_runs(runs['du-5-totalshield'],
             runs['du-5-totalshield_1'],
             runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp'],
             runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp_1'],
             runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp_2'],
             runs, key='total')

# plot_trans(runs, 'total', 'open', color='black', n_bins=800)

# plot_trans(runs, 'parallel', 'open', color='blue')
# plot_trans(runs, 'perpendicular', 'open', color='red', add_plot=True)

plt.figure()
for key in ['0124-open', '0124-Al', '0124-poly']:
    plt.plot(runs[key].spectra['unfiltered']['E'], label=key.replace('_', '-'))
plt.xlim(0, 650)
plt.ylim(bottom=0)
plt.legend()

plt.figure()
for key in ['0124-open_1', '0124-Wpart', '0124-Woreo', '0124-BNSIS']:
    x = runs[key].spectra['unfiltered']['E']
    x_out = runs['0124-open_1'].spectra['unfiltered']['E']
    plt.plot(x,
             drawstyle='steps-mid',
             label=key.replace('_', '-'))
    print(key.replace('_', '-'), sum(x[260:320]), np.sqrt(sum(x[260:320])))
plt.xlim(0, 650)
plt.ylim(bottom=0)
plt.legend()

"""
