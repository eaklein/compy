# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:46:40 2021

@author: Avram
"""

import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.close('all')
plt.rcParams.update({'font.size': 20}) #increase font size of plot labels
import xmltodict
from bisect import bisect_left
from scipy.stats import gaussian_kde
from copy import deepcopy

# set LaTEX print parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class compassRun:
    """ Python class for CoMPASS run """
    def __init__(self, key, folder='C:/CoMPASS/20210531/DAQ/', 
                 params={}, settings={}, spectra={}, data={}, verbose=0):
        """ Initialize TargetData class """
        self.key = key
        self.folder = folder
        self.params = params
        self.settings = {}
        self.spectra = {}
        self.data = {}
    
    def read_settings(self):
        """ read CoMPASS project folder and extract settings for each run """
        try:
            with open(self.folder + self.key + '/settings.xml') as f:
                settings_xml = xmltodict.parse(f.read())
            # read board parameters
            for entry in settings_xml['configuration']['board']['parameters']['entry']:
                self.settings[entry['key']] = entry['value']['value']['#text']
            # read acquisition time in min
            self.settings['t_meas'] = float(settings_xml['configuration']['acquisitionMemento']
                                            ['timedRunDuration'])/1000/60
        except:
            verboseprint(f'WARNING: Settings file could not be found for {self.key}.')
        
        # store certain settings in params dictionary
        # read measurement time
        self.params = {}
        self.params['t_meas'] = self.settings['t_meas']
        # read parameters for TOF spectra
        self.params['TOF'] = {}
        self.params['TOF']['n_bins'] = int(float(self.settings['SW_PARAMETER_DIFFERENCE_BINCOUNT']))
        self.params['TOF']['t_lo'] = float(self.settings['SW_PARAMETER_TIME_DIFFERENCE_CH_T0'])/1000
        self.params['TOF']['t_hi'] = float(self.settings['SW_PARAMETER_TIME_DIFFERENCE_CH_T1'])/1000
        # read parameters for energy spectra
        self.params['E'] = {}
        self.params['E']['n_bins'] = int(float(self.settings['SRV_PARAM_CH_SPECTRUM_NBINS'].split('_')[1]))
        # read parameters for PSD spectra
        self.params['PSD'] = {}
        self.params['PSD']['n_bins'] = int(float(self.settings['SW_PARAMETER_PSDBINCOUNT']))
        
        for filt_key in ['unfiltered', 'filtered']:
            filt_upper = filt_key.upper()
            self.params[filt_key] = {}
            try:
                # read raw CH0 data location
                self.params[filt_key][
                    'file_CH0'] = [file for file in os.listdir(
                                   self.folder + self.key + '/' + filt_upper + '/')
                                   if file.endswith(".csv") and ('CH0' in file)]
                 # read raw CH1 data location
                self.params[filt_key][
                    'file_CH1'] = [file for file in os.listdir(
                                   self.folder + self.key + '/' + filt_upper + '/') 
                                   if file.endswith(".csv") and ('CH1' in file)]
                # read saved TOF spectra location
                self.params[filt_key][
                    'file_data_TOF'] = [file for file in os.listdir(
                                        self.folder + self.key + '/' + filt_upper + '/') 
                                        if file.endswith(".txt") and ('TOF' in file)]
                # read saved E spectra location
                self.params[filt_key][
                    'file_data_E'] = [file for file in os.listdir(
                                      self.folder + self.key + '/' + filt_upper + '/') 
                                      if file.endswith(".txt") and ('CH0' in file) and ('E' in file)]
                # read saved PSD spectra location
                self.params[filt_key][
                    'file_data_PSD'] = [file for file in os.listdir(
                                        self.folder + self.key + '/' + filt_upper + '/') 
                                        if file.endswith(".txt") and ('CH0' in file) and ('PSD' in file)]
            except:
                verboseprint(f'WARNING: Could not find {filt_upper} folder for {self.key}.')

    def read_spectra(self, modes=['E', 'TOF', 'PSD']):
        """ read data from CoMPASS saved histogram """
        for filt_key in ['unfiltered', 'filtered']:
            self.spectra[filt_key] = {}
            for mode in modes:
                key_data = 'file_data_' + mode # can choose mode either E or TOF
                try:
                    self.spectra[filt_key][
                        mode] = np.array(np.loadtxt(self.folder + self.key + '/' 
                                                    + filt_key.upper() + '/'
                                                    + max(self.params[filt_key][key_data])))
                    verboseprint(f'Read in CoMPASS spectrum for {self.key} (mode: {mode}, {filt_key})')
                except:
                    # print(f'ERROR: Unable to open CoMPASS histogram for {key} (mode: {mode})')
                    pass
                
    def read_data(self, filtered=['unfiltered', 'filtered']):
        """ read raw data from CoMPASS csv files """
        for filt_key in filtered:
            try:
                self.data[filt_key] = {}
                # check to read Channel 0 (detector) data
                if len(self.params[filt_key]['file_CH0']) > 0:
                    verboseprint(f'Reading in {filt_key} CH0 data for key: {self.key}.')
                    self.data[filt_key][
                        'CH0'] = pd.read_csv(self.folder + self.key + '/' 
                                             + filt_key.upper() + '/'
                                             + self.params[filt_key]['file_CH0'][0], sep=';',
                                             on_bad_lines='skip')
                    # do not read Channel 1 (pulse) data if TOF was already calculated for run
                    # if 'TOF' in self.data[filt_key]['CH0']:
                    #     break                     
                else:
                    verboseprint(f'Did not find {filt_key} CH0 data for key: {self.key}.')
                # # check to read Channel 1 (pulse) data
                # if len(self.params[filt_key]['file_CH1']) > 0:
                #     verboseprint(f'Reading in {filt_key} CH1 data for key: {self.key}.')
                #     self.data[filt_key][
                #         'CH1'] = pd.read_csv(self.folder + self.key + '/'
                #                              + filt_key.upper() + '/'
                #                              + self.params[filt_key]['file_CH1'][0], sep=';')
                # else:
                #     verboseprint(f'Did not find {filt_key} CH1 data for key: {self.key}.')
                if self.data == {} or ((self.data[filt_key]['CH0'].size == 0) and 
                                       (self.data[filt_key]['CH1'].size == 0)):
                    verboseprint(f'WARNING: no data found for {self.key}.')
            except:
                verboseprint(f'WARNING: no data found for {self.key}.')
  
    def add_TOF(self, filtered=['unfiltered', 'filtered']):
        """ add TOF column to raw data dataframe"""
        for filt_key in filtered:
            # check if un-/filtered CH0 data present
            if (filt_key in self.data) and ('CH0' in self.data[filt_key]):
                # check that TOF not yet calculated
                if 'TOF' not in self.data[filt_key]['CH0']: 
                    # check to read Channel 1 (pulse) data
                    if len(self.params[filt_key]['file_CH1']) > 0:
                        verboseprint(f'Reading in {filt_key} CH1 data for key: {self.key}.')
                        self.data[filt_key][
                            'CH1'] = pd.read_csv(self.folder + self.key + '/'
                                                 + filt_key.upper() + '/'
                                                 + self.params[filt_key]['file_CH1'][0], sep=';',
                                                 on_bad_lines='skip')
                    else:
                        verboseprint(f'Did not find {filt_key} CH1 data for key: {self.key}.')
                    # check if pulse data exists and CH0/CH1 are not empty
                    if ('CH1' in self.data[filt_key]) and (
                    not any(self.data[filt_key][ch].empty for ch in ['CH0', 'CH1'])):
                        verboseprint(f'Calculating {filt_key} TOF for key: {self.key}')
                        # check that timing information was saved for run
                        if all('TIMETAG' in self.data[filt_key][ch] for ch in ['CH0', 'CH1']):
                            try:
                                self.data[filt_key][
                                    'CH0']['TOF'] = calc_TOF(self.data[filt_key]['CH1']['TIMETAG'], 
                                                             self.data[filt_key]['CH0']['TIMETAG'])
                                # rewrite CH0 csv file with TOF added
                                self.data[filt_key][
                                    'CH0'].to_csv(self.folder + self.key + '/' + filt_key.upper()
                                                  + '/' + self.params[filt_key]['file_CH0'][0], 
                                                  sep=';', index=False)
                            except:
                                print('No data found or another issue has arose.')
                                continue
                        else:
                            print('Unable to add TOF. Timetag not found in data.')
                    else:
                        print('Data is empty for either CH0, CH1, or both.')
                else:
                    print('TOF was already calculated.')
            else:
                print(f'WARNING: No Ch.0 {filt_key} data found to calculate TOF.')
        
    def user_filter(self):
        """ perform user cut of energy spectrum """
        print(f'\nUser energy cut requested for TOF spectrum for {self.key}.')
        self.data['user'] = {}
        E_lo = click.prompt('Set E_lo channel for pulse area cut', default=95)
        E_hi = click.prompt('Set E_hi channel for pulse area cut', default=135)
        # copy all of CH1 pulse data
        self.data['user'][
            'CH1'] = self.data['unfiltered']['CH1'].copy()
        # perform energy cut on unfiltered detector signal data
        self.data['user'][
            'CH0'] = self.data['unfiltered']['CH0'].loc[
                    (self.data['unfiltered']['CH0']['ENERGY'] > E_lo) &
                    (self.data['unfiltered']['CH0']['ENERGY'] < E_hi)].copy()
        # if TOF not already calculated, calculate TOF
        if 'TOF' not in self.data['user']['CH0']:
            self.data['user'][
                'CH0']['TOF'] = calc_TOF(self.data['unfiltered']['CH1']['TIMETAG'], 
                                         self.data['user']['CH0']['TIMETAG'])
        print(f'Successfully performed user energy cut of TOF spectrum for {self.key}.\n')
            
    def plot_TOF(self, t_lo=0, t_hi=200, n_bins=400, filtered='unfiltered', norm=True, add=False):
        """ plot manually calculated TOF spectrum """
        if ('CH0' in self.data[filtered]) and (self.data[filtered]['CH0'].size >= 0):
            try:
                x = self.data[filtered]['CH0'].TOF
                if add == False:
                    # create new figure
                    plt.figure(figsize=(16, 9))
                if norm == True:
                    # normalize counts
                    weights = [1/self.params['t_meas']]*len(x)
                    plt.ylabel('COUNTS/MINUTE')                    
                else:
                    weights = [1]*len(x)
                    plt.ylabel('COUNTS')
                plt.hist(x, range=[t_lo, t_hi], bins=n_bins, weights=weights,
                         histtype='step', label=(self.key + ' ' + filtered).replace('_', '-'), lw=2)
                plt.xlim(t_lo, t_hi)
                plt.yscale('log')
                plt.legend(loc='upper right')
                plt.xlabel('TIME (us)')
                plt.tight_layout()
            except:
                verboseprint(f'No TOF data to plot for {self.key}!')
        else:
            verboseprint(f'No {filtered} CH0 data found for {self.key}!')

    def plot_spectrum(self, mode='TOF', filtered='unfiltered', add=False):
        """ plot CoMPASS spectrum """
        if add == False:
            # create new figure
            plt.figure(figsize=(16, 9))
        # select data based on mode
        y = self.spectra[filtered][mode]
        if mode == 'TOF':
            t_lo, t_hi, n_bins = (self.params['TOF']['t_lo'], self.params['TOF']['t_hi'],
                                  self.params['TOF']['n_bins'])
            x = np.linspace(t_lo, t_hi, n_bins)
            plt.xlim(self.params['TOF']['t_lo'], self.params['TOF']['t_hi'])
            plt.xlabel('TIME (us)')
        elif mode == 'PSD':
            x = np.arange(self.params['PSD']['n_bins'])
            plt.xlabel('PSD Channel')
            plt.xlim(min(x), max(x))
        else:
            x=np.arange(1, len(y)+1)
            plt.xlim(0, len(y))
            plt.xlabel('CHANNEL')
        plt.errorbar(x, y, yerr=[np.sqrt(i) for i in y], 
                     capsize=2, drawstyle='steps-mid', label=self.key + ' ' + filtered)
        plt.ylabel('COUNTS')
        plt.ylim(bottom=1)
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()

    def make_hist(self, filtered='unfiltered', mode='TOF', ch='CH0', val_lo=0, val_hi=200, n_bins=400):
        """ create a histogram from TOF values """
        hist, bin_edges = np.histogram(self.data[filtered][ch][mode].to_numpy(), range=[val_lo, val_hi], bins=n_bins)
        return hist, bin_edges
    
def select_keys(folder):
    """ select keys to process """
    keys = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
    print(f'Available keys to process are: {keys}.')
    bool_all = click.confirm('\nWould you like to process all keys?', default='n')
    # process all runs
    if bool_all:
        keys_select = keys
    # manually select runs
    else:
        bool_date = click.confirm('\nWould you like to process by date?', default='y')
        if not bool_date:
            keys_select = []
            while True:
                key = input('Type \'options\' to see all available options.'
                            '\nPress \'enter\' to end key selection.'
                            '\nEnter key name: ')
                if not key and len(keys_select) != 0:       # click 'enter' to end key selection
                    break
                elif not key and len(keys_select) == 0:     # if 'enter', but no keys selected
                    print('You must enter at least one key name!')
                elif key == 'options':                      # show available key options
                    print(keys)
                else:                                       # if key selected
                    while key not in keys:                  # if user entry not available key, print warning
                        key = input('\nThat key does not exist.'
                                    '\nType \'options\' to see all available options.'
                                    '\nPress \'enter\' to end key selection.'
                                    '\nEnter key name: ')
                        if key == 'options':
                            print(keys)
                    keys_select.append(key)                 # if good key, append to list
        else:
            keys_select = []
            while True:
                date = input('(Note: \nPress \'enter\' to end key selection.)'
                             '\nEnter run date: ')
                if not date and len(keys_select) != 0:                  # click 'enter' to end key selection
                    break
                elif not any([key.startswith(date) for key in keys]
                           ) and len(keys_select) != 0:                 
                    print('Bad key provided.')         
                elif not any([key.startswith(date) for key in keys]
                           ) and len(keys_select) == 0:                 # if 'enter', but no keys selected
                    print('Bad key provided. You must enter at least one key name!')
                elif any([key.startswith(date) for key in keys]):
                    [keys_select.append(key) for key in keys if (key.startswith(date)) and (key not in keys_select)] # if good key, append to list
    return keys_select
           
def merge_copy(d1, d2):
    """ merge nested dicts """
    return {k: merge_copy(d1[k], d2[k]) if 
            k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict) else 
            merge_vals(d1[k], d2[k]) for k in d2}

def merge_vals(x, y):
    """ combine keys in dict """
    if x == y:
        return x
    elif isinstance(x, list) and isinstance(y, list):
        return [*x, *y]
    else:
        return [x, y]

def merge_runs(keys, runs={}, merge_key=''):
    """ merge data from CoMPASS runs """
    # choose key for merged run
    if merge_key == '':
        merge_key = click.prompt('Which key should be used for the merged run?')
    # initialize merged run with first run provided
    run_merged = deepcopy(runs[keys[0]])
    run_merged.key = merge_key
    # iterate over additional keys to merge
    for key in keys[1:]:
        run = runs[key]
        # check settings
        if run.settings != run_merged.settings:
            print('Runs have different settings.')
            settings_key = keys[0]
            # settings_key = click.prompt("Please select:", type=click.Choice([run.key, run_merged.key], 
            #                                                                 case_sensitive=False))
            run_merged.settings = runs[settings_key].settings
        # check folder
        run_merged.folder = [[run.folder, run_merged.folder] if run.folder != run_merged.folder \
                             else run.folder]
        # check if spectral parameters are equal
        if not all([run.params['E']==run_merged.params['E'], run.params['TOF']==run_merged.params['TOF']]):
            print(f'Spectra parameters for {run.key} and {run_merged.key} are not the same.' '\n' \
                  f'Will keep spectra parameters from {run_merged.key} but will not store any spectra.')
        else:
        # if yes, merge spectra
            print(f'Merging spectra for {run.key} and {run_merged.key}.')
            run_merged.spectra = {}
            for filtered in ['unfiltered', 'filtered']:
                run_merged.spectra[filtered] = {}
                for key in run.spectra[filtered].keys() & run_merged.spectra[filtered].keys():
                    run_merged.spectra[filtered][key] = [xi + yi for xi, yi in \
                                                         zip(run.spectra[filtered][key], 
                                                             run_merged.spectra[filtered][key])]
        # merge params
        print(f'Merging parameters for {run.key} and {run_merged.key}.')
        t_meas = run_merged.params['t_meas']
        # print('\nMerged run time:', run_merged.params['t_meas'])
        run_merged.params = merge_copy(run.params, run_merged.params)
        run_merged.params['t_meas'] = t_meas
        # print('Run time:', run.params['t_meas'])
        run_merged.params['t_meas'] += run.params['t_meas']
        # print('\nMerged run time:', run_merged.params['t_meas'])
        # merge data
        for filtered in ['unfiltered', 'filtered']:
            # run_merged.data[filtered] = {}
            [run.add_TOF(filtered=[filtered]) for run in [run, run_merged] \
             if 'TOF' not in run.data[filtered]['CH0']]
            run_merged.data[filtered]['CH0'] = pd.concat([run.data[filtered]['CH0'], 
                                                          run_merged.data[filtered]['CH0']], 
                                                         axis=0)
    runs[run_merged.key] = run_merged
    return run_merged

def initialize(folder=[], keys=[]):
    """ start up the CoMPASS Companion!"""
    print('\nWelcome to CoMPASS Companion!')
    if folder == []:
        folder = click.prompt('\nPlease enter a project folder path', 
                              default='C:/CoMPASS/20210531/DAQ/')
    if keys == []:
        keys = select_keys(folder)
    verbose = click.confirm('\nVerbose Mode?', default=True)
    return folder, keys, verbose

def process_runs(keys, folder='C:/CoMPASS/20210531/DAQ/', runs={}, verbose=0):
    bool_TOF =  click.confirm('Would you like to perform manual TOF calculation?', default='Y')
    runs = runs
    for key in keys:
        if key != 'CoMPASS':
            print('\n' + f'Processing Key: {key}...')
            run = compassRun(key, folder, verbose=verbose)
            run.read_settings()
            run.read_spectra()
            run.read_data()
            if bool_TOF:
                run.add_TOF()
            runs[key] = run
    return runs

def calc_TOF(t_pulse, t_signal):
    "calculate TOF from pulse and signal time arrays"
    tof = []
    for t in t_signal:
        idx = bisect_left(t_pulse, t)
        if idx == len(t_pulse):
            t0 = t_pulse.iloc[-1]
        else:
            t0 = t_pulse[idx-1]
        tof.append((t - t0)/1e6) # convert to ps to us
    return tof

def calc_trans(counts_in, counts_out, t_meas_in, t_meas_out):
    """ calculate transmission and propagate error """
    vals_trans = [(x/t_meas_in)/(y/t_meas_out) if 
                  y!= 0 else 0 for x, y in zip(counts_in, counts_out)]
    err_trans = [(x/t_meas_in)/(y/t_meas_out)*np.sqrt((1/np.sqrt(x))**2 + (1/np.sqrt(y))**2) if 
                 y!= 0 else 0 for x, y in zip(counts_in, counts_out)]
    return np.array(vals_trans), np.array(err_trans)

def calc_atten(data, thick, err_thick={}, keys=[], key_ref='target_out', bin_lo=90, bin_hi=135):
    """ calc transuation in number of counts """    
    if keys == []:
        keys = list(data.keys())
        keys.remove(key_ref)
    if err_thick == {}:
        err_thick = {key:0. for key in keys}
    trans = {}
    err_trans = {}
    mu = {}
    err_mu = {}
    c_in = sum(data[key_ref][bin_lo:bin_hi])
    for key in keys:
        c_out = sum(data[key][bin_lo:bin_hi])
        trans[key] = c_out / c_in
        err_trans[key] = trans[key] * np.sqrt(1/c_out + 1/c_in)
        mu[key] = -1*np.log(trans[key])/thick[key]
        err_mu[key] = np.sqrt((err_trans[key]/trans[key])**2 + (err_thick[key]/thick[key])**2)
        print(f'{key}:'.ljust(20) + f'mu = {mu[key]*10:.2f} +/- {err_mu[key]*10:.2f}' + ' [cm-1]')
    return (trans, mu), (err_trans, err_mu)

def plot2D(runs, key, var1, var2, filtered='unfiltered'):
    run = runs[key]
    data = run.data[filtered]['CH0']
    data['PSD'] = 1 - (data['ENERGYSHORT'] / data['ENERGY'])
    plt.figure(figsize=(16, 9))
    x = data[var1]
    y = data[var2]
    plt.scatter(x, y, s=0.1, c='b', label=key)
    plt.xlim(left=0)
    plt.ylim([0., 1.])
    plt.xlabel(var1, labelpad=10)
    plt.ylabel(var2, labelpad=10)
    plt.legend()

def plotEvsPSD(runs, key, filter_params={}, w=1.0):
    data = runs[key].data['unfiltered']['CH0']
    data['PSD'] = 1 - (data['ENERGYSHORT'] / data['ENERGY'])
    data = dfFilter(data, filter_params)
    heatmap, xedges, yedges = np.histogram2d(data['ENERGY']*w, data['PSD'], 
                                             bins=400, range=[[0, 4000], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cmap = mpl.cm.get_cmap("plasma").copy()
    cmap.set_under(color='white')
    plt.figure(figsize=(12, 12))
    plt.imshow(heatmap.T, extent=extent, origin='lower', vmin=0.5, vmax=25, cmap=cmap, aspect=2000)
    plt.xlabel('Energy [ADC]')
    plt.ylabel('PSD')
    plt.tight_layout()

def dfFilter(df, filter_params):
    df = df.copy()
    for key, value in filter_params.items():
        if value != ('0.0', '0.0'):
            df = df.loc[(df[key] >= float(value[0])) & (df[key] <= float(value[1]))]
    return df

###################################################################################################

# set folder of CoMPASS runs
folder = 'C:/Users/Avram/Dropbox (MIT)/Resonances/data/CoMPASS/20210531/DAQ/'
folder = 'C:/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/Experiments/IAP-2022/DAQ/'
# folder = 'C:/CoMPASS/Lincoln-new/DAQ/'
# folder = 'C:/Users/Avram/Dropbox (MIT)/MIT/research/NRTA/Experiments/DU-studies/DAQ/'

if __name__ == '__main__':
    folder, keys, verbose = initialize(folder=folder)
    verboseprint = print if verbose else lambda *a, **k: None       # set verbose mode
    runs = process_runs(keys, folder=folder, verbose=verbose)       # process CoMPASS files, reqs .CSV saving

    bool_ex = False

    """ EXAMPLES """   

    if bool_ex:
        # choose key
        key = 'U1Pb9'
        run = runs[key]
        
        """ Manual filter on neutron peak in energy spectrum """
        run.user_filter()

        """ process another run and add to dictionary"""
        keys_new = ['W_nozzle', 'W_part']
        runs = process_runs(keys_new, runs=runs)
        
        """ merge two runs """
        run_merged = merge_runs(run1=runs[keys_new[0]],
                                run2=runs[keys_new[1]], 
                                runs=runs)              # include runs dictionary to add merged run
        
        runs[keys_new[0]].plot_TOF('unfiltered', add=False)        
        runs[keys_new[1]].plot_TOF('unfiltered', add=True) 
        runs['merged'].plot_TOF('unfiltered', add=True) 
        
        """ Manual adition of TOF if not done during initial processing """
        run.add_TOF()   # manually add TOF for run
        
        """ plot pulse area for all counts """
        run.plot_spectrum(mode='E', filtered='unfiltered')
        
        """ compare manually calculated TOF spectrum for all / neutron counts """
        run.plot_TOF(filtered='user', add=False)
        run.plot_TOF(filtered='filtered', add=True)     # use add to plot on existing figure
        
        """ plot TOF for all keys overlaid """
        plt.figure(figsize=(16, 9))
        for key in [#'w-double', 'w-double-empty', 'w-double-nolead', 'w-double-nolead_1', 
                    #'w-single', 'w-single-empty', 'w-single_1']:
                    'w-single', 'w-double']:
            runs[key].plot_TOF(norm=True, filtered='filtered', add=True)     # use 'norm' to normalize to meas. time in min.

        for i, key in enumerate(runs):
            run = runs[key]
            if i != 0:
                run.plot_spectrum(filtered='filtered', add=True)     # use add to plot on existing figure
            else:
                run.plot_spectrum(filtered='filtered', add=False)     # use add to plot on existing figure
                
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

def plot_trans(runs, key_target, key_open, t_lo=0., t_hi=200., n_bins=400, t_offset=10.0,
               color='black', add_plot=False):
    """ calculate transmission and plot """
    target_in = runs[key_target].data['filtered']['CH0']['TOF']
    target_out = runs[key_open].data['filtered']['CH0']['TOF']
    t_meas_in = runs[key_target].params['t_meas']
    t_meas_out = runs[key_open].params['t_meas']
    counts_in, __ = np.histogram(target_in, bins=n_bins, range=[t_lo, t_hi])
    counts_out, __ = np.histogram(target_out, bins=n_bins, range=[t_lo, t_hi])
    bins = np.linspace(t_lo, t_hi, n_bins+1)[:-1] + (t_hi-t_lo)/n_bins/2 - t_offset
    vals_trans, vals_errs = calc_trans(counts_in, counts_out, t_meas_in, t_meas_out)
    if not add_plot:
        plt.figure(figsize=(16, 9))
    plt.errorbar(x=bins, y=vals_trans, yerr=vals_errs, 
                 lw=2, elinewidth=0.5, capsize=1, color=color, label=key_target + ' transmission' )
    plt.xlim([max(0, t_lo), t_hi-t_offset])
    plt.xlabel(r'TIME [$\mu$s]', labelpad=10)
    plt.ylabel(r'TRANSMISSION', labelpad=10)
    plt.legend()
    plt.tight_layout()
    return vals_trans, vals_errs, bins

# merge all runs that only differ by end underscore
key_stems = list(set([key.rsplit('_', maxsplit=1)[0] for key in keys]))
key_stems.sort(key=len)
keys_all = []

if 'w-double-nolead' in key_stems:
    key_stems.remove('w-double-nolead') # fix issue with double-nolead
    [keys.remove(key) for key in keys if key.startswith('w-double-nolead')]
stemkeys = [key for key in keys]
for stem in key_stems[::-1]:
    keys_stem = [key for key in stemkeys if key.startswith(stem) and not key.endswith('all')]
    if len(keys_stem) == 0:
        continue
    elif len(keys_stem) > 1:
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
    vals_raw  = np.array(runs[key].spectra['filtered']['TOF'])
    t = runs[key].params['t_meas']
    vals_err = np.sqrt(vals_raw) / t
    vals = vals_raw / t
    plt.errorbar(x=np.linspace(0, 192, 513)[1:], y=vals, yerr=vals_err, 
                 marker='s', linestyle='None', drawstyle='steps-mid', label=key.replace('_', '-'))
plt.xlim(25, 185)
plt.xlabel(r'TIME [$\mu$s]')
plt.ylabel('COUNTS/MINUTE')
plt.ylim(0, 3.5)
plt.legend()
plt.tight_layout()

# merge selected runs
run_merged =  merge_runs([key for key in keys if key.startswith('du-1')], runs, 
                         merge_key='1 mm DU')
run_merged =  merge_runs([key for key in keys if key.startswith('du-all')], runs, 
                         merge_key='13 mm DU')

keys_selected = [key for key in keys_all if key.startswith('w-single')]

# plot TOF histogram for selected keys
plt.figure(figsize=(16, 9))
for key in keys_selected:
    if not key.endswith('background'):
        vals_raw, bins  = np.histogram(runs[key].data['filtered']['CH0']['TOF'], bins=512, range=[0, 192])
        t = runs[key].params['t_meas']
        vals_err = np.sqrt(vals_raw) / t
        vals = vals_raw / t
        plt.errorbar(x=np.linspace(0, 192, 513)[1:], y=vals, yerr=vals_err, 
                     marker='s', linestyle='None', drawstyle='steps-mid', label=key.replace('_', '-'))
plt.xlim(25, 185)
plt.xlabel(r'TIME [$\mu$s]')
plt.ylabel('COUNTS/MINUTE')
plt.ylim(0, 3.5)
plt.legend()
plt.tight_layout()

plot_trans(runs, key_target='1 mm DU', key_open='no-target', color='blue', n_bins=800)
plot_trans(runs, key_target='13 mm DU', key_open='no-target', color='red', n_bins=800)

"""
# merge_runs(runs['du-5-totalshield'], runs['du-5-totalshield_1'], runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp'], runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp_1'], runs, key='total')
# merge_runs(runs['total'], runs['du-5-perp_2'], runs, key='total')

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