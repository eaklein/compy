# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 19:25:42 2021

@author: Avram
"""

import os
os.chdir("C:/Users/Avram/Dropbox (MIT)/MIT/Research/Neutron_Imaging/scripts") #set scripts directory
import click
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
plt.close('all')
plt.rcParams.update({'font.size': 20}) #increase font size of plot labels
import xmltodict
from bisect import bisect_left
import copy
import matplotlib.ticker as mtick

plt.rcParams.update({'font.size': 28, 
                     'font.family': 'serif', 
                     'font.serif': ['Computer Modern Roman'],
                     'font.weight': 'bold'})
plt.rcParams['text.usetex'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class compassRun:
    """ Python class for CoMPASS run 
    
    """
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
            with open(self.folder + self.key + '/settings.xml') as fd:
                settings_xml = xmltodict.parse(fd.read())
            # read board parameters
            for entry in settings_xml['configuration']['board']['parameters']['entry']:
                self.settings[entry['key']] = entry['value']['value']['#text']
            # read acquisition time in min
            self.settings['t_meas'] = float(settings_xml['configuration']['acquisitionMemento']
                                            ['timedRunDuration'])/1000/60
        except:
            verboseprint(f'WARNING: Settings file could not be found for {self.key}.')
        # store certain settings in params dictionary
        for filt_key in ['unfiltered', 'filtered']:
            filt_upper = filt_key.upper()
            try:
                self.params[filt_key] = {}
                # read raw CH0 data
                self.params[filt_key]['file_CH0'] = [file for file in 
                                                     os.listdir(self.folder + self.key + '/' + filt_upper + '/') 
                                                     if file.endswith(".csv") and ('CH0' in file)]
                 # read raw CH1 data
                self.params[filt_key]['file_CH1'] = [file for file in 
                                                     os.listdir(self.folder + self.key + '/' + filt_upper + '/') 
                                                     if file.endswith(".csv") and ('CH1' in file)]
                # read saved TOF spectra
                self.params[filt_key]['file_data_TOF'] = [file for file in 
                                                          os.listdir(self.folder + self.key + '/' + filt_upper + '/') 
                                                          if file.endswith(".txt") and ('TOF' in file)]
                # read saved E spectra
                self.params[filt_key]['file_data_E'] = [file for file in 
                                                        os.listdir(self.folder + self.key + '/' + filt_upper + '/') 
                                                        if file.endswith(".txt") and ('CH0' in file) and ('E' in file)]
                # read saved PSD spectra
                self.params[filt_key]['file_data_PSD'] = [file for file in 
                                                          os.listdir(self.folder + self.key + '/' + filt_upper + '/') 
                                                          if file.endswith(".txt") and ('CH0' in file) and ('PSD' in file)]
                # read measurement time
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
            except:
                verboseprint(f'WARNING: Could not find {filt_upper} folder for {self.key}.')

    def read_spectra(self, modes=['E', 'TOF', 'PSD']):
        """ read data from CoMPASS saved histogram """
        for filt_key in ['unfiltered', 'filtered']:
            self.spectra[filt_key] = {}
            for mode in modes:
                key_data = 'file_data_' + mode # can choose mode either E or TOF
                try:
                    self.spectra[filt_key][mode] = np.array(np.loadtxt(self.folder + self.key + '/' 
                                                                       + filt_key.upper() + '/'
                                                                       + self.params[filt_key][key_data][0]))
                    verboseprint(f'Read in CoMPASS spectrum for {self.key} (mode: {mode}, {filt_key})')
                except:
                    pass
                    # print(f'ERROR: Unable to open CoMPASS histogram for {key} (mode: {mode})')

    def read_data(self, filtered=['unfiltered', 'filtered']):
        """ read raw data from CoMPASS csv files """
        for filt_key in filtered:
            try:
                self.data[filt_key] = {}
                if len(self.params[filt_key]['file_CH0']) > 0:
                    verboseprint(f'Reading in {filt_key} CH0 data for key: {self.key}.')
                    self.data[filt_key]['CH0'] = pd.read_csv(self.folder + self.key + '/' + filt_key.upper() + '/' +
                                                               self.params[filt_key]['file_CH0'][0], sep=';')
                else:
                    verboseprint(f'Did not find {filt_key} CH0 data for key: {self.key}.')
                if len(self.params[filt_key]['file_CH1']) > 0:
                    verboseprint(f'Reading in {filt_key} CH1 data for key: {self.key}.')
                    self.data[filt_key]['CH1'] = pd.read_csv(self.folder + self.key + '/' + filt_key.upper() + '/' + 
                                                               self.params[filt_key]['file_CH1'][0], sep=';')
                else:
                    verboseprint(f'Did not find {filt_key} CH1 data for key: {self.key}.')
                if self.data == {} or ((self.data[filt_key]['CH0'].size == 0) and 
                                       (self.data[filt_key]['CH1'].size == 0)):
                    verboseprint(f'WARNING: no data found for {self.key}.')
            except:
                verboseprint(f'WARNING: no data found for {self.key}.')
  

    def add_TOF(self, filtered=['unfiltered', 'filtered']):
        """ add TOF column to raw data dataframe"""
        for filt_key in filtered:
            # check if data exists
            if ('CH0' in self.data[filt_key]) and ('CH1' in self.data[filt_key]):
                if (not self.data[filt_key]['CH0'].empty) and (not self.data[filt_key]['CH1'].empty):
                    # check that TOF was not already calculated
                    if 'TOF' not in self.data[filt_key]['CH0']:
                        verboseprint(f'Calculating {filt_key} TOF for key: {self.key}')
                        # check that timing information was saved
                        if 'TIMETAG' in self.data[filt_key]['CH1']:
                            self.data[filt_key]['CH0']['TOF'] = calc_TOF(self.data[filt_key]['CH1']['TIMETAG'], 
                                                                         self.data[filt_key]['CH0']['TIMETAG'])
                        else:
                            print('Unable to add TOF. Timetag not found in CH1 data.')
                    else:
                        print('TOF was already calculated.')
            else:
                print(f'WARNING: No {filtered} data found to calculate TOF.')
        
    def user_filter(self):
        """ perform user cut of energy spectrum """
        print(f'\nUser energy cut requested for TOF spectrum for {self.key}.')
        self.data['user'] = {}
        E_lo = click.prompt('Set E_lo channel for pulse area cut', default=95)
        E_hi = click.prompt('Set E_hi channel for pulse area cut', default=135)
        self.data['user']['CH1'] = self.data['unfiltered']['CH1'].copy()
        self.data['user']['CH0'] = self.data['unfiltered']['CH0'].loc[(self.data['unfiltered']['CH0']['ENERGY'] > E_lo)
                                                                      & (self.data['unfiltered']['CH0']['ENERGY'] < E_hi)].copy()
        if 'TOF' not in self.data['user']['CH0']:
            self.data['user']['CH0']['TOF'] = calc_TOF(self.data['unfiltered']['CH1']['TIMETAG'], 
                                                                 self.data['user']['CH0']['TIMETAG'])
        print(f'Successfully performed user energy cut of TOF spectrum for {self.key}.\n')
            
    def plot_TOF(self, t_lo=0, t_hi=200, n_bins=400, filtered='unfiltered', norm=True, add=False):
        """ plot manually calculated TOF spectrum """
        if ('CH0' in self.data[filtered]) and (self.data[filtered]['CH0'].size >= 0):
            try:
                x = self.data[filtered]['CH0'].TOF
                if add == False:
                    plt.figure(figsize=(16, 9))
                if norm == True:
                    weights = [1/self.params['t_meas']]*len(x)
                    plt.ylabel('COUNTS/MINUTE')                    
                else:
                    weights = [1]*len(x)
                    plt.ylabel('COUNTS')
                plt.hist(x, range=[t_lo, t_hi], bins=n_bins, weights=weights,
                         histtype='step', label=self.key + ' ' + filtered, lw=2)
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
            plt.figure(figsize=(16, 9))
        y=self.spectra[filtered][mode]
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
    if bool_all:        # process all runs
        keys_select = keys
    else:               # manually select runs
        keys_select = []
        # process user input
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

def merge_runs(run1, run2, runs={}, key=''):
    """ merge data from CoMPASS runs """
    # choose key
    if key == '':
        key = click.prompt('Which key should be used for the merged run?')
    run_merged = compassRun(key=key)
    # check settings
    if run1.settings != run2.settings:
        print('Runs have different settings.')
        run_merged.settings = click.Choice([run1.key, run2.key], case_sensitive=False)
    # check folder
    run_merged.folder = [[run1.folder, run2.folder] if run1.folder != run2.folder else run1.folder]
    # check if spectral parameters are equal
    if not all([run1.params['E']==run2.params['E'], run1.params['TOF']==run2.params['TOF']]):
        print(f'Spectra parameters for {run1.key} and {run2.key} are not the same.' '\n' \
              f'Will keep spectra parameters from {run1.key} but will not store any spectra.')
    else:
    # if yes, merge spectra
        print(f'Merging spectra for {run1.key} and {run2.key}.')
        run_merged.spectra = {}
        for filtered in ['unfiltered', 'filtered']:
            run_merged.spectra[filtered] = {}
            for key in run1.spectra[filtered].keys() & run2.spectra[filtered].keys():
                run_merged.spectra[filtered][key] = [xi + yi for xi, yi in zip(run1.spectra[filtered][key], 
                                                                                run2.spectra[filtered][key])]
    # merge params
    run_merged.params = merge_copy(run1.params, run2.params)
    run_merged.params['t_meas'] = run1.params['t_meas'] + run2.params['t_meas']    
    # merge data
    for filtered in ['unfiltered', 'filtered']:
        run_merged.data[filtered] = {}
        for ch in ['CH0', 'CH1']:
            [run.add_TOF(filtered) for run in [run1, run2] if 'TOF' not in run.data[filtered][ch]]
            run_merged.data[filtered][ch] = pd.concat([run1.data[filtered][ch], run2.data[filtered][ch]], axis=0)
    runs[run_merged.key] = run_merged
    return run_merged

def initialize(folder=[], keys=[]):
    """ start up the CoMPASS Companion!"""
    print('\nWelcome to CoMPASS Companion!')
    if folder == []:
        folder = click.prompt('\nPlease enter a project folder path', default='C:/CoMPASS/20210531/DAQ/')
    if keys == []:
        keys = select_keys(folder)
    verbose = click.confirm('\nVerbose Mode?', default=False)
    return folder, keys, verbose

def process_runs(keys, folder='C:/CoMPASS/20210531/DAQ/', runs={}, verbose=0):
    bool_TOF =  click.confirm('Would you like to perform manual TOF calculation?', default='Y')
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
    vals_trans = [(x/t_meas_in)/(y/t_meas_out) if y!= 0 else 0 for x, y in zip(counts_in, counts_out)]
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
    heatmap, xedges, yedges = np.histogram2d(data['ENERGY']*w, data['PSD'], bins=400, range=[[0, 4000], [0, 1]])
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

def calc_atten_PSD(runs, thick, err_thick={}, keys=[], key_ref='no_target',
                   filtered='unfiltered', filter_params={}):
    """ calc transuation in number of counts from PSD vs. E plot """    
    if keys == []:
        keys = list(runs.keys())
        keys.remove(key_ref)
    if err_thick == {}:
        err_thick = {key:0. for key in keys}
    trans = {}
    err_trans = {}
    mu = {}
    err_mu = {}
    data_in_raw = runs[key_ref].data[filtered]['CH0']
    data_in_raw['PSD'] = 1 - (data_in_raw['ENERGYSHORT'] / data_in_raw['ENERGY'])
    data_in = dfFilter(data_in_raw, filter_params)
    t_meas_in = runs[key_ref].params['t_meas']
    c_in = len(data_in)
    print(f'Counts In: {c_in/t_meas_in} n/min.')
    for key in [key for key in keys if key != key_ref]:
        print('\n'f'{key}''\n'+'='*30)
        print(f'Thickness: {thick[key]:.2f} +/- {err_thick[key]:.2f}')
        data_out_raw = runs[key].data[filtered]['CH0']
        data_out_raw['PSD'] = 1 - (data_out_raw['ENERGYSHORT'] / data_out_raw['ENERGY'])
        data_out = dfFilter(data_out_raw, filter_params)
        t_meas_out = runs[key].params['t_meas']
        c_out = len(data_out)
        print(f'Counts Out: {c_out/t_meas_out} n/min.')
        trans[key] = (c_out/t_meas_out) / (c_in/t_meas_in)
        err_trans[key] = trans[key] * np.sqrt(1/c_out + 1/c_in)
        print(f'Transmission: {trans[key]:.2f} +/- {err_trans[key]:.2f}')
        mu[key] = -1*np.log(trans[key])/thick[key]
        err_mu[key] = mu[key] * np.sqrt((err_trans[key]/(trans[key]*np.log(trans[key])))**2 + (err_thick[key]/thick[key])**2)
        print(f'mu = {mu[key]*10:1.2e} +/- {err_mu[key]*10:1.0e}' + ' [cm-1]') #f'{key}:'.ljust(20) + 
        # plotEvsPSD(runs, key)
    return (trans, mu), (err_trans, err_mu)    

###################################################################################################

# folder = 'C:/Users/Avram/Dropbox (MIT)/Resonances/data/CoMPASS/20210531/DAQ/'''
folder = 'C:/CoMPASS/Lincoln-new/DAQ/'

if __name__ == '__main__':
    folder, keys, verbose = initialize(folder=folder) #'
    #keys = [i for i in keys if not i.startswith('20210922')]
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
        for key in runs:
            runs[key].plot_TOF(norm=True, add=True)     # use 'norm' to normalize to meas. time in min.

        for i, key in enumerate(runs):
            run = runs[key]
            if i != 0:
                run.plot_spectrum(filtered='filtered', add=True)     # use add to plot on existing figure
            else:
                run.plot_spectrum(filtered='filtered', add=False)     # use add to plot on existing figure

# key = 'aluminum'
# filtered = 'filtered'
# var1 = 'ENERGY'
# var2 = 'PSD'
# plot2D(key, var1, var2, filtered='unfiltered')

# for key in keys:
#     data = runs[key].data[filtered]['CH0']
#     print(f'Key: {key}, No. of Counts: {len(data)}')
#     # data['PSD'] = 1 - (data['ENERGYSHORT'] / data['ENERGY'])

thick = {'al':4.65, 'poly':4.65, 'SIS':3.75, 'BN_SIS':3.35, 'W_BN':4.00, 'coreshell_nozzle':3.2, 'W_oreo':3.3, 'W_part':3.6}
err_thick = {'al':0.01, 'poly':0.01, 'SIS':0.01, 'test_lowest_gain':1, 'BN_SIS':0.01, 'W_BN':0.2, 'coreshell_nozzle':0.1, 'W_oreo':0.1, 'W_part':0.2}
mass = {'aluminum':56.75, 'poly':19.60, 'SIS':13.70, 'test_lowest_gain':0., 'BN_SIS':19.57, 'W_BN':61.14, 'coreshell_nozzle':50.0, 'W_oreo':66.44, 'W_part':124.5}

# ADC-to-E (keV) factor 
w = 1.0

# set filters for E vs. PSD plot to select fast neutrons
# NOTE: use SIS as key, but can use any since filtered cut was same for all runs
filter_params = {'PSD': (
                         0.10, 0.225,
                         # runs['SIS'].settings['SW_PARAMETER_CH_PSDLOWCUT'], 
                         # runs['SIS'].settings['SW_PARAMETER_CH_PSDHIGHCUT']
                         ),
                 'ENERGY': (1000/w,
                            #runs['SIS'].settings['SW_PARAMETER_CH_ENERGYLOWCUT'], 
                            4000/w#runs['SIS'].settings['SW_PARAMETER_CH_ENERGYHIGHCUT']
                            )}

# calculate attenuation from LaBr data PSD vs. E plot
(trans, mu), (err_trans, err_mu)  = calc_atten_PSD(runs, thick, err_thick, 
                                                   # keys=thick.keys(), 
                                                   key_ref='target_out',
                                                   filter_params=filter_params)

print('\n\n')
for (key, val), (key, err) in zip(mu.items(), err_mu.items()):
    print(f'{key} {10*val:1.2e} +/- {10*err:1.1e}')

# plot E vs. PSD for selected data sets w/ or w/o filtering
plotEvsPSD(runs, key='target_out', filter_params=filter_params)

plotEvsPSD(runs, key='target_out')

# plot2D('test_lowest_gain', var1, var2, filtered='unfiltered')
# plot2D(runs, key='SIS', var1='ENERGY', var2='ENERGY', filtered='unfiltered')

# MAKE 2D BAR PLOT WITH ERROR BARS AND X-AXIS LABELS
plot_keys = ['SIS', 'BN_SIS', 'W_BN', 'coreshell_nozzle', 'W_oreo', 'W_part']
plot_labels = ['SIS', 'BN/SIS', 'W/BN', 'W coreshell', 'W Oreo', 'W Part']
x_pos = np.arange(len(plot_keys))
plot_mus = []
plot_errs = []
fig, ax = plt.subplots()
for key in plot_keys:
    plot_mus.append(mu[key]*10)
    plot_errs.append(err_mu[key]*10)
ax.bar(x_pos, plot_mus, yerr=plot_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('$\mu$ [cm$^{-1}$]', labelpad=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_labels, rotation=45)
ax.set_title(r'Fast Neutron Attenuation Coefficient [cm$^{-1}$] for Selected Materials', pad=20)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax.yaxis.grid(True)

filter_params = {'ENERGY': (0, 8), 'TOF': (0, 200)}
thick = {'Al_Poly':9.30, 'al':4.65, 'poly':4.65, 'SIS':3.75, 'BN_SIS':3.35, 'W_BN':4.00, 'coreshell_nozzle':3.2, 'W_oreo':3.3, 'W_part':3.6}

thick_keys = [4.65, 3.35, 4.65, 3.75, 4.00, 3.2, 3.3, 3.6, 9.30, 9.30, 9.30]

key_ref = 'target_out' 
# key_ref = 'open_20210818'
data_in_raw_ref = runs[key_ref].data['unfiltered']['CH0']
data_in_ref = dfFilter(data_in_raw_ref, filter_params)
mu_Al = 1.8e-1 # 9.14e-2 # 1.72e-2

plt.figure()
for key in ['target_out', 'open_20210818']:
    t_meas = runs[key].settings['t_meas']
    print(f't_meas: {t_meas}')
    data_in_raw = runs[key].data['unfiltered']['CH0']
    data_in = dfFilter(data_in_raw, filter_params)
    plt.hist(data_in_raw['TOF'], bins=400, histtype='step', range=[0, 200], label=key)
plt.legend()
plt.xlabel('TOF (us)')
plt.ylabel('Counts')

mus_rel = {}
for key, thick in zip(runs.keys(), thick_keys):
    # print(key, thick)
    data_in_raw = runs[key].data['unfiltered']['CH0']
    # data_in_raw['PSD'] = 1 - (data_in_raw['ENERGYSHORT'] / data_in_raw['ENERGY'])
    data_in = dfFilter(data_in_raw, filter_params)
    mu_rel = -1*np.log(len(data_in)/len(data_in_ref))/thick/mu_Al #10600 for 0->30, 17458 for n
    mus_rel[key] = mu_rel
    # print(f'{key} {len(data_in)}')
    print(f'{key} {mu_rel:1.2e}')
    plt.hist(data_in['TOF'], bins=400, histtype='step', range=[0, 200], label=key)
# plt.legend()

plot_keys = ['poly', 'SIS', 'BN_SIS', 'W_BN_5050_blend', 'W_nozzle', 'W_oreo', 'W_part']
plot_labels = ['LDPE', 'SIS', 'BN/SIS', 'W/BN', 'W coreshell', 'W Oreo', 'W Part']
x_pos = np.arange(len(plot_keys))
plot_mus = []
fig, ax = plt.subplots()
for i, key in enumerate(plot_keys):
    plot_mus.append(mus_rel[key])
ax.bar(x_pos, plot_mus, align='center', alpha=0.5, color='blue')
ax.set_ylabel('Relative Attenuation', labelpad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_labels)#, rotation=45)
ax.set_title(r'Fast Neutron Attenuation for Selected Materials', pad=20)
ax.axhline(1, 0, 1, color='r', lw=5)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
# ax.yaxis.grid(True)