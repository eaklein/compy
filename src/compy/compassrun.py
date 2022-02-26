# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:46:40 2021
@author: E. A. Klein
"""

import struct
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
import xmltodict

from .utilities import calc_TOF

# set plotting environment
plt.close('all')
plt.rcParams.update({'font.size': 20})

# set LaTEX print parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class CompassRun:
    """
    A class for storing information from a CoMPASS data acquisition.

    Attributes
    ----------
    key : str
        unique run key assigned at creation
    folder : str
        DAQ folder location for CoMPASS project
    params : dictionary
        data acquisition parameters
    settings : dict
        CoMPASS settings read from xml file
    spectra : dict
        auto-generated spectra from CoMPASS
    data : dictionary
        un-/filtered data acquired and stored by CoMPASS
    t_meas : float
        acquisition time in minutes
    file_fmt : string
        file format of saved CoMPASS data (e.g., csv, bin)

    Methods
    -------
    read_settings():
        Read CoMPASS project folder and extract settings for each run.
    read_spectra(modes=['E', 'TOF', 'PSD']):
        Read data from CoMPASS saved histogram.
    add_spectra(filtered=['unfiltered', 'filtered'], modes=['E', 'TOF']):
        Add spectra for specified filtered data and modes
    read_data(filtered=['unfiltered', 'filtered']):
        Read raw data from CoMPASS-generated files.
    add_tof(filtered=['unfiltered', 'filtered']):
        Add TOF column to raw data dataframe.
    user_filter(e_lo=95, e_hi=135, prompt=False):
        Perform user cut of energy spectrum.
    plot_tof(t_lo=0, t_hi=200, n_bins=400,
             filtered='unfiltered', norm=True, add=False):
        Plot manually calculated TOF spectrum.
    plot_spectrum():
        Plot spectrum auto-generated by CoMPASS.
    make_hist(filtered='unfiltered', mode='TOF', ch='CH0',
              val_lo=0, val_hi=200, n_bins=400):
        Create a histogram from TOF values.
    """

    def __init__(self, key, params=None, settings=None, spectra=None, data=None,
                 t_meas=0., file_fmt='.csv',
                 folder='C:/Users/Avram/Dropbox (MIT)/Resonances/data/CoMPASS/'):
        """Constructs all the necessary attributes for the compassRun class.

        Parameters
        ----------
            key : str
                unique run key assigned at creation
            folder : str
                DAQ folder location for CoMPASS project
            params : dictionary
                data acquisition parameters
            settings : dict
                CoMPASS settings read from xml file
            spectra : dict
                auto-generated spectra from CoMPASS
            data : dictionary
                un-/filtered data acquired and stored by CoMPASS
            t_meas : float
                acquisition time in minutes
            file_fmt : string
                file format of saved CoMPASS data (e.g., csv, bin)
        """
        self.key = key
        self.folder = folder
        self.params = params
        self.settings = settings
        self.spectra = spectra
        self.data = data
        self.t_meas = t_meas
        self.file_fmt = file_fmt

    def read_settings(self):
        """Read CoMPASS project folder and extract settings for each run.

        """
        if self.settings is None:
            self.settings = {}
        if self.params is None:
            self.params = {}
        for filt_key in ['unfiltered', 'filtered']:
            self.params[filt_key] = {}
        try:
            xml_fname = Path(self.folder) / self.key / 'settings.xml'
            with open(xml_fname) as f:
                settings_xml = xmltodict.parse(f.read())
            # read board parameters
            for entry in (settings_xml
                          ['configuration']
                          ['board']
                          ['parameters']
                          ['entry']
                          ):
                self.settings[entry['key']] = entry['value']['value']['#text']
            # read acquisition time and convert to minutes
            self.t_meas = float(
                settings_xml
                ['configuration']
                ['acquisitionMemento']
                ['timedRunDuration']
                )
            self.t_meas /= 1000*60
            self.file_fmt = (
                settings_xml
                ['configuration']
                ['acquisitionMemento']
                ['fileFormatList']
                )
        except FileNotFoundError:
            print(
                f'ERROR: Settings file could not be found for {self.key} '
                f'at {xml_fname}'
            )
            return -1
        # store certain settings in params dictionary
        # read parameters for TOF spectra
        self.params['TOF'] = {}
        self.params['TOF']['n_bins'] = int(float(
            self.settings['SW_PARAMETER_DIFFERENCE_BINCOUNT']
        ))
        self.params['TOF']['t_lo'] = float(
            self.settings['SW_PARAMETER_TIME_DIFFERENCE_CH_T0']
        )
        self.params['TOF']['t_lo'] /= 1000  # convert to us
        self.params['TOF']['t_hi'] = float(
            self.settings['SW_PARAMETER_TIME_DIFFERENCE_CH_T1']
        )
        self.params['TOF']['t_hi'] /= 1000  # convert to us
        # read parameters for energy spectra
        self.params['E'] = {}
        self.params['E']['n_bins'] = int(
            self.settings['SRV_PARAM_CH_SPECTRUM_NBINS'].split('_')[1]
        )
        # read parameters for PSD spectra
        self.params['PSD'] = {}
        self.params['PSD']['n_bins'] = int(float(
            self.settings['SW_PARAMETER_PSDBINCOUNT']
        ))
        # read files from run directory
        print('Reading files from directory.')
        file_fmt = '.' + self.file_fmt.lower()
        for filt_key in ['unfiltered', 'filtered']:
            filt_upper = filt_key.upper()
            self.params[filt_key] = {}
            folder_filt = Path(self.folder) / self.key / filt_upper
            try:
                folder_filt.glob("*")
            except OSError:
                print(
                    f'WARNING: Cannot find {filt_upper} folder for {self.key}.'
                )
                break
            # read raw CH0 data location
            [print(str(file)) for file in Path(folder_filt).glob('*' + file_fmt)]
            self.params[filt_key]['file_CH0'] = [
		        str(file) for file in Path(folder_filt).glob('*' + file_fmt) if
		        'CH0' in str(file)
	        ]
            # read raw CH1 data location
            self.params[filt_key]['file_CH1'] = [
		        str(file) for file in Path(folder_filt).glob('*' + file_fmt) if
		        'CH1' in str(file)
	        ]
            # read saved TOF spectra location
            self.params[filt_key]['file_data_TOF'] = [
		        str(file) for file in Path(folder_filt).glob('*' + file_fmt) if
		        'TOF' in str(file)
	        ]
            # read saved E spectra location
            self.params[filt_key]['file_data_E'] = [
		        str(file) for file in Path(folder_filt).glob('*' + file_fmt) if
		        ('CH0' in str(file)) and ('E' in str(file))
            ]
            # read saved PSD spectra location
            self.params[filt_key]['file_data_PSD'] = [
                str(file) for file in Path(folder_filt).glob('*' + file_fmt) if
		        ('CH0' in str(file)) and ('PSD' in str(file))
            ]

    def read_spectra(self, filtered=['unfiltered', 'filtered'],
                     modes=['E', 'TOF', 'PSD']):
        """Read data from CoMPASS saved histograms.

        Parameters
        ----------
            filtered : list[str]
                specifies un-/filtered folders from which to read data
            modes : list[str]
                list of histogram types to read-in
        """
        if self.spectra is None:
            self.spectra = {}
        for filt_key in filtered:
            self.spectra[filt_key] = {}
            for mode in modes:
                key_data = 'file_data_' + mode
                if key_data in self.params[filt_key]:
                    self.spectra[filt_key][mode] = {}
                    try:
                        self.spectra[filt_key][mode]['vals'] = np.array(
                            np.loadtxt(
                                Path(self.folder) / self.key / filt_key.upper() /
                                max(self.params[filt_key][key_data])
                            )
                        )
                    except:
                        print('WARNING: unable to open CoMPASS histogram for '
                              f'{self.key} (mode: {mode})')
                        break
                    if mode == 'TOF':
                        self.spectra[filt_key][mode]['bins'] = np.linspace(
                            self.params['TOF']['t_lo'],
                            self.params['TOF']['t_hi'],
                            self.params['TOF']['n_bins']
                        )
                    else:
                        self.spectra[filt_key][mode]['bins'] = np.arange(
                            self.params[mode]['n_bins']
                        )
                    verbose('Read in CoMPASS spectrum for '
                            f'{self.key} (mode: {mode}, {filt_key})')


    def add_spectra(self, filtered=['unfiltered', 'filtered'],
                    modes=['TOF', 'E']):
        """Add spectrum using stored data.

        Parameters
        ----------
            filtered : str
                specifies whether to use filtered or unfiltered data
            modes : list[str]
                which modes for which to make spectra
        """
        for filt_key in filtered:
            data = self.data[filt_key]['CH0'].copy()
            x = pd.DataFrame()
            for mode in modes:
                print(f'Adding {filt_key} {mode} spectrum...', end="")
                if mode not in self.spectra[filt_key]:
                    self.spectra[filt_key][mode] = {}
                if mode == 'TOF':
                    if mode not in data:
                        self.add_tof(filtered)
                    try:
                        x = data['TOF']
                        n_bins = self.params['TOF']['n_bins']
                        x_lo = self.params['TOF']['t_lo']
                        x_hi = self.params['TOF']['t_hi']
                    except:
                        print('Could not add spectrum. No TOF!')
                elif mode == 'E':
                    try:
                        x = data['ENERGY']
                        n_bins = self.params['TOF']['n_bins']
                        x_lo = 0
                        x_hi = n_bins-1
                    except:
                        print('Could not add spectrum. No energy!')
                elif mode == 'PSD':
                    if 'PSD' not in data:
                        self.add_psd([filt_key])
                    try:
                        x = data['PSD']
                        n_bins = self.params['TOF']['n_bins']
                        x_lo = 0
                        x_hi = n_bins-1
                    except:
                        print('Could not add spectrum. No PSD!')
                if len(x.index) > 0:
                    hist, bin_edges = np.histogram(x, bins=n_bins,
                                                   range=[x_lo, x_hi])
                    bins = bin_edges[:-1] + 0.5*(x_hi-x_lo)/n_bins
                    self.spectra[filt_key][mode]['vals'] = hist
                    self.spectra[filt_key][mode]['bins'] = bins
                    print("Done!")


    def read_data(self, filtered=['unfiltered', 'filtered']):
        """Read raw data from CoMPASS-generated files.

        Parameters
        ----------
            filtered : list[str]
                specifies un-/filtered folders from which to read data
        """
        if self.data is None:
            self.data = {}
        file_fmt = '.' + self.file_fmt.lower()
        for filt_key in filtered:
            self.data[filt_key] = {}
            # attempt to read Ch.0 (detector) data if it exists
            if 'file_CH0' in self.params[filt_key]:
                verbose(f'Reading {filt_key} CH0 data for key: {self.key}...',
			end="")
                try:
                    print(self.folder, self.key)
                    fname = (Path(self.folder) / self.key / filt_key.upper() /
                                self.params[filt_key]['file_CH0'][0])
                    print(fname)
                    if file_fmt == '.csv':
                        self.data[filt_key]['CH0'] = pd.read_csv(fname,
                            sep=';', on_bad_lines='skip')
                    elif file_fmt == 'bin':
                        with open(fname, "rb") as f:
                            byte = f.read()
                        data = struct.unpack(('<'+'HHQHHII'*(len(byte)//24)),
                                             byte)
                        self.data[filt_key]['CH0'] = pd.DataFrame(
                            np.reshape(np.array(data), [-1, 7])[:, 2:5],
                            columns=['TIMETAG', 'ENERGY', 'ENERGYSHORT']
                        )
                    if self.data[filt_key]['CH0'].empty:
                        print('file was empty.')
                    else:
                        print("Done!")
                except:
                    print(f'ERROR: unable to read in {filt_key} CH0 data '
                          f'for {self.key}.')
                    break
                # attempt to read Ch.1 (pulse) data if TOF not yet calculated
                if 'TOF' in self.data[filt_key]['CH0']:
                    continue
                if len(self.params[filt_key]['file_CH1']) > 0:
                    verbose(
                        f'Reading {filt_key} CH1 data for key: {self.key}...',
                        end=""
                    )
                    fname = (Path(self.folder) / self.key / filt_key.upper() /
                                self.params[filt_key]['file_CH1'][0])
                    if file_fmt == '.csv':
                        self.data[filt_key]['CH1'] = pd.read_csv(
                            fname, sep=';', on_bad_lines='skip'
                        )
                    elif file_fmt == 'bin':
                        with open(fname, "rb") as f:
                            byte = f.read()
                        data = struct.unpack(('<'+'HHQHHII'*(len(byte)//24)),
                                             byte)
                        self.data[filt_key]['CH1'] = pd.DataFrame(
                            np.reshape(np.array(data), [-1, 7])[:, 2:5],
                            columns=['TIMETAG', 'ENERGY', 'ENERGYSHORT']
                        )
                    if self.data[filt_key]['CH1'].empty:
                        print('no data found.')
                    else:
                        print("Done!")
                else:
                    print(f'Did not find {filt_key} CH1 data for key: '
			  f'{self.key}.')
            else:
                print(f'Did not find {filt_key} CH0 data for key: '
		      f'{self.key}.')

    def add_tof(self, filtered=['unfiltered', 'filtered']):
        """Add TOF column to raw data dataframe.

        Parameters
        ----------
            filtered : list[str]
                specifies un-/filtered folders from which to read data
        """
        for filt_key in filtered:
            # check if un-/filtered CH0 data present and TOF not yet calculated
            if (filt_key in self.data) and (
		    'CH0' in self.data[filt_key]) and (
                    'TOF' in self.data[filt_key]['CH0']):
                print('TOF already calculated!')
            elif filt_key not in self.data:
                print(f'Could not calculate TOF. No {filt_key} data found!')
            elif 'CH0' not in self.data[filt_key]:
                print(f'Could not calculate TOF. No {filt_key} Ch.0 data found!')
            else:
                # attempt to read in Channel 1 (pulse) data if not yet done
                if ((len(self.params[filt_key]['file_CH1']) > 0) and
                        not(('CH1' not in self.data[filt_key])
                            or self.data[filt_key]['CH1'].empty)):
                    verbose(f'Reading {filt_key} CH1 data for key: {self.key}.')
                    fname = (Path(self.folder) / self.key / filt_key.upper() /
                             self.params[filt_key]['file_CH1'][0])
                    try:
                        self.data[filt_key]['CH1'] = pd.read_csv(
                            fname, sep=';', on_bad_lines='skip'
                        )
                    except:
                        print(f'ERROR: unable to read in {filt_key} CH1 data '
                              f'for {self.key}.')
                        break
                else:
                    print(f'WARNING: TOF not calculated--{filt_key} CH1 data'
                          f'non-existent or empty for {self.key}.')
                    break
                # check that neither CH0/CH1 are empty and timing info saved
                if (all((not self.data[filt_key][ch].empty) and
                        ('TIMETAG' in self.data[filt_key][ch]))
                        for ch in ['CH0', 'CH1']):
                    verbose(f'Calculating {filt_key} TOF for {self.key}')
                    try:
                        # calculate TOF using TIMETAG info
                        self.data[filt_key]['CH0']['TOF'] = calc_TOF(
                            self.data[filt_key]['CH1']['TIMETAG'],
                            self.data[filt_key]['CH0']['TIMETAG']
                        )
                    except:
                        print('ERROR: issue arose calculating TOF.')
                        break
                    try:
                        # rewrite CH0 csv file with TOF added
                        fname = (Path(self.folder) / self.key / filt_key.upper()
                                 / self.params[filt_key]['file_CH0'][0])
                        self.data[filt_key]['CH0'].to_csv(fname, sep=';', 
                                                          index=False)
                    except:
                        print('ERROR: issue arose writing TOF to file.')
                        break
                else:
                    print('ERROR: data empty or timetag not found.')


    def add_psd(self, filtered=['unfiltered', 'filtered'], file_write=True):
        """Add PSD column to raw data dataframe.

        Parameters
        ----------
            filtered : list[str]
                specifies un-/filtered folders from which to read data
            file_write : bool
                specifies whether to write new dataframe with PSD to file
        """
        for filt_key in filtered:
            # check if un-/filtered CH0 data present and TOF not yet calculated
            if 'CH0' not in self.data[filt_key]:
                print('Could not calculate PSD. Ch.0 data not found!')
            elif ('CH0' in self.data[filt_key]) and (
                    'PSD' in self.data[filt_key]['CH0']):
                print('PSD already calculated!')
            else:
                # check that CH0 is not empty
                if not self.data[filt_key]['CH0'].empty:
                    verbose(f'Calculating {filt_key} PSD for {self.key}')
                    try:
                        # calculate PSD
                        data = self.data[filt_key]['CH0']
                        self.data[filt_key]['CH0']['PSD'] = (
                            1 - data['ENERGYSHORT'] / data['ENERGY'])
                    except:
                        print('ERROR: issue arose calculating PSD.')
                        break
                    if 'TOF' not in self.data[filt_key]['CH0']:
                        file_write = click.confirm(
                            '\nNo TOF data was found in dataframe. Are you sure'
                            ' you want to overwrite data file to incldue PSD?',
                            default=False
                        )
                    if file_write:
                        fname = (Path(self.folder) / self.key / filt_key.upper()
                                 / self.params[filt_key]['file_CH0'][0])
                        try:
                            # rewrite CH0 csv file with PSD added
                            self.data[filt_key]['CH0'].to_csv(fname, sep=';', 
                                                              index=False)
                        except:
                            print('ERROR: issue arose writing PSD to file.')
                            break
                else:
                    print('ERROR: Ch.0 data empty.')


    def user_filter(self, e_lo=95, e_hi=135, prompt=False):
        """Perform user cut of energy spectrum.

        Parameters
        ----------
            e_lo: int
                energy low cut (ADC)
            e_hi : int
                energy hi cut (ADC)
            prompt : bool
                flag whether to ask user for cut bounds (True)
                or use defaults (False) if not provided
        """
        print(f'\nUser energy cut requested for TOF spectrum for {self.key}.')
        self.data['user'] = {}
        # choose to prompt user or else use default values
        if prompt:
            e_lo = click.prompt('Set e_lo channel for pulse area cut',
                                default=e_lo)
            e_hi = click.prompt('Set e_hi channel for pulse area cut',
                                default=e_hi)
        # copy all of CH1 pulse data
        self.data['user']['CH1'] = self.data['unfiltered']['CH1'].copy()
        # perform energy cut on unfiltered detector signal data
        self.data['user']['CH0'] = (
            self.data['unfiltered']['CH0']
            .loc[(self.data['unfiltered']['CH0']['ENERGY'] > e_lo) and
                 (self.data['unfiltered']['CH0']['ENERGY'] < e_hi)]
            .copy()
        )
        # if TOF not already calculated, calculate TOF
        if 'TOF' not in self.data['user']['CH0']:
            self.data['user']['CH0']['TOF'] = calc_TOF(
                self.data['unfiltered']['CH1']['TIMETAG'],
                self.data['user']['CH0']['TIMETAG']
            )
        print('Successfully performed user energy cut of TOF spectrum '
              f'for {self.key}.\n')

    def plot_tof(self, t_lo=0., t_hi=200., n_bins=400, color='blue',
                 filtered='unfiltered', norm=True, add=False):
        """Plot manually calculated TOF spectrum.

        Parameters
        ----------
            t_lo : float
                lower bound on TOF
            t_hi : float
                upper bound on TOF
            n_bins : int
                number of histogram bins
            color : str
                color of plot
            filtered : str
                choice of un-/filtered or user filter data to plot
            norm : bool
                whether to normalize data by measurement time
            add : bool
                whether to add plot to existing figure
        """
        if (('CH0' in self.data[filtered]) and
                (self.data[filtered]['CH0'].size >= 0)):
            try:
                x = self.data[filtered]['CH0'].TOF
            except:
                print(f'No TOF data to plot for {self.key}!')
                return
            if not add:
                plt.figure(figsize=(16, 9))
            if norm:
                weights = [1/self.t_meas]*len(x)
                plt.ylabel('COUNTS/MINUTE')
            else:
                weights = [1]*len(x)
                plt.ylabel('COUNTS')
            plt.hist(x, range=[t_lo, t_hi], bins=n_bins, weights=weights,
                     histtype='step', lw=2, color=color,
                     label=(self.key + ' ' + filtered).replace('_', '-'))
            plt.xlim(t_lo, t_hi)
            plt.yscale('log')
            plt.legend(loc='upper right')
            plt.xlabel('TIME (us)')
            plt.tight_layout()
        else:
            print(f'No {filtered} CH0 data found for {self.key}!')

    def plot_spectrum(self, mode='TOF', filtered='unfiltered', add=False):
        """Plot spectrum auto-generated by CoMPASS.

        Parameters
        ----------
            mode : str
                list of histogram types to read-in
            filtered : str
                specifies un-/filtered folder from which to read data
            add : bool
                whether to add plot to existing figure
        """
        if add is False:
            plt.figure(figsize=(16, 9))
        x = self.spectra[filtered][mode]['bins']
        y = self.spectra[filtered][mode]['vals']
        if mode == 'TOF':
            plt.xlim(self.params['TOF']['t_lo'], self.params['TOF']['t_hi'])
            plt.xlabel('TIME (us)')
        else:
            plt.xlim(min(x), max(x))
            plt.xlabel('CHANNEL')
        plt.errorbar(x, y, yerr=[np.sqrt(i) for i in y],
                     capsize=2, drawstyle='steps-mid',
                     label=self.key + ' ' + filtered)
        plt.ylabel('COUNTS')
        plt.ylim(bottom=1)
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.tight_layout()

    def make_hist(self, mode='TOF', filtered='unfiltered', ch='CH0',
                  val_lo=0., val_hi=200., n_bins=400):
        """Create a histogram from TOF values.

        Parameters
        ----------
            mode : str
                list of histogram types to read-in
            filtered : str
                specifies un-/filtered folder from which to read data
            ch : str
                specifies from which channel to read data
            val_lo : float
                lower bound on independent variable
            val_hi :
                upper bound on independent variable
            n_bins : int
                number of histogram bins

        Returns
        -------
            hist : array
                the values of the histogram
            bin_edges : array
                the bin edges
        """
        hist, bin_edges = np.histogram(self.data[filtered][ch][mode].to_numpy(),
                                       range=[val_lo, val_hi], bins=n_bins)
        return hist, bin_edges


def initialize(folders=None, keys=None):
    """Start up CoMPy, the CoMPASS Companion.

    Parameters
    ----------
        folders : list[str]
            DAQ folders from which to select keys
        keys : list[str]
            run keys chosen for processing data

    Returns
    -------
        folders : list[str]
            DAQ folders from which to select keys
        keys : list[str]
            run keys chosen for processing data
        VERBOSE : bool
            flag for printing verbose information about data processing
    """
    print('\nWelcome to CoMPy, the CoMPASS companion for Python!')
    if folders is None:
        folders = []
        while True:
            new_folder = input('\nPlease enter a project folder path '
                           '(ending in DAQ):\n')
            if new_folder and not new_folder.endswith('/DAQ'):
                new_folder += '/DAQ'
            if (not new_folder) and (len(folders) == 0):
                print('You must enter at least one valid folder name!')
                continue
            elif not new_folder:
                break
            try:
                Path(new_folder).resolve()
            except:
                print('The system cannot find the specified folder. '
                      'Please select another folder.')
            if (Path(new_folder).is_dir()) and (new_folder.endswith('DAQ')):
                folder_name = str(Path(new_folder).resolve())
                folders.append(folder_name)
                print(f'appending {folder_name}...')
            else:
                print(f'{new_folder} is not a valid CoMPASS directory.')
    if keys is None:
        keys = select_keys(folders)
    verbose_flag = click.confirm('\nVerbose Mode?', default=True)
    global verbose
    if verbose_flag:
        def verbose(*args, **kwargs):
            """Print additional details about processing data."""
            print(*args, **kwargs)
    else:
        def verbose(*args, **kwargs):
            """Do not print additional details about processing data."""
            return
    return keys, verbose_flag


def select_keys(folders):
    """Select keys to process.

    Parameters
    ----------
        folders : list[str]
            list of run folders to check for individual runs

    Returns
    -------
        keys_select : list[tuple(str, str)]
            list of tuples of (key, folder)
    """
    keys_select = []
    n_folders = len(folders)
    for i, folder in enumerate(folders):
        # folder_key = folder.rsplit('/', maxsplit=3)[1]
        # keys_select[folder_key] = []
        keys_folder = [item.name for item in Path(folder).glob("*")
                       if Path(folder, item.name).is_dir()]
        print('\nAvailable keys to process are: ')
        pprint(keys_folder, compact=True)
        bool_all = click.confirm('\nWould you like to process all keys?',
                                 default=False)
        # process all runs
        if bool_all:
            keys_new = [(key, folder) for key in keys_folder]
            keys_select.extend(keys_new)
            continue
        # manually select runs
        bool_date = click.confirm('\nWould you like to process by date?',
                                  default=False)
        if not bool_date:
            while True:
                n_keys = len(keys_select)
                key_input = input(
                    '\nType \'options\' to see all available options.'
                    '\nPress \'enter\' to end key selection.'
                    '\nEnter key name: '
                )
                # click 'enter' to end key selection for folder
                if not key_input:
                    break
                # if 'enter', but on last folder and no keys selected
                if (not key_input) and (i == n_folders-1) and (n_keys == 0):
                    print('You must enter at least one key name!')
                # print all key options
                elif key_input == 'options':
                    print(keys_folder)
                # if key selected
                else:
                    # if user entry not available key, print warning
                    while key_input not in keys_folder:
                        key_input = input(
                            '\nThat key does not exist.'
                            '\nType \'options\' to see all available options.'
                            '\nPress \'enter\' to end key selection.'
                            '\nEnter key name: '
                        )
                        if key_input == 'options':
                            print(keys_folder)
                    # if good key, append to list
                    keys_select.append((key_input, folder))
        else:
            while True:
                n_keys = len(keys_select)
                date = input('(Note: \nPress \'enter\' to end key selection.)'
                             '\nEnter run date: ')
                # click 'enter' to end key selection
                if ((not date) and (n_keys == 0)):
                    break
                if (n_keys != 0 and
                        not any(key.startswith(date) for key in keys_folder)):
                    print('Bad key provided.')
                # if 'enter', but no keys selected
                elif ((not any(key.startswith(date) for key in keys_folder))
                      and n_keys == 0):
                    print('Bad key provided. '
                          'You must enter at least one key name!')
                # if good key, append to list
                elif any(key.startswith(date) for key in keys_folder):
                    for key in keys_folder:
                        if ((key.startswith(date)) and
                                ((key, folder) not in keys_select)):
                            keys_select.append((key, folder))
    return keys_select


def process_runs(key_tuples, runs=None):
    """Read in settings, spectra, and data for specified runs."""
    if runs is None:
        runs = {}
    bool_TOF = click.confirm('Would you like to manually calculate TOF?',
                             default='Y')
    for (key, folder) in key_tuples:
        if key != 'CoMPASS':
            print('\n' + f'Processing Key: {key}...')
            run = CompassRun(key=key, folder=folder)
            err = run.read_settings()
            if err == -1:
                continue
            run.read_data()
            run.read_spectra()
            if bool_TOF:
                run.add_tof()
            runs[key] = run
    return runs
