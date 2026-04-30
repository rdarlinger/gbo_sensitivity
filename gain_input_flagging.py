import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dateutil
import datetime, pickle
import glob
import time
import re
# from outriggers_vlbi_pipeline.query_database import get_event_data
from chime_frb_api.backends import frb_master
from baseband_analysis.core.dedispersion import incoherent_dedisp, coherent_dedisp
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import scrunch
from baseband_analysis.core.flagging import get_RFI_channels

import pandas as pd
import glob
import datetime
import numpy as np
import warnings
import h5py
import csv

import matplotlib.pylab as pylab
params = {'axes.labelsize':20,
         'axes.titlesize':20,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)

import baseband_analysis
from ch_util.ephemeris import CasA, CygA, TauA, VirA, chime, gbo, hco, kko, unix_to_datetime
from ch_util import tools

from baseband_analysis.pipelines.config import backends_dict
from astropy.coordinates import SkyCoord
import astropy.units as u

import importlib
from baseband_analysis.analysis import beamform
from baseband_analysis.analysis.toa import get_TOA
import ch_util
from baseband_analysis.pipelines import outrigger_beamform
from baseband_analysis.core import calibration as cal
import baseband_analysis.core.bbdata as bbdata
import json
from scipy.stats import median_abs_deviation
from ch_util.tools import Blank
from IPython.display import Image
from baseband_analysis.core.signal import tiedbeam_baseband_to_power
import argparse
from ch_util.tools import get_correlator_inputs, fringestop_time, cmap, KKOAntenna, GBOAntenna, HCOAntenna
from matplotlib.colors import LogNorm
from caput.time import skyfield_wrapper
import scipy.linalg as la
from datetime import timedelta, timezone
from astropy.time import Time
import bisect
import os
import subprocess

from astropy.coordinates import EarthLocation
from scipy.optimize import minimize_scalar
#from outriggers_vlbi_pipeline.src.outriggers_vlbi_pipeline.multibeamform import get_best_gain_calibrator 
from outriggers_vlbi_pipeline.multibeamform import get_best_gain_calibrator 


def _get_src_transit_from_hdf5(path_to_hdf5, src, obs = chime):
    """"""
    files = sorted(os.listdir(path_to_hdf5))
    start_data = h5py.File(path_to_hdf5+files[0], 'r')
    end_data = h5py.File(path_to_hdf5+files[-1], 'r')
    start_time = unix_to_datetime(start_data['index_map']['time']['ctime'][0])
    print("start time to look for source", start_time)
    end_time = unix_to_datetime(end_data['index_map']['time']['ctime'][-1])
    print("end time to look for source", end_time)

    del start_data
    del end_data

    return obs.transit_times(src, start_time, end_time)

def _find_hdf5_from_unix(path_to_hdf5, transit_times):
    """Find h5 file within a directory of h5 files containing timestamp closest to the one you specified."""

    if type(transit_times)!=np.ndarray:
        transit_times = np.asarray(transit_times)

    files = sorted(os.listdir(path_to_hdf5))
    filenames = []
    file_idxs = []
    unix_times = []
    transit_idxs = []

    for t in transit_times:

        for file_idx, f in enumerate(files):
            _data = h5py.File(path_to_hdf5+f, 'r')
            
            # Index times contained in the file
            _times = _data['index_map']['time']['ctime']
            start=_times[0]
            if _times[-1] < t:
                pass

            else:
                diff_t = np.abs(_times - t)
                idx_t = np.where(diff_t == np.min(diff_t))[0][0]

                filenames.append(f)
                file_idxs.append(file_idx)
                unix_times.append(_times[idx_t])
                transit_idxs.append(idx_t)

                break
                
    return filenames, file_idxs, unix_times, transit_idxs


def _get_connected_inputs(unix_timestamp, correlator='FCG',inputmap = None):
    """Retrieve connected inputs at a given time.

    Parameters:
    -----------
    unix_timestamp: float
        Time at which you wish to know what feeds are connected.
    correlator: str
        Correlator string, default set to allenby correlator.

    Returns: list
        List of indices of connected inputs.
    """

    if inputmap is None:
        datetime = unix_to_datetime(unix_timestamp)
        inputs = pd.read_pickle('/arc/projects/chime_frb/rdarlinger/gboinputs_correct.pkl')

    else:
        inputs = inputmap

    connected_inputs = []
    for antenna in inputs:
        if type(antenna) == KKOAntenna: # check if Antenna is connected.
            connected_inputs.append(antenna.id)
        elif type(antenna) == GBOAntenna:
            connected_inputs.append(antenna.id)
        elif type(antenna) == HCOAntenna:
            connected_inputs.append(antenna.id)
        else:
            continue
    return connected_inputs



def get_gains_from_N2(path_to_h5_files, transit_times=None, src_str="cyga", gains_output_dir = '/arc/projects/chime_frb/rdarlinger/input_fix_gain_check/', 
                      correlator = 'FCG', obs=gbo, badinps=None,input_pkl_file='/arc/projects/chime_frb/rdarlinger/gboinputs_correct.pkl', 
                      plot_output_dir="/arc/projects/chime_frb/rdarlinger/input_fix_gain_check/plots/",
                      median=None,percent_of_band=0.1,max_dev=10000.):
    if src_str == "cyga":
        src = CygA
    elif src_str == "casa":
        src = CasA
    elif src_str == "taua":
        src = TauA
    elif src_str == "vira":
        src = VirA
    numfiles = len(os.listdir(path_to_h5_files))
    #if numfiles == 1:
       # return
    if transit_times ==None:
        transit_times = _get_src_transit_from_hdf5(path_to_h5_files, src)
    
    print("Transit times:",transit_times)
    filenames, file_idxs, unix_times, transit_idxs = _find_hdf5_from_unix(path_to_h5_files, transit_times)
    print("Filenames:", filenames)

    for i, f in enumerate(filenames):
        _data = h5py.File(path_to_h5_files+f)
        _index_map = _data['index_map']
        _timestamps = _index_map['time']['ctime']
        _transit_time = _timestamps[transit_idxs[i]:transit_idxs[i]+1]

        filename = unix_to_gain_name(_transit_time[0], src_str)
        filepath = os.path.join(gains_output_dir, filename)
        if os.path.isfile(filepath):
            cmd= f"rm {filepath}"
            subprocess.run(cmd, shell=True, check=True)
            
        if not os.path.isfile(filepath):
            print('Calculating gains for {0} transit at unix time {1}, {2}'.format(src_str, unix_times[i], filepath))
            _freqs = _data['index_map']['freq']

            _vis = _data['vis']
            
            
            if not input_pkl_file: 
                datetime = unix_to_datetime(unix_times[i])
                _inputmap = get_correlator_inputs(datetime, correlator=correlator)
            else: 
                _inputmap = pd.read_pickle(input_pkl_file)
                
            _connected_inps= _get_connected_inputs(
                unix_times[i],
                correlator=correlator,
                inputmap=_inputmap
            )
            
            
            if badinps is not None:
                flagged_ids = badinps
            else:
                flagged_ids = []# [15,16,21,22,23,24,30,31,32,36,37,38,43,
                  #44,63,64,67,68,70,71,78,79,80,84,85,88,89,93,94,
                  #112,113,116,117,130,131,145,146,
                  #153,154,155,156,157,158,159,165,
                  #166,180,181,186,187,201,202,214,
                  #215,216,240,241,243,246,247
                  #]

            gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)
            
            # Looking for new channels to flag 
            if median is not None: 
                print('Looking for bad channels...')
                flagged_inputs_new = filter_bad_inputs(gains,median,percent_of_band=percent_of_band,max_dev=max_dev)
                if len(flagged_inputs_new) > 0: 
                    
                    flagged_ids = list(flagged_inputs_new) + flagged_ids
                    gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)
                    num_flagged_ids=len(flagged_ids)
                    print("Number of bad inputs:", num_flagged_ids)
                else:
                    num_flagged_ids=len(flagged_ids)
                    print("Number of bad inputs:", num_flagged_ids)
            else: 
                print(f'Flagging any inputs with gains that exceed {max_dev} medians over {percent_of_band}')
                median = get_simple_median(gains)
                flagged_inputs_new = [] #filter_bad_inputs(gains,median,percent_of_band=percent_of_band,max_dev=max_dev)
                if len(flagged_inputs_new) > 0: 
                    flagged_ids = list(flagged_inputs_new) + flagged_ids
                    num_flagged_ids=len(flagged_ids)
                    print("Number of bad inputs:", num_flagged_ids)
                    gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)
                else:
                    num_flagged_ids=len(flagged_ids)
                    print("Number of bad inputs:", num_flagged_ids)
            #print("Bad inputs:", flagged_ids)
            print("Beginning writeout process...")
            if gains_output_dir is not None:
                filename = unix_to_gain_name(_transit_time[0], src_str)
                filepath = os.path.join(gains_output_dir, filename)
                #print(
                   # f'{dt.datetime.now().strftime("%Y%m%dT%H%M%S")}: Saving gains to {filepath}'
                #)
                with h5py.File(filepath, "w") as h5pyfile:
                    h5pyfile.attrs['src_name'] = src_str
                    h5pyfile.create_dataset("gain", data = gains)
                    h5pyfile.create_dataset("weight", data = weights)
                    h5pyfile.create_dataset("gain_err", data = gain_err)
                    h5pyfile.create_dataset("index_map/freq", data = _freqs)
                    h5pyfile.create_dataset("index_map/input", data = _index_map['input'])
                    h5pyfile.create_dataset("flagged_ids", data =flagged_ids)
            file=h5py.File(filepath, "r")
            gain=np.asarray(file['gain'])
            
            datetime_obj = unix_to_datetime(unix_times[i]).astimezone(timezone.utc)
            filepath_i=os.path.join(plot_output_dir, f"gain_{datetime_obj}.png")
            plt.imshow(np.abs(gain), norm=LogNorm(vmin=1,vmax=10))
            plt.title(f"Gain for {datetime_obj}")
            plt.xlabel("Input")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.savefig(filepath_i)
            plt.close()



    return num_flagged_ids, filepath

def filter_bad_inputs(gains, median, percent_of_band=0.1, max_dev = 5.): 
    '''
    Basic bad input filter that looks at median over all inputs (from a good gain file) 
    and masks any inputs that fall beyond 5 medians away. 
    
    Parameters:
    -----------
    gains: ndarray 
        Array of shape (nfreq, ninps) where ninps corresponds to the number of inputs. 
    median: str
        Path to .npy file containing median over E and S pol from a good gain file. 
    percent_of_band: float 
        Percent of bandwidth that must be above specified threshold to determine a determine a bad feed.
    max_dev: float 
        Number of medians away to determine flagging, default 5. 
    
    Returns: 
    --------
    flagged_inputs: ndarray 
        List containing ch_id of flagged inputs. 
    
    '''
    
    # Load in median previously computed
    med_E, med_S = np.load(median)
    
    ninps = gains.shape[-1]
    nfreqs = gains.shape[0]
    
    E_pol = np.abs(gains[:,:ninps//2])
    S_pol = np.abs(gains[:,ninps//2:])
        
    flagged_inputs = []
    # Iterate over inputs, and flag 
    for i in range(ninps//2): 
        
        bad_freqs_E = np.where(E_pol[:,i] > max_dev*med_E)[0]
        bad_outliers_E =[] # np.where(E_pol[:,i] > 200*med_E)[0]
        if (len(bad_freqs_E)> percent_of_band*nfreqs) or (len(bad_outliers_E)>0):
            flagged_inputs.append(i)
            
        bad_freqs_S = np.where(S_pol[:,i] > max_dev*med_S)[0]
        bad_outliers_S = np.where(S_pol[:,i] > 200*med_S)[0]
        if (len(bad_freqs_S)> percent_of_band*nfreqs) or (len(bad_outliers_S)>0):
            flagged_inputs.append(i+ninps//2)
    
    return flagged_inputs

def _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i): 
    # Get good x and y inputs
    _good_inputs_x = [
            inp.id
            for inp in _inputmap
            if (inp.id in _connected_inps and inp.pol == "E")
        ]

    _good_inputs_y = [
            inp.id
            for inp in _inputmap
            if (inp.id in _connected_inps and inp.pol == "S")
        ]
    # Remove flagged inputs
    for ch_id in flagged_ids:
        if ch_id in _good_inputs_x:
            _good_inputs_x.remove(ch_id)
        if ch_id in _good_inputs_y:
            _good_inputs_y.remove(ch_id)

    _good_inputs_x = np.asarray(_good_inputs_x)
    _good_inputs_y = np.asarray(_good_inputs_y)

    print("Mapping connected inputs")
    _connected_inps = np.concatenate((_good_inputs_x, _good_inputs_y))


    _n_good_inputs_x, _n_good_inputs_y = len(_good_inputs_x), len(_good_inputs_y)
    _good_inputs = [_good_inputs_x, _good_inputs_y]
    _n_good_inputs = [_n_good_inputs_x, _n_good_inputs_y]
    _num_ant = len(_inputmap)

    # Index correlator products
    print("Indexing correlator products")
    _corprods_x = [cmap(_good_inputs_x[i], _good_inputs_x[j], _num_ant) for i in range(_n_good_inputs_x)
                                                                     for j in range(i, _n_good_inputs_x)]
    _corprods_y = [cmap(_good_inputs_y[i], _good_inputs_y[j], _num_ant) for i in range(_n_good_inputs_y)
                                                                         for j in range(i, _n_good_inputs_y)]
    _vis_pp = [_vis[:, _corprods_x, transit_idxs[i]:transit_idxs[i]+1], _vis[:, _corprods_y, transit_idxs[i]:transit_idxs[i]+1]]
    _n_vis = [_vis_pp[pp].shape[1] for pp in range(2)]

    # Update product map
    _prod_map = [np.empty(_n_vis[pp], dtype=[("input_a", "u2"), ("input_b", "u2")]) for pp in range(2)]

    print("Making product map")
    for pp in range(2):
        _row_index, _col_index = np.triu_indices(_n_good_inputs[pp]) # row and col indices for upper triangular matrix
        _prod_map[pp]["input_a"], _prod_map[pp]["input_b"] = _row_index, _col_index



    _good_correlator_inputs = [[_inputmap[i] for i in _good_inputs[pp]] for pp in range(2)]

    print("Fringestopping...")
    # Fringestop visibilities
    _vis_fs = [
        fringestop_time(
            timestream=_vis_pp[pp],
            times=_transit_time,
            freq=_freqs['centre'],
            feeds=_good_correlator_inputs[pp],
            src=src,
            prod_map=_prod_map[pp],
            obs=obs,
            static_delays=False # Current workaround
        ) for pp in range(2)
                 ]


    print("Solving gains... ")
    # Calculate gains
    _ev, _g, _g_err = [], [], []
    for pp in range(2):
        evalues, gains, gain_errors = solve_gain(_vis_fs[pp])
        _ev.append(evalues)
        _g.append(gains)
        _g_err.append(gain_errors)



    # Set all flagged/unconnected inputs to zero
    gains = np.zeros(shape=(len(_freqs),len(_inputmap)), dtype=np.complex64)

    weights = np.zeros(shape=(len(_freqs),len(_inputmap)), dtype=np.float64)

    gain_err = np.zeros(shape=(len(_freqs),len(_inputmap)), dtype=np.complex64)

    for pp in range(2):

        if pp == 0: # XX pol
            gains[:, _good_inputs_x] = invert_no_zero(_g[pp][:,:,0])
            weights[:, _good_inputs_x] = _ev[pp][:,:,0]
            gain_err[:, _good_inputs_x] = _g_err[pp][:,:,0] * (invert_no_zero(_g[pp][:,:,0])) ** 2

        if pp == 1: # YY pol
            gains[:, _good_inputs_y] = invert_no_zero(_g[pp][:,:,0])
            weights[:, _good_inputs_y] = _ev[pp][:,:,0]
            gain_err[:, _good_inputs_y] = _g_err[pp][:,:,0] * (invert_no_zero(_g[pp][:,:,0])) ** 2

            
    return gains, weights, gain_err

def mat2utvec(A):
    """Vectorizes its upper triangle of the (hermitian) matrix A.

    Parameters
    ----------
    A : 2d array
        Hermitian matrix

    Returns
    -------
    1d array with vectorized form of upper triangle of A

    Example
    -------
    if A is a 3x3 matrix then the output vector is
    outvector = [A00, A01, A02, A11, A12, A22]

    See also
    --------
    utvec2mat
    """

    iu = np.triu_indices(np.size(A, 0)) # Indices for upper triangle of A

    return A[iu]

def utvec2mat(n, utvec):
    """Recovers a hermitian matrix a from its upper triangle vectorized version.

     Parameters
     ----------
     n : int
         order of the output hermitian matrix
     utvec : 1d array
         vectorized form of upper triangle of output matrix

    Returns
    -------
    A : 2d array
        hermitian matrix
    """

    iu = np.triu_indices(n)
    A = np.zeros((n, n), dtype=complex)
    A[iu] = utvec # Filling uppper triangle of A
    A = A+np.triu(A, 1).conj().T # Filling lower triangle of A
    return A

def rankN_approx(A, rank=1):
    """Create the rank-N approximation to the matrix A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to approximate
    rank : int, optional

    Returns
    -------
    B : np.ndarray
        Low rank approximation.
    """

    N = A.shape[0]

    evals, evecs = la.eigh(A, subset_by_index=[N - rank, N - 1])

    return np.dot(evecs, evals * evecs.T.conj())


def eigh_no_diagonal(A, niter=5, eigvals=None):
    """Eigenvalue decomposition ignoring the diagonal elements.

    The diagonal elements are iteratively replaced with those from a rank=1 approximation.

    Parameters
    ----------
    A : np.ndarray[:, :]
        Matrix to decompose.
    niter : int, optional
        Number of iterations to perform.
    eigvals : (lo, hi), optional
        Indices of eigenvalues to select (inclusive).

    Returns
    -------
    evals : np.ndarray[:]
    evecs : np.ndarray[:, :]
    """

    Ac = A.copy()

    if niter > 0:
        Ac[np.diag_indices(Ac.shape[0])] = 0.0

        for i in range(niter):
            Ac[np.diag_indices(Ac.shape[0])] = rankN_approx(Ac).diagonal()

    return la.eigh(Ac, subset_by_index=eigvals)

def eigh_special(A, zero_indices, niter=5):

    Ac = A.copy()

    if niter > 0:
        Ac[zero_indices] = 0.0

        for i in range(niter):
            Ac[zero_indices] = rankN_approx(Ac)[zero_indices]

    return la.eigh(Ac)

def invert_no_zero(x):
    """Return the reciprocal, but ignoring zeros.
    Where `x != 0` return 1/x, or just return 0. Importantly this routine does
    not produce a warning about zero division.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    r : np.ndarray
        Return the reciprocal of x.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x == 0, 0.0, 1.0 / x)

def solve_gain(data, zero_indices=None):
    """
    Steps through each time/freq pixel, generates a Hermitian matrix and
    calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : np.ndarray[nfreq, nprod, ntime]
        Visibility array to be decomposed

    Returns
    -------
    evalue : np.ndarray[nfreq, nfeed, ntime]
        Eigenvalues obtained from eigenvalue decomposition
        of the visibility matrix.
    gain : np.ndarray[nfreq, nfeed, ntime]
        Gain solution for each feed, time, and frequency
    gain_error : np.ndarray[nfreq, nfeed, ntime]
        Error on the gain solution for each feed, time, and frequency
    """

    Nfeeds = int((2 * data.shape[1])**0.5)
    Nfreqs, Ntimes = data.shape[0], data.shape[-1]

    # Create empty arrays to store the outputs
    gain = np.zeros((data.shape[0], Nfeeds, data.shape[-1]), complex)*np.NaN
    gain_error = np.zeros((data.shape[0], Nfeeds, data.shape[-1]), float)*np.NaN
    evalue = np.zeros((data.shape[0], Nfeeds, data.shape[-1]), float)*np.NaN

    # indices of autos
    i_autos = [k*Nfeeds-(k*(k-1))//2 for k in range(Nfeeds)]

    # Set up normalisation matrix
    norm = (data[:, i_autos, :].real)**0.5
    norm = invert_no_zero(norm)

    # Pre-generate the array of inverted norms
    inv_norm = invert_no_zero(norm)

    # Iterate over frequency/time and solve gains
    for fi in range(Nfreqs):
        if np.any(np.logical_or(np.isnan(data[fi]), np.isinf(data[fi]))):
            continue

        for ti in range(Ntimes):
            # Form correlation matrix
            cd = utvec2mat(Nfeeds, data[fi, :, ti])

            # Apply weighting
            w = norm[fi, :, ti]
            cd *= np.outer(w, w.conj())

            # Solve for eigenvectors and eigenvalues
            if zero_indices is None:
                evals, evecs = eigh_no_diagonal(cd, niter=5)
            else:
                evals, evecs = eigh_special(cd, zero_indices, niter=8)

            # Construct gain solutions
            if evals[-1] > 0:
                sign0 = (1.0 - 2.0 * (evecs[0, -1].real < 0.0))
                gain[fi, :, ti] = sign0 * inv_norm[fi, :, ti] * evecs[:, -1] * evals[-1]**0.5

                gain_error[fi, :, ti] = (inv_norm[fi, :, ti] *
                                         1.4826 * np.median(np.abs(evals[:-1] - np.median(evals[:-1]))) /
                                         ((Nfeeds-1)*evals[-1])**0.5)

                evalue[fi, :, ti] = evals

    return evalue, gain, gain_error

def check_if_exists(date, path='/data/kko/acquisition/daily_gain_solutions/daily_monitoring_plots/'):
    return os.path.exists(os.path.join(path,'{0}.png'.format(date)))

def unix_to_gain_name(unix_t, src):
    datetime_obj = unix_to_datetime(unix_t).astimezone(timezone.utc)
    s = str(src)
    src_string = (s.replace('_', '')).lower()
    return f'gain_{datetime_obj.strftime("%Y%m%dT%H%M%S.%fZ")}_{src_string}.h5'

def get_bad_inputs(path_to_txt_file): 
    
    try: 
        glob.glob(path_to_txt_file)[0]
    except: 
        return []
    
    bad_inputs_list = []
    with open(path_to_txt_file, 'r') as file:
        for line in file:
            parts = line.split('ch_id')
            if len(parts) > 1:
                chid = parts[1].split(',')[0].strip()
                bad_inputs_list.append(int(chid))
    return bad_inputs_list

def get_simple_median(gains): 
    
    ninps = gains.shape[-1]
    E_pol = gains[:,:ninps//2]
    S_pol = gains[:,ninps//2:]
    median_E = np.median(np.abs(E_pol), axis=1)
    median_S = np.median(np.abs(E_pol), axis=1)

    stacked_median = np.vstack((median_E,median_S))
    print("Stacked median max is:", np.max(stacked_median))
    np.save('tmp.npy',stacked_median)
    return 'tmp.npy'
    
''' if __name__ == "__main__":

    while True: 
        telescope = os.environ.get("SITE_NAME")
        output_dir = os.environ.get("GAINS_PATH")
        correlator = os.environ.get("CORRELATOR")
        bad_inputs_path = os.environ.get("BAD_INPS")
        input_pkl_file = os.environ.get("INPUTMAP_PKL")
        sleep_time_sec = os.environ.get("SLEEP_TIME_SECONDS")
        median = os.environ.get("MEDIAN")
        percent_of_band = float(os.environ.get("PERCENT"))
        max_dev = float(os.environ.get("MAX_DEV"))
        
        print(telescope,output_dir,correlator,bad_inputs_path,input_pkl_file)

        acq_path = f'/data/{telescope}/n_squared/'
        
        bad_inputs_list = get_bad_inputs(bad_inputs_path)

        obs_dict = {'kko':kko, 'gbo':gbo, 'hco':hco}

        # Get telescope, correlator and obs
        try:
            obs = obs_dict[telescope]
        except:
            raise OSError(
                "Unsupported telescope. Available options are: chime, tone, kko, hco and gbo. Please try again."
            )

        acq_names_list = sorted(os.listdir(acq_path))[-6:]

        print('Searching for data in the following folders: {0}'.format(acq_names_list))
        sources = [CasA, CygA]
        for acq_name in acq_names_list[:]:
            for src in sources:
                try:
                    print(acq_path)
                    print(acq_name)
                    _ = get_gains_from_N2(
                            acq_path=acq_path,
                            acq_name=acq_name,
                            src=src,
                            gains_output_dir=output_dir,
                            correlator=correlator,
                            obs=obs,
                            badinps=bad_inputs_list,
                            input_pkl_file=input_pkl_file,
                            median=median,
                            percent_of_band=percent_of_band,
                            max_dev=max_dev
                     )
                except Exception as e:
                    print(e)
                    print('Skipping {0} for source {1}'.format(acq_name, src))
                    continue
        
        print("Script finished. Sleeping for 1 minute before restarting...\n")
        time.sleep(float(sleep_time_sec))  # Sleep for 60 seconds before restarting '''

def get_eval_spec(evals, freqs):

    nant = evals.shape[-1]
    evals_x = (evals[:, nant//2:nant])
    evals_y = (evals[:, :nant//2])

    # Sort in increasing order, we will compute max / std(remaining)
    evals_x.sort(axis = 1)
    evals_y.sort(axis = 1)

    ratio = np.ones(shape=(2,evals.shape[0]), dtype=np.float64)

    for j in range(freqs.size):

        ratio[0,j] = evals_x[j, -1]/np.std(evals_x[j, :-2])
        ratio[1,j] = evals_y[j, -1]/np.std(evals_y[j, :-2])

    return ratio



def plot_diagnostic(
    date,
    gains_dir = '/arc/projects/chime_frb/rdarlinger/gain_solutions',
    plots_dir = '/arc/projects/chime_frb/rdarlinger/gain_solutions/diagnositc_plots',
    ):

    # Get gains
    gain_files = sorted(os.listdir(gains_dir))
    files_use = []
    print('\nSearching for files beginning with gain_{0}{1:02d}{2:02d}'.format(date.year,date.month,date.day))
    for f in gain_files:
        if f.startswith('gain_{0}{1:02d}{2:02d}'.format(date.year,date.month,date.day)):
            files_use.append(f)
            print('Found file {0}'.format(f))


    num_gainfiles = len(files_use)
    # Determine which sources are present
    endings = [f[-8:-3] for f in files_use]
    print('Found gain files for {0}'.format(endings))

    fig, axs = plt.subplots(4,2,figsize=(10,15), constrained_layout=True)

    for i in range(4):

        pol = ['XX', 'YY']
        colors = ['dodgerblue','darkorange']

        try:

            f = files_use[i]
            # Read in gain file
            print('Reading {0}'.format(f))
            _data = h5py.File('{0}{1}'.format(gains_dir, f))
            _gains = _data['gain']
            _freqs = _data['index_map']['freq']['centre']
            _evals = _data['weight']

            _eval_spec = get_eval_spec(_evals, _freqs)


            # First plot is waterfall of gains, useful to see if any feeds are acting up
            im = axs[i,0].imshow(
                np.abs(_gains),
                aspect='auto',
                norm=LogNorm(vmin=1,vmax=10),
                extent=[0,256,400,800],
                origin='upper',
                cmap='inferno'
            )
            axs[i,0].text(10, 750, endings[i], fontweight='bold', fontsize='12', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
            axs[i,0].set_ylabel('ν, MHz')
            axs[i,0].set_xlabel('Input Number')
            plt.colorbar(
                im,
                ax=axs[i,0],
                orientation="vertical",
            )

            for j in range(2):
                # Second plot is eval spectrum, useful to get a sense of how the system as a whole is doing
                axs[i,1].plot(
                    _freqs,
                    _eval_spec[j,:],
                    '.',
                    label=pol[j],
                    color=colors[j],
                    linestyle='None'
                )
            axs[i,1].set_yscale('log')
            axs[i,1].grid()
            axs[i,1].legend(loc='upper left')
            axs[i,1].set_ylabel(r'$\lambda_1 / \sigma_\lambda$')
            axs[i,1].set_xlabel('ν, MHz')
            axs[i,1].set_ylim(10,1e3)

            fig.suptitle(date)

            axs[0,0].set_title('Gains')
            axs[0,1].set_title('Eigenvalue Spectrum')

            print('Plotted gains from {0}'.format(f))

        except:
            continue

    savefig_to = os.path.join(plots_dir,'{0}.png'.format(date))
    plt.savefig(savefig_to, dpi=100, format='png')
    plt.close()
    print('Diagnostic plot saved to {0}'.format(savefig_to))
    
def find_files(file, file_path, toa_from_singlebeam, src_str="cyga", telescope="gbo"):
    """
    Finds transit time of source closest to toa specified and finds corresponding file to that transit time plus checks if there are 20 minutes on either side
    of the given transit time
    NOTE: Make sure to run get_best_gain_calibrator first to find the proper calibrator based on the solar transit
    
    NOTE: For use, you need a file of the names of the N2 files in the container folder which you can 
    
    get by "datatrail ls gbo.acquisition.processed commissioning"
    to look at the container files, then "datatrail ps gbo.acquisition.processed 20241204T174051Z_gbo_corr --show-files > name.txt" to get the txt file
    Parameters
    -------------
    file: str
        file path to .txt file with names of N2 files
    file_path: str
        file path to name of file to output the names of the correct file to
            in the form of just the root name of the files to put them into
    toa_from_singlebneam: float
        TOA from the just_toa or run_waterfall_from_singlebeam function
    src_str: str
        Name of source to find correct time and files for, cyga or casa for gbo
    telescope: str
        Name of telescope to find files for 
    Returns
    ---------
    file_transit_inital: : array
        The transit time needed to make gains based on the time provided.
       
    Pulls files from datatrail using created file. """
    obs_dict = {'kko':kko, 'gbo':gbo, 'hco':hco, 'chime':chime}
    obs = obs_dict[telescope]
    if src_str == "cyga":
        src = CygA
    elif src_str == "casa":
        src = CasA
    elif src_str == "taua":
        src = TauA
    elif src_str == "vira":
        src = VirA
        
    with open(file, 'r') as f:
        lines = f.readlines()
    file_2=None
    common_path = None
    file_data=[]

    for line in lines:
        # Look for the common path line
        if 'Common Path:' in line:
            # Extract the path after the colon
            common_path_pre = line.split('Common Path:')[1]
            # Capture the sequence of numbers and letters before the first underscore in the common path
            match_common_path = re.match(r'.*/(\w+)_(\w+)_', common_path_pre)  # This looks for word characters followed by an underscore
            if match_common_path:
                common_path=match_common_path.group(1)
                print("Container file is", common_path)
        # find filenames 
        match_file_ids = re.match(r'\s*│\s*-\s*(\d{8}_\d{4}\.h5)', line.strip())
        if match_file_ids:
            name = match_file_ids.group(1)
            file_id_float = float(name.split('_')[0])
            file_data.append((file_id_float, name))
    file_data.sort(key=lambda x: x[0])
    file_path=file_path+"_"+common_path+".txt"

    # Split into separate lists (now aligned)
    file_ids = [fid for fid, _ in file_data]
    file_names = [fname for _, fname in file_data]
    iso_time = f"{common_path[:4]}-{common_path[4:6]}-{common_path[6:8]}T{common_path[9:11]}:{common_path[11:13]}:{common_path[13:15]}"

    print(f"Formatted ISO time: {iso_time}")

    # Convert the formatted string to an astropy Time object
    time_obj = Time(iso_time, format='isot', scale='utc')

    # Convert to Unix time
    common_path_unix_time = time_obj.unix
    print("Common path unix time",common_path_unix_time)
    time_to_match=toa_from_singlebeam
    print("Time to match is", time_to_match)
    
    start_time = min(file_ids)+common_path_unix_time # or specify manually
    print("Start time in unix time is:", start_time)

    end_time = max(file_ids)+common_path_unix_time
    print('End time in unix time is:', end_time)
    
    while True:
        transit_times = np.asarray(chime.transit_times(src, start_time, end_time))  #For GBO observations, need to get transit time at CHIME
        print("Transit times are:", transit_times)

        index = bisect.bisect_left(transit_times, time_to_match)
        file_transit_initial = transit_times[index - 1]  # previous transit
        print("Closest transit time is:", file_transit_initial)

        file_transit = file_transit_initial - common_path_unix_time
        print("Closest transit in time from container file is:", file_transit)

        index2 = bisect.bisect_left(file_ids, file_transit)
        file = file_ids[index2 - 1]
        matching_file_name = file_names[index2 - 1]

        # If the time gap between the file and needed transit is too large, try next day
        if file_transit - file > 86400:
            print("File is more than a day away from required time. Trying next day.")
            time_to_match += 86400  # move forward one day
            continue  # restart loop with updated time

        # Otherwise, valid match found; break out
        break

    #check for second file within ±20 mins
    file_2 = None
    if index2 < len(file_ids):
        if file_ids[index2] - file_transit < 1200:
            file_2 = file_ids[index2]
    if index2 - 1 >= 0:
        if file_transit - file_ids[index2 - 1] < 1200:
            file_2 = file_ids[index2 - 2]

    file_2_name = None
    if file_2 is not None:
        i2 = file_ids.index(file_2)
        file_2_name = file_names[i2]

    # Write both filenames to file_path.txt
    with open(file_path, "a") as f:
        f.write(f"{matching_file_name}\n")
        if file_2_name:
            f.write(f"{file_2_name}\n")
    print("File containing time:", file, "and next file time to make sure:", file_ids[index2], "File 2 is", file_2)
    
    subprocess.run(["pip", "install", "--upgrade", "datatrail-cli"])
    cmd = f"echo Y | datatrail pull -s {file_path} {telescope}.acquisition.processed {common_path}_{telescope}_corr"
    subprocess.run(cmd, shell=True, check=True)
    
    return [file_transit_initial]

def find_files_reduced(time, file_path="to_pull.txt", commissioning="/arc/projects/chime_frb/rdarlinger/commissioning_files.txt", telescope="gbo"):
    """
    
    Parameters
    -------------
    time: float
        time to create gains for
    file_path: str
        file path to output the name of the N2 file. By default it overwrites this file since it should be the same every time
    commissioning: str
        File path of the txt file with the names of the commissioning folders. Created using the command 
        "datatrail ls gbo.acquisition.processed commissioning > commissioning_files.txt"
    telescope: str
        Name of telescope to find files for 
    Returns
    ---------
    transit_time: array
        The transit time needed to make gains based on the time provided.
       
    Pulls files from datatrail using created file. """
    obs_dict = {'kko':kko, 'gbo':gbo, 'hco':hco, 'chime':chime}
    obs = obs_dict[telescope]
    src_str=get_best_gain_calibrator(time, "gbo") #find the best calibrator for the given time
    print("Source",src_str)
    if src_str == "cyga":
        src = CygA
    elif src_str == "casa":
        src = CasA
    elif src_str == "taua":
        src = TauA
    elif src_str == "vira":
        src = VirA
       
    with open(commissioning, 'r') as f:
        lines = f.read()
    just_gbo_corr = re.findall(fr'\S+_{telescope}_corr\b',  lines)
    target_date = datetime.datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y%m%d")
    print("Target date:", target_date)

    # Filter to only those matching the same date prefix
    same_day_files = [f for f in just_gbo_corr if f.startswith(target_date)]
    same_day_files.sort()
    print("Files on that day:")
    for f in same_day_files:
        print(f)
    for f in same_day_files:
        # Example filename: 20251107T102346Z_gbo_corr
        time_str = f[:16]  # '20251107T102346Z'
        start_time = datetime.datetime.strptime(time_str, "%Y%m%dT%H%M%SZ")
        # Compute end time 80 minutes later
        end_time = start_time + timedelta(minutes=20)
        start_unix = start_time.timestamp()
        end_unix = end_time.timestamp()
        transit_times = np.asarray(chime.transit_times(src, start_unix, end_unix))  #For GBO observations, need to get transit time at CHIME
        if transit_times.size > 0:  # if any transit times found
            found_file = f
            print("Found file:", found_file)
            
    file_name="00000000_0000.h5"
    with open(file_path, "w") as f:
        f.write(file_name)
    subprocess.run(["pip", "install", "--upgrade", "datatrail-cli"])
    cmd = f"echo Y | datatrail pull -s {file_path} {telescope}.acquisition.processed {found_file}"
    subprocess.run(cmd, shell=True, check=True)