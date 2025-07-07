import numpy as np
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dateutil
import datetime, pickle
import glob
import time
import re
import csv
# from outriggers_vlbi_pipeline.query_database import get_event_data
from chime_frb_api.backends import frb_master
from baseband_analysis.core.dedispersion import incoherent_dedisp, coherent_dedisp
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import scrunch
from baseband_analysis.core.flagging import get_RFI_channels
from baseband_analysis.dev.Morphology_utils import get_structure_max_DM
from baseband_analysis.dev.Morphology_dev import get_signal_time_range
from baseband_analysis.analysis.snr import get_snr

import pandas as pd
import glob
import datetime
import numpy as np
import warnings
import h5py
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
from baseband_analysis.dev.Morphology_dev import get_best_downsamp

from astropy.coordinates import EarthLocation
from scipy.optimize import minimize_scalar
from sensitivity import *

os.environ["CHIME_FRB_ACCESS_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoicmRhcmxpbmdlciIsImV4cCI6MTc0NzE1ODAwMSwiaXNzIjoiZnJiLW1hc3RlciIsImlhdCI6MTc0NzE1NjIwMX0.UZeE1kjENxWo2Oz7NQkR5tjeVu-_S0xxMreXceUvN-0"
os.environ["CHIME_FRB_REFRESH_TOKEN"] = "1950d439627e9eece51ad947d8bd057e26a7d6f1a2c06f45"

frb_master_base_url = "http://frb-vsop.chime:8001"
master = frb_master.FRBMaster(base_url=frb_master_base_url)

def frb_rerun_pipeline(events, telescope="gbo"):
    """
    Runs all of the funcions needed from sensitivity.py to rerun old FRB data with offline gains

    Parameters:
    -----------
    events: list
        Event ids in a list format to cycle through
    
    Returns:
    ________
    fig: matplotlib.pyplot.figure
        Diagnostic plots for each FRB including: original waterfall from singlebeam, new waterfall from new beamforming, choice of bins in S/N plot, auto correlation SNR, fraction masked (Both in gain and in beamforming), visual of gain (soon to included cross correlation)

    """
    master = frb_master.FRBMaster()
    for event_id in events:
        path = '/arc/projects/chime_frb/rdarlinger/FRB_rerun_pipeline_results/event_{}/'.format(event_id)
        if not os.path.isdir(path):
            print("Making new dir: ", path)
            os.mkdir(path)
        print("Getting information for {}".format(event_id))
        event = master.events.get_event(event_id, full_header=True)
        source_name = f"FRB_{event_id}"
        print("Source name is", source_name)
        dm = event['event_best_data']['dm']
        event_snrs = []
        for beam in event['event_beam_header']:
            event_snrs.append(beam['snr'])
        snr = np.nanmax(event_snrs)
        dt = datetime.datetime.strptime(event['event_best_data']['timestamp_utc'].split('.')[0], "%Y%m%d%H%M%S")
        date = dt.strftime("%Y-%m-%d")
        print("Max S/N is", snr)
        print("Date of FRB is", date)
        print("DM is", dm)
        singlebeam=f"/arc/projects/chime_frb/data/chime/baseband/processed/{dt.strftime('%Y/%m/%d')}/astro_{event_id}/singlebeam_{event_id}.h5"
        print("Singlebeam location is", singlebeam)
        with h5py.File(singlebeam, "r") as f:
            loc_data=f['tiedbeam_locations'][:]
            ra=loc_data[0][0]
            dec=loc_data[0][1]
            print("RA is", ra, "Dec is", dec)
        fig, bad=waterfall_pulsar2(dt, singlebeam, "chime", source_name, DM=dm, snr=snr)
        fig.savefig(f"{path}/temp_waterfall.png") #use as the spot to look at these waterfalls before continuing with the code
        print("Finding Structure Maximizing DM")
        data = BBData.from_file(singlebeam)
        # S/N max DM
        dm_range_snr = 5
        downsample = 1 #you'll need to change this until you have enough S/N
        (
                freq_id,
                freq,
                power,
                _,
                _,
                valid_channels,
                _,
                DM,
                downsampling_factor,
        ) = get_snr(
                data,
                DM=None,
                downsample=downsample,
                fill_missing_time=None,
                diagnostic_plots=False,
                spectrum_lim=True,
                return_full=True,
                DM_range=dm_range_snr,
                raise_missing_signal=True,
        )
        print(f"The S/N maximizing DM is {DM} pc/cc")

        profile, start, end = get_signal_time_range(
                power, downsampling_factor, None, False
                )
        downsample = get_best_downsamp(profile, downsampling_factor)
        print(f"We will use {downsample} as downsampling factor from now on")

        if downsample != downsampling_factor:
            print(
                "New downsampling factor != original factor. Going to re-run get_snr as a result to get a new power array"
            )
            (
                freq_id,
                freq,
                power,
                _,
                _,
                valid_channels,
                _,
                DM_dsamp,
                downsampling_factor,
            ) = get_snr(
                data,  ### Change here to DM_dsamp as it changes if DM was none
                DM=DM,
                downsample=downsample,
                fill_missing_time=None,
                diagnostic_plots=False,
                spectrum_lim=True,
                return_full=True,
                DM_range=None,
                raise_missing_signal=True,
            )
    
        # profile after any downsampling changes made
        profile, start, end = get_signal_time_range(
                power, downsampling_factor, None, False
                )
        t_res = 2.56e-6 * downsampling_factor

        print("Running structure maximizing DM script")
        plt.close("all")
        DM_corr, DM_err = get_structure_max_DM(
                    power[..., start:end],
                    freq,
                    t_res=2.56e-6 * downsampling_factor,
                    DM_range=dm_range_snr,
                    diagnostic_plots = True
                )
        os.rename('/arc/projects/chime_frb/rdarlinger/DM_Search.pdf',path+'/DM_Search.pdf')
        os.rename('/arc/projects/chime_frb/rdarlinger/Waterfall_5sig.pdf',path+'/Waterfall_5sig.pdf')
        DM = DM + DM_corr
        print(f"The structure maximizing DM is {DM} pc/cc +/- {DM_err} pc/cc")
        fig2, bad=waterfall_pulsar2(dt, singlebeam, "chime", source_name, DM=DM, snr=snr)
        fig2.savefig(f"{path}/structure_maxmized_waterfall.png") #use as the spot to look at these waterfalls before continuing with the code
        print("Creating gains for day of burst")
        #make gain for that day
        plt.close("all")
        commissioning_file='/arc/projects/chime_frb/rdarlinger/commissioning_files.txt' #find corr file to pull from
        with open(commissioning_file, 'r') as f:
            content= f.read()

        timestamps = re.findall(r'\d{8}T\d{6}Z(?=_gbo_corr)', content)
        dt_list = [datetime.datetime.strptime(ts, "%Y%m%dT%H%M%SZ") for ts in timestamps]
        dt_list = sorted(dt_list)
        idx = bisect.bisect_left(dt_list, dt)
        if idx > 0:
            previous_dt = dt_list[idx - 1]
            print("Previous datetime:", previous_dt, "with original datetime being", dt)
            formatted_dt = previous_dt.strftime("%Y%m%dT%H%M%SZ")
        else:
            print("No earlier datetime found.")
        output_file = f"n_squared_{formatted_dt}.txt"
        with open(output_file, "w") as f:
            subprocess.run([
                "datatrail", "ps",
                f"{telescope}.acquisition.processed",
                f"{formatted_dt}_{telescope}_corr",
                "--show-files"
            ], stdout=f, check=True)
        
        toa=just_toa(singlebeam, DM=DM)
        transit_time=find_files(output_file, f"{path}/n_squared_files", toa)
        flagged, filepath = get_gains_from_N2(f"/arc/projects/chime_frb/data/{telescope}/n_squared/{formatted_dt}_{telescope}_corr/", transit_times=transit_time, gains_output_dir=path, plot_output_dir=path)
        
        command = f"echo y | datatrail pull gbo.event.baseband.raw {event_id}"
        subprocess.run(command, shell = True, check=True)
        command = f"echo y | datatrail pull chime.event.baseband.raw {event_id}"
        subprocess.run(command, shell = True, check=True)
        print(f"Using gain file: {filepath}")
        flagged_freq, flagged_input= process_data([event_id], "gbo", gain=filepath, out_file=path, save_dir=path, DMs=[DM], source_info= [source_name, ra, dec])
        print("Number of flagged frequencies:", flagged_freq, "Number of flagged inputs:", flagged_input)
        flagged_freq_chime, flagged_input_chime= process_data([event_id], "chime", out_file=path, save_dir=path, DMs=[DM], source_info= [source_name, ra, dec])
        print("Number of flagged frequencies at CHIME:", flagged_freq_chime, "Number of flagged inputs at CHIME:", flagged_input_chime)
        snrc, snrg=SNR(event_id, f"{path}SNR_{event_id}.csv", "/arc/projects/chime_frb/rdarlinger/power.h5")
        event_data = {
            "event_id": event_id,
            "timestamp": date,
            "DM": DM,
            "Flagged inputs": flagged,
            "Flagged frequencies": flagged_freq,
            "Flagged CHIME frequencies": flagged_freq_chime,
            "Flagged CHIME inputs": flagged_input_chime,
            "SNR GBO": snrg,
            "SNR CHIME": snrc
        }
        with open(f"{path}event_data.csv", mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=event_data.keys())
            writer.writeheader()
            writer.writerow(event_data)

    
    
    
if __name__ == "__main__":
    frb_rerun_pipeline([412590956, 424530814, 432660091 ]) #350136130, 358105468, 366503638, 378287810, 383577603, 388211354, 397220423,