import numpy as np
import pytz
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
import datetime
import h5py
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
import pytz

from astropy.coordinates import EarthLocation
import astropy.units as u
from scipy.optimize import minimize_scalar
from matplotlib.colors import LogNorm


# Load in pulsar dataframe for pulsar beamforming
pulsar_df = pd.read_csv('/arc/home/rdarlinger/outriggers_vlbi_pipeline/src/outriggers_vlbi_pipeline/calibrators/known_pulsars.csv', comment='#')

def chunked(seq, size):
    '''Used in pulsar beamforming'''
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

def freq_id_from_filename(path):
    '''Used in pulsar beamforming'''
    _, filename = os.path.split(path)
    return int(re.split(r"\.|_", filename)[2])

def waterfall_pulsar2(
    date, singlebeam, telescope, src_name, 
    downsample_factor_ms = 1.0, DM = None, 
    save_dir = None, pulsar_df = pulsar_df,
    snr = None, gain=None
):
    """
    Wrapper of the _waterfall_pulsar function. Plots a waterfall and integrated intensity profile for a given singlebeam, dedispersing 

    Parameters:
    -----------
    date: datetime.date
        Date of event to waterfall.
    singlebeam: str
        Path to saved singlebeam
    telescope: str
        Telescope that was used to beamform the pulsar
    src_name: str
        Name of pulsar being waterfalled
    downsample_factor_ms: float
        Amount to downsample by in milliseconds. Sweet spot is around 1 ms for pulse widths of ~1-10s of milliseconds.  
    DM: float
        If None, DM will be determined using available databases.
    save_dir: str
        Directory to save plot if specified.
    pulsar_df: Pandas.DataFrame
        Table containing pulsar information. Default's to DataFrame available at 'outriggers_vlbi_pipeline/calibrators/known_pulsars.csv'
    snr : float
        Detected SNR of FRB/pulsar burst/pulse
    
    Returns:
    ________
    fig: matplotlib.pyplot.figure
        
    toa: float
        time of arrival for given pulse
    """
    print("Load in data...")
    data = BBData.from_file(singlebeam)
    event_id = data.attrs["event_id"]
    if data.get("tiedbeam_power") is None:
        tiedbeam_baseband_to_power(data)

    print("Get DM...")
    if DM is None: 
        try: 
            DM = pulsar_df[pulsar_df['name']==src_name]['DM'].values[0]
        except: 
            master = frb_master.FRBMaster()
            frb_master_event = master.events.get_event(event_id)
            DM = frb_master_event['measured_parameters'][-1]['dm']
    print("DM={0:.2f} pc/cc...".format(DM))

    print("Coherently dedisperse...")
    coherent_dedisp(data, DM, time_shift=False, write=True)
    print("Incoherently dedisperse...")
    bb_chime, freqs_chime, freq_id = incoherent_dedisp(data, DM, fill_wfall=False)
    print("Calculate power...")
    power = np.abs(bb_chime) ** 2

    date_str = "{0}-{1:02d}-{2:02d}".format(date.year, date.month, date.day)

    print("Plot!")
    fig = _waterfall_pulsar2(
        power = power,
        event_id = event_id,
        downsample_factor_ms = downsample_factor_ms,
        src_name = src_name,
        telescope = telescope,
        date_str = date_str,
        DM = DM,
        save_dir = save_dir,
        snr = snr, gain=gain
    )
    return fig

def _waterfall_pulsar2(
    power,
    event_id,
    downsample_factor_ms,
    src_name,
    telescope,
    date_str,
    DM,
    save_dir = None,
    snr = None, gain=None
):
    """
    Simple waterfall plotting function for a given array.

    Parameters:
    -----------
    power: ndarray
        Power array of shape (nfreq, npol, ntime)
    event_id: str
        Event id of event 
    downsample_factor_ms: float
        Downsampling factor in milliseconds.
    src_name: str
        Name of source being waterfalled. 
    date_str: str
        Date to include in the title
    DM: float
        DM used to dedisperse baseband data. 
    save_dir: str
        Directory to save plot if specified (default None).
    
    Returns: 
    --------
    fig0: matplotlib.pyplot.figure
        Waterfall and integrated intensity figure. 
    I: Power of beamformed data
        
    Saves:
    -------
    Power in h5 file named power.h5
    """
    freqs = np.linspace(800, 400, power.shape[0], endpoint=True)
    tscrunch = int(downsample_factor_ms / 2.56e-3)

    # Do some rough masking to get rid of really bad frequency channels. 
    mask = np.zeros(freqs.size, dtype=bool)
    #mask[np.logical_and(freqs > 640, freqs < 660)] = 1
    #mask[np.logical_and(freqs > 750, freqs < 765)] = 1
    
    nframes = power.shape[-1]
    time = nframes * 2.56e-3

    XX_power = power[:, 0, :]
    YY_power = power[:, 1, :]

    I = XX_power + YY_power

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        I /= np.nanstd(I, axis=1).reshape(-1, 1)

    I[mask, :] = np.nanmean(I, axis=1)[mask, np.newaxis]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        I -= np.nanmean(I, axis=1).reshape(-1, 1)

        
    # Downsample
    I = scrunch(I, tscrunch=tscrunch, fscrunch=1)
    
    print("Remove RFI...")
    channels_rfi = get_RFI_channels(I, diagnostic_plots=False, thres_mean=5, thres_std=3)
    I[channels_rfi, :] = np.nan
    
    vmin,vmax=np.nanpercentile(I, [5,95])
    
    # Set the positions of the plot elements
    # (left, bottom, width, height)
    waterfall = (0, 0, 1.0, 0.7)
    timeseries = (0, 0.7, 1.0, 0.3)
    
    fig0 = plt.figure(figsize=(8, 10))
    
    ax = plt.axes(waterfall)
    im = ax.imshow(
        I,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
        aspect="auto",
        extent=[0, int(time), 400, 800],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")

    ax = plt.axes(timeseries)
    times = np.linspace(0, time, I.shape[-1])
    ax.plot(times, np.nansum(I, axis=0), color="black")
    ax.set_ylabel("Arbitrary Intensity")
    ax.set_xlim([0, np.max(time)])
    ax.set_xticklabels([])
    ax.grid()

    if snr is not None:
        fig0.suptitle(
            telescope.upper()
            + ", "
            + src_name
            + ", "
            + str(event_id)
            + "\n"
            + date_str
            + ", "
            + "DM (pc/cc): "
            + str(round(DM, 2))
            + ", "
            + "SNR: {0:.1f}".format(snr),
            fontsize=18,
            y=1.05,
        )
    else:
        fig0.suptitle(
            telescope.upper()
            + ", "
            + src_name
            + ", "
            + str(event_id)
            + ", "
            + date_str
            + ", "
            + "DM (pc/cc): "
            + str(round(DM, 2)),
            fontsize=18,
            y=1.05,
        )
    fig0.patch.set_alpha(1)

    if save_dir is not None:
        print("Saving plot to: {}".format("{}/{}_{}_{}_{}_waterfall.png".format(save_dir, event_id, src_name, date_str, telescope)))
        plt.savefig("{}/{}_{}_{}_{}_waterfall.png".format(save_dir, event_id, src_name, date_str, telescope), dpi=300, bbox_inches="tight")

        file_path="power_9_20.h5"
        with h5py.File(file_path, 'a') as f:
            # Ensure event group and telescope group exist (create if not)
            event_group = f.require_group(str(event_id))
            telescope_group = event_group.require_group(telescope)
            subgroup = telescope_group.require_group(gain)
            # Overwrite Power dataset if it exists
            if "Power" in subgroup:
                del subgroup["Power"]

            # Write new Power dataset and Date attribute
            subgroup.create_dataset("Power", data=np.array(I))
            subgroup.attrs['Date'] = date_str

    plt.close()

    return I

def process_data(events, telescope):
    '''Takes event ids and a telescope to beamform a pulsar and save the singlebeams, waterfalls
    and power
    
    Parameters
    -------
    events: ndarray
        array of evenet ids to beform
    telescope: str
        name of telescopoe i.e. gbo or chime
        
    Returns
    ------
    dates: ndarray
        Dates of the data
        
    Saves
    ------
    Singlebeam: .h5 file
        singlebeam saved in specified directory
    Waterfall: .png file
        watefall png of data and intensity
    Power: .h5 file
        power of events saved from the use of the waterfall_pulsar function
        '''
    master = frb_master.FRBMaster()
    snrs = []
    dms = []
    source_names = []
    event_ids_filt = []
    dates = []
    for event_id in events:
        if len(str(event_id)) == 10:
            continue
        print("Getting information for {}".format(event_id))
        event = master.events.get_event(event_id, full_header=True)
        source_name = event['event_best_data']['source_category_name']
        dm = event['event_best_data']['dm']
        event_snrs = []
        for beam in event['event_beam_header']:
            event_snrs.append(beam['snr'])
        snr = np.nanmax(event_snrs)
        dt = datetime.datetime.strptime(event['event_best_data']['timestamp_utc'].split('.')[0], "%Y%m%d%H%M%S")
        date = dt.strftime("%Y-%m-%d")

        # Save only pulsars
        if 'B' in source_name:
            print("{}, {}, {}, {}".format(source_name, dm, snr, date))
            snrs.append(snr)
            dms.append(dm)
            source_names.append(source_name)
            event_ids_filt.append(event_id)
            dates.append(date)
    snrs = np.array(snrs)
    dms = np.array(dms)
    source_names = np.array(source_names)
    dates = np.array(dates)
    event_ids_filt = np.array(event_ids_filt)
    
    
    fns = []
    exceptions = []
    exception_ids = []
    for event_id in event_ids_filt:
        try:
            # event_id = event_ids_filt[snrs > 40.][0]
            mm = np.where(event_ids_filt == event_id)[0][0]
            telescope= telescope

            if telescope == 'gbo':
                telescope_rotation_angle = tools._GBO_ROT 
                obs = gbo
            elif telescope == 'chime':
                telescope_rotation_angle = tools._CHIME_ROT 
                obs = chime
            else:
                raise Exception('Telescope must be one of "gbo" or "chime" right now.')

            dt = dateutil.parser.parse(dates[mm])
            print("Get location of converted data for the given event")
            pattern = '/arc/projects/chime_frb/data/{}/baseband/raw/{}/{}/{}/astro_{}/*h5'.format(
                telescope, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), event_id
            )
            # pattern = '/arc/projects/chime_frb/bandersen/{}/baseband/raw/{}/{}/{}/astro_{}/*h5'.format(
            #     telescope, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), event_id
            # )

            src_name = source_names[mm]
            snr = snrs[mm]
            print("Source: {0}, SNR: {1:.1f}".format(src_name, snr))
            print("Location of data: {}".format(pattern))

            print("Grab location of the source")
            src_info = pulsar_df[pulsar_df['name']==src_name]
            src_ra = src_info["ra"].values[0]
            src_dec = src_info["dec"].values[0]
            coord = SkyCoord(
                ra = src_ra, 
                dec = src_dec, 
                frame = 'icrs',
                unit=(u.deg, u.deg),
            )
            ra_frb = coord.ra.deg # deg
            dec_frb = coord.dec.deg # deg
            print("RA = {0:.3f} deg, Dec = {1:.3f} deg".format(ra_frb, dec_frb))

            print("Grab corresponding calibration files")
            dt_gains = dt # dateutil.parser.parse("2023-06-27")
            if telescope == 'gbo':
                # cal_files = glob.glob('/arc/projects/chime_frb/bandersen/gbo/gains/*{}*h5'.format(dt_gains.strftime("%Y%m%d")))
                cal_files = glob.glob('/arc/projects/chime_frb/rdarlinger/gain_solutions/gain_20241118T000857.798131Z_cyga.h5') 
                #use specific gain for gbo ^^
                gain_d=datetime.datetime(2024, 11,18)
                gain_date = gain_d.strftime("%Y-%m-%d")
            elif telescope == 'chime':
                cal_files = glob.glob('/arc/projects/chime_frb/data/chime/daily_gain_solutions/hdf5_files/*{}*h5'.format(dt_gains.strftime("%Y%m%d")))
                gain_date=dt_gains.strftime("%Y%m%d")
            else:
                raise Exception('Telescope must be one of "gbo" or "chime" right now.')
            cal_h5 = cal_files[0]
            gain = cal_h5.split('/')[-1].split('.h5')[0]
            print("Calibration file: {}".format(cal_h5))

            print("Set output location for singlebeam")
            out_file = '/arc/projects/chime_frb/rdarlinger/singlebeams/singlebeam_{0}_{1}_{2}.h5'.format(telescope, event_id,gain_date)
            print('Saving singlebeam to', out_file)

            if os.path.exists(out_file):
                singlebeam=out_file
            else:
    #         ################## Beamforming #####################

                print("Grab converted h5 files")
                filenames = glob.glob(pattern)
                filenames.sort(key=freq_id_from_filename)
                if len(filenames) == 0:
                    raise OSError(f"Looked in {pattern} but raw baseband data not found.")
                print(f"Found Raw BBData ({len(filenames)}) :{filenames[0:3]}")
                first_data = BBData.from_acq_h5(filenames[0])

                date_unix = first_data["time0"]["ctime"][0]

                date = datetime.datetime.utcfromtimestamp(date_unix)

                ra = ra_frb
                dec = dec_frb
                print(f"Beamforming at {(ra_frb,dec_frb)}")

                print("Load in inputs to speed up call")
                #date = datetime.datetime(2024,8,20)
                inputs = tools.get_correlator_inputs(date, correlator=telescope)
                # inputs = correlator_inputs.copy()
                # with open('./gbo_256ants_chutil_23.6.0+1.gf8c7162.pkl', 'rb') as file:
                #     inputs = list(pickle.load(file))

                # Load gains
                gains = cal.read_gains(cal_h5)

                datas = []
                for ii, files in enumerate(chunked(filenames, 8)):
                    print(
                        f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}: Processing freq_ids: {[freq_id_from_filename(f) for f in files]}'
                    )
                    data = BBData.from_acq_h5(files)
                    cal.apply_calibration(data, gains, inputs=inputs)
                    static_centroid = tools.ArrayAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                    )

                    if telescope == 'gbo':
                        # flagged_ids = [10, 13, 15, 21, 26, 38, 40, 41, 56, 63, 67, 74, 78, 84, 86, 104, 108, 111, 118, 120, 123, 127, 130, 138, 147, 154, 155, 156, 158, 165, 168, 171, 175, 180, 186, 199, 201, 232, 240, 243, 245, 248, 249, 252, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]
                        flagged_ids = [10, 13, 15, 21, 26, 38, 40, 41, 56, 63, 67, 74, 78, 84, 86, 104, 108, 111, 118, 120, 123, 127, 130, 138, 147, 154, 155, 156, 158, 165, 168, 171, 175, 180, 186, 199, 201, 232, 240, 243, 245, 248, 249, 252]
                        # flagged_ids = np.arange(254)
                        for iid in flagged_ids:
                            antenna = Blank(id=inputs[iid].id, input_sn=inputs[iid].input_sn, corr=inputs[iid].corr)
                            inputs[iid] = antenna
                        # inputs = list(np.delete(inputs, flagged_ids))

                    beamform.tied_array(
                        data,
                        ra=ra,
                        dec=dec,
                        correlator_inputs=inputs,
                        reference_feed=static_centroid,
                        obs=obs,
                        telescope_rotation=telescope_rotation_angle,
                        static_delays=False,
                    )
                    del data["baseband"]
                    datas.append(data)
                    print(
                        f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}: We have processed {len(datas)} groups so far!'
                    )

                beamformed = bbdata.concatenate(datas)
                beamformed.attrs["gain_cal_h5"] = cal_h5
                beamformed.attrs["src_name"] = src_name
                beamformed.save(out_file)
                beamformed.close()
                print(
                    f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}: Wrote output to {out_file}'
                )

                ################## Plot #####################

                print("Get singlebeam -- assuming the singlebeam has been saved")
                singlebeam = out_file


            print("Specify save directory")
            save_dir = '/arc/projects/chime_frb/rdarlinger/waterfalls'

            # Create figure
            print(dt, singlebeam, telescope, src_name)
            intensity = waterfall_pulsar2(
                dt, singlebeam, telescope, src_name, 
                downsample_factor_ms = 0.1, DM = None, 
                save_dir = save_dir, pulsar_df = pulsar_df,
                snr = snr, gain=gain_date
            )

            date_str = dt.strftime("%Y-%m-%d")
            fn = "{}/{}_{}_{}_{}_waterfall.png".format(save_dir, event_id, src_name, date_str, telescope)
            fns.append(fn)
        except Exception as e:
            print(e)
            exceptions.append(e)
            exception_ids.append(event_id)
            continue
    return dates 
    
def SNR(event_id, file_path, power_path, telescope1="chime", telescope2="gbo", pulse_start=None, pulse_end=None):
    '''Finds SNR values for two telescopes and saves the values
    Parameters
    --------
    event_id: float
        event id of event to calculate the SNR values for
    file_path: str
        file path of csv to save SNR values to
    power_path: str
        file path of power.h5 file saved from process_data function
    telescope1: str
        name of telescope to calculate SNR values for first
    telescope2: str
        name of telescope to calculate SNR values for second, must be outrigger to work with plot function
    pulse_start: float
        specified bin to start the pulse if needed
    pulse_end: float
        specified bin to end the pulse if needed
    
    Returns
    -------
    simple_snr: float
        snr of the first telescope given
    simple_snrg: float
        snr of the second telsecope given
        
    Saves
    -------
    file: .csv file
        File at specified file path with SNR data'''
    telescope=telescope1
    with h5py.File(power_path, 'r') as f: #grabbing the previously saved power data from _waterfall_pulsar2 function
        if str(event_id) in f:
            event=f[str(event_id)]
            for key in event:
                if key == telescope:
                    power1=event[telescope]['Power'][:]
                    date=event[telescope].attrs['Date']
    power=np.nansum(power1, axis=0)
    noise = np.mean(power[50:100])
    mad_off_pulses = median_abs_deviation(power[50:100]) #np.nanstd(off_pulses, axis = 0)
    power=(power-noise)/mad_off_pulses
    peak_bin = np.argmax(power)
    print(peak_bin)
    peak = power[peak_bin]
    if pulse_start == None and pulse_end == None:
        pulse_start = np.where(power[:peak_bin] < 0.1 * peak)[0][-1]
        pulse_end = np.where(power[peak_bin:] < 0.1 * peak)[0][0] +peak_bin
    plt.plot(power)
    plt.axvline(pulse_start, color="red")
    plt.axvline(pulse_end, color="red")
    plt.title("After Standardize")
    plt.show()
    print(pulse_start,pulse_end)
        
    
    simple_snr = np.nansum(power[pulse_start:pulse_end],axis=-1)
    #signal=np.max(power)
    print(simple_snr, np.mean(power[50:100]), median_abs_deviation(power[50:100]))
    
    
    data = {'ID': [event_id], 'Telescope': [telescope], "SNR": [simple_snr], "Date": [date], "Peak": [peak]}

    df = pd.DataFrame(data)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
    telescope=telescope2
    with h5py.File(power_path, 'r') as f: #grabbing the previously saved power data from _waterfall_pulsar2 function
        if str(event_id) in f:
            event=f[str(event_id)]
            for key in event:
                if key == telescope:
                    power1g=event[telescope]['Power'][:]
                    dateg=event[telescope].attrs['Date']
    powerg=np.nansum(power1g, axis=0)
    noiseg = np.mean(powerg[50:100])
    mad_off_pulsesg = median_abs_deviation(powerg[50:100]) #np.nanstd(off_pulses, axis = 0)
    powerg=(powerg-noiseg)/mad_off_pulsesg
    peak_bing = np.argmax(powerg)
    print(peak_bing)
    peakg = powerg[peak_bing]
    plt.plot(powerg)
    plt.axvline(pulse_start+31, color="red") #31 is so it lines up with pulse since it is not 1 to 1 from CHIME
    plt.axvline(pulse_end+31, color="red")
    plt.title("After Standardize")
    plt.show()
    print(pulse_start+31,pulse_end+31)
        
    
    simple_snrg = np.nansum(powerg[pulse_start+31:pulse_end+31],axis=-1)
    #signal=np.max(power)
    print(simple_snrg, np.mean(powerg[50:100]), median_abs_deviation(powerg[50:100]))
    
    
    datag = {'ID': [event_id], 'Telescope': [telescope], "SNR": [simple_snrg], "Date": [dateg], "Peak": [peakg]}

    df = pd.DataFrame(datag)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
    
    return simple_snr, simple_snrg    

def plot_SNR(snr_file, out_file):
    '''Plot the SNR ratio of two telescopes
    
    Parameters
    ------
    snr_file: str
        file path of file created in SNR function with SNR data
    out_file: str
        file path to save the SNR plot to
    
    Saves
    ------
    Plot: .png file
        Saves plot of snr ratios at specified out path
    '''
    # Read the CSV file
    df = pd.read_csv(snr_file)

    # Sort the dataframe to ensure pairs are adjacent and consistent
    df = df.sort_values(by=["ID", "Telescope"])

    # Get the SNR values as an array
    snr = df["SNR"].values

    # Compute successive ratios: telescope / chime
    r = [snr[i] / snr[i-1] for i in range(1, len(snr), 2)]
    dates=pd.to_datetime(df["Date"])
    dates=dates.values[::2]
    print(dates)
    plt.plot(dates, r, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('GBO/CHIME All SNR')
    plt.title('Offline GBO Gains')

    # Rotate the x-axis labels to make them more readable
    plt.xticks(rotation=60)
    plt.xticks(dates) 

    plt.tight_layout()  # Ensures everything fits without overlap
    plt.savefig(out_file)
    plt.show()
    return
    
def run_waterfall_from_singlebeam(singlebeam, telescope, event_i):
    '''Run just the waterfall from a given singlebeam
    
    Paramaters
    ------
    singlebeam: str
        file path to .h5 file of the singlebeam
    telescope: str
        name of telescope to waterfall event for
    event_i: float
        event id to waterfall for
        
    Returns:
    -------
    toa: float
        time of arrival for event
        
    Saves
    -------
    Waterfall: .png
        Waterfall saved at specified directory in function'''
    event = master.events.get_event(event_i, full_header=True) 
    dt = datetime.datetime.strptime(event['event_best_data']['timestamp_utc'].split('.')[0], "%Y%m%d%H%M%S")
    date = dt.strftime("%Y-%m-%d")
    source_name = event['event_best_data']['source_category_name']
    dm = event['event_best_data']['dm']
    src_name="B0329+54"
    save_dir="/arc/projects/chime_frb/rdarlinger/waterfalls/temp"
    toa = waterfall_pulsar2(
                dt, singlebeam, telescope, src_name, 
                downsample_factor_ms = 0.1, DM = None, 
                save_dir = save_dir, pulsar_df = pulsar_df
            )
    return toa

def just_toa(singlebeam, DM=None):
    '''Gets just the time of arrival from a singlebeam
    
    Parameters
    --------
    singlebeam: str
        file path of singlebeam to get TOA from
    DM: float
        If None, DM is pulled from the pulsar_df form or measured parameters from the event id
    
    Returns
    ------
    toa: float
        TOA from singlebeam'''
    data = BBData.from_file(singlebeam)
    event_id = data.attrs["event_id"]
    if data.get("tiedbeam_power") is None:
        tiedbeam_baseband_to_power(data)
    if DM is None: 
        try: 
            DM = pulsar_df[pulsar_df['name']==src_name]['DM'].values[0]
        except: 
            master = frb_master.FRBMaster()
            frb_master_event = master.events.get_event(event_id)
            DM = frb_master_event['measured_parameters'][-1]['dm']
    print("DM={0:.2f} pc/cc...".format(DM))
    toa=get_TOA(data, DM=DM) #In unix time
    print(toa)
    return toa

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



def get_gains_from_N2(path_to_h5_files, transit_times=None, src_str="cyga", gains_output_dir = '/arc/projects/chime_frb/rdarlinger/gain_solutions/', 
                      correlator = 'FCG', obs=gbo, badinps=None,input_pkl_file='/arc/projects/chime_frb/rdarlinger/gboinputs_correct.pkl', 
                      plot_output_dir="/arc/projects/chime_frb/rdarlinger/gain_solutions/plots/",
                      median=None,percent_of_band=0.1,max_dev=10.):
    if src_str == "cyga":
        src = CygA
    elif src_str == "casa":
        src = CasA
    elif src_str == "taua":
        src = TauA
    elif src_str == "vira":
        src = VirA
    numfiles = len(os.listdir(path_to_h5_files))
    if numfiles == 1:
        return
    if transit_times ==None:
        transit_times = _get_src_transit_from_hdf5(path_to_h5_files, src)
    
    print(transit_times)
    filenames, file_idxs, unix_times, transit_idxs = _find_hdf5_from_unix(path_to_h5_files, transit_times)
    print(filenames)

    for i, f in enumerate(filenames):
        _data = h5py.File(path_to_h5_files+f)
        _index_map = _data['index_map']
        _timestamps = _index_map['time']['ctime']
        _transit_time = _timestamps[transit_idxs[i]:transit_idxs[i]+1]

        filename = unix_to_gain_name(_transit_time[0], src.names)
        filepath = os.path.join(gains_output_dir, filename)
        
        if not os.path.isfile(filepath):
            print('Calculating gains for {0} transit at unix time {1}, {2}'.format(src.names, unix_times[i], filepath))
            _freqs = _data['index_map']['freq']

            _vis = _data['vis']
            
            
            if input_pkl_file == '': 
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
                print('Flagging bad inputs:', badinps)
                flagged_ids = badinps
            else:
                flagged_ids =  [15,16,21,22,23,24,30,31,32,36,37,38,43,
                  44,63,64,67,68,70,71,78,79,80,84,85,88,89,93,94,
                  112,113,116,117,130,131,145,146,
                  153,154,155,156,157,158,159,165,
                  166,180,181,186,187,201,202,214,
                  215,216,240,241,243,246,247
                  ]

            gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)
            
            # Looking for new channels to flag 
            if median is not None: 
                print('Looking for bad channels...')
                flagged_inputs_new = filter_bad_inputs(gains,median,percent_of_band=percent_of_band,max_dev=max_dev)
                if len(flagged_inputs_new) > 0: 
                    print('Found bad input(s):', flagged_inputs_new)
                    
                    flagged_ids = list(flagged_inputs_new) + flagged_ids
                    print('Recalculating gains, final list of bad inputs:', flagged_ids)
                    gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)

            else: 
                print(f'Flagging any inputs with gains that exceed {max_dev} medians over {percent_of_band}')
                median = get_simple_median(gains)
                flagged_inputs_new = filter_bad_inputs(gains,median,percent_of_band=percent_of_band,max_dev=max_dev)
                if len(flagged_inputs_new) > 0: 
                    print('Found bad input(s):', flagged_inputs_new)
                    flagged_ids = list(flagged_inputs_new) + flagged_ids
                    print('Recalculating gains, final list of bad inputs:', flagged_ids)
                    print("Number of bad inputs:", len(flagged_ids))
                    gains, weights, gain_err = _solve_gain_wrapper(_vis, _inputmap, _connected_inps,_transit_time,_freqs,src, obs, flagged_ids,transit_idxs,i)

            
            print("Beginning writeout process...")
            if gains_output_dir is not None:
                filename = unix_to_gain_name(_transit_time[0], src.names)
                filepath = os.path.join(gains_output_dir, filename)
                #print(
                   # f'{dt.datetime.now().strftime("%Y%m%dT%H%M%S")}: Saving gains to {filepath}'
                #)
                with h5py.File(filepath, "w") as h5pyfile:
                    h5pyfile.attrs['src_name'] = src.names
                    h5pyfile.create_dataset("gain", data = gains)
                    h5pyfile.create_dataset("weight", data = weights)
                    h5pyfile.create_dataset("gain_err", data = gain_err)
                    h5pyfile.create_dataset("index_map/freq", data = _freqs)
                    h5pyfile.create_dataset("index_map/input", data = _index_map['input'])
                    h5pyfile.create_dataset("flagged_ids", data =flagged_ids)
            file=h5py.File(filepath, "r")
            gain=np.asarray(file['gain'])
            print(gain.shape)
            
            datetime_obj = unix_to_datetime(unix_times[i]).astimezone(timezone.utc)
            filepath=os.path.join(plot_output_dir, f"gain_{datetime_obj}.png")
            plt.imshow(np.abs(gain), norm=LogNorm(vmin=1,vmax=10))
            plt.title(f"Gain for {datetime_obj}")
            plt.xlabel("Input")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.savefig(filepath)
        else: 
            print(filepath, ' already exists!')



    return

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

    print("Good inputs", _good_inputs_x, _good_inputs_y)
    print('Removing flagged inputs: {0}'.format(flagged_ids))
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

    evals, evecs = la.eigh(A, eigvals=(N - rank, N - 1))

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

    return la.eigh(Ac, eigvals=eigvals)

def eigh_special(A, zero_indices, niter=5):

    Ac = A.copy()

    if niter > 0:
        Ac[zero_indices] = 0.0

        for i in range(niter):
            Ac[zero_indices] = rankN_approx(Ac)[zero_indices]

    return la.eigh(Ac, eigvals=None)

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
    File: Creates file with names of the required h5 files in it for use to pull from datatrail
       
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
    file_path="/arc/projects/chime_frb/rdarlinger/"+file_path+"_"+common_path+".txt"

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
    
    transit_times = np.asarray(chime.transit_times(src, start_time, end_time)) #For GBO observations, need to get transit time at CHIME
    print("Transit times are:", transit_times)
    index = bisect.bisect_left(transit_times, time_to_match)
    file_transit_initial=transit_times[index - 1] #finds the previous transit
    print("Closest transit time is:", file_transit_initial)
    file_transit=file_transit_initial-common_path_unix_time #get into same units as the file_ids
    print("Closest transit in time from container file is:", file_transit)
    index2= bisect.bisect_left(file_ids, file_transit)
    file=file_ids[index2-1]
    matching_file_name = file_names[index2-1]
    if index2 < len(file_ids):  # check bounds
        if file_ids[index2] - file_transit < 1200:
            file_2 = file_ids[index2]

    if index2 - 1 >= 0:
        if file_transit - file_ids[index2 - 1] < 1200:
            file_2 = file_ids[index2 - 2]
    file_2_name=None
    if file_2 is not None:
        i2 = file_ids.index(file_2)
        file_2_name =file_names[i2]

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

def find_transits(src, latitude_deg, longitude_deg, elevation_m=0,
                          start_time=None, end_time=None):
    """
    Find all transit times (hour angle = 0) for a star between start and end time using Astropy.

    Parameters:
        star : skyfield.starlib.Star
            Skyfield Star object.
        latitude_deg : float
            Latitude of the telescope (degrees).
        longitude_deg : float
            Longitude of the telescope (degrees).
        elevation_m : float
            Elevation of the telescope (meters).
        start_time : float
            Start time as UNIX timestamp (seconds since epoch).
        end_time : float
            End time as UNIX timestamp (seconds since epoch).

    Returns:
        List of transit times in UNIX time.
    """
    if start_time is None or end_time is None:
        raise ValueError("Both start_time and end_time must be provided as UNIX timestamps.")

    # Convert times to Astropy Time objects
    t_start = Time(start_time, format='unix')
    t_end = Time(end_time, format='unix')

    # Convert Skyfield Star to Astropy SkyCoord
    ra = src.ra.radians * u.rad
    dec = src.dec.radians * u.rad
    coord = SkyCoord(ra=ra, dec=dec, frame="icrs")

    # Observer location
    location = EarthLocation(lat=latitude_deg * u.deg,
                             lon=longitude_deg * u.deg,
                             height=elevation_m * u.m)

    # Step through time in 1 sidereal day increments
    SIDEREAL_DAY = 0.9972696  # in solar days
    step = SIDEREAL_DAY * u.day
    current_time = t_start
    transits = []

    while current_time < t_end:
        # Minimize hour angle over ±6 hours around current time
        def ha_abs(dt_sec):
            t = current_time + dt_sec * u.s
            lst = t.sidereal_time("apparent", longitude=location.lon)
            ha = (lst - coord.ra).wrap_at(180 * u.deg)
            return abs(ha.deg)

        result = minimize_scalar(ha_abs, bounds=(-21600, 21600), method="bounded")
        transit_time = current_time + result.x * u.s

        # Check if result is in range
        if t_start <= transit_time <= t_end:
            transits.append(transit_time.unix)

        current_time += step

    return transits


def SNR_gain_diff(event_id, file_path, power_path, telescope1="gbo", telescope2="chime", pulse_start=None, pulse_end=None):
    '''Finds SNR values for two telescopes and saves the values
    Parameters
    --------
    event_id: float
        event id of event to calculate the SNR values for
    file_path: str
        file path of csv to save SNR values to
    power_path: str
        file path of power.h5 file saved from process_data function
    telescope1: str
        name of telescope to calculate SNR values for first
    telescope2: str
        name of telescope to calculate SNR values for second, must be outrigger to work with plot function
    pulse_start: float
        specified bin to start the pulse if needed
    pulse_end: float
        specified bin to end the pulse if needed
    
    Returns
    -------
    simple_snr: float
        snr of the first telescope given
    simple_snrg: float
        snr of the second telsecope given
        
    Saves
    -------
    file: .csv file
        File at specified file path with SNR data'''
    telescope=telescope1
    with h5py.File(power_path, 'r') as f: #grabbing the previously saved power data from _waterfall_pulsar2 function
        print("Top-level keys in file:", list(f.keys()))
        if str(event_id) in f:
            event=f[str(event_id)]
            print("Found event:", str(event_id))
            if telescope in event:
                telescope_group = event[telescope]
                print("Checking telescope:", telescope)
                print("Gain keys:", list(telescope_group.keys()))
                for gain_key in telescope_group:  # Loop over gain groups
                    gain_group = telescope_group[gain_key]
                    power1 = gain_group['Power'][:]
                    date = gain_key  # assuming gain_key is a date-like string
                    power = np.nansum(power1, axis=0)
                    noise = np.mean(power[50:100])
                    mad_off_pulses = median_abs_deviation(power[50:100])
                    power=(power-noise)/mad_off_pulses
                    peak_bin = np.argmax(power)
                    print(peak_bin)
                    peak = power[peak_bin]
                    if pulse_start == None and pulse_end == None:
                        pulse_start = np.where(power[:peak_bin] < 0.1 * peak)[0][-1]
                        pulse_end = np.where(power[peak_bin:] < 0.1 * peak)[0][0] +peak_bin
                    plt.plot(power)
                    plt.axvline(pulse_start, color="red")
                    plt.axvline(pulse_end, color="red")
                    plt.title("After Standardize")
                    plt.show()
                    print(pulse_start,pulse_end)


                    simple_snr = np.nansum(power[pulse_start:pulse_end],axis=-1)
                    #signal=np.max(power)
                    print(simple_snr, np.mean(power[50:100]), median_abs_deviation(power[50:100]))


                    data = {'ID': [event_id], 'Telescope': [telescope], "SNR": [simple_snr], "Date": [date], "Peak": [peak]}

                    df = pd.DataFrame(data)

                    if os.path.exists(file_path):
                        df.to_csv(file_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(file_path, mode='w', header=True, index=False)
    telescope=telescope2
    with h5py.File(power_path, 'r') as f: #grabbing the previously saved power data from _waterfall_pulsar2 function
        if str(event_id) in f:
            print("Found event:", str(event_id))
            event=f[str(event_id)]
            if telescope in event:
                print("Checking telescope:", telescope)
                telescope_group = event[telescope]
                print("Gain keys:", list(telescope_group.keys()))
                for gain_key in telescope_group:  # Loop over gain groups
                    gain_group = telescope_group[gain_key]
                    power1c=event[telescope]['Power'][:]
                    datec=event[telescope].attrs['Date']
                    powerc=np.nansum(power1c, axis=0)
                    noisec = np.mean(powerc[50:100])
                    mad_off_pulsesc = median_abs_deviation(powerc[50:100]) #np.nanstd(off_pulses, axis = 0)
                    powerc=(powerc-noisec)/mad_off_pulsesc
                    peak_binc = np.argmax(powerc)
                    print(peak_binc)
                    peakc = powerc[peak_binc]
                    plt.plot(powerc)
                    plt.axvline(pulse_start-31, color="red") #31 is so it lines up with pulse since it is not 1 to 1 from CHIME
                    plt.axvline(pulse_end-31, color="red")
                    plt.title("After Standardize")
                    plt.show()
                    print(pulse_start-31,pulse_end-31)


                    simple_snrc = np.nansum(powerc[pulse_start-31:pulse_end-31],axis=-1)
                    #signal=np.max(power)
                    print(simple_snrc, np.mean(powerc[50:100]), median_abs_deviation(powerc[50:100]))


                    datac = {'ID': [event_id], 'Telescope': [telescope], "SNR": [simple_snrc], "Date": [datec], "Peak": [peakc]}

                    df = pd.DataFrame(datac)

                    if os.path.exists(file_path):
                        df.to_csv(file_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(file_path, mode='w', header=True, index=False)
    
    return    

def plot_SNR_gain_diff(snr_file, out_file):
    '''Plot the SNR ratio of two telescopes
    
    Parameters
    ------
    snr_file: str
        file path of file created in SNR function with SNR data
    out_file: str
        file path to save the SNR plot to
    
    Saves
    ------
    Plot: .png file
        Saves plot of snr ratios at specified out path
    '''
    # Read the CSV file
    df = pd.read_csv(snr_file)

    # Sort the dataframe to ensure pairs are adjacent and consistent
    df = df.sort_values(by=["ID", "Telescope"])

    # Get the SNR values as an array
    snr = df["SNR"].values

    # Compute successive ratios: telescope / chime
    r = [snr[i] / snr[4] for i in range(0, len(snr))]
    dates=pd.to_datetime(df["Date"])
    dates=dates.values[:4]
    print(dates)
    plt.plot(dates, r, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('GBO/CHIME All SNR')
    plt.title('Offline GBO Gains All on 9/20 Pulse')

    # Rotate the x-axis labels to make them more readable
    plt.xticks(rotation=60)
    plt.xticks(dates) 

    plt.tight_layout()  # Ensures everything fits without overlap
    plt.savefig(out_file)
    plt.show()
    return