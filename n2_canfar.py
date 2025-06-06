import numpy as np
from baseband_analysis.core import BBData
from baseband_analysis.core.signal import tiedbeam_baseband_to_power
from ch_util.ephemeris import CasA, CygA, TauA, VirA, chime, gbo, hco, kko, unix_to_datetime
from chime_frb_api.backends import frb_master
from baseband_analysis.analysis.toa import get_TOA
import bisect
import subprocess

# Load in pulsar dataframe for pulsar beamforming
pulsar_df = pd.read_csv('/arc/home/rdarlinger/outriggers_vlbi_pipeline/src/outriggers_vlbi_pipeline/calibrators/known_pulsars.csv', comment='#')

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