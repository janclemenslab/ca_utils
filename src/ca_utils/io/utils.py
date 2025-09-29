import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import scipy
import h5py
import yaml
from glob import glob
import logging
from collections import namedtuple
from .scanimagetiffile import ScanImageTiffFile
from typing import List, Optional, Dict, Any


Trial = namedtuple(
    "Trial",
    [
        "file_names",
        "frames_first",
        "frames_last",
        "nb_frames",
        "frame_width",
        "frame_height",
        "nb_channels",
        "channel_names",
        "frame_rate_hz",
        "volume_rate_hz",
        "nb_slices",
        "frame_zindex",
    ],
)


def load_prot(prot_file_name: str) -> Dict[str, Any]:
    with open(prot_file_name, mode="r") as f:
        prot = yaml.load(f, Loader=yaml.FullLoader)
    return prot


def parse_stim_log(logfile_name) -> List[Dict[str, Any]]:
    """Reconstruct playlist from log file.

    Args:
        logfilename
    Returns:
        dict with playlist entries
    """
    with open(logfile_name, "r") as f:
        logs = f.read()
    log_lines = logs.strip().split("\n")
    session_log = []
    for current_line in log_lines:
        head, _, dict_str = current_line.partition(": ")
        if dict_str.startswith("cnt:"):  # indicates playlist line in log file
            dict_items = dict_str.strip().split("; ")
            dd = dict()
            for dict_item in dict_items:
                key, val = dict_item.strip(";").split(": ")
                val = val.replace("nan", "np.nan")
                try:
                    dd[key.strip()] = eval(val.strip())
                except (ValueError, NameError):
                    dd[key.strip()] = val.strip()
            session_log.append(dd)
    return session_log


def make_df_multi_index(logs_stims, channel_names=None) -> pd.DataFrame:
    """Convert logged playlist with list entries into multi-index DataFrame."""
    keys = list(logs_stims[0].keys())
    keys.remove("cnt")
    [log_stim.pop("cnt") for log_stim in logs_stims]

    if channel_names is None:
        channel_names = ["left_sound", "right_sound", "odor1", "odor2", "piezo"]
    n = dict()
    for lg in logs_stims:
        for key, val in lg.items():
            for cnt, v in enumerate(val):
                try:
                    n[(key, channel_names[cnt])].append(v)
                except KeyError:
                    n[(key, channel_names[cnt])] = []
                    n[(key, channel_names[cnt])].append(v)
    df = pd.DataFrame(n, index=list(range(len(logs_stims))))
    return df


def parse_files(path) -> List[ScanImageTiffFile]:
    """Load all tif files in the path.

    Args:
        path
    Returns:
        list of ScanImageTiffFile objects
    """
    tif_files = glob(path + "_*.tif")
    tif_files.sort()
    logging.info(f"Found {len(tif_files)} tif files.")

    files = []
    for tif_file in tif_files:
        try:
            files.append(ScanImageTiffFile(tif_file))
        except:
            logging.debug(f"Failed loading '{tif_file}'")
    return files


def parse_trial_files(path):
    """Get file info for each trial

    Args:
        path
    Returns:
        list of Trial objects:
            Trial(file_names, frames_first, frames_last, nb_frames, frame_width, frame_height, nb_channels, channel_names)
    """
    files = parse_files(path)
    # assemble file numbers/names and frame-numbers for each trial
    trial_starttime = np.concatenate([f.description["nextFileMarkerTimestamps_sec"] for f in files])  # start time of the current trial for each frame
    file_index = np.concatenate([f.description["acquisitionNumbers"] for f in files]) - 1  # file number for each frame, -1 for 0-based indexing
    frame_numbers = np.concatenate([f.description["frameNumbers"] for f in files]) - 1  # running number of frames in session, -1 for 0-based indexing

    trial_uni, trial_index = np.unique(trial_starttime, return_inverse=True)
    nb_trials = len(files)

    file_onsets = np.where(np.diff(file_index) > 0)[0].astype(np.uintp) + 1  # plus 1 since we want the first frame *after* the change
    file_onsets = np.pad(file_onsets, (1, 0), mode="constant", constant_values=(0, len(file_index)))  # append first frame as first file onset
    file_offsets = np.pad(file_onsets[1:], (0, 1), mode="constant", constant_values=(0, len(file_index)))  # append last frame as last file offset

    # probably don't need this if we only care about the first and last frame for that trial from each file
    # within trial frame number
    frame_index = np.zeros(len(file_index), dtype=np.uintp)
    for onset, offset in zip(file_onsets, file_offsets):
        frame_index[onset:offset] = np.arange(0, offset - onset)

    trials = []
    # add info about which frame from which files belong to that trial
    for trial_number in range(nb_trials):
        idx = np.where(trial_index == trial_number)[0]  # get global frame numbers for current trial
        frm = frame_index[idx]  # get within-file frame numbers for current trial
        uni_files = np.unique(file_index[idx])  # which files contribute to the current trial

        file_names = [files[ii].name for ii in uni_files]
        framenumbers = [frm[file_index[idx] == ii] for ii in uni_files]
        frames_first = [int(f[0]) for f in framenumbers]
        frames_last = [int(f[-1] + 1) for f in framenumbers]  # +1 since we we use it as a range (exclusive bounds), otherwise we would miss the last frame
        nb_frames = sum([int(last - first) for first, last in zip(frames_first, frames_last)])
        # add some metadata to the trial info from the first file in each trial
        frame_width = int(files[uni_files[0]].metadata["hRoiManager.pixelsPerLine"])
        frame_height = int(files[uni_files[0]].metadata["hRoiManager.linesPerFrame"])
        frame_rate_hz = files[uni_files[0]].metadata["hRoiManager.scanFrameRate"]
        volume_rate_hz = files[uni_files[0]].metadata["hRoiManager.scanVolumeRate"]

        nb_channels = 1
        channel_idx = files[uni_files[0]].metadata["hChannels.channelSave"]
        if isinstance(channel_idx, list):
            nb_channels = len(channel_idx)

        channel_names = ["gcamp", "tdtomato"][:nb_channels]
        nb_slices = files[uni_files[0]].metadata["hStackManager.numSlices"]
        frame_zindex = np.mod(np.array(frame_numbers[idx[::nb_channels]]), nb_slices)
        trials.append(
            Trial(
                file_names,
                frames_first,
                frames_last,
                nb_frames,
                frame_width,
                frame_height,
                nb_channels,
                channel_names,
                frame_rate_hz,
                volume_rate_hz,
                nb_slices,
                frame_zindex,
            )
        )
    return trials


def parse_daq(ypos, zpos, next_trigger, sound=None, channel_names=None) -> Dict[str, Any]:
    """Get timing of frames and trials from ca recording.d_ypos.

    Args:
        ypos - y-position of the pixel scanner (resets indicate frame onsets)
        zpos - z-position of the piezo scanner
        next_trigger - recording of the next file trigger from scanimage to partition trials
        sound: [samples, channels]
        channel_names: name for each channel in sound

    Returns dict with the following keys:
        frame_onset_samples - sample number for the onset of each frame (inferred from the y-pos reset)
        frame_offset_samples - sample number for the offset of each frame (inferred from the y-pos reset)
        trial_onset_samples - onset of each trial (inferred from peaks in the next_trigger signals)
        trial_offset_samples - offset of each trial (last sample number added as offset for final trial)
        sound_onset_samples - onset of first sound event in the trial (None if no sound provided)
        sound_offset_samples - offset of last sound event in the trial (None if no sound provided)
        frame_zpos - average zpos for each frame
    """
    d_ypos = -np.diff(ypos)  # neg. since we want offsets
    # samples at which each frame has been stopped being acquired (y pos resets)
    frame_offset_samples, _ = find_peaks(d_ypos, height=np.max(d_ypos) / 2)
    frame_onset_samples = np.zeros_like(frame_offset_samples)
    frame_onset_samples[1:] = frame_offset_samples[:-1] + 1  # samples after frame offset

    height = np.max(-d_ypos[: frame_onset_samples[1]]) / 2
    tmp = find_peaks(-d_ypos[: frame_onset_samples[1]], height=height)
    frame_onset_samples[0] = tmp[0] - 1

    d_nt = np.diff(next_trigger)
    trial_onset_samples, _ = find_peaks(d_nt, height=np.max(d_nt) / 2)

    # from these construct trial offset samples:
    trial_offset_samples = np.append(trial_onset_samples[1:], len(next_trigger))  # add last sample as final offset

    # import matplotlib.pyplot as plt
    # plt.ion()

    # detect sound onsets and offset from DAQ recording of the sound
    timing_dict = dict()
    if sound is not None and channel_names is not None:
        # use these to infer within trial sound onset
        for channel, channel_name in enumerate(channel_names):
            if channel_name is None:
                continue
            onset_samples = np.zeros((len(trial_onset_samples),), dtype=np.intp)
            offset_samples = np.zeros((len(trial_onset_samples),), dtype=np.intp)
            for cnt, (trial_start_sample, trial_end_sample) in enumerate(zip(trial_onset_samples, trial_offset_samples)):
                trial_sound = sound[trial_start_sample:trial_end_sample, channel]
                trial_sound[:10] = 0
                trial_sound = np.convolve(np.abs(trial_sound), np.ones((10,)) / 10)  # compute the envelope

                thres = 0.01  # TODO: specify threshold for each channel
                # print(np.mean(trial_sound), np.std(trial_sound))
                # plt.clf()
                # plt.plot(trial_sound)
                if np.any(trial_sound >= thres):
                    onset_samples[cnt] = int(trial_start_sample + np.argmax(trial_sound >= thres))
                    trial_sound = sound[onset_samples[cnt] : trial_end_sample, channel]
                    offset_samples[cnt] = int(onset_samples[cnt] + len(trial_sound) - 1 - np.argmax(trial_sound[::-1] >= thres))
                    # plt.axvline(onset_samples[cnt] - trial_start_sample, c='r', alpha=0.1)
                    # plt.axvline(offset_samples[cnt] - trial_start_sample, c='r', alpha=0.1)
                else:
                    onset_samples[cnt] = -1
                    offset_samples[cnt] = -1

                # plt.pause(0.001)
                # plt.show()
                # breakpoint()

            timing_dict[f"{channel_name}_onset_samples"] = onset_samples
            timing_dict[f"{channel_name}_offset_samples"] = offset_samples

    # get avg z-pos for each frame
    frame_zpos = np.zeros_like(frame_onset_samples, dtype=float)
    for cnt, (on, off) in enumerate(zip(frame_onset_samples, frame_offset_samples)):
        frame_zpos[cnt] = np.mean(zpos[on:off])

    d = dict()
    d["frame_offset_samples"] = frame_offset_samples
    d["frame_onset_samples"] = frame_onset_samples
    d["trial_onset_samples"] = trial_onset_samples
    d["trial_offset_samples"] = trial_offset_samples
    d["frame_zpos"] = frame_zpos
    d.update(timing_dict)
    return d


def get_pixel_zpos(zpos, frame_shape, frame_onset_samples, frame_offset_samples, smooth_zpos=False):
    """Get z-position for each get_pixel_zpos.

    Args:
        zpos for each samples
        frame_shape - [width, height]
        frame_onset_samples
        frame_offset_samples
        smooth_zpos=False - smooth zpos using Savitzky-Golay filter
    Returns:
        array with shape [nb_frames, width, height] with z-positions for each pixel in each frame
    """
    if smooth_zpos:
        zpos = savgol_filter(zpos, 21, 3)
    pxPerFrame = np.prod(frame_shape)
    nb_frames = len(frame_onset_samples)
    pixels_zpos = np.empty((nb_frames, *frame_shape))
    for cnt, (t0, t1) in enumerate(zip(frame_onset_samples, frame_offset_samples)):
        zpix = np.interp(np.linspace(0, t1 - t0, pxPerFrame), np.arange(0, t1 - t0), zpos[t0:t1])
        pixels_zpos[cnt, ...] = np.reshape(zpix, frame_shape)
    return pixels_zpos


def samples2frames(frame_samples, samples):
    """Get next frame after sample.

    Args:
        frame_samples: sample numbers of the frames
        samples: single sample number or list of samples
    Returns:
        list of frame at or after samples
    """
    try:
        len(samples)
    except TypeError:
        samples = np.array(
            [
                samples,
            ],
            dtype=np.uintp,
        )
    frames = np.zeros_like(samples)
    frame_samples = np.append(frame_samples, np.max(samples) + 1)  # why +1?

    for cnt, sample in enumerate(samples):
        frames[cnt] = np.argmax(frame_samples > sample)
    return frames


def find_nearest(arr, val):
    """Find index of occurrence of item in arr closest to val."""
    return (np.abs(np.array(arr) - val)).argmin()


def interpolator(x, y, fill_value="extrapolate"):
    return scipy.interpolate.interp1d(x, y, fill_value=fill_value)


def parse_trial_timing(daq_file_name, frame_shapes=None, channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse DAQ file to get frame precise timestamps and sound onset/offset information.

    Args:
        daq_file_name ([type]): [description]
        frame_shapes ([type], optional): [description]. Defaults to None.
        channel_names (List[str], optional): Same as nb channels in daq file - will ignore the first 3 since they are 2P timing related. Defaults to None.

    Returns:
        [type]: [description]
    """
    # this will be wrong in most cases - so smake channel_names a required arg? and allow providing it when initiating the session?
    idx_y_pos = 0
    idx_z_pos = 1
    idx_next = 6
    if channel_names is not None:
        idx_y_pos = channel_names.index("y pos feedback")
        idx_z_pos = channel_names.index("piezo pos feeback")
        idx_next = channel_names.index("next trigger")

    # parse DAQ data for synchronization
    with h5py.File(daq_file_name, "r") as f:
        data = f["samples"][:]
        ypos = data[:, idx_y_pos]  # y pos of the scan pixel
        zpos = data[:, idx_z_pos]  # z pos of the scan pixel
        # next_trigger = data[:, 5]  # should be "6"
        next_trigger = data[:, idx_next]  # should be "6"
        # sound = np.sum(data[:, 3:5], axis=1)  # pool left and right channel
        # sound = data[:, 3:]
        daq_stamps = f["systemtime"][:][:, 0]
        daq_sampleinterval = f["samplenumber"][:][:, 0]
        try:
            fs = f.attrs["rate"]
            logging.info(f"Using saved sampling rate of: {fs} Hz.")
        except KeyError:
            fs = 10_000
            logging.warning(f"Sampling rate of recording not found in DAQ file - defaulting to {fs} Hz.")

    if channel_names is None:
        nb_channels = data.shape[1]
        channel_names = [f"channel{channel}" for channel in range(nb_channels)]
    else:
        channel_names = channel_names.copy()  # copy so we don't overwrite info in the protocol outside of this function

    # set channels to ignore to None
    print(channel_names)
    to_ignore = ["y pos feedback", "piezo pos feeback", "line sync", "start trigger", "next trigger"]
    for ii, c in enumerate(channel_names):
        if c in to_ignore:
            channel_names[ii] = None

    print(data.shape, channel_names)
    d = parse_daq(ypos, zpos, next_trigger, data, channel_names)

    daq_samplenumber = np.cumsum(daq_sampleinterval)
    ip = interpolator(daq_samplenumber, daq_stamps)

    # get frame times for each trial
    nb_trials = len(d["trial_onset_samples"])
    log_keys = [
        "trial_onset_time",
        "trial_offset_time",
        "trial_onset_frame",
        "trial_offset_frame",
        "frameoffset_ms",
        "frameonset_ms",
        "frame_zpos",
        "pixels_zpos",
    ]
    for channel_name in channel_names:
        if channel_name is None:
            continue
        log_keys.append(f"{channel_name}_onset_ms")
        log_keys.append(f"{channel_name}_offset_ms")
        log_keys.append(f"{channel_name}_onset_frame")
        log_keys.append(f"{channel_name}_offset_frame")

    logs = {key: [None for _ in range(nb_trials)] for key in log_keys}

    for cnt in range(nb_trials):
        logs["trial_onset_frame"][cnt] = samples2frames(d["frame_offset_samples"], d["trial_onset_samples"][cnt])[0]
        logs["trial_offset_frame"][cnt] = samples2frames(d["frame_offset_samples"], d["trial_offset_samples"][cnt])[0]

        logs["trial_onset_time"][cnt] = ip(d["trial_onset_samples"][cnt])
        logs["trial_offset_time"][cnt] = ip(d["trial_offset_samples"][cnt])
        logs["frame_zpos"][cnt] = d["frame_zpos"][logs["trial_onset_frame"][cnt] : logs["trial_offset_frame"][cnt]]

        frame_onset = (d["frame_onset_samples"][logs["trial_onset_frame"][cnt] : logs["trial_offset_frame"][cnt]] - d["trial_onset_samples"][cnt]) / fs * 1000
        logs["frameonset_ms"][cnt] = frame_onset.tolist()
        frame_offset = (d["frame_offset_samples"][logs["trial_onset_frame"][cnt] : logs["trial_offset_frame"][cnt]] - d["trial_onset_samples"][cnt]) / fs * 1000
        logs["frameoffset_ms"][cnt] = frame_offset.tolist()

        for channel_name in channel_names:
            if channel_name is None:
                continue
            logs[f"{channel_name}_onset_ms"][cnt] = float((d[f"{channel_name}_onset_samples"][cnt] - d["trial_onset_samples"][cnt]) / fs * 1000)
            logs[f"{channel_name}_offset_ms"][cnt] = float((d[f"{channel_name}_offset_samples"][cnt] - d["trial_onset_samples"][cnt]) / fs * 1000)
            # get these rel. to time of frame midpoint?
            logs[f"{channel_name}_onset_frame"][cnt] = find_nearest(logs["frameoffset_ms"][cnt], logs[f"{channel_name}_onset_ms"][cnt])
            logs[f"{channel_name}_offset_frame"][cnt] = find_nearest(logs["frameoffset_ms"][cnt], logs[f"{channel_name}_offset_ms"][cnt])

        if frame_shapes:  # get zpos per pixel
            logs["pixels_zpos"][cnt] = get_pixel_zpos(
                zpos.copy(),
                frame_shapes[cnt],
                d["frame_onset_samples"][logs["trial_onset_frame"][cnt] : logs["trial_offset_frame"][cnt]],
                d["frame_offset_samples"][logs["trial_onset_frame"][cnt] : logs["trial_offset_frame"][cnt]],
                smooth_zpos=True,
            ).astype(np.float16)
        else:
            logs["pixels_zpos"][cnt] = None

    return logs
