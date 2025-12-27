"""Represent a single calcium imaging experiment."""

import numpy as np
import pandas as pd
from .utils import parse_trial_timing, parse_trial_files, parse_stim_log, make_df_multi_index, load_prot
from .scanimagetiffile import ScanImageTiffFile
import os
from typing import List, Optional
import logging
import xarray as xr


class Session:
    """Session object.

    Construction:
        s = Session(path)

    Methods:
        stack(trial_number, split_channels, split_volumes) - returns 3D, 4D, or 5D matrix depending on params
        argfind(column_title, pattern, channel=None, op='==') - returns matching trial numbers
        find(column_title, pattern, channel=None, op='==') - return matching rows from log table

    Attributes:
        path
        log - pandas DataFrame, one row per trial, with the following columns:
           PLAYLIST INFO (copied straight from playlist logs)
            MODE: List[int]
            cnt: int
            delayPost: List[float]
            freq: List[float]
            intensity: List[float]
            silencePost: List[float]
            silencePre: List[float]
            stimFileName: List[str]
           FILE INFO
            file_names: List[str] list of tif file names
            frames_first: List[int] *first* frame in tif each file contributing to that trial
            frames_last: List[int] *last* frame in tif each file contributing to that trial
           PER FRAME INFO
            frametimes_ms: List[float] time (in millisecond) for each frame rel. to trial start
            frame_zindex: List[int] slice index for each frame
            frame_zpos: List[float] avg z-pos for each frame
           PER STACK INFO
            nb_frames: int total number of frames in trial
            frame_width
            frame_height
            nb_channels: number of channels in stack (typically two channels, one for PMT in the microscope)
            channel_names
            frame_rate_hz: frame rate as defined in scanimage, actual frame rate may differ slightly but these values should be good enough for most use cases
            volume_rate_hz: volume rate as defined in scanimage. should be close to frame_rate_hz * nb_slices. otherwise should be close to fra, actual volume rate may differ slightly but this values should be good enough for most use cases
            nb_slices
            stimonset_ms
            stimoffset_ms
            stimonset_frame
            stimoffset_frame
    """

    def __init__(self, path, with_pixel_zpos=False, analog_out_channel_names=None, analog_in_channel_names=None) -> None:
        """Intialize a session.

        Args:
            path ([type]): [description]
            with_pixel_zpos (bool, optional): [description]. Defaults to False.
            analog_out_channel_names ([type], optional): Names of the output channels specified in the playlist. Defaults to None.
            analog_in_channel_names ([type], optional): Names of the analog input channesl in *_daq.h5 file. Defaults to None.
        """

        self.path = path
        self._log_file_name = self.path + "_daq.log"
        self._daq_file_name = self.path + "_daq.h5"
        self._prot_file_name = self.path + "_prot.yml"

        # open protocol file
        self.prot = None
        if os.path.exists(self._prot_file_name):
            logging.info(f"   Parsing protocol file {self._prot_file_name}.")
            self.prot = load_prot(self._prot_file_name)

        # gather information from logs and tif files
        self._logs_files = parse_trial_files(self.path)
        frame_shapes = None
        if with_pixel_zpos:
            frame_shapes = [(lf.frame_width, lf.frame_height) for lf in self._logs_files]

        # drop last file if it only contains a single volume or layer
        last_nb_frames = self._logs_files[-1].nb_frames
        nb_slices = self._logs_files[-1].nb_slices
        if last_nb_frames < nb_slices * 2:              # number of frames must be at least two times the number of slices
            self._logs_files = self._logs_files[:-1]
            logging.info("Last frame has less than two volumens - dropping.")

        self._logs_files = pd.DataFrame(self._logs_files)
        self.nb_trials = len(self._logs_files)
        # get info from daq files
        if analog_in_channel_names is None:
            try:
                analog_in_channel_names = self.prot["DAQ"]["analog_chans_in_info"]
            except:
                analog_in_channel_names = self.prot["DAQ"]["analog_chans_in"]

        self._logs_timing = parse_trial_timing(self._daq_file_name, frame_shapes, analog_in_channel_names)
        self._logs_timing = pd.DataFrame(self._logs_timing).iloc[: self.nb_trials]
        # print(self._logs_timing.iloc)

        if analog_out_channel_names is None:
            try:
                analog_out_channel_names = (
                    self.prot["DAQ"]["analog_chans_out_info"] + self.prot["DAQ"]["digital_chans_out_info"]
                )
            except:
                analog_out_channel_names = self.prot["DAQ"]["analog_chans_out"] + self.prot["DAQ"]["digital_chans_out"]

        tmp = parse_stim_log(self._log_file_name)
        self._logs_stims = make_df_multi_index(tmp, analog_out_channel_names).iloc[: self.nb_trials]
        self.log = pd.concat((self._logs_stims, self._logs_files, self._logs_timing), axis=1)

        del self._logs_timing
        del self._logs_files
        del self._logs_stims

        self.log.index.name = "trial"

        # session-wide information
        self.nb_trials = len(self.log)

    def __repr__(self) -> str:
        return f"Session in {self.path} with {self.nb_trials} trials."

    def stack(
        self,
        trial_number: Optional[int] = None,
        split_channels: bool = True,
        split_volumes: bool = True,
        force_dims: bool = False,
        use_zarr: bool = False,
    ) -> np.ndarray:
        """Load stack for a specific trial or for all trials.

        Gathers frames across files and reshape according to number of channels and/or volumes.

        Args:
            trial_number (int, optional): Trial to load. If not provided or None, will load concatenate stacks across all trials. Defaults to None.
            split_channels (bool, optional): reshape channel-interleaved tif to [time, [volume], x, y, channel]. Defaults to True.
            split_volumes (bool, optional): reshape channel-interleaved tif to [time, volume, x, y, [channel]]. Defaults to False.
            force_dims (bool, optional): [description]. Defaults to False.
        Returns:
            np.ndarray of shape [time, width, heigh, channels]
        """
        if trial_number is None:
            stack, frame_times = self._all_trials_stack()
            # Stack metadata together, transform to dict
            metadata = {}
            for key in self.log.columns:
                # For list or array elements
                if isinstance(self.log[key][0], list) or isinstance(self.log[key][0], np.ndarray):
                    # Make list of lists for fields with unequal length
                    if key in ["file_names", "frames_first", "frames_last"]:
                        metadata[key] = [ el for el in self.log[key] ]
                    else:
                        # Stack other fields to array 
                        metadata[key] = np.hstack(self.log[key])
                else:
                    # For other data types, convert to array
                    metadata[key] = self.log[key].to_numpy()
        else:
            stack, frame_times = self._single_trial_stack(trial_number)
            metadata = self.log.loc[trial_number].to_dict()
        stack = self._reshape(stack, split_channels, split_volumes, force_dims, use_zarr)

        # volume times correspond to the time of the first frame for each volume
        nb_volumes = stack.shape[0]
        nb_layers = stack.shape[1]
        frame_times = frame_times[::nb_layers]
        frame_times = frame_times[:nb_volumes]

        stack = xr.DataArray(
            stack,
            dims=["time", "z", "x", "y", "channel"],
            coords={
                "time": frame_times,
                "z": np.arange(stack.shape[1], dtype=int),
                "channel": ["gcamp", "tdtomato"][: stack.shape[-1]],
            },
            attrs=metadata,
        )
        # Add infos about the stimuli to the dataset attributes (works for single trials and all trials)
        stack.attrs["stim_info"] = self.stim_info(trial=stack)

        return stack

    def _all_trials_stack(self, use_zarr: bool = False) -> np.ndarray:
        nb_trials = len(self.log)
        trial = self.log.loc[0]
        total_nb_frames = self.log.loc[nb_trials-1, "trial_offset_frame"]   # Set the last trial offset frame as the total number of frames 
        if use_zarr:
            filename = "tmp.mmap"
            stack = np.memmap(
                filename, dtype=np.int16, mode="w+", shape=(total_nb_frames, trial.frame_width, trial.frame_height)
            )
            stack[:] = 0
        else:
            stack = np.zeros((total_nb_frames, trial.frame_width, trial.frame_height), dtype=np.int16)

        last_idx = 0
        frame_times = -np.ones(total_nb_frames)
        for trial_number in range(self.nb_trials):
            trial_stack, stack_frame_times = self._single_trial_stack(trial_number)
            stack[last_idx : int(last_idx + trial_stack.shape[0]), ...] = trial_stack
            # Add last time value to frame times of current stack (not for the first stack)
            if trial_number > 0:
                stack_frame_times = stack_frame_times + frame_times[last_idx-1]
            frame_times[last_idx : int(last_idx + trial_stack.shape[0])] = stack_frame_times[: trial_stack.shape[0]]
            last_idx += trial_stack.shape[0]

        return stack, frame_times

    def _single_trial_stack(self, trial_number: int) -> np.ndarray:
        """Loads the stack for a single trial.

        See `stack` for args.
        """
        trial = self.log.loc[trial_number]
        stack = np.zeros((int(trial.nb_frames), int(trial.frame_width), int(trial.frame_height)), dtype=np.int16)
        frame_times = -np.ones(int(trial.nb_frames))
        last_idx = 0
        # gather frames for the trial across files
        for file_name, first_frame, last_frame in zip(trial.file_names, trial.frames_first, trial.frames_last):
            with ScanImageTiffFile(file_name) as f:
                d = f.data(beg=np.uint32(first_frame), end=np.uint32(last_frame))
                stack[last_idx : int(last_idx + d.shape[0]), ...] = d
                last_idx += d.shape[0]

        frame_times = np.array(trial["frameonset_ms"]) / 1_000
        stack = stack[: len(frame_times), ...]
        return stack, frame_times

    def _reshape(
        self,
        stack: np.ndarray,
        split_channels: bool = True,
        split_volumes: bool = False,
        force_dims: bool = False,
        use_zarr: bool = False,
    ) -> np.ndarray:
        # reshape to split channels
        trial = self.log.loc[0]
        if split_channels:
            stack = stack.reshape((-1, int(trial.nb_channels), int(trial.frame_width), int(trial.frame_height)))
            stack = stack.transpose((0, 2, 3, 1))  # reorder from [frames, channels, x, y] to [frames, x, y, channels]

        # split by planes into volumes
        if split_volumes:
            if split_channels:
                nb_volumes = int(np.floor(stack.shape[0] / trial.nb_slices) * trial.nb_slices)
                stack = stack[:nb_volumes, ...]
                stack = stack.reshape((-1, int(trial.nb_slices), *stack.shape[1:]))
            else:
                nb_volumes = int(np.floor(stack.shape[0] / trial.nb_slices / trial.nb_channels) * trial.nb_slices)
                stack = stack[: nb_volumes * trial.nb_channels, ...]
                stack = stack.reshape((-1, trial.nb_slices, *stack.shape[1:]))

        if force_dims:
            if not split_channels:
                stack = stack[..., np.newaxis]
            if not split_volumes:
                stack = stack[:, np.newaxis, ...]

        return stack

    def stim_info(self, *, trial=None, trial_number=None):
        """Return stimulus info as DataFrame

        Must specify:
        - either trial (stack with attached metadata for a single trial)
        - or the trial number (will load associated trial)

        Args:
            trial (xarray.DataSet, optional): _description_. Defaults to None.
            trial_number (int, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: For each stimulus channel, info on stim name, onset/offset, and intensity,
        """
        if trial is not None:
            trial_log = trial.attrs
        elif trial_number is not None:
            trial_log = self.log.loc[trial_number]

        stims = [k.rstrip("_onset_ms") for k in trial_log.keys() if "_onset_ms" in k]

        keys = ["name", "onset_seconds", "offset_seconds", "intensity"]
        stim_info = {key: [] for key in keys}
        for stim in stims:
            stim_info["onset_seconds"].append(trial_log[stim + "_onset_ms"] / 1_000)
            stim_info["offset_seconds"].append(trial_log[stim + "_offset_ms"] / 1_000)
            stim_info["name"].append(trial_log[("stimFileName", stim)])
            stim_info["intensity"].append(trial_log[("intensity", stim)])

        stim_info = pd.DataFrame(stim_info, index=stims)
        return stim_info

    def argfind(self, column_title, pattern, channel=None, op="==") -> List[int]:
        """Get trial numbers of matching rows in playlist.

        Args:
            column_title:
            pattern:
            channel=None:
            op='==': any of the standard comparison operators ('==', '>', '>=', '<', '<=', ') or 'in' for partial string matching.
        Returns:
            list of indices
        """
        if isinstance(pattern, str):
            pattern = '"' + pattern + '"'

        if channel is None:
            channels = [channel for channel in self.log[column_title]]
        else:
            channels = [channel]

        matches = []
        for channel in channels:
            for x, idx in zip(self.log[(column_title, channel)], self.log.index):
                if isinstance(x, str):
                    x = '"' + x + '"'

                if op == "in":
                    out = eval("{0}{1}{2}".format(pattern, op, x))
                else:
                    out = eval("{0}{1}{2}".format(x, op, pattern))

                if out:
                    matches.append(idx)
        return matches

    def find(self, column_title, pattern, channel=None, op="=="):
        """Get matching rows from playlist (see argmatch for details)."""
        matches = self.argfind(column_title, pattern, channel, op)
        return self.log.loc[matches]
