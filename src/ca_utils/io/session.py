"""Represent a single calcium imaging experiment."""
import numpy as np
import pandas as pd
from .utils import parse_trial_timing, parse_trial_files, parse_stim_log, make_df_multi_index
from .scanimagetiffile import ScanImageTiffFile


class Session():
    """Session object.

    Construction:
        s = Session(path)

    Methods:
        stack(trial_number) - returns 4D matrix [time steps, frame width, frame height, channels]
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
            frame_avgzpos: List[float] avg z-pos for each frame
           PER STACK INFO
            nb_frames: int total number of frames in trial
            frame_width
            frame_height
            nb_channels: number of channels in stack (typically two channels, one for PMT in the microscope)
            channel_names
            frame_rate_hz: frame rate as defined in scanimage, actual frame rate may differ slightly but this values should be good enough for most use cases
            volume_rate_hz: volume rate as defined in scanimage. should be close to frame_rate_hz * nb_slices. otherwise should be close to fra, actual volume rate may differ slightly but this values should be good enough for most use cases
            nb_slices
            stimonset_ms
            stimoffset_ms
            stimonset_frame
            stimoffset_frame
    """

    def __init__(self, path):
        """Init."""
        self.path = path
        self._log_file_name = self.path + '_daq.log'
        self._daq_file_name = self.path + '_daq.h5'

        # gather information from logs and data files
        self._logs_timing = parse_trial_timing(self._daq_file_name)
        self._logs_files = parse_trial_files(self.path)
        tmp = parse_stim_log(self._log_file_name)

        self._logs_stims = make_df_multi_index(tmp)
        self.log = pd.concat((self._logs_stims,
                               pd.DataFrame(self._logs_files),
                               pd.DataFrame(self._logs_timing)), axis=1)
        self.log.index.name = 'trial'

        # session-wide information
        self.nb_trials = len(self.log)

    def __repr__(self):
        return f"Session in {self.path} with {self.nb_trials} trials."

    def stack(self, trial_number: int) -> np.ndarray:
        """Load stack for a specific trial.

        Will gather frames across files and reshape according to number of channels.

        Args:
            trial_number
        Returns:
            np.ndarray of shape [time, width, heigh, channels]
        """
        trial = self.log.loc[trial_number]
        stack = np.zeros((trial.nb_frames, trial.frame_width, trial.frame_height), dtype=np.int16)
        last_idx = 0
        # gather frames across files
        for file_name, first_frame, last_frame in zip(trial.file_names, trial.frames_first, trial.frames_last):
            with ScanImageTiffFile(file_name) as f:
                d = f.data(beg=int(first_frame), end=int(last_frame))  # last_frame is +1 since slice-indices are not inclusive
                stack[last_idx:int(last_idx + d.shape[0]), ...] = d
                last_idx += d.shape[0]
        # reshape to split channels
        stack = stack.reshape((int(trial.nb_frames / trial.nb_channels), trial.nb_channels, trial.frame_width, trial.frame_height)).transpose((0, 2, 3, 1))
        return stack

    def argfind(self, column_title, pattern, channel=None, op='=='):
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
                if op == 'in':
                    out = eval('{0}{1}{2}'.format(pattern, op, x))
                else:
                    out = eval('{0}{1}{2}'.format(x, op, pattern))
                if out:
                    matches.append(idx)
        return matches

    def find(self, column_title, pattern, channel=None, op='=='):
        """Get matching rows from playlist.
        See argmatch for details.
        """
        matches = self.argfind(column_title, pattern, channel, op)
        return self.log.loc[matches]
