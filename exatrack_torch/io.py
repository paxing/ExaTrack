# -*- coding: utf-8 -*-
"""
io.py
-----
Data input/output utilities: reading, padding, segmenting, and exporting tracks.

Functions
---------
read_table              : load tracks from CSV/pkl files into lists of arrays
padding                 : homogenise variable-length tracks with zero-padding
segment_tracks          : split long tracks into fixed-length segments for batching
_segment_tracks_core    : Numba-accelerated inner loop for segment_tracks
TrackSegmentSequence    : torch Dataset wrapping segment_tracks for the training loop
ExaTrack_2_DataFrame    : convert ExaTrack predictions back to a pandas DataFrame
correct_state_predictions_padding : fix state prediction indices after padding

Dependencies: numpy, pandas, numba, torch
"""

import numpy as np
import pandas as pd
import torch
from numba import njit
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_table(paths,
               lengths=np.arange(4, 40),
               dist_th=np.inf,
               frames_boundaries=[-np.inf, np.inf],
               fmt='csv',
               colnames=['POSITION_X', 'POSITION_Y', 'POSITION_T', 'TRACK_ID'],
               opt_colnames=[],
               remove_no_disp=True):
    """
    Read single-particle tracks from one or more tabular files.

    Parameters
    ----------
    paths             : str or list of str — path(s) to the file(s)
    lengths           : accepted track lengths (others are discarded or split)
    dist_th           : maximum allowed displacement between consecutive frames
    frames_boundaries : [min, max] frame indices to include
    fmt               : 'csv', 'pkl', or a separator string (e.g. ' ', '\\t')
    colnames          : [X, Y, T, TRACK_ID] column names
    opt_colnames      : list of additional columns to collect
    remove_no_disp    : if True, discard tracks with >5% zero displacements

    Returns
    -------
    tracks, frames, track_IDs, opt_metrics
    """
    if isinstance(paths, (str, np.str_)):
        paths = [paths]

    colnames = list(colnames)
    tracks, frames, track_IDs = [], [], []
    opt_metrics = {m: [] for m in opt_colnames}

    for path in paths:
        if fmt == 'csv':
            data = pd.read_csv(path, sep=',', low_memory=False)
        elif fmt == 'pkl':
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path, sep=fmt, low_memory=False)

        if not isinstance(colnames[3], (str, np.str_)):
            None_ID = (data[colnames[3]] == 'None') + pd.isna(data[colnames[3]])
            data = data.drop(data[np.any(None_ID, 1)].index)
            new_ID = data[colnames[3][0]].astype(str)
            for k in range(1, len(colnames[3])):
                new_ID = new_ID + '_' + data[colnames[3][k]].astype(str)
            data['unique_ID'] = new_ID
            colnames[3] = 'unique_ID'

        try:
            None_ID = ((data[colnames[3]] == 'None')
                       + pd.isna(data[colnames[3]]))
            max_ID = np.max(data[colnames[3]][
                (~None_ID)].astype(int))
            data.loc[None_ID, colnames[3]] = np.arange(
                max_ID + 1, max_ID + 1 + np.sum(None_ID))
        except Exception:
            None_ID = (data[colnames[3]] == 'None') + pd.isna(data[colnames[3]])
            data = data.drop(data[None_ID].index)

        data = data[colnames + opt_colnames].dropna()

        try:
            for ID, track in data.groupby(colnames[3]):
                track = track.sort_values(colnames[2], axis=0)
                track_mat = track.values[:, :4].astype('float64')
                dists2 = (track_mat[1:, :2] - track_mat[:-1, :2]) ** 2
                if remove_no_disp:
                    if len(dists2) > 0 and np.mean(dists2 == 0) > 0.05:
                        continue
                dists = np.sum(dists2, axis=1) ** 0.5

                if (track_mat[0, 2] >= frames_boundaries[0]
                        and track_mat[0, 2] <= frames_boundaries[1]
                        and not np.any(dists > dist_th)):
                    if np.any(len(track_mat) == np.array(lengths)):
                        tracks.append(track_mat[:, 0:2])
                        frames.append(track_mat[:, 2])
                        track_IDs.append(track_mat[:, 3])
                        for m in opt_colnames:
                            opt_metrics[m].append(track[m].values)
                    elif len(track_mat) > np.max(lengths):
                        for k in range(len(track_mat) // np.max(lengths)):
                            l = np.max(lengths)
                            tracks.append(track_mat[l * k:l * (k + 1), 0:2])
                            frames.append(track_mat[l * k:l * (k + 1), 2])
                            for m in opt_colnames:
                                opt_metrics[m].append(
                                    track[m].values[l * k:l * (k + 1)])
        except Exception as e:
            import logging
            logging.error(f'Error reading {path}: {e}')

    return tracks, frames, track_IDs, opt_metrics


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def padding(track_list, LocErr_list=None, dt_list=None, batch_size=None):
    """
    Pad variable-length tracks to a common length for batched processing.

    Parameters
    ----------
    track_list  : list of (track_len_i, nb_dims) arrays
    LocErr_list : list of per-frame localisation errors (optional)
    dt_list     : list of per-frame durations (optional)
    batch_size  : if given, pad the batch dimension to a multiple of this

    Returns
    -------
    padded_tracks, padded_LocErrs, padded_dts, mask
    """
    max_len = max(t.shape[0] for t in track_list)
    nb_tracks = len(track_list)
    if batch_size is not None:
        nb_tracks = int(np.ceil(nb_tracks / batch_size)) * batch_size

    padded_tracks = np.zeros(
        (nb_tracks, max_len, track_list[0].shape[1]),
        dtype=track_list[0].dtype)

    padded_dts = None
    if dt_list is not None:
        padded_dts = np.zeros(
            (nb_tracks, max_len + 1), dtype=dt_list[0].dtype)

    padded_LocErrs = None
    if LocErr_list is not None:
        if LocErr_list[0].ndim == 2:
            padded_LocErrs = np.zeros(
                (nb_tracks, max_len, LocErr_list[0].shape[1]),
                dtype=LocErr_list[0].dtype)
        else:
            padded_LocErrs = np.zeros(
                (nb_tracks, max_len), dtype=LocErr_list[0].dtype)

    mask = np.zeros((nb_tracks, max_len), dtype=track_list[0].dtype)

    for i, track in enumerate(track_list):
        cur_len = track.shape[0]
        if cur_len < 1:
            raise Warning('Tracks of 1 time point were discarded.')
        padded_tracks[i, :cur_len] = track
        padded_tracks[i, cur_len:] = track[-1]
        mask[i, :cur_len] = 1
        if dt_list is not None:
            dts = dt_list[i]
            padded_dts[i, :cur_len] = dts
            padded_dts[i, cur_len:] = dts[-1]
        if LocErr_list is not None:
            LocErrs = LocErr_list[i]
            padded_LocErrs[i, :cur_len] = LocErrs
            padded_LocErrs[i, cur_len:] = LocErrs[-1]

    return padded_tracks, padded_LocErrs, padded_dts, mask


# ---------------------------------------------------------------------------
# Track segmentation (Numba-accelerated)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _segment_tracks_core(tracks_flat, locerrs_flat, dts_flat,
                          track_offsets, dt_offsets,
                          batch_size, segment_length, min_segment_length,
                          max_nb_batches, mean_dt):
    """
    Numba-JIT inner loop for segment_tracks.

    Splits variable-length tracks into fixed-length segments and packs them
    into batched arrays. Uses the leftmost-column-with-minimum-next-row
    placement strategy to reproduce the original np.where C-order placement.
    """
    nb_dims = tracks_flat.shape[1]
    locerr_dims = locerrs_flat.shape[1]

    track_batches = np.zeros((max_nb_batches, batch_size, segment_length, nb_dims))
    locerr_batches = np.zeros((max_nb_batches, batch_size, segment_length, locerr_dims))
    dt_batches = np.full((max_nb_batches, batch_size, segment_length + 1), mean_dt)
    mask_batches = np.zeros((max_nb_batches, batch_size, segment_length))
    isfirst_batches = np.ones((max_nb_batches, batch_size))

    next_row = np.zeros(batch_size, dtype=np.int64)
    nb_tracks = len(track_offsets) - 1

    for t in range(nb_tracks):
        track_start = track_offsets[t]
        track_end = track_offsets[t + 1]
        track_len = track_end - track_start
        dt_start = dt_offsets[t]
        dt_track_len = dt_offsets[t + 1] - dt_start

        nb_segments = track_len // segment_length
        if track_len % segment_length > min_segment_length:
            nb_segments += 1
        if nb_segments == 0:
            continue

        min_row, min_col = next_row[0], 0
        for c in range(1, batch_size):
            if next_row[c] < min_row:
                min_row, min_col = next_row[c], c

        batch_ID, index_ID = min_row, min_col

        for i in range(nb_segments):
            seg_start = i * (segment_length - 1)
            seg_end_unclipped = (i + 1) * segment_length - i
            seg_end = min(seg_end_unclipped, track_len)
            seg_len = seg_end - seg_start
            last_track_idx = track_start + seg_end - 1
            row = batch_ID + i

            track_batches[row, index_ID, :seg_len] = \
                tracks_flat[track_start + seg_start:track_start + seg_end]
            for k in range(nb_dims):
                pad_val = tracks_flat[last_track_idx, k]
                for j in range(seg_len, segment_length):
                    track_batches[row, index_ID, j, k] = pad_val

            locerr_batches[row, index_ID, :seg_len] = \
                locerrs_flat[track_start + seg_start:track_start + seg_end]
            for k in range(locerr_dims):
                pad_val = locerrs_flat[last_track_idx, k]
                for j in range(seg_len, segment_length):
                    locerr_batches[row, index_ID, j, k] = pad_val

            dt_end = min(seg_end_unclipped + 1, dt_track_len)
            dt_seg_len = dt_end - seg_start
            last_dt_idx = dt_start + dt_end - 1
            dt_batches[row, index_ID, :dt_seg_len] = \
                dts_flat[dt_start + seg_start:dt_start + dt_end]
            pad_val_dt = dts_flat[last_dt_idx]
            for j in range(dt_seg_len, segment_length + 1):
                dt_batches[row, index_ID, j] = pad_val_dt

            for j in range(seg_len):
                mask_batches[row, index_ID, j] = 1.0
            if i != 0:
                isfirst_batches[row, index_ID] = 0.0

        next_row[index_ID] = batch_ID + nb_segments

    return track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches


def segment_tracks(track_list, LocErr_list, dt_list, batch_size,
                   segment_length=20, min_segment_length=4,
                   cutoff_batch_treshhold=0.5, shuffle=False):
    """
    Split long tracks into fixed-length segments and pack them into batches.

    Parameters
    ----------
    track_list             : list of (track_len, nb_dims) arrays
    LocErr_list            : list of localisation errors (or None)
    dt_list                : list of frame durations
    batch_size             : number of tracks per batch
    segment_length         : number of frames per segment
    min_segment_length     : minimum trailing segment length to include
    cutoff_batch_treshhold : fraction threshold for keeping partial batches
    shuffle                : if True, shuffle track order

    Returns
    -------
    track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches
    """
    cutoff_batch_treshhold = np.clip(
        cutoff_batch_treshhold, 0.5 / batch_size, 1.0)

    if LocErr_list is None:
        LocErr_list = [np.ones(len(track)) for track in track_list]

    nb_tracks = len(track_list)
    nb_dims = track_list[0].shape[1]
    locerr_is_1d = (LocErr_list[0].ndim == 1)
    locerr_dims = 1 if locerr_is_1d else LocErr_list[0].shape[1]

    if shuffle:
        perm = np.random.permutation(nb_tracks)
        track_list = [track_list[i] for i in perm]
        LocErr_list = [LocErr_list[i] for i in perm]
        dt_list = [dt_list[i] for i in perm]

    track_lens = np.fromiter((len(t) for t in track_list),
                              dtype=np.int64, count=nb_tracks)
    dt_lens = np.fromiter((len(d) for d in dt_list),
                           dtype=np.int64, count=nb_tracks)

    track_offsets = np.empty(nb_tracks + 1, dtype=np.int64)
    track_offsets[0] = 0
    np.cumsum(track_lens, out=track_offsets[1:])

    dt_offsets = np.empty(nb_tracks + 1, dtype=np.int64)
    dt_offsets[0] = 0
    np.cumsum(dt_lens, out=dt_offsets[1:])

    total_track_len = int(track_offsets[-1])
    total_dt_len = int(dt_offsets[-1])

    tracks_flat = np.zeros((total_track_len, nb_dims), dtype=np.float64)
    locerrs_flat = np.zeros((total_track_len, locerr_dims), dtype=np.float64)
    dts_flat = np.zeros(total_dt_len, dtype=np.float64)

    for i in range(nb_tracks):
        s, e = int(track_offsets[i]), int(track_offsets[i + 1])
        tracks_flat[s:e] = track_list[i]
        if locerr_is_1d:
            locerrs_flat[s:e, 0] = LocErr_list[i]
        else:
            locerrs_flat[s:e] = LocErr_list[i]
        ds, de = int(dt_offsets[i]), int(dt_offsets[i + 1])
        dts_flat[ds:de] = dt_list[i]

    mean_dt = float(dts_flat.mean()) if total_dt_len > 0 else 0.0
    max_track_length = int(track_lens.max())
    max_nb_batches = (nb_tracks // batch_size + 1) * (max_track_length // segment_length + 1) * 2

    (track_batches, locerr_batches, dt_batches,
     mask_batches, isfirst_batches) = _segment_tracks_core(
        tracks_flat, locerrs_flat, dts_flat,
        track_offsets, dt_offsets,
        int(batch_size), int(segment_length), int(min_segment_length),
        int(max_nb_batches), mean_dt)

    nb_batches = int(np.argmin(mask_batches[:, :, 0].mean(axis=1)
                               >= cutoff_batch_treshhold))
    track_batches = track_batches[:nb_batches]
    locerr_batches = locerr_batches[:nb_batches]
    dt_batches = dt_batches[:nb_batches]
    mask_batches = mask_batches[:nb_batches]
    isfirst_batches = isfirst_batches[:nb_batches]

    if locerr_is_1d:
        locerr_batches = locerr_batches[..., 0]

    return track_batches, locerr_batches, dt_batches, mask_batches, isfirst_batches


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper (replaces Keras Sequence)
# ---------------------------------------------------------------------------

class TrackSegmentSequence(Dataset):
    """
    PyTorch Dataset that pre-segments tracks into fixed-length batches.

    Each __getitem__ call returns one pre-computed batch as torch tensors
    (tracks, LocErrs, dts, masks, isfirsts). The training loop can iterate
    directly over this dataset.

    Replaces the Keras Sequence of the same name.
    """

    def __init__(self, track_list, LocErr_list, dt_list,
                 batch_size, segment_length=20, min_segment_length=4,
                 cutoff_batch_treshhold=0.5, shuffle=False):
        self.track_list = track_list
        if LocErr_list is None:
            LocErr_list = [np.ones(len(track)) for track in track_list]
        if dt_list is None:
            dt_list = [np.ones(len(track)) for track in track_list]
        self.LocErr_list = LocErr_list
        self.dt_list = dt_list
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        self.cutoff_batch_treshhold = cutoff_batch_treshhold
        self.shuffle = shuffle
        self.batch_size = batch_size

        self._resegment()

    def _resegment(self):
        tracks, LocErrs, dts, masks, isfirsts = segment_tracks(
            self.track_list, self.LocErr_list, self.dt_list, self.batch_size,
            self.segment_length, self.min_segment_length,
            self.cutoff_batch_treshhold, self.shuffle)
        self.tracks = torch.tensor(tracks, dtype=torch.float64)
        self.LocErrs = torch.tensor(LocErrs, dtype=torch.float64)
        self.dts = torch.tensor(dts, dtype=torch.float64)
        self.masks = torch.tensor(masks, dtype=torch.float64)
        self.isfirsts = torch.tensor(isfirsts, dtype=torch.float64)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return (self.tracks[idx],
                self.LocErrs[idx],
                self.dts[idx],
                self.masks[idx],
                self.isfirsts[idx])

    def on_epoch_end(self):
        """Call at the end of each epoch to optionally re-shuffle segments."""
        if self.shuffle:
            self._resegment()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def ExaTrack_2_DataFrame(track_list, frame_list, track_ID_list,
                          opt_metrics, state_preds, all_masks):
    """
    Convert ExaTrack state predictions to a pandas DataFrame.

    Parameters
    ----------
    track_list   : list of position arrays
    frame_list   : list of frame index arrays
    track_ID_list: list of track ID arrays
    opt_metrics  : dict of optional per-frame metrics
    state_preds  : (nb_tracks, track_len, nb_states) state probability arrays
    all_masks    : (nb_tracks, track_len) validity masks

    Returns
    -------
    pandas DataFrame with columns:
        POSITION_X, POSITION_Y[, POSITION_Z], FRAME, TRACK_ID,
        STATE_0, ..., STATE_{n-1}, STATE_MISLABELED,
        [optional metric columns]
    """
    nb_rows = int(np.sum(all_masks))
    nb_dims = track_list[0].shape[1]
    nb_states = state_preds.shape[-1]
    opt_colnames = list(opt_metrics.keys())

    track_array = np.zeros((nb_rows, nb_dims))
    frame_array = np.zeros((nb_rows, 1))
    state_pred_array = np.zeros((nb_rows, nb_states))
    track_ID_array = np.zeros((nb_rows, 1))
    opt_metrics_array = np.zeros((nb_rows, len(opt_colnames)))

    idx = 0
    for i in range(len(track_list)):
        track_length = int(np.sum(all_masks[i]))
        track_array[idx:idx + track_length] = track_list[i]
        frame_array[idx:idx + track_length] = frame_list[i][:, None]
        state_pred_array[idx:idx + track_length] = \
            state_preds[i][all_masks[i].astype(bool)]
        track_ID_array[idx:idx + track_length] = track_ID_list[i][:, None]
        for j, col in enumerate(opt_colnames):
            opt_metrics_array[idx:idx + track_length, j] = opt_metrics[col][i]
        idx += track_length

    data = np.concatenate(
        (track_array, frame_array, track_ID_array,
         state_pred_array, opt_metrics_array), axis=1)

    state_names = [f'STATE_{s}' for s in range(nb_states - 1)] + ['STATE_MISLABELED']
    columns = (['POSITION_X', 'POSITION_Y', 'POSITION_Z'][:nb_dims]
               + ['FRAME', 'TRACK_ID']
               + state_names
               + opt_colnames)

    return pd.DataFrame(data, columns=columns)


def correct_state_predictions_padding(state_preds, all_masks, sequence_length):
    """Shift state predictions to align with actual track positions after padding."""
    max_length = state_preds.shape[1]
    for i in range(len(state_preds)):
        track_length = int(np.sum(all_masks[i]))
        if track_length <= sequence_length:
            state_preds[i, :track_length] = state_preds[i, -track_length:]
        elif track_length < max_length:
            state_preds[i, track_length - sequence_length:track_length] = \
                state_preds[i, -sequence_length:]
        state_preds[i, track_length:] = 0
