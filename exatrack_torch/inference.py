# -*- coding: utf-8 -*-
"""
inference.py
------------
Post-fitting inference: hidden variable extraction and forward-backward smoothing.

PyTorch conversion notes
------------------------
- pred_model.predict(inputs, batch_size=N) →
      with torch.no_grad(): model(inputs, return_all=True)  [batched]
- pred_model.weights[7].assign(tf.transpose(w)) →
      model.rnn_layer.transition_rates.data.copy_(w.t())
- pred_model.weights[8].assign(tf.transpose(w)) →
      model.rnn_layer.transition_shapes.data.copy_(w.t())
- numpy-only functions (marginalise_variable, extract_hidden_variables,
  extract_smooth_hidden_variables post-processing) are unchanged.
"""

import numpy as np
import torch
from scipy.special import softmax as scipy_softmax

from .config import dtype


# ---------------------------------------------------------------------------
# Helper: batched prediction
# ---------------------------------------------------------------------------

def _predict_all(pred_model, inputs, batch_size, device='cpu'):
    """
    Run pred_model(return_all=True) in batches, return concatenated numpy arrays:
    (LPs, All_states, All_coefs, All_biases, All_LPs).
    """
    tracks, LocErrs, dts, masks, is_first = inputs
    nb_tracks = tracks.shape[0]

    LPs_list, states_list, coefs_list, biases_list, lps_list = [], [], [], [], []

    pred_model.eval()
    with torch.no_grad():
        for start in range(0, nb_tracks, batch_size):
            end = min(start + batch_size, nb_tracks)
            batch_tracks   = torch.tensor(tracks[start:end],    dtype=dtype, device=device)
            batch_LocErrs  = torch.tensor(LocErrs[start:end],   dtype=dtype, device=device)
            batch_dts      = torch.tensor(dts[start:end],       dtype=dtype, device=device)
            batch_masks    = torch.tensor(masks[start:end],     dtype=dtype, device=device)
            batch_isfirst  = torch.tensor(is_first[start:end],  dtype=dtype, device=device)

            out_lp, out_states, out_coefs, out_biases, out_lps = pred_model(
                batch_tracks, batch_LocErrs, batch_dts,
                batch_masks, batch_isfirst, return_all=True)

            LPs_list.append(out_lp.cpu().numpy())
            states_list.append(out_states.cpu().numpy())
            coefs_list.append(out_coefs.cpu().numpy())
            biases_list.append(out_biases.cpu().numpy())
            lps_list.append(out_lps.cpu().numpy())

    return (np.concatenate(LPs_list,  axis=0),
            np.concatenate(states_list, axis=0),
            np.concatenate(coefs_list,  axis=0),
            np.concatenate(biases_list, axis=0),
            np.concatenate(lps_list,    axis=0))


# ---------------------------------------------------------------------------
# Gaussian marginalisation (pure numpy — unchanged from TF version)
# ---------------------------------------------------------------------------

def marginalise_variable(All_coefs, All_biases, integrate_index):
    """
    Given nb_gaussians=2 Gaussians characterised by coefficients and biases,
    integrate (marginalise) over the hidden variable at `integrate_index`.
    """
    keep_index = 1 - integrate_index
    nb_dims = All_biases.shape[-1]

    C1 = All_coefs[..., 0:1, integrate_index]
    C2 = All_coefs[..., 1:,  integrate_index]

    coefs1  = All_coefs[..., 0, :]  / (C1 + 1e-30)
    coefs2  = All_coefs[..., 1, :]  / (C2 + 1e-30)
    biases1 = All_biases[..., 0, :] / (C1 + 1e-30)
    biases2 = All_biases[..., 1, :] / (C2 + 1e-30)

    var1 = 1.0 / (C1 ** 2 + 1e-30)
    var2 = 1.0 / (C2 ** 2 + 1e-30)
    var3 = var1 + var2
    std3 = np.sqrt(var3)

    coefs3  = (coefs1  - coefs2)  / std3
    biases3 = (biases1 - biases2) / std3

    var4    = var1 * var2 / var3
    std4    = var4 ** 0.5
    coefs4  = (coefs1  * var2 + coefs2  * var1) / (var3 * std4)
    biases4 = (biases1 * var2 + biases2 * var1) / (var3 * std4)

    LogConstant = -nb_dims * np.log(np.abs(C1 * C2 * std4 * std3))[:, :, :, 0]

    return coefs3, biases3, coefs4, biases4, LogConstant


def extract_hidden_variables(All_coefs, All_biases, All_LPs, nb_dims, sequence_length):
    """
    Estimate the distribution of the hidden variables at a given time step
    given the prior positions (online estimate, equivalent to filtering in
    the context of Kalman filters).
    """
    All_coefs  = np.array(All_coefs)
    All_biases = np.array(All_biases)
    All_LPs    = np.array(All_LPs)

    nb_tracks, track_len, nb_sequences = All_LPs.shape
    nb_states = nb_sequences // sequence_length

    # ---- anomalous variable ------------------------------------------------
    integrate_index = 0
    coefs3, biases3, coefs4, biases4, LogConstant = marginalise_variable(
        All_coefs, All_biases, integrate_index)

    All_LPs_ano = (All_LPs + LogConstant
                   - nb_dims * np.log(
                       np.abs(coefs4[..., integrate_index]) + 1e-30))
    All_LPs_ano = All_LPs_ano.reshape(
        (nb_tracks, track_len, sequence_length, nb_states))

    ano_MAP = -biases3 / coefs3[:, :, :, 1:]
    ano_MAP = ano_MAP.reshape(
        (nb_tracks, track_len, sequence_length, nb_states, nb_dims))
    ano_var = 1 / (coefs3[..., 1:] ** 2 + 1e-50)
    ano_var = ano_var.reshape(
        (nb_tracks, track_len, sequence_length, nb_states, 1))

    w_ano          = scipy_softmax(All_LPs_ano, axis=2)[..., None]
    anomalous_mean = np.sum(ano_MAP * w_ano, axis=2)
    anomalous_var  = (np.sum((ano_var + ano_MAP ** 2) * w_ano, axis=2)
                      - np.sum(ano_MAP * w_ano, axis=2) ** 2)
    anomalous_std  = anomalous_var ** 0.5

    # ---- position variable -------------------------------------------------
    integrate_index = 1
    All_LPs_pos = (All_LPs
                   - nb_dims * np.log(
                       np.abs(All_coefs[..., integrate_index, integrate_index])
                       + 1e-30))
    pos_MAP = -All_biases[..., 0, :] / All_coefs[..., 0, :1]
    pos_var = 1 / (All_coefs[..., 0, :1] ** 2 + 1e-50)

    w_pos        = scipy_softmax(All_LPs_pos, axis=2)[..., None]
    pos_mean     = np.sum(pos_MAP * w_pos, axis=2)
    position_var = (np.sum((pos_var + pos_MAP ** 2) * w_pos, axis=2)
                    - np.sum(pos_MAP * w_pos, axis=2) ** 2)
    position_std = position_var ** 0.5

    return pos_mean, anomalous_mean, position_std, anomalous_std


def extract_smooth_hidden_variables(tracks, LocErrs, dts, masks, pred_model,
                                    batch_size, sequence_length, motion_types,
                                    reference_dt, device='cpu'):
    """
    Variable-dt-aware version: returns per-step VELOCITY (dt-independent)
    for directed states instead of the step-wise displacement.

    Parameters
    ----------
    motion_types : iterable of length nb_states (physical states only, NOT
                   including the mislinking state).
                   1 for directed, 0 for confined.
    device       : torch device string or torch.device
    """
    motion_types = np.array(motion_types)
    nb_dims  = tracks.shape[-1]
    is_first = np.ones(tracks.shape[0])

    # Forward pass
    LPs, preds_1, All_coefs_1, All_biases_1, All_LPs_1 = _predict_all(
        pred_model, (tracks, LocErrs, dts, masks, is_first),
        batch_size, device)

    # Reverse pass — transpose Gamma shape/rate matrices
    tr = pred_model.rnn_layer.transition_rates.data.clone()
    ts = pred_model.rnn_layer.transition_shapes.data.clone()
    pred_model.rnn_layer.transition_rates.data.copy_(tr.t())
    pred_model.rnn_layer.transition_shapes.data.copy_(ts.t())

    inverse_dts = np.ascontiguousarray(
        np.concatenate((dts[:, -1:], dts[:, :-1]), axis=1)[:, ::-1])

    LPs, preds_2, All_coefs_2, All_biases_2, All_LPs_2 = _predict_all(
        pred_model,
        (np.ascontiguousarray(tracks[:, ::-1]),
         np.ascontiguousarray(LocErrs[:, ::-1]),
         inverse_dts,
         np.ascontiguousarray(masks[:, ::-1]),
         is_first),
        batch_size, device)

    # Restore transition matrices
    pred_model.rnn_layer.transition_rates.data.copy_(tr)
    pred_model.rnn_layer.transition_shapes.data.copy_(ts)

    # Per-pass marginals
    pos_mean_1, anomalous_mean_1, position_std_1, anomalous_std_1 = \
        extract_hidden_variables(All_coefs_1, All_biases_1, All_LPs_1,
                                 nb_dims, sequence_length)
    pos_mean_2, anomalous_mean_2, position_std_2, anomalous_std_2 = \
        extract_hidden_variables(All_coefs_2, All_biases_2, All_LPs_2,
                                 nb_dims, sequence_length)

    # Convert directed-state ano (velocity * dt) → velocity
    isdir_state  = motion_types[None, None, :, None]
    isconf_state = 1.0 - isdir_state

    nb_time = anomalous_mean_1.shape[1]

    fwd_dt = dts[:,         1:1 + nb_time][:, :, None, None]
    rev_dt = inverse_dts[:, 1:1 + nb_time][:, :, None, None]

    fwd_scale = isdir_state / fwd_dt * reference_dt + isconf_state
    rev_scale = isdir_state / rev_dt * reference_dt + isconf_state

    anomalous_mean_1 = anomalous_mean_1 * fwd_scale
    anomalous_std_1  = anomalous_std_1  * fwd_scale
    anomalous_mean_2 = anomalous_mean_2 * rev_scale
    anomalous_std_2  = anomalous_std_2  * rev_scale

    # Align time axes
    pos_mean_1     = pos_mean_1[:, 1:]
    position_std_1 = position_std_1[:, 1:]
    pos_mean_2     = pos_mean_2[:, :0:-1]
    position_std_2 = position_std_2[:, :0:-1]

    motion_type_sign = (-1 * (motion_types == 1) + 1 * (motion_types == 0))
    motion_type_sign = motion_type_sign[None, None, :, None]

    anomalous_mean_1 = anomalous_mean_1[:, 1:]
    anomalous_std_1  = anomalous_std_1[:, 1:]
    anomalous_mean_2 = motion_type_sign * anomalous_mean_2[:, :0:-1]
    anomalous_std_2  = anomalous_std_2[:, :0:-1]

    def optimal_estimator(x1, x2, var1, var2):
        w1 = 1 / var1
        w2 = 1 / var2
        return (w1 * x1 + w2 * x2) / (w1 + w2)

    position_mean = optimal_estimator(
        pos_mean_1, pos_mean_2, position_std_1 ** 2, position_std_2 ** 2)
    position_std  = (1 / (1 / position_std_1 ** 2
                          + 1 / position_std_2 ** 2)) ** 0.5

    anomalous_mean = optimal_estimator(
        anomalous_mean_1, anomalous_mean_2,
        anomalous_std_1 ** 2, anomalous_std_2 ** 2)
    anomalous_std  = (1 / (1 / anomalous_std_1 ** 2
                           + 1 / anomalous_std_2 ** 2)) ** 0.5

    mean_preds = (preds_1 + preds_2[:, ::-1]) / 2

    return position_mean, position_std, anomalous_mean, anomalous_std, mean_preds
