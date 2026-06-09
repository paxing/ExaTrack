# -*- coding: utf-8 -*-
"""
emg_constraints.py
------------------
Physical model definition for 2-channel EMG gesture decoding.

Adapts the ExaTrack CGP framework to classify hand gestures from 2 EMG
channels into N states (e.g. rest, flexion, extension).

Model structure
---------------
Each state is described by 6 Gaussians per time step acting on:

  Observed :  s1_i, s2_i               (2 channels — the raw EMG signals)
  Hidden   :  r1_i, u1_i, r2_i, u2_i  (4 hidden — real signal + activation per channel)
  Next     :  r1_{i+1}, u1_{i+1}, r2_{i+1}, u2_{i+1}

  Column order: [r1, u1, r2, u2,  r1+, u1+, r2+, u2+]

  G1: localisation noise ch1  →  s1 linked to r1
  G2: signal dynamics ch1     →  r1 evolves toward u1
  G3: activation drift ch1    →  u1 drifts slowly
  G4: localisation noise ch2  →  s2 linked to r2
  G5: signal dynamics ch2     →  r2 evolves toward u2
  G6: activation drift ch2    →  u2 drifts slowly

Physical parameters (per gesture state, columns of all_params):
  0: log_sigma1  — log measurement noise channel 1
  1: log_d1      — log signal fluctuation amplitude channel 1
  2: logit_l1    — logit confinement factor channel 1
  3: log_q1      — log activation drift std channel 1
  4: log_sigma2  — log measurement noise channel 2
  5: log_d2      — log signal fluctuation amplitude channel 2
  6: logit_l2    — logit confinement factor channel 2
  7: log_q2      — log activation drift std channel 2

Initial parameters (per gesture state, columns of all_initial_params):
  0: log_init_spread1 — log initial spread of r1
  1: log_init_spread2 — log initial spread of r2
"""

import numpy as np
import torch
import torch.nn as nn

from exatrack_torch.config import dtype, minval
from exatrack_torch.integration import get_sequences, RNN_reccurence_formula
from exatrack_torch.layers import (
    Initial_layer_constraints,
    Custom_RNN_layer,
    Final_layer,
    IsfirstMaskLayer,
)
from exatrack_torch.models import MLE_loss
from exatrack_torch.constraints import transition_param_function


# ---------------------------------------------------------------------------
# EMG constraint function
# ---------------------------------------------------------------------------

def emg_constraint_function(all_params, all_initial_params, LocErrs, dts,
                             nb_dims, reference_dt, LocErr_function, dtype):
    """
    Constraint function for 2-channel EMG gesture decoding.

    Parameters
    ----------
    all_params         : (nb_states, 8) tensor
    all_initial_params : (nb_states, 2) tensor
    LocErrs            : (nb_tracks, segment_len) — pass ones (not used)
    dts                : (nb_tracks, segment_len+1)
    nb_dims            : int, must be 1
    reference_dt       : float
    LocErr_function    : callable — kept for interface parity
    dtype              : torch dtype

    Returns
    -------
    Same 15-tuple as tracking constraint_function.
    """
    device = all_params.device

    nb_states                  = all_params.shape[0]
    integration_variable_index = torch.tensor(2, dtype=torch.int32, device=device)
    nb_hidden_vars             = 4
    nb_obs_vars                = 2
    nb_gaussians               = 6
    nb_transition_gaussians    = 2

    # ---- normalise dts to (track_len+1, nb_tracks, 1) ----------------------
    dts = dts.to(dtype)
    if dts.dim() == 2:
        dts = dts.unsqueeze(-1)
    dts = dts.mean(dim=-1, keepdim=True)
    dts = dts.permute(1, 0, 2)                     # (track_len+1, nb_tracks, 1)

    reference_dt = (torch.tensor(reference_dt, dtype=dtype, device=device)
                    if not isinstance(reference_dt, torch.Tensor)
                    else reference_dt.to(dtype=dtype, device=device))

    track_len = dts.shape[0] - 1
    nb_tracks  = dts.shape[1]

    # ---- per-state physical parameters -------------------------------------
    log_sigma1 = all_params[:, 0][None, None, :]
    log_d1     = all_params[:, 1][None, None, :]
    logit_l1   = all_params[:, 2][None, None, :]
    log_q1     = all_params[:, 3][None, None, :]
    log_sigma2 = all_params[:, 4][None, None, :]
    log_d2     = all_params[:, 5][None, None, :]
    logit_l2   = all_params[:, 6][None, None, :]
    log_q2     = all_params[:, 7][None, None, :]

    log_init_spread1 = all_initial_params[:, 0][None, :]
    log_init_spread2 = all_initial_params[:, 1][None, :]

    # ---- time-step scaling -------------------------------------------------
    dt_ratio      = dts / reference_dt
    dt_sqrt_ratio = torch.sqrt(dt_ratio)

    def _channel(log_d, logit_l, log_q):
        l_ref = torch.sigmoid(logit_l)
        d     = torch.exp(log_d) * dt_sqrt_ratio[:track_len] + 1e-20
        q     = torch.exp(log_q) * dt_sqrt_ratio[:track_len] + 1e-20
        l_ref_c     = -torch.log(1.0 - l_ref)
        l_c         = l_ref_c * dt_ratio[:track_len]
        l           = -torch.expm1(-l_c) + 1e-20
        one_minus_l = torch.exp(-l_c) + 1e-20
        well_dist   = d / torch.sqrt(2 * (1 - torch.exp(-2 * l_c)) + 1e-20)
        dt_ratio_next      = dt_ratio[1:]
        ano_rescale        = dt_ratio_next / dt_ratio[:track_len]
        g_std  = d / (2 * l_c + 1e-20) ** 0.5 * (1 - torch.exp(-2 * l_c) + 1e-20) ** 0.5
        inv_d  = 1.0 / (g_std + 1e-20)
        g_c0   = one_minus_l * inv_d
        g_c1   = l           * inv_d + 1.1e-20
        inv_q  = 1.0 / q
        return d, q, l_c, g_std, inv_d, g_c0, g_c1, inv_q, ano_rescale, well_dist

    d1, q1, l_c1, g1_std, inv_d1, g1_c0, g1_c1, inv_q1, ano1, well1 = _channel(log_d1, logit_l1, log_q1)
    d2, q2, l_c2, g2_std, inv_d2, g2_c0, g2_c1, inv_q2, ano2, well2 = _channel(log_d2, logit_l2, log_q2)

    sigma1_b = torch.exp(log_sigma1).expand(track_len, nb_tracks, nb_states) + 1e-20
    sigma2_b = torch.exp(log_sigma2).expand(track_len, nb_tracks, nb_states) + 1e-20
    zeros    = torch.zeros(track_len, nb_tracks, nb_states, dtype=dtype, device=device)

    # ====================================================================
    # Recurrent hidden-variable coefficients
    # Column order: [r1, u1, r2, u2,  r1+, u1+, r2+, u2+]  (8 columns)
    # Shape: (track_len, 6, nb_tracks, nb_states, 8)
    # ====================================================================
    g1_row = torch.stack([-1/sigma1_b, zeros,  zeros, zeros,  zeros,  zeros, zeros, zeros], dim=-1)
    g2_row = torch.stack([g1_c0, g1_c1, zeros, zeros, -inv_d1, zeros, zeros, zeros], dim=-1)
    g3_row = torch.stack([zeros, ano1*inv_q1, zeros, zeros, zeros, -inv_q1, zeros, zeros], dim=-1)
    g4_row = torch.stack([zeros, zeros, -1/sigma2_b, zeros, zeros, zeros, zeros, zeros], dim=-1)
    g5_row = torch.stack([zeros, zeros, g2_c0, g2_c1, zeros, zeros, -inv_d2, zeros], dim=-1)
    g6_row = torch.stack([zeros, zeros, zeros, ano2*inv_q2, zeros, zeros, zeros, -inv_q2], dim=-1)

    hidden_vars = torch.stack([g1_row, g2_row, g3_row, g4_row, g5_row, g6_row], dim=1)

    # ====================================================================
    # Observation coefficients
    # Shape: (track_len, 6, nb_tracks, nb_states, 2)
    # ====================================================================
    obs_zero = zeros.unsqueeze(-1)
    obs_s1   = (1 / sigma1_b).unsqueeze(-1)
    obs_s2   = (1 / sigma2_b).unsqueeze(-1)

    obs_vars = torch.stack([
        torch.cat([obs_s1,   obs_zero], dim=-1),   # G1: s1
        torch.cat([obs_zero, obs_zero], dim=-1),   # G2: none
        torch.cat([obs_zero, obs_zero], dim=-1),   # G3: none
        torch.cat([obs_zero, obs_s2],   dim=-1),   # G4: s2
        torch.cat([obs_zero, obs_zero], dim=-1),   # G5: none
        torch.cat([obs_zero, obs_zero], dim=-1),   # G6: none
    ], dim=1)

    # ====================================================================
    # Initial hidden-variable coefficients
    # Shape: (4, nb_tracks, nb_states, 4)
    # ====================================================================
    z0 = zeros[0]
    init_spread1 = torch.exp(log_init_spread1).expand_as(d1[0])
    init_spread2 = torch.exp(log_init_spread2).expand_as(d2[0])

    init_r1 = torch.stack([1/init_spread1,  z0,           z0,           z0], dim=-1)
    init_u1 = torch.stack([z0, -1/well1[0],              z0,           z0], dim=-1)
    init_r2 = torch.stack([z0,              z0,  1/init_spread2,        z0], dim=-1)
    init_u2 = torch.stack([z0,              z0,            z0, -1/well2[0]], dim=-1)

    initial_hidden_vars = torch.stack([init_r1, init_u1, init_r2, init_u2], dim=0)

    # ====================================================================
    # Transition Gaussians (re-initialise u1 and u2 at state switches)
    # Shape: (track_len, 2, nb_tracks, nb_states, 4)
    # ====================================================================
    transition_hidden_vars = torch.stack([init_u1, init_u2], dim=0)[None].expand(
        track_len, 2, nb_tracks, nb_states, 4)

    # ====================================================================
    # Scaffolding tensors
    # ====================================================================
    Gaussian_stds = torch.ones(
        (track_len, nb_gaussians, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    biases = torch.zeros(
        (track_len, nb_gaussians, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)
    initial_obs_vars = torch.zeros(
        (nb_hidden_vars, nb_tracks, nb_states, nb_obs_vars), dtype=dtype, device=device)
    initial_Gaussian_stds = torch.ones(
        (nb_hidden_vars, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    initial_biases = torch.zeros(
        (nb_hidden_vars, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)
    transition_Gaussian_stds = torch.ones(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    transition_biases = torch.zeros(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)

    # ====================================================================
    # Log normalisation factors
    # ====================================================================
    Log_factors = (- torch.log(sigma1_b + 1e-20)
                   - torch.log(g1_std   + 1e-20)
                   - torch.log(q1       + 1e-20)
                   - torch.log(sigma2_b + 1e-20)
                   - torch.log(g2_std   + 1e-20)
                   - torch.log(q2       + 1e-20))

    initial_anomalous_factor = (
          (- torch.log(d1 + 1e-20) + 0.5 * torch.log(2*(1-torch.exp(-2*l_c1))+1e-20))
        + (- torch.log(d2 + 1e-20) + 0.5 * torch.log(2*(1-torch.exp(-2*l_c2))+1e-20)))

    initial_Log_factors    = (Log_factors[0]
                               - log_init_spread1 - log_init_spread2
                               + initial_anomalous_factor[0])
    transition_Log_factors = Log_factors + initial_anomalous_factor

    return (hidden_vars, obs_vars, Gaussian_stds, biases,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index,
            Log_factors, initial_Log_factors, transition_Log_factors)


# ---------------------------------------------------------------------------
# EMG_Initial_layer
# ---------------------------------------------------------------------------

class EMG_Initial_layer(Initial_layer_constraints):
    """
    Subclass of Initial_layer_constraints that:
      1. Wires in emg_constraint_function
      2. Overrides forward() to build an 8-column mislinking row instead of
         the 5-column one hardcoded in the parent.
    """

    def __init__(self, nb_states, params, initial_params, initial_fractions,
                 max_linking_distance, reference_dt,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 sequence_length=3, carryover=True,
                 LocErr_type='Constant'):

        super().__init__(
            nb_states,
            nb_gaussians=6, nb_obs_vars=2, nb_hidden_vars=4,
            params=params, initial_params=initial_params,
            initial_fractions=initial_fractions,
            max_linking_distance=max_linking_distance,
            constraint_function=emg_constraint_function,
            reference_dt=reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=carryover,
            LocErr_type=LocErr_type)

    def _init_carryover_buffers(self, nb_tracks, nb_hidden_vars_out, device,
                                nb_dims=1):
        """
        Override to use nb_dims for the bias last dimension.
        nb_dims=1 for envelope model, nb_freq_bins for spectral model.
        Called from EMG_Initial_layer.forward with the correct nb_dims.
        """
        nb_sequences = self.sequence_length * (self.nb_states + 1)
        self.register_buffer('carryout_coefs',
            torch.zeros(nb_hidden_vars_out, nb_tracks, nb_sequences,
                        nb_hidden_vars_out, dtype=dtype, device=device))
        self.register_buffer('carryout_biases',
            torch.zeros(nb_hidden_vars_out, nb_tracks, nb_sequences,
                        nb_dims, dtype=dtype, device=device))
        self.register_buffer('carryout_LP',
            torch.zeros(nb_tracks, nb_sequences, dtype=dtype, device=device))
        self.carryover_initialized = True

    def _mislinking_row(self, param_vars):
        """8-column mislinking parameter row for EMG."""
        _dev = param_vars.device
        log_d_mis = torch.log(self.max_linking_distance_param.to(dtype))
        log_q_mis = torch.log(torch.tensor(0.00001, dtype=dtype, device=_dev))
        neg15     = torch.tensor(-15., dtype=dtype, device=_dev)
        return torch.stack([
            param_vars[-1][0],  # log_σ1
            log_d_mis,          # log_d1
            neg15,              # logit_l1
            log_q_mis,          # log_q1
            param_vars[-1][4],  # log_σ2
            log_d_mis,          # log_d2
            neg15,              # logit_l2
            log_q_mis,          # log_q2
        ]).unsqueeze(0)

    def forward(self, inputs, input_LocErrs, input_dts):
        # --- everything identical to parent up to the mislinking append ---
        nb_tracks      = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        constraint_fn  = self.constraint_function
        reference_dt   = self.reference_dt

        param_vars         = self._apply_constraint(self.param_vars)
        initial_param_vars = self._apply_constraint(self.initial_param_vars)
        initial_fractions  = torch.softmax(self.initial_fractions, dim=-1)

        param_vars = (self.vary_params * param_vars
                      + (1 - self.vary_params) * param_vars.detach())
        initial_param_vars = (self.vary_initial_params * initial_param_vars
                              + (1 - self.vary_initial_params) * initial_param_vars.detach())
        initial_fractions = (self.vary_initial_fractions * initial_fractions
                             + (1 - self.vary_initial_fractions) * initial_fractions.detach())

        param_vars, initial_param_vars, initial_fractions = self.duplicate_states(
            param_vars, initial_param_vars, initial_fractions)

        # --- only difference: 8-column mislinking row ---
        param_vars         = torch.cat((param_vars, self._mislinking_row(param_vars)), dim=0)
        initial_param_vars = torch.cat((initial_param_vars, initial_param_vars[-1:]), dim=0)
        nb_states          = self.nb_states + 1

        # nb_dims = last dimension of input tensor.
        # For envelope model: inputs.shape[-1] = 1  (scalar per channel)
        # For spectral model: inputs.shape[-1] = nb_freq_bins  (e.g. 17)
        # The channels live in inputs.shape[-2] = nb_obs_vars = 2
        nb_dims          = inputs.shape[-1]
        LocErr_function  = self.LocErr_function
        sequence_length  = self.sequence_length

        # --- rest verbatim from parent forward (layers.py lines 472-565) ---
        (hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases,
         initial_hidden_var_coefs, initial_obs_var_coefs,
         initial_Gaussian_stds, initial_biases,
         transition_hidden_var_coefs, transition_Gaussian_stds,
         transition_biases, integration_variable_index,
         Log_factors, initial_Log_factors,
         transition_Log_factors) = constraint_fn(
            param_vars, initial_param_vars, input_LocErrs, input_dts,
            nb_dims, reference_dt, LocErr_function, dtype)

        hidden_var_coefs = hidden_var_coefs / Gaussian_stds
        obs_var_coefs    = obs_var_coefs    / Gaussian_stds
        biases           = biases           / Gaussian_stds

        current_hidden_var_coefs = hidden_var_coefs[..., :nb_hidden_vars]
        next_hidden_var_coefs    = hidden_var_coefs[..., nb_hidden_vars:]

        reccurent_obs_var_coefs         = obs_var_coefs
        reccurent_hidden_var_coefs      = current_hidden_var_coefs
        reccurent_next_hidden_var_coefs = next_hidden_var_coefs
        reccurent_biases                = biases

        initial_hidden_var_coefs = initial_hidden_var_coefs / initial_Gaussian_stds
        initial_obs_var_coefs    = initial_obs_var_coefs    / initial_Gaussian_stds
        initial_biases           = initial_biases           / initial_Gaussian_stds

        current_initial_hidden_var_coefs = initial_hidden_var_coefs[..., :nb_hidden_vars]
        next_initial_hidden_var_coefs    = torch.zeros(
            (nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars),
            dtype=dtype, device=inputs.device)

        transition_hidden_var_coefs = transition_hidden_var_coefs / transition_Gaussian_stds
        transition_biases           = transition_biases           / transition_Gaussian_stds

        transition_hidden_var_coefs = transition_hidden_var_coefs.repeat(
            1, 1, 1, sequence_length * nb_states, 1)
        transition_biases = transition_biases.repeat(
            1, 1, 1, nb_states * sequence_length, 1)

        biases_t0                   = reccurent_biases[0]
        obs_var_coefs_t0            = reccurent_obs_var_coefs[0]
        current_hidden_var_coefs_t0 = reccurent_hidden_var_coefs[0]
        next_hidden_var_coefs_t0    = reccurent_next_hidden_var_coefs[0]

        biases_t0      = biases_t0 + torch.sum(obs_var_coefs_t0[..., None] * inputs[0], dim=-2)
        initial_biases = initial_biases + torch.sum(initial_obs_var_coefs[..., None] * inputs[0], dim=-2)

        current_hidden_var_coefs_t0 = torch.cat(
            (current_initial_hidden_var_coefs, current_hidden_var_coefs_t0), dim=0)
        next_hidden_var_coefs_t0 = torch.cat(
            (next_initial_hidden_var_coefs, next_hidden_var_coefs_t0), dim=0)
        biases_t0 = torch.cat((initial_biases, biases_t0), dim=0)

        current_hidden_var_coefs_t0 = current_hidden_var_coefs_t0.repeat(1, 1, sequence_length, 1)
        next_hidden_var_coefs_t0    = next_hidden_var_coefs_t0.repeat(1, 1, sequence_length, 1)
        biases_t0                   = biases_t0.repeat(1, 1, sequence_length, 1)

        Next_coefs, Next_biases, LC = RNN_reccurence_formula(
            current_hidden_var_coefs_t0,
            next_hidden_var_coefs_t0,
            biases_t0,
            self.initial_sequence_phase_1,
            self.initial_sequence_phase_2,
            nb_dims,
            dtype=dtype)

        init_log_fractions = initial_fractions.log().repeat(1, sequence_length)
        init_log_factors   = (nb_dims * initial_Log_factors).repeat(1, sequence_length)

        LP = (LC + init_log_factors + init_log_fractions
              + float(np.log(1 / sequence_length)))

        Log_factors            = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors

        if self.carryover and not self.carryover_initialized:
            self._init_carryover_buffers(nb_tracks, Next_coefs.shape[0],
                                         device=inputs.device, nb_dims=nb_dims)

        return inputs, [
            Next_coefs, Next_biases, LP,
            Log_factors, transition_Log_factors,
            reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
            reccurent_next_hidden_var_coefs, reccurent_biases,
            transition_hidden_var_coefs, transition_biases,
        ]


# ---------------------------------------------------------------------------
# EMGSegmentModel
# ---------------------------------------------------------------------------

class EMGSegmentModel(nn.Module):
    """
    Full EMG gesture classification model.

    Input shapes:
      inputs        : (batch, segment_len, 2)   — 2 EMG channels (nb_dims=1 each)
      input_LocErrs : (batch, segment_len)       — pass ones
      input_dts     : (batch, segment_len+1)     — inter-sample intervals
      input_mask    : (batch, segment_len)       — validity mask
      input_isfirst : (batch,)                   — 1 for first segment of trial
    """

    def __init__(self, segment_len, nb_states, params, initial_params,
                 transition_rates, transition_shapes, initial_fractions,
                 batch_size, reference_dt,
                 sequence_length=3, max_linking_distance=1,
                 estimated_density=1e-5,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 vary_transition_shapes=None, vary_transition_rates=None):
        super().__init__()

        self.segment_len     = segment_len
        self.nb_dims         = 1
        self.sequence_length = sequence_length
        self.reference_dt    = reference_dt

        self.init_layer = EMG_Initial_layer(
            nb_states, params, initial_params, initial_fractions,
            max_linking_distance, reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=True,
            LocErr_type='Constant')

        self.isfirst_mask = IsfirstMaskLayer()

        self.rnn_layer = Custom_RNN_layer(
            batch_size, transition_shapes, transition_rates,
            estimated_density, nb_states,
            self.init_layer.recurrent_sequence_phase_1,
            self.init_layer.recurrent_sequence_phase_2,
            self.init_layer.transition_sequence,
            transition_param_function,
            sequence_length=sequence_length,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates,
            carryover=True)

        self.final_layer = Final_layer(
            self.init_layer.final_sequence_phase_1,
            nb_dims=self.nb_dims,
            sequence_length=sequence_length)

    def forward(self, inputs, input_LocErrs, input_dts, input_mask,
                input_isfirst, return_all=False):
        device = next(self.parameters()).device
        inputs        = inputs.to(device)
        input_LocErrs = input_LocErrs.to(device)
        input_dts     = input_dts.to(device)
        input_mask    = input_mask.to(device)
        input_isfirst = input_isfirst.to(device)

        # inputs: (batch, seg_len, 2)
        # We need transposed: (seg_len, 1, batch, 1, nb_obs_vars=2, nb_dims=1)
        # so that input_i.shape[-1]=1 (nb_dims) and input_i.shape[-2]=2 (channels).
        # Channels go into the nb_obs_vars axis (second-to-last), NOT the last axis.
        reshaped   = inputs[:, None, :, None, :, None]    # (batch, 1, seg_len, 1, 2, 1)
        transposed = reshaped.permute(2, 1, 0, 3, 4, 5)  # (seg_len, 1, batch, 1, 2, 1)

        transposed, initial_states = self.init_layer(
            transposed, input_LocErrs, input_dts)

        (Prev_coefs, Prev_biases, LP,
         Log_factors, transition_Log_factors,
         rec_obs_coefs, rec_hid_coefs,
         rec_next_hid_coefs, rec_biases,
         trans_hid_coefs, trans_biases) = initial_states

        softmax_inv_Fractions = self.init_layer.initial_fractions
        log_ds            = self.init_layer.param_vars[:, 1]
        anomalous_factors = self.init_layer.param_vars[:, 2]
        isdir             = torch.zeros_like(log_ds)   # all confined for EMG

        Prev_coefs  = self.isfirst_mask(Prev_coefs,  self.init_layer.carryout_coefs,
                                         input_isfirst[None, :, None, None])
        Prev_biases = self.isfirst_mask(Prev_biases, self.init_layer.carryout_biases,
                                         input_isfirst[None, :, None, None])
        LP          = self.isfirst_mask(LP,          self.init_layer.carryout_LP,
                                         input_isfirst[:, None])

        sliced_inputs = transposed[1:]
        sliced_mask   = input_mask[:, 1:]

        (Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var,
         All_motion_states, All_coefs, All_biases, All_LPs,
         motion_states) = self.rnn_layer(
            sliced_inputs, input_dts, self.reference_dt, sliced_mask,
            Prev_coefs, Prev_biases, LP,
            Log_factors, transition_Log_factors,
            rec_obs_coefs, rec_hid_coefs,
            rec_next_hid_coefs, rec_biases,
            trans_hid_coefs, trans_biases,
            log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
            isfirst=input_isfirst)

        states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]
        outputs, All_states = self.final_layer(states)

        if self.init_layer.carryover_initialized:
            self.init_layer.carryout_coefs.data.copy_(Prev_coefs.detach())
            self.init_layer.carryout_biases.data.copy_(Prev_biases.detach())
            self.init_layer.carryout_LP.data.copy_(LP.detach())
        if self.rnn_layer.carryover:
            self.rnn_layer.carryout_segment_len.data.copy_(segment_len.detach())
            self.rnn_layer.carryout_gamma_dist_mean.data.copy_(gamma_dist_mean.detach())
            self.rnn_layer.carryout_gamma_dist_var.data.copy_(gamma_dist_var.detach())

        if return_all:
            return outputs, All_states, All_coefs, All_biases, All_LPs
        return outputs


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_emg_model(segment_len, nb_states, params, initial_params,
                    transition_rates, transition_shapes, initial_fractions,
                    batch_size, reference_dt,
                    sequence_length=3, max_linking_distance=1,
                    estimated_density=1e-5,
                    vary_params=None, vary_initial_params=None,
                    vary_initial_fractions=None,
                    vary_transition_shapes=None, vary_transition_rates=None):
    """
    Build an EMG gesture classification model.

    Parameters
    ----------
    segment_len     : int — number of time samples per segment
    nb_states       : int — number of gesture classes (e.g. 3)
    params          : (nb_states, 8) array
                      [log_σ1, log_d1, logit_l1, log_q1,
                       log_σ2, log_d2, logit_l2, log_q2]
    initial_params  : (nb_states, 2) array
                      [log_init_spread1, log_init_spread2]
    transition_rates  : (nb_states, nb_states)
    transition_shapes : (nb_states, nb_states)
    initial_fractions : (1, nb_states+1)
    batch_size      : int
    reference_dt    : float — reference inter-sample interval (seconds)

    Returns
    -------
    model, model  — same object twice (mirrors build_segment_model convention)

    Example
    -------
    >>> nb_states, seg_len, batch = 3, 50, 32
    >>> reference_dt = 1/2000.
    >>> params = np.array([
    ...     [np.log(0.1), np.log(0.05), 0.0, np.log(0.02),
    ...      np.log(0.1), np.log(0.05), 0.0, np.log(0.02)],  # rest
    ...     [np.log(0.1), np.log(0.3),  1.0, np.log(0.1),
    ...      np.log(0.1), np.log(0.2),  0.5, np.log(0.08)],  # flexion
    ...     [np.log(0.1), np.log(0.2),  0.5, np.log(0.08),
    ...      np.log(0.1), np.log(0.3),  1.0, np.log(0.1)],   # extension
    ... ], dtype='float64')
    >>> initial_params    = np.array([[np.log(60), np.log(60)]] * nb_states)
    >>> initial_fractions = np.array([[0.0] * nb_states + [-5.0]])
    >>> transition_rates  = 3 * np.eye(nb_states)
    >>> transition_shapes = np.zeros((nb_states, nb_states))
    >>> model, pred_model = build_emg_model(
    ...     seg_len, nb_states, params, initial_params,
    ...     transition_rates, transition_shapes, initial_fractions,
    ...     batch, reference_dt)
    """
    if vary_params is None:
        vary_params = np.ones(params.shape, dtype='float64')
    if vary_initial_params is None:
        vary_initial_params = np.ones(initial_params.shape, dtype='float64')
    if vary_initial_fractions is None:
        vary_initial_fractions = np.ones(initial_fractions.shape, dtype='float64')
    if vary_transition_shapes is None:
        vary_transition_shapes = np.ones(transition_shapes.shape, dtype='float64')
    if vary_transition_rates is None:
        vary_transition_rates = np.ones(transition_rates.shape, dtype='float64')

    model = EMGSegmentModel(
        segment_len, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt,
        sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates)

    return model, model