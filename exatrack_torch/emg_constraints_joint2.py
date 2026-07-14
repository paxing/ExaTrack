# -*- coding: utf-8 -*-
"""
emg_constraints_joint2.py
--------------------------
Two-channel EMG model with a SHARED (r, u) hidden pair — same 2-hidden-variable
structure as emg_constraints_ratio.py (proven to work with the library's
Custom_RNN_layer, which hardcodes indexing up to Prev_coefs[1,...] and
therefore REQUIRES nb_hidden_vars >= 2).

Unlike emg_constraints_ratio.py (which observes a single precomputed ratio),
this model observes CH1 and CH2 SEPARATELY, each through its own learned
per-state coupling coefficient to the shared "r" hidden variable:

    o1_i (CH1) = k1 * r_i + noise(sigma1)
    o2_i (CH2) = k2 * r_i + noise(sigma2)
    r_i evolves confined toward u_i  (same as the ratio model's r dynamics)
    u_i drifts slowly                (same as the ratio model's u dynamics)

Per-state parameters (columns of all_params):
    0: log_sigma1  — CH1 observation noise
    1: log_sigma2  — CH2 observation noise
    2: log_d       — diffusion of r per reference step
    3: logit_l     — confinement rate of r toward u per reference step
    4: log_q       — diffusion of u (slow drift) per reference step
    5: k1          — CH1 coupling to r for this state (learned, not log)
    6: k2          — CH2 coupling to r for this state (learned, not log)

Initial parameters: (nb_states, 1) — [log_init_spread] for r, same as ratio model.

Status: structurally closer to the proven ratio model than the previous
single-hidden-variable attempt (which crashed — Custom_RNN_layer requires
nb_hidden_vars >= 2). Still NOT execution-verified end-to-end. Run the
mandatory sanity check in the training script before a full run.
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
# Constraint function
# ---------------------------------------------------------------------------

def emg_joint2_constraint_function(all_params, all_initial_params, LocErrs, dts,
                                    nb_dims, reference_dt, LocErr_function, dtype):
    """
    all_params         : (nb_states, 7) [log_sigma1, log_sigma2, log_d, logit_l, log_q, k1, k2]
    all_initial_params : (nb_states, 1) [log_init_spread]
    """
    device    = all_params.device
    nb_states = all_params.shape[0]

    integration_variable_index = torch.tensor(1, dtype=torch.int32, device=device)
    nb_hidden_vars          = 2   # (r, u) — matches ratio model, satisfies library's >=2 requirement
    nb_obs_vars             = 2   # o1 (CH1), o2 (CH2)
    nb_transition_gaussians = 1   # re-init u at state switches

    dts = dts.to(dtype)
    if dts.dim() == 2:
        dts = dts.unsqueeze(-1)
    dts = dts.mean(dim=-1, keepdim=True).permute(1, 0, 2)

    reference_dt = (torch.tensor(reference_dt, dtype=dtype, device=device)
                    if not isinstance(reference_dt, torch.Tensor)
                    else reference_dt.to(dtype=dtype, device=device))

    track_len = dts.shape[0] - 1
    nb_tracks = dts.shape[1]

    log_sigma1 = all_params[:, 0][None, None, :]
    log_sigma2 = all_params[:, 1][None, None, :]
    log_d      = all_params[:, 2][None, None, :]
    logit_l    = all_params[:, 3][None, None, :]
    log_q      = all_params[:, 4][None, None, :]
    k1         = all_params[:, 5][None, None, :]
    k2         = all_params[:, 6][None, None, :]
    log_init_spread = all_initial_params[:, 0][None, :]

    dt_ratio      = dts / reference_dt
    dt_sqrt_ratio = torch.sqrt(dt_ratio)

    l_ref       = torch.sigmoid(logit_l)
    d           = torch.exp(log_d) * dt_sqrt_ratio[:track_len] + 1e-20
    q           = torch.exp(log_q) * dt_sqrt_ratio[:track_len] + 1e-20
    l_ref_c     = -torch.log(1.0 - l_ref + 1e-20)
    l_c         = l_ref_c * dt_ratio[:track_len]
    l           = -torch.expm1(-l_c) + 1e-20
    one_minus_l = torch.exp(-l_c) + 1e-20
    well_dist   = d / torch.sqrt(2 * (1 - torch.exp(-2 * l_c)) + 1e-20)

    dt_ratio_next = dt_ratio[1:]
    if dt_ratio_next.shape[0] < track_len:
        dt_ratio_next = torch.cat([dt_ratio_next, dt_ratio_next[-1:]], dim=0)
    ano_rescale = (dt_ratio_next[:track_len] / (dt_ratio[:track_len] + 1e-20))

    g_std  = d / (2 * l_c + 1e-20) ** 0.5 * (1 - torch.exp(-2 * l_c) + 1e-20) ** 0.5
    inv_d  = 1.0 / (g_std + 1e-20)
    g_c0   = one_minus_l * inv_d
    g_c1   = l           * inv_d + 1.1e-20
    inv_q  = 1.0 / (q + 1e-20)

    sigma1_b = torch.exp(log_sigma1).expand(track_len, nb_tracks, nb_states) + 1e-20
    sigma2_b = torch.exp(log_sigma2).expand(track_len, nb_tracks, nb_states) + 1e-20
    k1_b     = k1.expand(track_len, nb_tracks, nb_states)
    k2_b     = k2.expand(track_len, nb_tracks, nb_states)
    zeros    = torch.zeros(track_len, nb_tracks, nb_states, dtype=dtype, device=device)

    # ── recurrent hidden-variable coefficients ────────────────────────────
    # Column layout: [r, u,  r+, u+]   (identical structure to the ratio model)
    # G1: obs CH1     → -k1/sigma1 * r  = 0 (folded with obs coef 1/sigma1 below)
    # G2: r dynamics  → g_c0*r + g_c1*u - inv_d*r+ = 0
    # G3: u drift     → ano_rescale*inv_q*u - inv_q*u+ = 0
    # G4: obs CH2     → -k2/sigma2 * r = 0
    g1_row = torch.stack([-k1_b/sigma1_b, zeros,             zeros,  zeros], dim=-1)
    g2_row = torch.stack([g_c0,           g_c1,              -inv_d, zeros], dim=-1)
    g3_row = torch.stack([zeros,          ano_rescale*inv_q, zeros,  -inv_q], dim=-1)
    g4_row = torch.stack([-k2_b/sigma2_b, zeros,             zeros,  zeros], dim=-1)

    hidden_vars = torch.stack([g1_row, g2_row, g3_row, g4_row], dim=1)
    # shape: (track_len, 4, nb_tracks, nb_states, 4)

    # ── recurrent observation coefficients ────────────────────────────────
    # shape: (track_len, 4, nb_tracks, nb_states, 2) — obs_vars axis = [o1, o2]
    obs_zero = zeros.unsqueeze(-1)
    obs_o1   = (1.0 / sigma1_b).unsqueeze(-1)
    obs_o2   = (1.0 / sigma2_b).unsqueeze(-1)

    obs_g1 = torch.cat([obs_o1,   obs_zero], dim=-1)   # G1 depends on o1 only
    obs_g2 = torch.cat([obs_zero, obs_zero], dim=-1)   # G2 depends on neither
    obs_g3 = torch.cat([obs_zero, obs_zero], dim=-1)   # G3 depends on neither
    obs_g4 = torch.cat([obs_zero, obs_o2  ], dim=-1)   # G4 depends on o2 only
    obs_vars = torch.stack([obs_g1, obs_g2, obs_g3, obs_g4], dim=1)

    # ── initial hidden-variable coefficients ──────────────────────────────
    # shape: (2, nb_tracks, nb_states, 2) — identical structure to ratio model
    z0          = zeros[0]
    init_spread = torch.exp(log_init_spread).expand(nb_tracks, nb_states)
    init_r      = torch.stack([1.0 / init_spread,   z0              ], dim=-1)
    init_u      = torch.stack([z0,                  -1.0/well_dist[0]], dim=-1)
    initial_hidden_vars = torch.stack([init_r, init_u], dim=0)

    # ── transition hidden-variable coefficients ───────────────────────────
    # Re-initialise u whenever a state switch occurs.
    transition_hidden_vars = init_u[None, None].expand(track_len, 1, nb_tracks, nb_states, 2)

    # ── scaffolding tensors ────────────────────────────────────────────────
    Gaussian_stds = torch.ones(
        (track_len, nb_obs_vars + nb_hidden_vars, nb_tracks, nb_states, 1),
        dtype=dtype, device=device)
    biases = torch.zeros(
        (track_len, nb_obs_vars + nb_hidden_vars, nb_tracks, nb_states, nb_dims),
        dtype=dtype, device=device)
    initial_obs_vars = torch.zeros(
        (nb_hidden_vars, nb_tracks, nb_states, nb_obs_vars), dtype=dtype, device=device)
    initial_Gaussian_stds = torch.ones(
        (nb_hidden_vars, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    initial_biases = torch.zeros(
        (nb_hidden_vars, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)
    transition_Gaussian_stds = torch.ones(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, 1),
        dtype=dtype, device=device)
    transition_biases = torch.zeros(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, nb_dims),
        dtype=dtype, device=device)

    # ── log normalisation factors ─────────────────────────────────────────
    # G1 (obs CH1): -log(sigma1)   G2 (r dyn): -log(g_std)   G3 (u drift): -log(q)
    # G4 (obs CH2): -log(sigma2)
    # Combined per the ratio model's convention (single Log_factors tensor
    # summed across the "always-present" Gaussians G2/G3; G1/G4 obs-noise
    # terms folded via sigma1_b/sigma2_b directly in the obs coefficients,
    # matching how the tracking/ratio models handle their own obs Gaussian).
    Log_factors = (- torch.log(g_std + 1e-20)
                   - torch.log(q     + 1e-20))

    anom_factor = (- torch.log(d + 1e-20)
                   + 0.5 * torch.log(2 * (1 - torch.exp(-2 * l_c)) + 1e-20))

    initial_Log_factors    = Log_factors[0] - log_init_spread + anom_factor[0]
    transition_Log_factors = Log_factors + anom_factor

    return (hidden_vars, obs_vars, Gaussian_stds, biases,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index,
            Log_factors, initial_Log_factors, transition_Log_factors)


# ---------------------------------------------------------------------------
# Initial layer
# ---------------------------------------------------------------------------

class EMGJoint2_Initial_layer(Initial_layer_constraints):
    """nb_gaussians=4, nb_obs_vars=2, nb_hidden_vars=2."""

    def __init__(self, nb_states, params, initial_params, initial_fractions,
                 max_linking_distance, reference_dt,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 sequence_length=3, carryover=True,
                 LocErr_type='Constant'):
        super().__init__(
            nb_states,
            nb_gaussians=4, nb_obs_vars=2, nb_hidden_vars=2,
            params=params, initial_params=initial_params,
            initial_fractions=initial_fractions,
            max_linking_distance=max_linking_distance,
            constraint_function=emg_joint2_constraint_function,
            reference_dt=reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=carryover,
            LocErr_type=LocErr_type)

    def _init_carryover_buffers(self, nb_tracks, nb_hidden_vars_out, device,
                                nb_dims=1):
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
        """7-column mislinking parameter row — flat prior, k1=k2=0."""
        _dev      = param_vars.device
        log_d_mis = torch.log(self.max_linking_distance_param.to(dtype))
        log_q_mis = torch.log(torch.tensor(0.00001, dtype=dtype, device=_dev))
        neg15     = torch.tensor(-15., dtype=dtype, device=_dev)
        zero      = torch.tensor(0.,   dtype=dtype, device=_dev)
        return torch.stack([
            param_vars[-1][0],   # log_sigma1 — keep last state's noise
            param_vars[-1][1],   # log_sigma2
            log_d_mis,           # log_d      — wide well
            neg15,               # logit_l    — no confinement
            log_q_mis,           # log_q      — slow drift
            zero,                # k1 = 0
            zero,                # k2 = 0
        ]).unsqueeze(0)

    def forward(self, inputs, input_LocErrs, input_dts):
        """
        Full override — structurally identical to EMGRatio_Initial_layer.forward(),
        which is agnostic to nb_obs_vars/nb_hidden_vars values (only references
        them via self.nb_hidden_vars etc.), copied here since the base class's
        generic forward() assumes the 5-column tracking model.
        """
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

        param_vars         = torch.cat((param_vars, self._mislinking_row(param_vars)), dim=0)
        initial_param_vars = torch.cat((initial_param_vars, initial_param_vars[-1:]), dim=0)
        nb_states          = self.nb_states + 1

        nb_dims         = inputs.shape[-1]
        LocErr_function = self.LocErr_function
        sequence_length = self.sequence_length

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
# Model
# ---------------------------------------------------------------------------

class EMGJoint2SegmentModel(nn.Module):
    """
    Input shapes
    ------------
    inputs        : (batch, segment_len, 2, n_dims)  — [CH1, CH2] value(s) per step.
                    n_dims=1 for a scalar envelope, or n_freq_ac for per-bin
                    spectral features (log-amplitude or real/imag stacked).
    input_LocErrs : (batch, segment_len)
    input_dts     : (batch, segment_len+1)
    input_mask    : (batch, segment_len)
    input_isfirst : (batch,)
    """

    def __init__(self, segment_len, nb_states, params, initial_params,
                 transition_rates, transition_shapes, initial_fractions,
                 batch_size, reference_dt,
                 sequence_length=3, max_linking_distance=1,
                 estimated_density=1e-5, nb_dims=1,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 vary_transition_shapes=None, vary_transition_rates=None):
        super().__init__()
        self.segment_len     = segment_len
        self.nb_dims         = nb_dims   # 1 for scalar envelope, n_freq_ac for spectral
        self.sequence_length = sequence_length
        self.reference_dt    = reference_dt

        self.init_layer = EMGJoint2_Initial_layer(
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

        # (batch, seg_len, nb_obs_vars=2, n_freq_ac) →
        # (seg_len, 1, batch, 1, nb_obs_vars=2, n_freq_ac)
        # Keeping the last axis (":" instead of "None") lets nb_dims = n_freq_ac
        # flow through automatically via inputs.shape[-1] in the caller.
        reshaped   = inputs[:, None, :, None, :, :]
        transposed = reshaped.permute(2, 1, 0, 3, 4, 5)

        transposed, initial_states = self.init_layer(
            transposed, input_LocErrs, input_dts)

        (Prev_coefs, Prev_biases, LP,
         Log_factors, transition_Log_factors,
         rec_obs_coefs, rec_hid_coefs,
         rec_next_hid_coefs, rec_biases,
         trans_hid_coefs, trans_biases) = initial_states

        softmax_inv_Fractions = self.init_layer.initial_fractions
        log_ds            = self.init_layer.param_vars[:, 2]   # log_d column
        anomalous_factors = self.init_layer.param_vars[:, 3]   # reuse logit_l slot (unused by isdir path)
        isdir             = torch.zeros_like(log_ds)           # always confined, never directed

        Prev_coefs  = self.isfirst_mask(Prev_coefs,  self.init_layer.carryout_coefs,
                                         input_isfirst[None, :, None, None])
        Prev_biases = self.isfirst_mask(Prev_biases, self.init_layer.carryout_biases,
                                         input_isfirst[None, :, None, None])
        LP          = self.isfirst_mask(LP, self.init_layer.carryout_LP,
                                         input_isfirst[:, None])

        sliced_inputs = transposed[1:]
        sliced_mask   = input_mask[:, 1:]

        (Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var,
         All_motion_states, All_coefs, All_biases, All_LPs,
         motion_states) = self.rnn_layer(
            sliced_inputs, input_dts, self.reference_dt, sliced_mask,
            Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors,
            rec_obs_coefs, rec_hid_coefs, rec_next_hid_coefs, rec_biases,
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


def build_emg_joint2_model(segment_len, nb_states, params, initial_params,
                            transition_rates, transition_shapes, initial_fractions,
                            batch_size, reference_dt,
                            sequence_length=3, max_linking_distance=1,
                            estimated_density=1e-5, nb_dims=1,
                            vary_params=None, vary_initial_params=None,
                            vary_initial_fractions=None,
                            vary_transition_shapes=None, vary_transition_rates=None):
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

    model = EMGJoint2SegmentModel(
        segment_len, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt,
        sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        nb_dims=nb_dims,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates)

    return model, model
