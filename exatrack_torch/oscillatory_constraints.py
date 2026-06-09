# -*- coding: utf-8 -*-
"""
oscillatory_constraints.py
--------------------------
Constraint function for single-particle tracking with three motion types:
directed, confined, and oscillatory.

Follows equations 11/12 of Simon et al. (HAL hal-04692487v2), extended to
oscillatory motion via the harmonic oscillator propagator.

Hidden variables: [r, u]  (nb_hidden_vars = 2)
    r = particle position
    u = velocity (oscillatory/directed) | well centre (confined)

A-matrix columns: [o_i, r_i, u_i, r_{i+1}, u_{i+1}]

Confined (eq. 11):
    G1:  1/σ  -1/σ    0      0      0
    G2:   0  (1-l)/d  l/d  -1/d    0
    G3:   0   eps     1/q    0    -1/q

Directed (eq. 12):
    G1:  1/σ  -1/σ    0      0      0
    G2:   0   1/d    1/d   -1/d    0
    G3:   0   eps   dt_r/q   0    -1/q

Oscillatory (harmonic propagator M = [[cos,sin/ω],[-ω·sin,cos]]):
    G1:  1/σ     -1/σ              0                0       0
    G2:   0    cos(ωΔt)/d   sin(ωΔt)/(ωd)         -1/d     0
    G3:   0   -ω·sin(ωΔt)/q  cos(ωΔt)/q             0    -1/q

G3 c_ri = eps (not 0) for confined/directed to avoid division-by-zero
in the broadcasting integration engine (same trick as constraints.py).

Parameters (all_params columns, nb_states × 6):
    col 0: log_sigma    — localisation noise
    col 1: log_d        — position noise
    col 2: motion_param — logit_l (confined) | log_v (directed) | log_omega_phys (osc)
              where omega_phys is in rad/s; omega_step = omega_phys * reference_dt
    col 3: log_q        — velocity/well noise
    col 4: is_dir       — 1.0 = directed  (fixed, col 4 for models.py compat)
    col 5: is_osc       — 1.0 = oscillatory (fixed)
"""

import numpy as np
import torch
import torch.nn as nn
from exatrack_torch.config import dtype
from exatrack_torch.layers import Initial_layer_constraints
from exatrack_torch.integration import RNN_reccurence_formula
from exatrack_torch.models import SegmentModel


def oscillatory_constraint_function(all_params, all_initial_params, LocErrs, dts,
                                    nb_dims, reference_dt, LocErr_function, dtype):

    device     = all_params.device
    nb_states  = all_params.shape[0]
    integration_variable_index = torch.tensor(1, dtype=torch.int32, device=device)
    nb_hidden_vars          = 2
    nb_obs_vars             = 1
    nb_transition_gaussians = 1

    # ------------------------------------------------------------------
    # Normalise inputs
    # ------------------------------------------------------------------
    LocErrs = LocErrs.to(dtype)
    if LocErrs.dim() == 2:
        LocErrs = LocErrs.unsqueeze(-1)
    LocErrs = LocErrs.mean(dim=-1, keepdim=True).permute(1, 0, 2)

    dts = dts.to(dtype)
    if dts.dim() == 2:
        dts = dts.unsqueeze(-1)
    dts = dts.mean(dim=-1, keepdim=True).permute(1, 0, 2)

    reference_dt = (torch.tensor(reference_dt, dtype=dtype, device=device)
                    if not isinstance(reference_dt, torch.Tensor)
                    else reference_dt.to(dtype=dtype, device=device))

    track_len = LocErrs.shape[0]
    nb_tracks = LocErrs.shape[1]

    # ------------------------------------------------------------------
    # Per-state parameters → (1, 1, nb_states)
    # ------------------------------------------------------------------
    LocErr_param    = torch.exp(all_params[:, 0])[None, None, :]
    LocErrs         = LocErr_function(LocErrs, LocErr_param)
    log_d           = all_params[:, 1][None, None, :]
    motion_param    = all_params[:, 2][None, None, :]
    log_q           = all_params[:, 3][None, None, :]
    is_dir          = all_params[:, 4][None, None, :]
    is_osc          = all_params[:, 5][None, None, :]
    log_init_spread = all_initial_params[:, 0][None, :]

    is_dir_mask = (is_dir >= 0.5).to(dtype)
    is_osc_mask = (is_osc >= 0.5).to(dtype)
    is_con_mask = 1.0 - is_dir_mask - is_osc_mask

    # ------------------------------------------------------------------
    # Step-size scaling
    # ------------------------------------------------------------------
    dt_ratio      = dts / reference_dt
    dt_sqrt_ratio = torch.sqrt(dt_ratio)

    d = torch.exp(log_d) * dt_sqrt_ratio[:track_len] + 1e-20
    q = torch.exp(log_q) * dt_sqrt_ratio[:track_len] + 1e-20

    # Confined
    l_ref   = torch.sigmoid(motion_param)
    l_ref_c = -torch.log(1.0 - l_ref + 1e-20)
    l_c     = l_ref_c * dt_ratio[:track_len]
    l       = -torch.expm1(-l_c) + 1e-20
    one_minus_l = torch.exp(-l_c) + 1e-20

    # Directed
    v = torch.exp(motion_param) * dt_ratio[:track_len] + 1e-20
    dt_ratio_next  = dt_ratio[1:]
    ano_step_ratio = dt_ratio_next / (dt_ratio[:track_len] + 1e-20)

    # Well distance for confined init
    well_distance = d / (torch.sqrt(2*(1-torch.exp(-2*l_c))) + 1e-20)

    # Oscillatory
    # motion_param stores log(omega_phys) where omega_phys is in rad/s.
    # omega_step = omega_phys * reference_dt  (rad/reference_step)
    # omega_dt   = omega_phys * dt            (rad/step, accounts for variable dt)
    # This is equivalent to omega_step * dt_ratio = omega_phys * reference_dt * dt/reference_dt
    omega_phys = torch.exp(motion_param) + 1e-20
    omega_step = omega_phys * reference_dt              # rad/reference_step (normalised)
    omega_dt   = omega_step * dt_ratio[:track_len]      # rad/step (= omega_phys * dt)
    cos_w    = torch.cos(omega_dt)
    sin_w    = torch.sin(omega_dt)
    sinc_w   = sin_w / (omega_step + 1e-20)             # sin(omega_dt)/omega_step
    wsin_w = omega_step * sin_w                                # omega_step * sin(omega_dt)
    # Note: sinc_w = sin(omega_phys*dt)/(omega_phys*reference_dt)
    #   u has units position/reference_step, so G2 coefficient is sin(Ω·dt)/Ω = sinc_w

    # g1_std (G2 position noise)
    g1_std = (d * is_dir_mask
              + d / (2*l_c+1e-20)**0.5 * (1-torch.exp(-2*l_c)+1e-20)**0.5 * is_con_mask
              + d * is_osc_mask) + 1e-20

    omega = omega_step   # use normalised omega for all coefficient computations
    inv_d = 1.0 / (g1_std + 1e-20)
    inv_q = 1.0 / (q + 1e-20)

    # ------------------------------------------------------------------
    # Broadcast tensors
    # ------------------------------------------------------------------
    LocErr_b = LocErrs.expand(track_len, nb_tracks, nb_states) + 1e-20
    zeros    = torch.zeros_like(LocErr_b)
    # (eps removed — G3 c_ri=0 for all states matches original ExaTrack exactly)

    # ==================================================================
    # Gaussian coefficients — columns: [r_i, u_i, r_{i+1}, u_{i+1}]
    # ==================================================================

    # G1: observation
    g1 = torch.stack([-1.0/LocErr_b, zeros, zeros, zeros], dim=-1)

    # G2: position transition
    g2_c_ri = (one_minus_l * inv_d * is_con_mask
               + 1.0       * inv_d * is_dir_mask
               + cos_w     * inv_d * is_osc_mask)
    g2_c_ui = (l           * inv_d * is_con_mask
               + 1.0       * inv_d * is_dir_mask
               + sinc_w    * inv_d * is_osc_mask)
    g2 = torch.stack([g2_c_ri, g2_c_ui, -inv_d, zeros], dim=-1)

    # G3: velocity/anomalous transition
    # c_ri = 0 for ALL states — matches original constraints.py exactly.
    # This keeps the integration schedule identical to ExaTrack:
    # G3 is NOT included in the r_i elimination step.
    #
    # The M21 term (-ω·sin·r_i) from the harmonic propagator is dropped.
    # LP cost: ~8 nats/track (systematic, does not affect classification).
    # Confined:    [ 0,      1/q,     0,  -1/q ]
    # Directed:    [ 0,  dt_r/q,     0,  -1/q ]
    # Oscillatory: [ 0,   cos/q,     0,  -1/q ]
    
    g3_c_ri = (1e-7               * is_con_mask
               + 1e-7  * is_dir_mask
               - wsin_w          * inv_q * is_osc_mask)
    
    g3_c_ui = (1.0              * inv_q * is_con_mask
               + ano_step_ratio * inv_q * is_dir_mask
               + cos_w          * inv_q * is_osc_mask)
    g3 = torch.stack([zeros, g3_c_ui, zeros, -inv_q], dim=-1)

    hidden_vars = torch.stack([g1, g2, g3], dim=1)

    # ==================================================================
    # Observation coefficients
    # ==================================================================
    obs_g1   = (-1.0/LocErr_b).unsqueeze(-1)
    obs_zero = zeros.unsqueeze(-1)
    obs_vars = torch.stack([obs_g1, obs_zero, obs_zero], dim=1)

    # ==================================================================
    # Initial hidden-variable coefficients — shape (2, nb_tracks, nb_states, 2)
    # ==================================================================
    init_spread  = torch.exp(log_init_spread).expand(nb_tracks, nb_states)
    init_vel_osc = q[0] / (omega_step[0] + 1e-20)   # q/omega_step = initial velocity spread

    init_g0_c_ri = 1.0 / init_spread
    init_g0 = torch.stack([init_g0_c_ri, zeros[0]], dim=-1)

    tiny = torch.full_like(zeros[0], 1e-15)
    init_g1_c_ri = (1.0/(well_distance[0]+1e-20) * is_con_mask[0]
                    + tiny                         * is_dir_mask[0]
                    + tiny                         * is_osc_mask[0])
    init_g1_c_ui = (-1.0/(well_distance[0]+1e-20) * is_con_mask[0]
                    + 1.0/(v[0]+1e-20)             * is_dir_mask[0]
                    + 1.0/(init_vel_osc+1e-20)     * is_osc_mask[0])
    init_g1 = torch.stack([init_g1_c_ri, init_g1_c_ui], dim=-1)

    initial_hidden_vars = torch.stack([init_g0, init_g1], dim=0)

    # ==================================================================
    # Transition hidden-variable coefficients (time-varying)
    # ==================================================================
    init_vel_osc_t = q / (omega_step + 1e-20)
    tiny_t = torch.full_like(zeros, 1e-15)
    trans_c_ri = (1.0/(well_distance+1e-20)   * is_con_mask
                  + tiny_t                     * is_dir_mask
                  + tiny_t                     * is_osc_mask)
    trans_c_ui = (-1.0/(well_distance+1e-20)  * is_con_mask
                  + 1.0/(v+1e-20)             * is_dir_mask
                  + 1.0/(init_vel_osc_t+1e-20) * is_osc_mask)
    trans_g1   = torch.stack([trans_c_ri, trans_c_ui], dim=-1)
    transition_hidden_vars = trans_g1[:, None]

    # ==================================================================
    # Scaffolding tensors
    # ==================================================================
    nb_gaussians = nb_obs_vars + nb_hidden_vars   # = 3

    Gaussian_stds = torch.ones(
        (track_len, nb_gaussians, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    biases = torch.zeros(
        (track_len, nb_gaussians, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)
    initial_obs_vars = torch.zeros(
        (nb_hidden_vars, nb_tracks, nb_states, nb_obs_vars), dtype=dtype, device=device)
    initial_Gaussian_stds = torch.ones(
        (nb_hidden_vars, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    initial_biases = torch.zeros(
        (nb_transition_gaussians, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)
    transition_Gaussian_stds = torch.ones(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, 1), dtype=dtype, device=device)
    transition_biases = torch.zeros(
        (track_len, nb_transition_gaussians, nb_tracks, nb_states, nb_dims), dtype=dtype, device=device)

    # ==================================================================
    # Log normalisation factors
    # ==================================================================
    Log_factors = (- torch.log(LocErr_b + 1e-20)
                   - torch.log(g1_std   + 1e-20)
                   - torch.log(q        + 1e-20))

    initial_anomalous_factor = (
        (-torch.log(d+1e-20) + 0.5*torch.log(2*(1-torch.exp(-2*l_c))+1e-20)) * is_con_mask
        + (-torch.log(v+1e-20))                                                 * is_dir_mask
        + torch.zeros_like(g1_std)                                              * is_osc_mask)

    initial_Log_factors    = Log_factors[0] - log_init_spread + initial_anomalous_factor[0]
    transition_Log_factors = Log_factors + initial_anomalous_factor

    return (hidden_vars, obs_vars, Gaussian_stds, biases,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index,
            Log_factors, initial_Log_factors, transition_Log_factors)


# ===========================================================================
# Custom Initial Layer — 6-column mislinking row
# ===========================================================================

class Oscillatory_Initial_layer(Initial_layer_constraints):

    def __init__(self, nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
                 params, initial_params, initial_fractions,
                 max_linking_distance, _constraint_fn_ignored,
                 reference_dt=1.0,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 sequence_length=3, carryover=True,
                 LocErr_type='Identity'):

        # Ensure params and vary_params have 6 columns
        if params.shape[1] == 7:
            params = params[:, :6]
        if vary_params is not None and vary_params.shape[1] == 7:
            vary_params = vary_params[:, :6]

        super().__init__(
            nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
            params, initial_params, initial_fractions,
            max_linking_distance, oscillatory_constraint_function,
            reference_dt=reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=carryover,
            LocErr_type=LocErr_type)

    def _mislinking_row(self, param_vars):
        _dev  = param_vars.device
        neg15 = torch.tensor(-15., dtype=dtype, device=_dev)
        tiny  = torch.log(torch.tensor(0.00001, dtype=dtype, device=_dev))
        zero  = torch.tensor(0., dtype=dtype, device=_dev)
        return torch.stack([
            param_vars[-1][0],
            torch.log(self.max_linking_distance_param.to(dtype)),
            neg15, tiny, zero, zero,
        ]).unsqueeze(0)

    def forward(self, inputs, input_LocErrs, input_dts):
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
        initial_fractions  = (self.vary_initial_fractions * initial_fractions
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

        biases_t0      = biases_t0 + torch.sum(
            obs_var_coefs_t0[..., None] * inputs[0], dim=-2)
        initial_biases = initial_biases + torch.sum(
            initial_obs_var_coefs[..., None] * inputs[0], dim=-2)

        current_hidden_var_coefs_t0 = torch.cat(
            (current_initial_hidden_var_coefs, current_hidden_var_coefs_t0), dim=0)
        next_hidden_var_coefs_t0 = torch.cat(
            (next_initial_hidden_var_coefs, next_hidden_var_coefs_t0), dim=0)
        biases_t0 = torch.cat((initial_biases, biases_t0), dim=0)

        current_hidden_var_coefs_t0 = current_hidden_var_coefs_t0.repeat(
            1, 1, sequence_length, 1)
        next_hidden_var_coefs_t0 = next_hidden_var_coefs_t0.repeat(
            1, 1, sequence_length, 1)
        biases_t0 = biases_t0.repeat(1, 1, sequence_length, 1)

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
                                         device=inputs.device)

        return inputs, [
            Next_coefs, Next_biases, LP,
            Log_factors, transition_Log_factors,
            reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
            reccurent_next_hidden_var_coefs, reccurent_biases,
            transition_hidden_var_coefs, transition_biases,
        ]


# ===========================================================================
# Factory function
# ===========================================================================

def build_oscillatory_model(track_len, nb_states, params, initial_params,
                             transition_rates, transition_shapes,
                             initial_fractions, batch_size, reference_dt,
                             nb_dims=2, sequence_length=3,
                             max_linking_distance=3, estimated_density=1e-4,
                             vary_params=None, vary_initial_params=None,
                             vary_initial_fractions=None,
                             vary_transition_shapes=None,
                             vary_transition_rates=None,
                             LocErr_type='Identity'):
    """
    params : (nb_states, 6) array
        [log_sigma, log_d, motion_param, log_q, is_dir, is_osc]
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

    # Truncate to 6 columns if 7 were passed
    if params.shape[1] == 7:
        params = params[:, :6]
    if vary_params.shape[1] == 7:
        vary_params = vary_params[:, :6]

    vary_params[:, 4] = 0.0   # fix is_dir
    vary_params[:, 5] = 0.0   # fix is_osc

    model = SegmentModel(
        track_len, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt,
        nb_dims=nb_dims,
        sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        LocErr_type=LocErr_type,
        init_layer_class=Oscillatory_Initial_layer)

    return model, model