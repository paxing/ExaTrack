# -*- coding: utf-8 -*-
"""
constraints.py
--------------
Physical model definition: mapping from parameters θ to A-matrix coefficients.

This module contains:
  - constraint_function(): maps (LocErr, d, l, q, model_type) → Gaussian
    coefficients for every time step.
  - transition_param_function(): maps transition parameters → Gamma
    distribution rates and shapes for the dwell-time model.

PyTorch conversion notes
------------------------
- @tf.function removed (always eager in PyTorch)
- tf.cast(x, dtype) → x.to(dtype)
- tf.transpose(x, [a,b,c]) → x.permute(a,b,c)
- tf.math.sigmoid → torch.sigmoid
- tf.math.exp / tf.exp → torch.exp
- tf.math.log → torch.log
- tf.math.expm1 → torch.expm1
- tf.sqrt → torch.sqrt
- tf.zeros_like → torch.zeros_like
- tf.fill(shape, val) → torch.full(shape, val, dtype=dtype)
- tf.broadcast_to(x, shape) → x.expand(shape)
- tf.ones(shape, dtype) → torch.ones(shape, dtype=dtype)
- tf.zeros(shape, dtype) → torch.zeros(shape, dtype=dtype)
- tf.concat([a,b], axis=k) → torch.cat([a,b], dim=k)
- tf.stack([a,b], axis=k) → torch.stack([a,b], dim=k)
- tf.shape(x)[i] → x.shape[i]  (always int in eager PyTorch)
- x.shape.rank → x.dim()
- tf.math.softmax(x, axis=k) → torch.softmax(x, dim=k)
- tf.constant([[1]*n], dtype=dtype) → torch.ones(1, n, dtype=dtype)
- tf.reduce_mean(x, axis=-1, keepdims=True) → x.mean(dim=-1, keepdim=True)
"""

import numpy as np
import torch
from .config import dtype


def constraint_function(all_params, all_initial_params, LocErrs, dts,
                        nb_dims, reference_dt, LocErr_function, dtype):
    """
    Vectorised, time-varying constraint function that makes the link between
    the model variables and the characteristic parameters of the Gaussians.

    Parameters
    ----------
    all_params         : (nb_states, 5) tensor
    all_initial_params : (nb_states, >=1) tensor
    LocErrs            : (nb_tracks, track_len) or (nb_tracks, track_len, 1)
    dts                : (nb_tracks, track_len+1) or (nb_tracks, track_len+1, 1)
    nb_dims            : int
    reference_dt       : scalar
    LocErr_function    : callable
    dtype              : torch dtype
    """

    # ------------------------------------------------------------------
    # Bookkeeping constants
    # ------------------------------------------------------------------
    nb_states                  = all_params.shape[0]
    integration_variable_index = torch.tensor(1, dtype=torch.int32)
    nb_hidden_vars             = 2
    nb_obs_vars                = 1
    nb_transition_gaussians    = 1

    # ------------------------------------------------------------------
    # Normalise LocErrs and dts to shape (track_len, nb_tracks, 1)
    # ------------------------------------------------------------------
    LocErrs = LocErrs.to(dtype)
    if LocErrs.dim() == 2:
        LocErrs = LocErrs.unsqueeze(-1)                             # (nb_tracks, track_len, 1)
    LocErrs = LocErrs.mean(dim=-1, keepdim=True)                    # (nb_tracks, track_len, 1)
    LocErrs = LocErrs.permute(1, 0, 2)                              # (track_len, nb_tracks, 1)

    dts = dts.to(dtype)
    if dts.dim() == 2:
        dts = dts.unsqueeze(-1)                                     # (nb_tracks, track_len+1, 1)
    dts = dts.mean(dim=-1, keepdim=True)                            # (nb_tracks, track_len+1, 1)
    dts = dts.permute(1, 0, 2)                                      # (track_len+1, nb_tracks, 1)

    reference_dt = torch.tensor(reference_dt, dtype=dtype) if not isinstance(reference_dt, torch.Tensor) else reference_dt.to(dtype)

    track_len = LocErrs.shape[0]
    nb_tracks = LocErrs.shape[1]

    # ------------------------------------------------------------------
    # Per-state parameters, broadcast-ready on (1, 1, nb_states)
    # ------------------------------------------------------------------
    LocErr_param    = torch.exp(all_params[:, 0])[None, None, :]
    LocErrs = LocErr_function(LocErrs, LocErr_param)

    log_d           = all_params[:, 1][None, None, :]
    ano             = all_params[:, 2][None, None, :]
    log_q           = all_params[:, 3][None, None, :]
    is_dir          = all_params[:, 4][None, None, :]
    log_init_spread = all_initial_params[:, 0][None, :]

    isdir_mask  = (is_dir >= 0.5).to(dtype)
    isconf_mask = 1.0 - isdir_mask

    # ------------------------------------------------------------------
    # Step-size scaling
    # ------------------------------------------------------------------
    dt_ratio      = dts / reference_dt                             # (track_len+1, nb_tracks, 1)
    dt_sqrt_ratio = torch.sqrt(dt_ratio)

    d_ref = torch.exp(log_d)
    q_ref = torch.exp(log_q)
    l_ref = torch.sigmoid(ano)
    v_ref = torch.exp(ano)

    d       = d_ref * dt_sqrt_ratio[:track_len] + 1e-20
    q       = q_ref * dt_sqrt_ratio[:track_len] + 1e-20
    l_ref_c = -torch.log(1.0 - l_ref)
    l_c     = l_ref_c * dt_ratio[:track_len]
    l       = -torch.expm1(-l_c) + 1e-20
    one_minus_l = torch.exp(-l_c) + 1e-20
    v       = v_ref * dt_ratio[:track_len] + 1e-20

    dt_ratio_next         = dt_ratio[1:]
    ano_step_ratio        = dt_ratio_next / dt_ratio[:-1]
    ano_rescale_per_state = ano_step_ratio * isdir_mask + (1.0 - isdir_mask)

    well_distance = d / torch.sqrt(2 * (1 - torch.exp(-2 * l_c)))

    initial_position_spread = torch.exp(log_init_spread).expand(d[0].shape)

    LocErr_b = LocErrs.expand(track_len, nb_tracks, nb_states) + 1e-20

    zeros = torch.zeros_like(LocErr_b)
    tiny  = torch.full((track_len, nb_tracks, nb_states), 1e-15, dtype=dtype)

    # ==================================================================
    # Recurrent hidden-variable coefficients
    # Final shape: (track_len, 3, nb_tracks, nb_states, 4)
    # ==================================================================

    # Gaussian 0 -- localisation error
    g0 = torch.stack([1.0 / LocErr_b, zeros, zeros, zeros], dim=-1)

    # Gaussian 1 -- diffusion + anomalous drift
    g1_std = d * isdir_mask + d / (2 * l_c) ** 0.5 * (1 - torch.exp(-2 * l_c)) ** 0.5 * isconf_mask

    inv_d  = 1.0 / g1_std
    g1_c0  = (one_minus_l * isconf_mask + isdir_mask) * inv_d
    g1_c1  = (l * isconf_mask + isdir_mask) * inv_d + 1.1e-20
    g1     = torch.stack([g1_c0, g1_c1, -inv_d, zeros], dim=-1)

    # Gaussian 2 -- anomalous-variable evolution
    inv_q  = 1.0 / q
    g2_c1  = ano_rescale_per_state * inv_q
    g2     = torch.stack([zeros, g2_c1, zeros, -inv_q], dim=-1)

    hidden_vars = torch.stack([g0, g1, g2], dim=1)   # (track_len, 3, nb_tracks, nb_states, 4)

    # ==================================================================
    # Recurrent observation coefficients
    # Final shape: (track_len, 3, nb_tracks, nb_states, 1)
    # ==================================================================
    obs_g0   = (-1.0 / LocErr_b).unsqueeze(-1)
    obs_zero = zeros.unsqueeze(-1)
    obs_vars = torch.stack([obs_g0, obs_zero, obs_zero], dim=1)

    # ==================================================================
    # Initial hidden-variable coefficients
    # Final shape: (2, nb_tracks, nb_states, 2)
    # ==================================================================
    init_g0 = torch.stack([1.0 / initial_position_spread, zeros[0]], dim=-1)

    init_g1_c0 = (1.0  / well_distance) * isconf_mask + tiny * isdir_mask
    init_g1_c1 = (-1.0 / well_distance) * isconf_mask + (1.0 / v) * isdir_mask
    init_g1    = torch.stack([init_g1_c0, init_g1_c1], dim=-1)

    initial_hidden_vars = torch.stack([init_g0, init_g1[0]], dim=0)

    # ==================================================================
    # Transition hidden-variable coefficients
    # Final shape: (track_len, 1, nb_tracks, nb_states, 2)
    # ==================================================================
    transition_hidden_vars = init_g1[:, None]

    # ==================================================================
    # Unit-std / zero-bias scaffolding tensors
    # ==================================================================
    Gaussian_stds = torch.ones((track_len, nb_obs_vars + nb_hidden_vars,
                                nb_tracks, nb_states, 1), dtype=dtype)
    biases = torch.zeros((track_len, nb_obs_vars + nb_hidden_vars,
                          nb_tracks, nb_states, nb_dims), dtype=dtype)
    initial_obs_vars      = torch.zeros((nb_hidden_vars,
                                         nb_tracks, nb_states, nb_obs_vars), dtype=dtype)
    initial_Gaussian_stds = torch.ones((nb_hidden_vars,
                                        nb_tracks, nb_states, 1), dtype=dtype)
    initial_biases        = torch.zeros((nb_transition_gaussians,
                                         nb_tracks, nb_states, nb_dims), dtype=dtype)
    transition_Gaussian_stds = torch.ones((track_len, nb_transition_gaussians,
                                           nb_tracks, nb_states, 1), dtype=dtype)
    transition_biases = torch.zeros((track_len, nb_transition_gaussians,
                                     nb_tracks, nb_states, nb_dims), dtype=dtype)

    # Log normalising factors
    Log_factors = (- torch.log(LocErrs + 1e-20)
                   - torch.log(g1_std)
                   - torch.log(q))

    initial_anomalous_factor = (
        (- torch.log(d)
         + 0.5 * torch.log(2 * (1 - torch.exp(-2 * l_c)) + 1e-20))
        * isconf_mask
        - torch.log(v) * isdir_mask)

    initial_Log_factors     = Log_factors[0] - log_init_spread + initial_anomalous_factor[0]
    transition_Log_factors  = Log_factors + initial_anomalous_factor

    return (hidden_vars, obs_vars, Gaussian_stds, biases,
            initial_hidden_vars, initial_obs_vars,
            initial_Gaussian_stds, initial_biases,
            transition_hidden_vars, transition_Gaussian_stds,
            transition_biases, integration_variable_index,
            Log_factors, initial_Log_factors, transition_Log_factors)


def transition_param_function(transition_shapes, transition_rates,
                               density, Fs, effective_ds,
                               dts, reference_dt, dtype):
    """
    The transition_param_function must define the initial transition parameters
    and their constraints, similarly to how constraint_function defines the
    constraints of the states.
    """
    nb_states = transition_shapes.shape[0]
    nb_time_points, nb_tracks = dts.shape

    transition_shapes = torch.exp(transition_shapes)
    transition_rates  = (torch.softmax(transition_rates, dim=1)
                         * transition_shapes / reference_dt)
    transition_rates  = transition_rates[None, None] * dts[..., None, None] + 1e-20

    new_transition_shapes = torch.cat(
        (transition_shapes,
         torch.ones(1, nb_states, dtype=dtype)), dim=0)
    new_transition_shapes = torch.cat(
        (new_transition_shapes,
         torch.ones(nb_states + 1, 1, dtype=dtype)), dim=1)

    mislinking_dwell_time = torch.tensor(
        [0.9 / nb_states] * nb_states, dtype=dtype)
    mislinking_dwell_time = torch.cat((mislinking_dwell_time,
                                        torch.tensor([0.1], dtype=dtype)), dim=0)
    mislinking_dwell_time = mislinking_dwell_time[None, None, None].expand(
        nb_time_points, nb_tracks, 1, nb_states + 1)

    mislinking_rates = (1 - torch.exp(
        -0.5 * density
        * torch.sum(
            Fs[None] * (effective_ds[:, None] ** 2
                        + effective_ds[None] ** 2) ** 0.5,
            dim=0)[:, None]))
    mislinking_rates = mislinking_rates[None, None].expand(
        nb_time_points, nb_tracks, nb_states, 1)

    new_transition_rates = torch.cat(
        (transition_rates, mislinking_rates), dim=3)
    new_transition_rates = torch.cat(
        (new_transition_rates, mislinking_dwell_time), dim=2)

    return new_transition_shapes, new_transition_rates
