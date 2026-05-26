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

This file is a verbatim copy of the original functions — no changes made.
Dependencies: config.py only.
"""

import numpy as np
import tensorflow as tf
from .config import dtype


@tf.function
def constraint_function(all_params, all_initial_params, LocErrs, dts,
                        nb_dims, reference_dt, LocErr_function, dtype):
    '''
    Vectorised, time-varying constraint function that makes the link between
    the model variables and the characteristic parameters of the Gaussians.

    Builds the per-step Gaussian coefficients, biases, std-rescaling factors
    and log-normalising factors describing the joint distribution
        p(observation_t, hidden_t, hidden_{t+1} | state_t)
    for all (time, track, state) triples in one pass. Designed to be used in the
    call of the layer `Initial_layer_constraints`.

    Parameters
    ----------
    all_params         : (nb_states, 5) — columns
                         [log_LocErr_unused, log_d, ano, log_q, is_directed_flag].
                         `log_d`  : log diffusion length per reference_dt.
                         `ano`    : in directed regime acts as log drift speed,
                                    in confined regime acts as a logistic well
                                    confinement (l = sigmoid(ano)).
                         `log_q`  : log std of the anomalous-variable noise.
                         `is_directed_flag` : 1 = directed motion, 0 = confined.
    all_initial_params : (nb_states, >=1), column 0 is log(initial spread).
    LocErr             : per-step localisation error. Accepted shapes
                         (nb_tracks, track_len),
                         (nb_tracks, track_len, 1) or
                         (nb_tracks, track_len, nb_dims). A trailing dim axis
                         is averaged out.
    dts                : per-step frame durations, shape (nb_tracks, track_len+1)
                         or (nb_tracks, track_len+1, 1). Must have one extra
                         time step relative to the track length to support
                         the directed-mode `dt_ratio_next` rescaling at
                         segment carryovers.
    nb_dims            : int, spatial dimensionality. Typically set to 2.
    reference_dt       : scalar, reference frame duration the parameters
                         are expressed in.
    dtype              : tensorflow dtype string (e.g. 'float64').
    '''

    # ------------------------------------------------------------------
    # Bookkeeping constants
    # ------------------------------------------------------------------
    nb_states                  = all_params.shape[0]
    integration_variable_index = tf.constant(1)
    nb_hidden_vars             = 2
    nb_obs_vars                = 1
    nb_transition_gaussians    = 1

    # ------------------------------------------------------------------
    # Normalise LocErrs and dts to shape (track_len, nb_tracks, 1)
    # ------------------------------------------------------------------
    LocErrs = tf.cast(LocErrs, dtype)
    if LocErrs.shape.rank == 2:
        LocErrs = LocErrs[..., None]                            # (nb_tracks, track_len, 1)
    LocErrs = tf.reduce_mean(LocErrs, axis=-1, keepdims=True)   # (nb_tracks, track_len, 1)
    LocErrs = tf.transpose(LocErrs, [1, 0, 2])                  # (track_len, nb_tracks, 1)

    dts = tf.cast(dts, dtype)
    if dts.shape.rank == 2:
        dts = dts[..., None]                                  # (nb_tracks, track_len+1, 1)
    dts = tf.reduce_mean(dts, axis=-1, keepdims=True)         # (nb_tracks, track_len+1, 1)
    dts = tf.transpose(dts, [1, 0, 2])                        # (track_len+1, nb_tracks, 1)

    reference_dt = tf.cast(reference_dt, dtype)

    # Dynamic shape helpers
    track_len = tf.shape(LocErrs)[0]
    nb_tracks = tf.shape(LocErrs)[1]

    # ------------------------------------------------------------------
    # Per-state parameters, broadcast-ready on (1, 1, nb_states)
    # ------------------------------------------------------------------
    LocErr_param    = tf.math.exp(all_params[:, 0][None, None, :])
    LocErrs = LocErr_function(LocErrs, LocErr_param)

    log_d           = all_params[:, 1][None, None, :]
    ano             = all_params[:, 2][None, None, :]
    log_q           = all_params[:, 3][None, None, :]
    is_dir          = all_params[:, 4][None, None, :]
    log_init_spread = all_initial_params[:, 0][None, :]

    # State-selection masks, shape (1, 1, nb_states)
    isdir_mask  = tf.cast(is_dir >= 0.5, dtype)
    isconf_mask = 1.0 - isdir_mask

    # ------------------------------------------------------------------
    # Step-size scaling from reference_dt to the actual dts.
    # All scaled tensors have shape (track_len, nb_tracks, nb_states).
    # ------------------------------------------------------------------
    dt_ratio      = dts / reference_dt                        # (track_len+1, nb_tracks, 1)
    dt_sqrt_ratio = tf.sqrt(dt_ratio)

    d_ref = tf.exp(log_d)            # (1, 1, nb_states)
    q_ref = tf.exp(log_q)
    l_ref = tf.math.sigmoid(ano)
    v_ref = tf.exp(ano)

    # Continuous-discrete conversion for l:
    #   ld = 1 - exp(-lc)   <=>   lc = -log(1 - ld)
    d       = d_ref * dt_sqrt_ratio[:track_len] + 1e-20
    q       = q_ref * dt_sqrt_ratio[:track_len] + 1e-20
    l_ref_c = -tf.math.log(1.0 - l_ref)
    l_c     = l_ref_c * dt_ratio[:track_len]
    l       = -tf.math.expm1(-l_c) + 1e-20
    one_minus_l = tf.math.exp(-l_c) + 1e-20
    v       = v_ref * dt_ratio[:track_len] + 1e-20

    # ------------------------------------------------------------------
    # Per-step rescaling of the ano_t coefficient in recurrent g2.
    # For directed states: E[ano_{t+1} | ano_t] = (dts[t+1]/dts[t]) * ano_t
    # For confined states: ano_t is the well anchor, dt-independent.
    # ------------------------------------------------------------------
    dt_ratio_next         = dt_ratio[1:]
    ano_step_ratio        = dt_ratio_next / dt_ratio[:-1]
    ano_rescale_per_state = ano_step_ratio * isdir_mask + (1.0 - isdir_mask)

    # Characteristic well distance for confined motion
    well_distance = d / tf.sqrt(2 * (1 - tf.math.exp(-2 * l_c)))

    # Initial position spread
    initial_position_spread = tf.broadcast_to(tf.exp(log_init_spread),
                                              tf.shape(d[0]))

    # LocErrs broadcast across states
    LocErr_b = tf.broadcast_to(LocErrs, (track_len, nb_tracks, nb_states)) + 1e-20

    zeros = tf.zeros_like(LocErr_b)
    tiny  = tf.fill((track_len, nb_tracks, nb_states), tf.constant(1e-15, dtype=dtype))

    # ==================================================================
    # Recurrent hidden-variable coefficients
    # Final shape: (track_len, 3, nb_tracks, nb_states, 4)
    # Last axis ordering: [pos_t, ano_t, pos_{t+1}, ano_{t+1}]
    # ==================================================================

    # Gaussian 0 -- localisation error: [1/LocErrs, 0, 0, 0]
    g0 = tf.stack([1.0 / LocErr_b, zeros, zeros, zeros], axis=-1)

    # Gaussian 1 -- diffusion + anomalous drift
    #   confined: [(1-l)/d, l/d, -1/d, 0]
    #   directed: [   1/d, 1/d, -1/d, 0]
    g1_std = d * isdir_mask + d / (2 * l_c) ** 0.5 * (1 - tf.math.exp(-2 * l_c)) ** 0.5 * isconf_mask

    inv_d  = 1.0 / g1_std
    g1_c0  = (one_minus_l * isconf_mask + isdir_mask) * inv_d
    g1_c1  = (l * isconf_mask + isdir_mask) * inv_d + 1.1e-20
    g1     = tf.stack([g1_c0, g1_c1, -inv_d, zeros], axis=-1)

    # Gaussian 2 -- anomalous-variable evolution: [0, g2_c1, 0, -1/q]
    inv_q  = 1.0 / q
    g2_c1  = ano_rescale_per_state * inv_q
    g2     = tf.stack([zeros, g2_c1, zeros, -inv_q], axis=-1)

    hidden_vars = tf.stack([g0, g1, g2], axis=1)   # (track_len, 3, nb_tracks, nb_states, 4)

    # ==================================================================
    # Recurrent observation coefficients
    # Final shape: (track_len, 3, nb_tracks, nb_states, 1)
    # Only Gaussian 0 depends on the observation: [-1/LocErrs, 0, 0]
    # ==================================================================
    obs_g0   = (-1.0 / LocErr_b)[..., None]
    obs_zero = zeros[..., None]
    obs_vars = tf.stack([obs_g0, obs_zero, obs_zero], axis=1)

    # ==================================================================
    # Initial hidden-variable coefficients
    # Final shape: (2, nb_tracks, nb_states, 2)
    # ==================================================================
    init_g0 = tf.stack([1.0 / initial_position_spread, zeros[0]], axis=-1)

    init_g1_c0 = (1.0  / well_distance) * isconf_mask + tiny * isdir_mask
    init_g1_c1 = (-1.0 / well_distance) * isconf_mask + (1.0 / v) * isdir_mask
    init_g1    = tf.stack([init_g1_c0, init_g1_c1], axis=-1)

    initial_hidden_vars = tf.stack([init_g0, init_g1[0]], axis=0)

    # ==================================================================
    # Transition hidden-variable coefficients
    # Final shape: (track_len, 1, nb_tracks, nb_states, 2)
    # ==================================================================
    transition_hidden_vars = init_g1[:, None]

    # ==================================================================
    # Unit-std / zero-bias scaffolding tensors
    # ==================================================================
    Gaussian_stds = tf.ones((track_len, nb_obs_vars + nb_hidden_vars,
                             nb_tracks, nb_states, 1), dtype=dtype)
    biases = tf.zeros((track_len, nb_obs_vars + nb_hidden_vars,
                       nb_tracks, nb_states, nb_dims), dtype=dtype)
    initial_obs_vars      = tf.zeros((nb_hidden_vars,
                                      nb_tracks, nb_states, nb_obs_vars), dtype=dtype)
    initial_Gaussian_stds = tf.ones((nb_hidden_vars,
                                     nb_tracks, nb_states, 1), dtype=dtype)
    initial_biases        = tf.zeros((nb_transition_gaussians,
                                      nb_tracks, nb_states, nb_dims), dtype=dtype)
    transition_Gaussian_stds = tf.ones((track_len, nb_transition_gaussians,
                                        nb_tracks, nb_states, 1), dtype=dtype)
    transition_biases = tf.zeros((track_len, nb_transition_gaussians,
                                  nb_tracks, nb_states, nb_dims), dtype=dtype)

    # Log normalising factors
    Log_factors = (- tf.math.log(LocErrs + 1e-20)
                   - tf.math.log(g1_std)
                   - tf.math.log(q))

    initial_anomalous_factor = (
        (- tf.math.log(d)
         + 0.5 * tf.math.log(2 * (1 - tf.math.exp(-2 * l_c)) + 1e-20))
        * isconf_mask
        - tf.math.log(v) * isdir_mask)

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
    '''
    The transition_param_function must define the initial transition parameters
    and their constraints, similarly to how constraint_function defines the
    constraints of the states.
    '''
    print('transition_shapes', transition_shapes)
    nb_states = transition_shapes.shape[0]
    nb_time_points, nb_tracks = dts.shape

    transition_shapes = tf.math.exp(transition_shapes)
    transition_rates  = (tf.math.softmax(transition_rates, axis=1)
                         * transition_shapes / reference_dt)
    transition_rates  = transition_rates[None, None] * dts[..., None, None] + 1e-20

    new_transition_shapes = tf.concat(
        (transition_shapes,
         tf.constant([[1] * nb_states], dtype=dtype)), axis=0)
    new_transition_shapes = tf.concat(
        (new_transition_shapes,
         tf.constant([[1]] * (nb_states + 1), dtype=dtype)), axis=1)

    mislinking_dwell_time = tf.constant(
        [0.9 / nb_states] * nb_states, dtype=dtype)
    mislinking_dwell_time = tf.concat((mislinking_dwell_time, [0.1]), axis=0)
    mislinking_dwell_time = tf.broadcast_to(
        mislinking_dwell_time[None, None, None],
        (nb_time_points, nb_tracks, 1, nb_states + 1))

    mislinking_rates = (1 - tf.math.exp(
        -0.5 * density
        * tf.reduce_sum(
            Fs[None] * (effective_ds[:, None] ** 2
                        + effective_ds[None] ** 2) ** 0.5,
            axis=0)[:, None]))
    mislinking_rates = tf.broadcast_to(
        mislinking_rates[None, None],
        (nb_time_points, nb_tracks, nb_states, 1))

    new_transition_rates = tf.concat(
        (transition_rates, mislinking_rates), axis=3)
    new_transition_rates = tf.concat(
        (new_transition_rates, mislinking_dwell_time), axis=2)

    return new_transition_shapes, new_transition_rates
