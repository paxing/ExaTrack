# -*- coding: utf-8 -*-
"""
layers.py
---------
All Keras layer classes for the ExaTrack model.

This file is a verbatim copy of the original layer classes.

Layers
------
transpose_layer           : utility transpose layer
IsfirstMaskLayer          : selects between fresh-init and carry-over state
CarryoverAssignLayer      : saves hidden state to buffers between batches
Initial_layer_constraints : first time step — initialises parameters, runs t=0
Custom_RNN_layer          : forward algorithm loop over all subsequent time steps
Final_layer               : integrates remaining hidden variables, outputs likelihood

Dependencies: config.py, gaussian_ops.py, integration.py, constraints.py
"""

import numpy as np
import tensorflow as tf
from .config import dtype, minval
from .gaussian_ops import norm_log_gaussian
from .integration import (
    RNN_reccurence_formula,
    transition_RNN_reccurence_formula,
    get_sequences,
)


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class transpose_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, x, perm):
        '''
        input dimensions: time point, gaussian, track, state, observed variable
        '''
        return tf.transpose(x, perm=perm)


class IsfirstMaskLayer(tf.keras.layers.Layer):
    """Element-wise   init_val * isfirst + prev_val * (1 - isfirst)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, init_val, prev_val, isfirst):
        return init_val * isfirst + prev_val * (1 - isfirst)


class CarryoverAssignLayer(tf.keras.layers.Layer):
    def __init__(self, carryout_variables, **kwargs):
        super().__init__(**kwargs)
        self.carryout_variables = carryout_variables

    def call(self, output, new_states):
        assign_ops = []
        for var, state in zip(self.carryout_variables, new_states):
            assign_ops.append(var.assign(tf.stop_gradient(state)))
        with tf.control_dependencies(assign_ops):
            return tf.identity(output)


# ---------------------------------------------------------------------------
# RNN cell (one time step of the forward algorithm)
# ---------------------------------------------------------------------------

@tf.function(jit_compile=False)
def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len,
             reshaped_Log_factors, reshaped_transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
             reccurent_next_hidden_var_coefs, reccurent_biases,
             transition_hidden_var_coefs, transition_biases,
             sequence_phase_1, sequence_phase_2,
             transition_mask, transition_sequence,
             transition_mean, transition_var,
             gamma_dist_mean, gamma_dist_var, states, dt_ratios):
    print('LP', LP)

    nb_dims    = input_i.shape[-1]
    nb_tracks  = LP.shape[0]
    nb_states  = reccurent_hidden_var_coefs.shape[2]
    sequence_length = LP.shape[1] // nb_states

    # ---- 1. replicate each hypothesis for all possible next states ----------
    Prev_coefs2  = tf.repeat(Prev_coefs,  nb_states, axis=2)
    Prev_biases2 = tf.repeat(Prev_biases, nb_states, axis=2)
    LP2          = tf.repeat(LP,          nb_states, axis=1)
    segment_len  = tf.repeat(segment_len, nb_states, axis=1)

    # ---- 2. transition integration -----------------------------------------
    alternative_Prev_coefs  = tf.concat((Prev_coefs2,
                                          tf.identity(transition_hidden_var_coefs)), axis=0)
    alternative_Prev_biases = tf.concat((Prev_biases2,
                                          tf.identity(transition_biases)), axis=0)

    transition_Prev_coefs, transition_Prev_biases, LC = \
        transition_RNN_reccurence_formula(
            current_hidden_var_coefs=alternative_Prev_coefs,
            next_hidden_var_coefs=tf.constant(0, dtype=dtype,
                                               shape=alternative_Prev_coefs.shape),
            biases=alternative_Prev_biases,
            transition_sequence=transition_sequence,
            nb_dims=nb_dims,
            dtype=dtype)

    LP2 += LC * transition_mask + reshaped_Log_factors

    # ---- 3. Gamma dwell-time transition probabilities ----------------------
    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates  = gamma_dist_mean / gamma_dist_var

    all_Prev_coefs = (  transition_Prev_coefs  * transition_mask[None, :, :, None]
                      + Prev_coefs2            * (1 - transition_mask[None, :, :, None]))
    all_prev_biases = (  transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2           * (1 - transition_mask[None, :, :, None]))

    transition_probas = tf.clip_by_value(
        (tf.compat.v1.distributions.Gamma(
            current_shapes, current_rates).prob(segment_len[:, :] + 0.5) + 1e-14)
        / (1 - tf.compat.v1.distributions.Gamma(
            current_shapes, current_rates).cdf(segment_len[:, :] + 0.5) + 1e-12),
        clip_value_min=1 - 20, clip_value_max=1 - 1e-10)

    non_transition_probas = tf.repeat(
        1 - tf.clip_by_value(
            tf.reduce_sum(
                tf.reshape(transition_probas * transition_mask,
                           shape=(nb_tracks, nb_states * sequence_length, nb_states)),
                axis=2),
            clip_value_min=1 - 20, clip_value_max=1 - 1e-10),
        nb_states, axis=1)

    transition_probas = (transition_probas * transition_mask
                         + non_transition_probas * (1 - transition_mask))
    all_LP = LP2 + tf.math.log(transition_probas)

    # ---- 4. fold observation into biases and run CGP recurrence ------------
    current_reccurent_obs_var_coefs         = tf.concat([reccurent_obs_var_coefs]         * (sequence_length * nb_states), axis=2)
    current_reccurent_hidden_var_coefs      = tf.concat([reccurent_hidden_var_coefs]      * (sequence_length * nb_states), axis=2)
    current_reccurent_next_hidden_var_coefs = tf.concat([reccurent_next_hidden_var_coefs] * (sequence_length * nb_states), axis=2)
    current_reccurent_biases                = tf.concat([reccurent_biases]                * (sequence_length * nb_states), axis=2)

    current_hidden_var_coefs = tf.concat(
        (all_Prev_coefs, tf.identity(current_reccurent_hidden_var_coefs)), axis=0)
    zero_tensor = tf.constant(0, dtype=dtype, shape=all_Prev_coefs.shape)
    next_hidden_var_coefs = tf.concat(
        (zero_tensor, tf.identity(current_reccurent_next_hidden_var_coefs)), axis=0)

    current_biases  = tf.identity(current_reccurent_biases)
    current_biases += tf.reduce_sum(
        current_reccurent_obs_var_coefs[:, :, :, :, None] * input_i, -2)
    biases = tf.concat((all_prev_biases, current_biases), axis=0)

    Next_coefs, Next_biases, LC = RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2,
        nb_dims=nb_dims, dtype=dtype)

    all_LP += LC

    # ---- 5. reduce transition hypotheses back to sequence_length per state -
    reshaped_Next_coefs = tf.reshape(
        Next_coefs,
        Next_coefs.shape[:2]
        + [sequence_length * nb_states, nb_states, Next_coefs.shape[-1]])

    transition_LPs = (
        tf.reshape(all_LP - 200 * (1 - transition_mask),
                   (nb_tracks, sequence_length * nb_states, nb_states))
        - nb_dims * tf.math.log(
            tf.math.abs(reshaped_Next_coefs[0, :, :, :, 0]
                        * reshaped_Next_coefs[1, :, :, :, 1]) + 1e-20))

    max_transition_LPs = tf.reduce_max(transition_LPs, axis=1, keepdims=True)
    transition_Ps      = tf.math.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps / tf.reduce_sum(transition_Ps, 1, keepdims=True)

    transition_states = tf.reduce_sum(
        states[:, :, None] * transition_weights[:, :, :, None, None], 1)

    transition_Next_coefs = tf.reshape(
        Next_coefs,
        Next_coefs.shape[:2]
        + [sequence_length * nb_states, nb_states, Next_coefs.shape[-1]])
    transition_Next_coefs = tf.reduce_sum(
        transition_Next_coefs * transition_weights[None, :, :, :, None], axis=2)

    transition_Next_biases = tf.reshape(
        Next_biases,
        Next_biases.shape[:2]
        + [sequence_length * nb_states, nb_states, nb_dims])
    transition_Next_biases = tf.reduce_sum(
        transition_Next_biases * transition_weights[None, :, :, :, None], axis=2)

    transition_LPs = (
        tf.math.log(tf.reduce_sum(transition_Ps, axis=1))
        + max_transition_LPs[:, 0]
        + nb_dims * tf.math.log(
            tf.math.abs(transition_Next_coefs[0, :, :, 0]
                        * transition_Next_coefs[1, :, :, 1]) + 1e-20))

    stable_LPs = tf.reshape(all_LP,
                             (nb_tracks, sequence_length * nb_states, nb_states))
    stable_weights = tf.reshape(
        (1 - transition_mask),
        (sequence_length * nb_states, nb_states))[None]
    stable_LPs = tf.reduce_sum(stable_LPs * stable_weights, 2)

    stable_states = tf.reduce_sum(
        states[:, :, None] * stable_weights[:, :, :, None, None], 2)

    stable_Next_coefs = tf.reduce_sum(
        tf.reshape(Next_coefs,
                   Next_coefs.shape[:2]
                   + [sequence_length * nb_states, nb_states, Next_coefs.shape[-1]])
        * stable_weights[None, :, :, :, None], axis=3)

    stable_Next_biases = tf.reduce_sum(
        tf.reshape(Next_biases,
                   Next_biases.shape[:2]
                   + [sequence_length * nb_states, nb_states, nb_dims])
        * stable_weights[None, :, :, :, None], axis=3)

    stable_segment_len = tf.reduce_sum(
        tf.reshape(segment_len, (nb_tracks, sequence_length * nb_states, nb_states))
        * stable_weights, axis=2)

    current_gamma_dist_mean = tf.concat([transition_mean, gamma_dist_mean], axis=1)
    current_gamma_dist_var  = tf.concat([transition_var,  gamma_dist_var],  axis=1)

    Next_coefs  = tf.concat([transition_Next_coefs,  stable_Next_coefs],  axis=2)
    Next_biases = tf.concat([transition_Next_biases, stable_Next_biases], axis=2)
    new_LP          = tf.concat([transition_LPs, stable_LPs], axis=1)
    current_segment_len = tf.concat(
        [tf.ones((nb_tracks, nb_states), dtype=dtype),
         stable_segment_len + dt_ratios[:, None]], axis=1)
    Next_states = tf.concat([transition_states, stable_states], axis=1)

    # ---- 6. merge oldest slab back into the buffer -------------------------
    saved_Next_coefs  = Next_coefs[:,  :, :-nb_states * 2]
    saved_Next_biases = Next_biases[:, :, :-nb_states * 2]
    saved_LP          = new_LP[:, :-nb_states * 2]
    saved_segment_len = current_segment_len[:, :-nb_states * 2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states ** 2 * 2]
    saved_gamma_dist_var  = current_gamma_dist_var[:,  :-nb_states ** 2 * 2]
    saved_states      = Next_states[:, :-nb_states * 2]

    nb_prev_gaussians = Next_coefs.shape[0]

    last_Next_coefs = tf.reshape(
        Next_coefs[:, :, -nb_states * 2:],
        (nb_prev_gaussians, nb_tracks, 2, nb_states, Next_coefs.shape[-1]))
    last_Next_biases = tf.reshape(
        Next_biases[:, :, -nb_states * 2:],
        (nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims))
    last_LP = (
        tf.reshape(new_LP[:, -nb_states * 2:], (nb_tracks, 2, nb_states))
        - nb_dims * tf.math.log(
            tf.math.abs(last_Next_coefs[0, :, :, :, 0]
                        * last_Next_coefs[1, :, :, :, 1]) + 1e-20))
    last_segment_len      = tf.reshape(current_segment_len[:, -nb_states * 2:],
                                        (nb_tracks, 2, nb_states))
    last_gamma_dist_mean  = tf.reshape(current_gamma_dist_mean[:, -nb_states ** 2 * 2:],
                                        (nb_tracks, 2, nb_states, nb_states))
    last_gamma_dist_var   = tf.reshape(current_gamma_dist_var[:,  -nb_states ** 2 * 2:],
                                        (nb_tracks, 2, nb_states, nb_states))
    last_states = tf.reshape(Next_states[:, -nb_states * 2:],
                              (nb_tracks, 2, nb_states, sequence_length, nb_states))

    last_LP_max = tf.reduce_max(last_LP, axis=1, keepdims=True)
    last_P      = tf.math.exp(last_LP - last_LP_max)
    sum_last_P  = tf.reduce_sum(last_P, 1, keepdims=True)

    weight_last_P  = tf.math.exp(last_LP - tf.reduce_max(last_LP, axis=1, keepdims=True))
    last_weights   = weight_last_P / tf.reduce_sum(weight_last_P, 1, keepdims=True)

    reduced_last_Next_coefs  = tf.reduce_sum(
        last_Next_coefs  * last_weights[None, :, :, :, None], axis=2)
    reduced_last_Next_biases = tf.reduce_sum(
        last_Next_biases * last_weights[None, :, :, :, None], axis=2)
    reduced_last_LPs = (
        tf.math.log(sum_last_P + 1e-100) + last_LP_max)[:, 0] + nb_dims * tf.math.log(
        tf.math.abs(reduced_last_Next_coefs[0, :, :, 0]
                    * reduced_last_Next_coefs[1, :, :, 1]) + 1e-20)
    reduced_last_segment_len     = tf.reduce_sum(last_segment_len * last_weights, axis=1)
    reduced_last_gamma_dist_mean = tf.reduce_sum(
        last_gamma_dist_mean * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_var  = tf.reduce_sum(
        (last_gamma_dist_var
         + (last_gamma_dist_mean - reduced_last_gamma_dist_mean[:, None]) ** 2)
        * last_weights[:, :, :, None], axis=1)
    reduced_last_gamma_dist_mean = tf.reshape(reduced_last_gamma_dist_mean,
                                               (nb_tracks, nb_states ** 2))
    reduced_last_gamma_dist_var  = tf.reshape(reduced_last_gamma_dist_var,
                                               (nb_tracks, nb_states ** 2))
    reduced_last_states = tf.reduce_sum(
        last_states * last_weights[:, :, :, None, None], axis=1)

    new_Next_coefs  = tf.concat((saved_Next_coefs,  reduced_last_Next_coefs),  axis=2)
    new_Next_biases = tf.concat((saved_Next_biases, reduced_last_Next_biases), axis=2)
    new_LPs         = tf.concat((saved_LP,          reduced_last_LPs),         axis=1)
    new_segment_len = tf.concat((saved_segment_len, reduced_last_segment_len), axis=1)
    new_gamma_dist_mean = tf.concat((saved_gamma_dist_mean, reduced_last_gamma_dist_mean), axis=1)
    new_gamma_dist_var  = tf.concat((saved_gamma_dist_var,  reduced_last_gamma_dist_var),  axis=1)
    new_states = tf.concat((saved_states, reduced_last_states), axis=1)

    current_states = states[:, :, -1:]
    new_states = tf.concat((new_states, current_states), axis=2)[:, :, 1:]

    return (new_Next_coefs, new_Next_biases, new_LPs, new_segment_len,
            new_gamma_dist_mean, new_gamma_dist_var, new_states)


# ---------------------------------------------------------------------------
# Initial layer
# ---------------------------------------------------------------------------

class Initial_layer_constraints(tf.keras.layers.Layer):
    def __init__(self, nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
                 params, initial_params, initial_fractions,
                 max_linking_distance, constraint_function, reference_dt,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 sequence_length=3, carryover=True,
                 LocErr_type='Linear', **kwargs):
        super().__init__(**kwargs)

        if vary_params is None:
            vary_params = np.ones(params.shape, dtype='float64')
        if vary_initial_params is None:
            vary_initial_params = np.ones(initial_params.shape, dtype='float64')
        if vary_initial_fractions is None:
            vary_initial_fractions = np.ones(initial_fractions.shape, dtype='float64')

        self.nb_states = nb_states
        self.nb_gaussians = nb_gaussians
        self.nb_obs_vars = nb_obs_vars
        self.nb_hidden_vars = nb_hidden_vars
        self.params = params
        self.initial_params = initial_params
        self.initial_fractions = initial_fractions
        self.constraint_function = constraint_function
        self.sequence_length = sequence_length
        self.max_linking_distance = max_linking_distance
        self.vary_params = vary_params
        self.vary_initial_params = vary_initial_params
        self.vary_initial_fractions = vary_initial_fractions
        self.reference_dt = reference_dt
        self.carryover = carryover
        self.LocErr_type = LocErr_type

        (initial_sequence_phase_1,
         initial_sequence_phase_2,
         recurrent_sequence_phase_1,
         recurrent_sequence_phase_2,
         final_sequence_phase_1,
         transition_sequence) = get_sequences(
            params, initial_params, constraint_function,
            nb_gaussians, nb_hidden_vars, dtype)

        self.initial_sequence_phase_1  = initial_sequence_phase_1
        self.initial_sequence_phase_2  = initial_sequence_phase_2
        self.recurrent_sequence_phase_1 = recurrent_sequence_phase_1
        self.recurrent_sequence_phase_2 = recurrent_sequence_phase_2
        self.transition_sequence        = transition_sequence
        self.final_sequence_phase_1     = final_sequence_phase_1

    def build(self, input_shape):
        _dtype = self.dtype
        self.param_vars = tf.Variable(
            self.params, dtype=_dtype, name='recurrence_variables',
            constraint=lambda w: tf.where(
                tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.initial_param_vars = tf.Variable(
            self.initial_params, dtype=_dtype, name='initial_variables',
            trainable=True,
            constraint=lambda w: tf.where(
                tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.max_linking_distance_param = tf.Variable(
            self.max_linking_distance, dtype=_dtype,
            name='max linking distance', trainable=False)
        initial_fractions = self.initial_fractions
        self.initial_fractions = tf.Variable(
            initial_fractions, dtype=_dtype, name='Fractions', trainable=True)

        nb_sequences = self.sequence_length * (self.nb_states + 1)
        if self.carryover:
            self.carryout_coefs = tf.Variable(
                np.zeros((self.nb_hidden_vars, input_shape[2],
                           nb_sequences, input_shape[5])),
                dtype=_dtype, trainable=False)
            self.carryout_biases = tf.Variable(
                np.zeros(self.carryout_coefs.shape),
                dtype=_dtype, trainable=False)
            self.carryout_LP = tf.Variable(
                np.zeros((input_shape[2], nb_sequences)),
                dtype=_dtype, trainable=False)

        if self.LocErr_type == 'Identity':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs
        elif self.LocErr_type == 'Linear':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs * LocErr_param
        elif self.LocErr_type == 'Photon':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs ** 0.5 * LocErr_param
        elif self.LocErr_type == 'Constant':
            def LocErr_function(LocErrs, LocErr_param):
                return LocErrs * 0 + LocErr_param
        else:
            raise ValueError(
                "Wrong LocErr_type, can be 'Identity', 'Linear', 'Photon' or 'Constant'.")
        self.LocErr_function = LocErr_function

    def call(self, inputs, input_LocErrs, input_dts):
        nb_tracks = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        _dtype = self.dtype
        constraint_function = self.constraint_function
        reference_dt = self.reference_dt

        param_vars         = self.param_vars
        initial_param_vars = self.initial_param_vars
        nb_states          = self.nb_states
        max_linking_distance = self.max_linking_distance_param
        vary_params          = self.vary_params
        vary_initial_params  = self.vary_initial_params
        initial_fractions    = tf.math.softmax(self.initial_fractions)
        vary_initial_fractions = self.vary_initial_fractions
        LocErr_function      = self.LocErr_function
        nb_dims              = inputs.shape[-1]

        param_vars = (vary_params * param_vars
                      + (1 - vary_params) * tf.stop_gradient(param_vars))
        initial_param_vars = (vary_initial_params * initial_param_vars
                              + (1 - vary_initial_params)
                              * tf.stop_gradient(initial_param_vars))
        initial_fractions = (vary_initial_fractions * initial_fractions
                             + (1 - vary_initial_fractions)
                             * tf.stop_gradient(initial_fractions))

        param_vars, initial_param_vars, initial_fractions = self.duplicate_states(
            param_vars, initial_param_vars, initial_fractions)

        # Add mislinking state
        param_vars = tf.concat(
            (param_vars,
             [[param_vars[-1][0],
               tf.math.log(tf.cast(max_linking_distance, dtype=_dtype)),
               -15.,
               tf.math.log(tf.cast(0.00001, dtype=_dtype)),
               0]]),
            axis=0)
        initial_param_vars = tf.concat(
            (initial_param_vars, [initial_param_vars[-1]]), axis=0)
        nb_states = nb_states + 1

        (hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases,
         initial_hidden_var_coefs, initial_obs_var_coefs,
         initial_Gaussian_stds, initial_biases,
         transition_hidden_var_coefs, transition_Gaussian_stds,
         transition_biases, integration_variable_index,
         Log_factors, initial_Log_factors,
         transition_Log_factors) = constraint_function(
            param_vars, initial_param_vars, input_LocErrs, input_dts,
            nb_dims, reference_dt, LocErr_function, _dtype)

        hidden_var_coefs = hidden_var_coefs / Gaussian_stds
        obs_var_coefs    = obs_var_coefs    / Gaussian_stds
        biases           = biases           / Gaussian_stds

        current_hidden_var_coefs = hidden_var_coefs[..., :nb_hidden_vars]
        next_hidden_var_coefs    = hidden_var_coefs[..., nb_hidden_vars:]

        reccurent_obs_var_coefs         = tf.identity(obs_var_coefs)
        reccurent_hidden_var_coefs      = tf.identity(current_hidden_var_coefs)
        reccurent_next_hidden_var_coefs = tf.identity(next_hidden_var_coefs)
        reccurent_biases                = tf.identity(biases)

        initial_hidden_var_coefs = initial_hidden_var_coefs / initial_Gaussian_stds
        initial_obs_var_coefs    = initial_obs_var_coefs    / initial_Gaussian_stds
        initial_biases           = initial_biases           / initial_Gaussian_stds

        current_initial_hidden_var_coefs = initial_hidden_var_coefs[..., :nb_hidden_vars]
        next_initial_hidden_var_coefs    = tf.zeros(
            (nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars), dtype=_dtype)

        transition_hidden_var_coefs = transition_hidden_var_coefs / transition_Gaussian_stds
        transition_biases           = transition_biases           / transition_Gaussian_stds

        sequence_length = self.sequence_length
        transition_hidden_var_coefs = tf.concat(
            [transition_hidden_var_coefs] * sequence_length * nb_states, 3)
        transition_biases = tf.concat(
            [transition_biases] * nb_states * sequence_length, 3)

        # First time step (t=0)
        biases                   = reccurent_biases[0]
        obs_var_coefs            = reccurent_obs_var_coefs[0]
        current_hidden_var_coefs = reccurent_hidden_var_coefs[0]
        next_hidden_var_coefs    = reccurent_next_hidden_var_coefs[0]

        biases        += tf.reduce_sum(obs_var_coefs[..., None] * inputs[0], -2)
        initial_biases += tf.reduce_sum(initial_obs_var_coefs[..., None] * inputs[0], -2)

        current_hidden_var_coefs = tf.concat(
            (current_initial_hidden_var_coefs, current_hidden_var_coefs), axis=0)
        next_hidden_var_coefs = tf.concat(
            (next_initial_hidden_var_coefs, next_hidden_var_coefs), axis=0)
        biases = tf.concat((initial_biases, biases), axis=0)

        current_hidden_var_coefs = tf.concat(
            [current_hidden_var_coefs] * sequence_length, axis=2)
        next_hidden_var_coefs = tf.concat(
            [next_hidden_var_coefs] * sequence_length, axis=2)
        biases = tf.concat([biases] * sequence_length, axis=2)

        sequence_phase_1 = self.initial_sequence_phase_1
        sequence_phase_2 = self.initial_sequence_phase_2

        Next_coefs, Next_biases, LC = RNN_reccurence_formula(
            current_hidden_var_coefs,
            next_hidden_var_coefs,
            biases,
            sequence_phase_1,
            sequence_phase_2,
            nb_dims,
            dtype=_dtype)

        init_log_fractions = tf.concat(
            [tf.math.log(initial_fractions)] * sequence_length, axis=1)
        init_log_factors = tf.concat(
            [nb_dims * initial_Log_factors] * sequence_length, axis=1)

        LP = (LC + init_log_factors + init_log_fractions
              + tf.math.log(np.array(1 / sequence_length)))

        Log_factors            = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors

        initial_states = [
            Next_coefs, Next_biases, LP,
            Log_factors, transition_Log_factors,
            reccurent_obs_var_coefs,
            reccurent_hidden_var_coefs,
            reccurent_next_hidden_var_coefs,
            reccurent_biases,
            transition_hidden_var_coefs,
            transition_biases,
        ]
        return inputs, initial_states

    def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
        return param_vars, initial_param_vars, initial_fractions


# ---------------------------------------------------------------------------
# Custom RNN layer (forward algorithm loop)
# ---------------------------------------------------------------------------

class Custom_RNN_layer(tf.keras.layers.Layer):
    def __init__(self, nb_tracks, transition_shapes, transition_rates,
                 density, nb_states,
                 sequence_phase_1, sequence_phase_2, transition_sequence,
                 transition_param_function,
                 sequence_length=3,
                 vary_transition_shapes=None, vary_transition_rates=None,
                 carryover=False, **kwargs):
        if vary_transition_rates is None:
            vary_transition_rates = tf.ones(transition_rates.shape, dtype=dtype)
        if vary_transition_shapes is None:
            vary_transition_shapes = tf.ones(transition_shapes.shape, dtype=dtype)

        self.sequence_phase_1       = sequence_phase_1
        self.sequence_phase_2       = sequence_phase_2
        self.transition_sequence    = transition_sequence
        self.nb_states              = nb_states + 1
        self.sequence_length        = sequence_length
        self.nb_tracks              = nb_tracks
        self.initial_transition_params = [transition_shapes, transition_rates]
        self.transition_param_function = transition_param_function
        self.density                = density
        self.vary_transition_shapes = vary_transition_shapes
        self.vary_transition_rates  = vary_transition_rates
        self.carryover              = carryover
        super().__init__(**kwargs)

    def build(self, input_shape):
        nb_states       = self.nb_states
        transition_shapes, transition_rates = self.initial_transition_params
        sequence_length = self.sequence_length
        nb_tracks       = self.nb_tracks

        self.transition_rates = tf.Variable(
            transition_rates, dtype=dtype, name='Transition rates', trainable=True,
            constraint=lambda w: tf.where(
                tf.greater_equal(w, tf.math.log(minval)), w, tf.math.log(minval)))
        self.transition_shapes = tf.Variable(
            transition_shapes, dtype=dtype, name='Transition shape', trainable=True)

        indices = tf.stack([
            tf.repeat(tf.constant(list(np.arange(nb_states)) * sequence_length), nb_states),
            tf.concat([tf.range(nb_states)] * nb_states * sequence_length, 0)
        ], axis=1)
        transition_mask = tf.cast((indices[:, 0] - indices[:, 1]) != 0, dtype=dtype)[None]
        self.indices         = indices
        self.transition_mask = transition_mask

        if self.carryover:
            self.carryout_segment_len = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states)),
                dtype=dtype, name='carryover_segment_length', trainable=False)
            self.carryout_gamma_dist_mean = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states ** 2)),
                dtype=dtype, name='carryover_gamma_dist_mean', trainable=False)
            self.carryout_gamma_dist_var = tf.Variable(
                np.zeros((nb_tracks, sequence_length * nb_states ** 2)),
                dtype=dtype, name='carryover_gamma_dist_var', trainable=False)

        self.built = True

    @tf.function(jit_compile=False)
    def call(self, inputs, input_dts, reference_dt, mask,
             Prev_coefs, Prev_biases, LP,
             Log_factors, transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
             reccurent_next_hidden_var_coefs, reccurent_biases,
             transition_hidden_var_coefs, transition_biases,
             log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
             isfirst=None):

        nb_tracks              = self.nb_tracks
        sequence_phase_1       = self.sequence_phase_1
        sequence_phase_2       = self.sequence_phase_2
        transition_sequence    = self.transition_sequence
        transition_mask        = self.transition_mask
        nb_states              = self.nb_states
        indices                = self.indices
        sequence_length        = self.sequence_length
        density                = self.density
        vary_transition_shapes = self.vary_transition_shapes
        vary_transition_rates  = self.vary_transition_rates

        transition_rates  = self.transition_rates
        transition_shapes = self.transition_shapes

        transition_shapes = (vary_transition_shapes * transition_shapes
                             + (1 - vary_transition_shapes)
                             * tf.stop_gradient(transition_shapes))
        transition_rates  = (vary_transition_rates  * transition_rates
                             + (1 - vary_transition_rates)
                             * tf.stop_gradient(transition_rates))

        ds           = tf.math.exp(log_ds)
        Fs           = tf.math.softmax(softmax_inv_Fractions[0, :-1])
        effective_ds = ds + 2 * tf.math.exp(anomalous_factors) * isdir

        dts_TN = tf.transpose(input_dts, [1, 0])
        transition_shapes_full, transition_rates_full = self.transition_param_function(
            transition_shapes, transition_rates, density,
            Fs, effective_ds, dts_TN, reference_dt, dtype)
        transition_rates_full[0, 0]

        oh_row = tf.cast(tf.one_hot(indices[:, 0], nb_states), dtype)
        oh_col = tf.cast(tf.one_hot(indices[:, 1], nb_states), dtype)
        oh_src = oh_col

        flat_Log_full       = tf.einsum('tns,ps->tnp', Log_factors,            oh_row)
        flat_trans_Log_full = tf.einsum('tns,ps->tnp', transition_Log_factors, oh_src)
        flat_Log_full = (flat_trans_Log_full * transition_mask
                         + flat_Log_full     * (1 - transition_mask))

        transition_rates_flat_full = tf.einsum(
            'tnij,pi,pj->tnp', transition_rates_full,  oh_row, oh_col)
        transition_shapes_flat = tf.einsum(
            'ij,pi,pj->p',     transition_shapes_full, oh_row, oh_col)

        transition_mean_full = (transition_shapes_flat[None, None]
                                / transition_rates_flat_full)
        transition_var_full  = (transition_shapes_flat[None, None]
                                / (transition_rates_flat_full ** 2))

        rec_obs_var_coefs_seq           = reccurent_obs_var_coefs[1:]
        rec_hidden_var_coefs_seq        = reccurent_hidden_var_coefs[1:]
        rec_next_hidden_var_coefs_seq   = reccurent_next_hidden_var_coefs[1:]
        rec_biases_seq                  = reccurent_biases[1:]
        transition_hidden_var_coefs_seq = transition_hidden_var_coefs[1:]
        transition_biases_seq           = transition_biases[1:]

        reccurent_hidden_var_coefs[0]

        flat_Log_seq        = flat_Log_full[1:]
        flat_trans_Log_seq  = flat_trans_Log_full[1:]
        transition_mean_seq = transition_mean_full[1:, :, :nb_states ** 2]
        transition_var_seq  = transition_var_full[1:,  :, :nb_states ** 2]

        segment_len     = tf.ones((nb_tracks, sequence_length * nb_states), dtype=dtype)
        gamma_dist_mean = transition_mean_full[0]
        gamma_dist_var  = transition_var_full[0]

        if self.carryover:
            br_isfirst_1 = tf.broadcast_to(isfirst[:, None], segment_len.shape)
            segment_len  = (br_isfirst_1 * segment_len
                            + (1 - br_isfirst_1) * self.carryout_segment_len)
            br_isfirst_2 = tf.broadcast_to(isfirst[:, None], gamma_dist_mean.shape)
            gamma_dist_mean = (br_isfirst_2 * gamma_dist_mean
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_mean)
            gamma_dist_var  = (br_isfirst_2 * gamma_dist_var
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_var)

        states_indices = tf.range(0, nb_states * sequence_length, dtype='int32') % nb_states
        states_indices = tf.repeat(states_indices[:, None], sequence_length, axis=1)
        states = tf.repeat(
            tf.one_hot(states_indices, nb_states, dtype=dtype)[None],
            nb_tracks, axis=0)

        nb_dims   = reccurent_biases.shape[4]
        num_steps = tf.shape(inputs)[0]

        All_states_ta = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=(nb_tracks, 1, nb_states))
        All_coefs_ta  = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=Prev_coefs.shape)
        All_biases_ta = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=Prev_biases.shape)
        All_LP_ta     = tf.TensorArray(
            dtype=dtype, size=num_steps, dynamic_size=False,
            element_shape=LP.shape)

        def body(i, Prev_coefs, Prev_biases, LP, segment_len,
                 gamma_dist_mean, gamma_dist_var, states,
                 All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta):

            log_w = LP - nb_dims * tf.math.log(
                tf.math.abs(Prev_coefs[0, :, :, 0]
                            * Prev_coefs[1, :, :, 1]) + 1e-20)
            max_log_w = tf.reduce_max(log_w, 1, keepdims=True)
            w = tf.math.exp(log_w - max_log_w)
            w = w / tf.reduce_sum(w, 1, keepdims=True)
            pred_states = tf.reduce_sum(
                w[:, :, None] * states[:, :, 0], 1, keepdims=True)

            All_states_ta = All_states_ta.write(i, pred_states)
            All_coefs_ta  = All_coefs_ta.write(i,  Prev_coefs)
            All_biases_ta = All_biases_ta.write(i, Prev_biases)
            All_LP_ta     = All_LP_ta.write(i,     LP)

            input_i = inputs[i]
            mask_i  = mask[:, i]

            rec_obs_i      = rec_obs_var_coefs_seq[i]
            rec_hid_i      = rec_hidden_var_coefs_seq[i]
            rec_next_hid_i = rec_next_hidden_var_coefs_seq[i]
            rec_bias_i     = rec_biases_seq[i]
            trans_hid_i    = transition_hidden_var_coefs_seq[i]
            trans_bias_i   = transition_biases_seq[i]

            log_factors_i       = flat_Log_seq[i]
            trans_log_factors_i = flat_trans_Log_seq[i]
            trans_mean_i        = transition_mean_seq[i]
            trans_var_i         = transition_var_seq[i]
            dt_ratios           = input_dts[:, i] / reference_dt

            (Next_coefs, Next_biases, Next_LP, Next_segment_len,
             Next_gamma_mean, Next_gamma_var, Next_states) = RNN_cell(
                input_i, Prev_coefs, Prev_biases, LP, segment_len,
                log_factors_i, trans_log_factors_i,
                rec_obs_i, rec_hid_i, rec_next_hid_i, rec_bias_i,
                trans_hid_i, trans_bias_i,
                sequence_phase_1, sequence_phase_2,
                transition_mask, transition_sequence,
                trans_mean_i, trans_var_i,
                gamma_dist_mean, gamma_dist_var, states, dt_ratios)

            (Next_LP - nb_dims * tf.math.log(
                tf.math.abs(Next_coefs[0, :, :, 0]
                            * Next_coefs[1, :, :, 1]) + 1e-20))[10:20]

            mask_coef   = mask_i[None, :, None, None]
            mask_scalar = mask_i[:, None]
            mask_state  = mask_i[:, None, None, None]

            Prev_coefs      = Next_coefs       * mask_coef   + Prev_coefs      * (1 - mask_coef)
            Prev_biases     = Next_biases      * mask_coef   + Prev_biases     * (1 - mask_coef)
            LP              = Next_LP          * mask_scalar + LP              * (1 - mask_scalar)
            segment_len     = Next_segment_len * mask_scalar + segment_len     * (1 - mask_scalar)
            gamma_dist_mean = Next_gamma_mean  * mask_scalar + gamma_dist_mean * (1 - mask_scalar)
            gamma_dist_var  = Next_gamma_var   * mask_scalar + gamma_dist_var  * (1 - mask_scalar)
            states          = Next_states      * mask_state  + states          * (1 - mask_state)

            return (i + 1, Prev_coefs, Prev_biases, LP, segment_len,
                    gamma_dist_mean, gamma_dist_var, states,
                    All_states_ta, All_coefs_ta, All_biases_ta, All_LP_ta)

        cond = lambda i, *_: i < num_steps

        (i, Prev_coefs, Prev_biases, LP, segment_len,
         gamma_dist_mean, gamma_dist_var, states,
         All_states_ta, All_coefs_ta,
         All_biases_ta, All_LP_ta) = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0), Prev_coefs, Prev_biases,
                       LP, segment_len, gamma_dist_mean,
                       gamma_dist_var, states, All_states_ta,
                       All_coefs_ta, All_biases_ta, All_LP_ta],
            parallel_iterations=1, swap_memory=True)

        All_states = tf.transpose(All_states_ta.stack(),
                                  perm=[1, 0, 2, 3])[:, :, 0, :]
        All_coefs  = tf.transpose(All_coefs_ta.stack(),
                                  perm=[2, 0, 3, 1, 4])
        All_biases = tf.transpose(All_biases_ta.stack(),
                                  perm=[2, 0, 3, 1, 4])
        All_LPs    = tf.transpose(All_LP_ta.stack(),
                                  perm=[1, 0, 2])
        All_states = All_states[:, sequence_length - 1:]

        return (Prev_coefs, Prev_biases, LP, segment_len,
                gamma_dist_mean, gamma_dist_var,
                All_states, All_coefs, All_biases, All_LPs, states)


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------

class Final_layer(tf.keras.layers.Layer):
    def __init__(self, sequence_phase_1, nb_dims, sequence_length, **kwargs):
        self.sequence_phase_1 = sequence_phase_1
        self.nb_dims          = nb_dims
        self.sequence_length  = sequence_length
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    @tf.function(jit_compile=False)
    def call(self, states):
        nb_dims = self.nb_dims
        Prev_coefs, Prev_biases, LP, All_states, last_states = states

        if Prev_coefs.shape[0] > 0:
            current_hidden_var_coefs = Prev_coefs
            zero_tensor = tf.constant(0, dtype=dtype, shape=Prev_coefs.shape)
            next_hidden_var_coefs = zero_tensor
            biases = Prev_biases

            Next_coefs, Next_biases, LC = RNN_reccurence_formula(
                current_hidden_var_coefs,
                next_hidden_var_coefs,
                biases,
                self.sequence_phase_1,
                [[], []],
                nb_dims=nb_dims,
                dtype=self.dtype)
            LP += LC

        log_weigths     = LP
        max_log_weigths = tf.reduce_max(log_weigths, 1, keepdims=True)
        weights         = tf.math.exp(log_weigths - max_log_weigths)
        weights         = weights / tf.reduce_sum(weights, 1, keepdims=True)
        pred_states     = tf.reduce_sum(weights[:, :, None, None] * last_states, 1)
        All_states      = tf.concat((All_states, pred_states), axis=1)
        output          = LP

        return output, All_states