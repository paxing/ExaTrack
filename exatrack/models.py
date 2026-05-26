# -*- coding: utf-8 -*-
"""
models.py
---------
Model assembly, loss function, and parameter utilities.

This file is a verbatim copy of the original functions.

Dependencies: config.py, constraints.py, layers.py
"""

import numpy as np
import tensorflow as tf
import pandas as pd

from .config import dtype, minval
from .constraints import constraint_function, transition_param_function
from .layers import (
    Initial_layer_constraints,
    Custom_RNN_layer,
    Final_layer,
    IsfirstMaskLayer,
    CarryoverAssignLayer,
    transpose_layer,
)


def MLE_loss(y_true, y_pred):
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims=True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims=True)) + max_LP
    return -tf.math.reduce_mean(pred)


class get_parameters(tf.keras.callbacks.Callback):
    def __init__(self, track_segmentation=True, layer_name='params'):
        super(get_parameters, self).__init__()
        self.layer_name = layer_name
        self.track_segmentation = track_segmentation

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.weights
        nb_states = weights[0].shape[0]
        if self.track_segmentation:
            shape_idx = 8
            rate_idx = 7
        else:
            shape_idx = 5
            rate_idx = 4
        transition_shapes = tf.math.exp(weights[shape_idx])
        transition_rates = tf.math.softmax(weights[rate_idx], axis=1) * transition_shapes
        transition_rates = np.round(transition_rates, 3)
        transition_shapes = np.round(transition_shapes, 3)
        transition_rates = [list(rates) for rates in transition_rates]
        transition_shapes = [list(shapes) for shapes in transition_shapes]
        model_types = weights[0][:, -1].numpy().astype(int)
        model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
        params = {
            'Model types': model_types_str,
            'anomalous factors': list(np.round(
                tf.sigmoid(weights[0][:, 2]) * (1 - weights[0][:, 4])
                + 2 ** 0.5 * tf.exp(weights[0][:, 2]) * weights[0][:, 4], 4)),
            'Localization errors': list(np.round(np.exp(weights[0][:, 0]), 3)),
            'd': list(np.round(np.exp(weights[0][:, 1]), 3)),
            'anomalous variation': list(np.round(np.exp(weights[0][:, 3]), 5)),
            'transition rates': transition_rates,
            'transition shapes': transition_shapes,
            'Fractions': list(np.round(tf.math.softmax(weights[2][0]), 3)),
        }
        print(params)


def build_segment_model(track_len,
                        nb_states,
                        params,
                        initial_params,
                        transition_rates,
                        transition_shapes,
                        initial_fractions,
                        batch_size,
                        reference_dt,
                        nb_dims=2,
                        sequence_length=3,
                        max_linking_distance=3,
                        estimated_density=0.001,
                        vary_params=None,
                        vary_initial_params=None,
                        vary_initial_fractions=None,
                        vary_transition_shapes=None,
                        vary_transition_rates=None,
                        nb_LocErr_dims=1,
                        LocErr_type='Linear'):

    nb_obs_vars = 1
    nb_independent_vars = nb_dims
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars

    inputs = tf.keras.Input(
        batch_shape=(batch_size, track_len, nb_independent_vars),
        name='tracks', dtype=dtype)
    if nb_LocErr_dims > 0:
        input_LocErrs = tf.keras.Input(
            batch_shape=(batch_size, track_len, nb_LocErr_dims),
            name='Localization errors', dtype=dtype)
    else:
        input_LocErrs = tf.keras.Input(
            batch_shape=(batch_size, track_len),
            name='Localization errors', dtype=dtype)
    input_dts = tf.keras.Input(
        batch_shape=(batch_size, track_len + 1),
        name='frame durations', dtype=dtype)
    input_mask = tf.keras.Input(
        batch_shape=(batch_size, track_len), name='masks', dtype=dtype)
    input_isfirst = tf.keras.Input(
        batch_shape=(batch_size,), name='isfirsts', dtype=dtype)

    reshaped_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, None, :, None, None, :], dtype=dtype)(inputs)
    transposed_inputs = transpose_layer(dtype=dtype)(
        reshaped_inputs, perm=[2, 1, 0, 3, 4, 5])

    Init_layer = Initial_layer_constraints(
        nb_states,
        nb_gaussians,
        nb_obs_vars,
        nb_hidden_vars,
        params,
        initial_params,
        initial_fractions,
        max_linking_distance,
        constraint_function,
        reference_dt=reference_dt,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        sequence_length=sequence_length,
        carryover=True,
        LocErr_type=LocErr_type,
        dtype=dtype)

    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)

    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir = Init_layer.param_vars[:, 4]

    (Prev_coefs, Prev_biases, LP,
     Log_factors, transition_Log_factors,
     reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
     reccurent_next_hidden_var_coefs, reccurent_biases,
     transition_hidden_var_coefs, transition_biases) = initial_states

    first_mask_layer = IsfirstMaskLayer(dtype=dtype)
    Prev_coefs  = first_mask_layer(Prev_coefs,  Init_layer.carryout_coefs,
                                   input_isfirst[None, :, None, None])
    Prev_biases = first_mask_layer(Prev_biases, Init_layer.carryout_biases,
                                   input_isfirst[None, :, None, None])
    LP          = first_mask_layer(LP,          Init_layer.carryout_LP,
                                   input_isfirst[:, None])

    sliced_inputs = tf.keras.layers.Lambda(
        lambda x: x[1:], dtype=dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(
        lambda x: x[:, 1:], dtype=dtype)(input_mask)

    layer = Custom_RNN_layer(
        batch_size,
        transition_shapes,
        transition_rates,
        estimated_density,
        nb_states,
        Init_layer.recurrent_sequence_phase_1,
        Init_layer.recurrent_sequence_phase_2,
        Init_layer.transition_sequence,
        transition_param_function,
        sequence_length=sequence_length,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        carryover=True,
        dtype=dtype)

    (Prev_coefs, Prev_biases, LP, segment_len,
     gamma_dist_mean, gamma_dist_var,
     All_motion_states, All_coefs, All_biases, All_LPs,
     motion_states) = layer(
        sliced_inputs, input_dts, reference_dt, sliced_mask,
        Prev_coefs, Prev_biases, LP,
        Log_factors, transition_Log_factors,
        reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
        reccurent_next_hidden_var_coefs, reccurent_biases,
        transition_hidden_var_coefs, transition_biases,
        log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
        isfirst=input_isfirst)

    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]

    carryover_layer = CarryoverAssignLayer(
        carryout_variables=[
            Init_layer.carryout_coefs,
            Init_layer.carryout_biases,
            Init_layer.carryout_LP,
            layer.carryout_segment_len,
            layer.carryout_gamma_dist_mean,
            layer.carryout_gamma_dist_var,
        ],
        dtype=dtype)

    F_layer = Final_layer(
        Init_layer.final_sequence_phase_1,
        nb_dims=nb_independent_vars,
        sequence_length=sequence_length,
        dtype=dtype)

    outputs, All_states = F_layer(states)
    outputs = carryover_layer(
        outputs,
        [Prev_coefs, Prev_biases, LP,
         segment_len, gamma_dist_mean, gamma_dist_var])

    model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=(outputs, All_states, All_coefs, All_biases, All_LPs),
        name="Diffusion_model")

    return model, pred_model


def get_model_params(model, track_segmentation=False):
    weights = model.weights
    nb_states = weights[-1].shape[0]
    if track_segmentation:
        shape_IDs, rates_IDs = 8, 7
    else:
        shape_IDs, rates_IDs = 5, 4
    transition_shapes = tf.math.exp(weights[shape_IDs]).numpy()
    transition_rates  = (tf.math.softmax(weights[rates_IDs], axis=1)
                         * transition_shapes).numpy()
    model_types = weights[0][:, -1].numpy().astype(int)
    model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
    anomalous_factors = (
        tf.sigmoid(weights[0][:, 2]) * (1 - weights[0][:, 4])
        + 2 ** 0.5 * tf.exp(weights[0][:, 2]) * weights[0][:, 4]).numpy()
    return {
        'Model types':        model_types_str,
        'anomalous factors':  anomalous_factors,
        'Localization errors': np.exp(weights[0][:, 0].numpy()),
        'd':                  np.exp(weights[0][:, 1].numpy()),
        'q':                  np.exp(weights[0][:, 3].numpy()),
        'transition rates':   transition_rates,
        'transition shapes':  transition_shapes,
        'Fractions':          tf.math.softmax(weights[2][0]).numpy(),
    }


def get_model_raw_params(model, track_segmentation=True, return_dict=False):
    weights = model.get_weights()
    params             = weights[0].copy()
    initial_params     = weights[1].copy()
    initial_fractions  = weights[2].copy()
    if track_segmentation:
        transition_shapes = weights[8].copy()
        transition_rates  = weights[7].copy()
    else:
        transition_shapes = weights[5].copy()
        transition_rates  = weights[4].copy()
    if return_dict:
        return {
            'params': params,
            'initial_params': initial_params,
            'initial_fractions': initial_fractions,
            'transition_shapes': transition_shapes,
            'transition_rates': transition_rates,
        }
    return params, initial_params, initial_fractions, transition_shapes, transition_rates


def equilibrium_distribution(P):
    """Compute stationary distribution of a row-stochastic Markov chain."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def build_abrupt_directed_motion_changes_model(
        segment_length,
        nb_states,
        params,
        initial_params,
        transition_rates,
        transition_shapes,
        initial_fractions,
        batch_size,
        reference_dt,
        nb_dims=2,
        sequence_length=3,
        max_linking_distance=3,
        estimated_density=0.001,
        abrupt_change_state=1,
        vary_params=None,
        vary_initial_params=None,
        vary_initial_fractions=None,
        vary_transition_shapes=None,
        vary_transition_rates=None,
        LocErr_type='Linear'):

    class Initial_layer_constraints_abrupt_change(Initial_layer_constraints):
        def duplicate_states(self, param_vars, initial_param_vars, initial_fractions):
            param_vars = tf.concat(
                (param_vars[:abrupt_change_state + 1],
                 param_vars[abrupt_change_state:]), 0)
            initial_param_vars = tf.concat(
                (initial_param_vars[:abrupt_change_state + 1],
                 initial_param_vars[abrupt_change_state:]), 0)
            initial_fractions = tf.concat(
                (initial_fractions[:, :abrupt_change_state + 1],
                 [[1e-10]],
                 initial_fractions[:, abrupt_change_state + 1:]), 1)
            return param_vars, initial_param_vars, initial_fractions

    def transition_param_function(transition_shapes, transition_rates,
                                   density, Fs, effective_ds,
                                   dts, reference_dt, dtype):
        print('transition_shapes', transition_shapes)
        nb_states = transition_shapes.shape[0]
        nb_time_points, nb_tracks = dts.shape

        directed_directed_transition_shape = tf.math.exp(
            transition_shapes[abrupt_change_state, abrupt_change_state])
        directed_directed_transition_rate = (
            transition_rates[abrupt_change_state, abrupt_change_state] - 4)

        transition_shapes = tf.math.exp(transition_shapes)
        transition_rates = transition_rates

        new_transition_shapes = tf.concat(
            (transition_shapes,
             tf.constant([[1] * nb_states], dtype=dtype)), axis=0)
        new_transition_shapes = tf.concat(
            (new_transition_shapes,
             tf.constant([[1]] * (nb_states + 1), dtype=dtype)), axis=1)

        mislinking_dwell_time = tf.constant(
            [0.9 / nb_states] * nb_states, dtype=dtype)
        mislinking_dwell_time = tf.concat(
            (mislinking_dwell_time, [0.1]), axis=0)[None]

        mislinking_rates = tf.math.log(
            1 - tf.math.exp(
                -0.5 * density
                * tf.reduce_sum(
                    Fs[None] * (effective_ds[:, None] ** 2
                                + effective_ds[None] ** 2) ** 0.5,
                    axis=0)[:, None]))

        new_transition_rates = tf.concat(
            (transition_rates, mislinking_rates), axis=1)
        new_transition_rates = tf.concat(
            (new_transition_rates, mislinking_dwell_time), axis=0)

        second_directed_state_on_rates = tf.stack(
            [-10] * abrupt_change_state
            + [directed_directed_transition_rate]
            + [-10] * (nb_states - abrupt_change_state))[:, None]
        new_new_transition_rates = tf.concat(
            (new_transition_rates[:, :abrupt_change_state + 1],
             second_directed_state_on_rates,
             new_transition_rates[:, abrupt_change_state + 1:]), 1)

        second_directed_state_off_rates = tf.concat(
            [new_new_transition_rates[abrupt_change_state, :abrupt_change_state],
             directed_directed_transition_rate[None],
             new_new_transition_rates[abrupt_change_state,
                                      abrupt_change_state:abrupt_change_state + 1],
             new_new_transition_rates[abrupt_change_state,
                                      abrupt_change_state + 2:]], axis=0)
        new_new_transition_rates = tf.concat(
            (new_new_transition_rates[:abrupt_change_state + 1],
             second_directed_state_off_rates[None],
             new_new_transition_rates[abrupt_change_state + 1:]), 0)

        second_directed_state_on_shapes = tf.stack(
            [1] * abrupt_change_state
            + [directed_directed_transition_shape]
            + [1] * (nb_states - abrupt_change_state))
        new_new_transition_shapes = tf.concat(
            (new_transition_shapes[:, :abrupt_change_state + 1],
             second_directed_state_on_shapes[:, None],
             new_transition_shapes[:, abrupt_change_state + 1:]), 1)
        second_directed_state_off_shapes = tf.concat(
            [new_new_transition_shapes[abrupt_change_state, :abrupt_change_state],
             directed_directed_transition_shape[None],
             new_new_transition_shapes[abrupt_change_state, abrupt_change_state + 1:]],
            axis=0)
        new_new_transition_shapes = tf.concat(
            (new_new_transition_shapes[:abrupt_change_state + 1],
             second_directed_state_off_shapes[None],
             new_new_transition_shapes[abrupt_change_state + 1:]), 0)

        new_new_transition_rates = (
            tf.math.softmax(new_new_transition_rates, axis=1)
            * new_new_transition_shapes / reference_dt)
        new_new_transition_rates = tf.broadcast_to(
            new_new_transition_rates[None, None],
            (nb_time_points, nb_tracks) + new_new_transition_rates.shape)
        new_new_transition_rates = new_new_transition_rates * dts[..., None, None]

        return new_new_transition_shapes, new_new_transition_rates

    nb_obs_vars = 1
    nb_independent_vars = nb_dims
    nb_hidden_vars = 2
    nb_gaussians = nb_obs_vars + nb_hidden_vars
    nb_states = nb_states + 1

    inputs = tf.keras.Input(
        batch_shape=(batch_size, segment_length, nb_independent_vars),
        dtype=dtype)
    input_LocErrs = tf.keras.Input(
        batch_shape=(batch_size, segment_length, nb_independent_vars),
        name='Localization errors', dtype=dtype)
    input_dts = tf.keras.Input(
        batch_shape=(batch_size, segment_length + 1),
        name='frame durations', dtype=dtype)
    input_mask = tf.keras.Input(
        batch_shape=(batch_size, segment_length), dtype=dtype)
    input_isfirst = tf.keras.Input(
        batch_shape=(batch_size,), name='isfirsts', dtype=dtype)

    reshaped_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, None, :, None, None, :], dtype=dtype)(inputs)
    transposed_inputs = transpose_layer(dtype=dtype)(
        reshaped_inputs, perm=[2, 1, 0, 3, 4, 5])

    Init_layer = Initial_layer_constraints_abrupt_change(
        nb_states,
        nb_gaussians,
        nb_obs_vars,
        nb_hidden_vars,
        params,
        initial_params,
        initial_fractions,
        max_linking_distance,
        constraint_function,
        reference_dt,
        vary_params=vary_params,
        vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        sequence_length=sequence_length,
        LocErr_type=LocErr_type,
        dtype=dtype)

    tensor1, initial_states = Init_layer(transposed_inputs, input_LocErrs, input_dts)

    softmax_inv_Fractions = Init_layer.initial_fractions
    log_ds            = Init_layer.param_vars[:, 1]
    anomalous_factors = Init_layer.param_vars[:, 2]
    isdir             = Init_layer.param_vars[:, 4]

    (Prev_coefs, Prev_biases, LP,
     Log_factors, transition_Log_factors,
     reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
     reccurent_next_hidden_var_coefs, reccurent_biases,
     transition_hidden_var_coefs, transition_biases) = initial_states

    first_mask_layer = IsfirstMaskLayer(dtype=dtype)
    Prev_coefs  = first_mask_layer(Prev_coefs,  Init_layer.carryout_coefs,
                                   input_isfirst[None, :, None, None])
    Prev_biases = first_mask_layer(Prev_biases, Init_layer.carryout_biases,
                                   input_isfirst[None, :, None, None])
    LP          = first_mask_layer(LP,          Init_layer.carryout_LP,
                                   input_isfirst[:, None])

    sliced_inputs = tf.keras.layers.Lambda(
        lambda x: x[1:], dtype=dtype)(transposed_inputs)
    sliced_mask = tf.keras.layers.Lambda(
        lambda x: x[:, 1:], dtype=dtype)(input_mask)

    layer = Custom_RNN_layer(
        batch_size, transition_shapes, transition_rates,
        estimated_density, nb_states,
        Init_layer.recurrent_sequence_phase_1,
        Init_layer.recurrent_sequence_phase_2,
        Init_layer.transition_sequence,
        transition_param_function,
        sequence_length=sequence_length,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        carryover=True,
        dtype=dtype)

    (Prev_coefs, Prev_biases, LP, segment_len,
     gamma_dist_mean, gamma_dist_var,
     All_motion_states, All_coefs, All_biases, All_LPs,
     motion_states) = layer(
        sliced_inputs, input_dts, reference_dt, sliced_mask,
        Prev_coefs, Prev_biases, LP,
        Log_factors, transition_Log_factors,
        reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
        reccurent_next_hidden_var_coefs, reccurent_biases,
        transition_hidden_var_coefs, transition_biases,
        log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
        isfirst=input_isfirst)

    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]

    carryover_layer = CarryoverAssignLayer(
        carryout_variables=[
            Init_layer.carryout_coefs,
            Init_layer.carryout_biases,
            Init_layer.carryout_LP,
            layer.carryout_segment_len,
            layer.carryout_gamma_dist_mean,
            layer.carryout_gamma_dist_var,
        ],
        dtype=dtype)

    F_layer = Final_layer(
        Init_layer.final_sequence_phase_1,
        nb_dims=nb_independent_vars,
        sequence_length=sequence_length,
        dtype=dtype)

    outputs, All_states = F_layer(states)
    outputs = carryover_layer(
        outputs,
        [Prev_coefs, Prev_biases, LP,
         segment_len, gamma_dist_mean, gamma_dist_var])

    model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=outputs, name="Diffusion_model")
    pred_model = tf.keras.Model(
        inputs=(inputs, input_LocErrs, input_dts, input_mask, input_isfirst),
        outputs=(outputs, All_states, All_coefs, All_biases, All_LPs),
        name="Diffusion_model")

    return model, pred_model


def model_to_DataFrame(model, dt):
    weights = model.weights
    nb_states = weights[0].shape[0]
    anomalous_factors = (
        tf.sigmoid(weights[0][:, 2]) * (1 - weights[0][:, 4])
        + tf.exp(weights[0][:, 2]) * weights[0][:, 4]).numpy()
    params = {
        'anomalous factors':  anomalous_factors,
        'Localization errors': np.exp(weights[0][:, 0]),
        'd':                  np.exp(weights[0][:, 1]),
        'transition rates':   tf.math.softmax(weights[4], axis=1).numpy(),
        'transition shapes':  tf.math.exp(weights[5]).numpy(),
        'Fractions':          tf.math.softmax(weights[2][0]).numpy(),
    }
    colnames, data = [], []
    for state in range(nb_states):
        colnames.append(f'D{state}')
        data.append(params['d'][state] ** 2 / (2 * dt))
    for state in range(nb_states):
        colnames.append(f'Fraction {state}')
        data.append(params['Fractions'][state])
    for state in range(nb_states):
        colnames.append(f'Anomalous factor {state}')
        data.append(params['anomalous factors'][state])
    for state in range(nb_states):
        colnames.append(f'Model type state {state}')
        data.append(['Confined', 'directed'][int(weights[0][:, 4][state])])
    for state in range(nb_states):
        colnames.append(f'Localization error {state}')
        data.append(params['Localization errors'][state])
    return pd.DataFrame([data], columns=colnames)