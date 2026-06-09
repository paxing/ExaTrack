# -*- coding: utf-8 -*-
"""
models.py
---------
Model assembly, loss function, and parameter utilities.

PyTorch conversion notes
------------------------
- tf.keras.Model (functional API) → nn.Module with forward()
- tf.keras.Input → removed (no graph-mode input specs needed)
- tf.keras.layers.Lambda → inline Python ops in forward()
- model.weights[i] → named attribute access (model.init_layer.param_vars etc.)
- tf.keras.callbacks.Callback → plain Python class with on_epoch_end()
- MLE_loss(y_true, y_pred) → MLE_loss(y_pred)  (y_true unused)
- tf.math.reduce_max → torch.max / .max()
- tf.math.reduce_sum → torch.sum
- tf.math.softmax → torch.softmax
- tf.math.exp / tf.sigmoid / tf.math.log → torch equivalents
- tf.concat → torch.cat
- tf.stack → torch.stack
- tf.broadcast_to → .expand()
- build_segment_model returns (model, model); use return_all=True for pred outputs
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from .config import dtype, minval
from .constraints import (constraint_function,
                          transition_param_function as _default_tpf)
from .layers import (
    Initial_layer_constraints,
    Custom_RNN_layer,
    Final_layer,
    IsfirstMaskLayer,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def MLE_loss(y_pred):
    max_LP     = y_pred.max(dim=1, keepdim=True).values
    reduced_LP = y_pred - max_LP
    pred       = torch.log(torch.exp(reduced_LP).sum(dim=1, keepdim=True)) + max_LP
    return -pred.mean()


# ---------------------------------------------------------------------------
# Parameter logger (replaces tf.keras.callbacks.Callback)
# ---------------------------------------------------------------------------

class get_parameters:
    """
    Prints model parameters at epoch end.
    Call on_epoch_end(epoch) manually from the PyTorch training loop.
    """
    def __init__(self, model, track_segmentation=True, layer_name='params'):
        self.model              = model
        self.layer_name         = layer_name
        self.track_segmentation = track_segmentation

    def on_epoch_end(self, epoch, logs=None):
        param_vars    = self.model.init_layer.param_vars.detach()
        init_frac     = self.model.init_layer.initial_fractions.detach()
        ts_raw        = self.model.rnn_layer.transition_shapes.detach()
        tr_raw        = self.model.rnn_layer.transition_rates.detach()

        t_shapes = torch.exp(ts_raw)
        t_rates  = torch.softmax(tr_raw, dim=1) * t_shapes
        t_rates  = np.round(t_rates.cpu().numpy(), 3)
        t_shapes = np.round(t_shapes.cpu().numpy(), 3)

        model_types     = np.clip(np.round(param_vars[:, -1].cpu().numpy()).astype(int), 0, 1)
        model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]
        ano    = param_vars[:, 2]
        is_dir = param_vars[:, 4]
        params = {
            'Model types': model_types_str,
            'anomalous factors': list(np.round(
                (torch.sigmoid(ano) * (1 - is_dir)
                 + 2 ** 0.5 * torch.exp(ano) * is_dir).cpu().numpy(), 4)),
            'Localization errors': list(np.round(
                np.exp(param_vars[:, 0].cpu().numpy()), 3)),
            'd': list(np.round(np.exp(param_vars[:, 1].cpu().numpy()), 3)),
            'anomalous variation': list(np.round(
                np.exp(param_vars[:, 3].cpu().numpy()), 5)),
            'transition rates':  [list(r) for r in t_rates],
            'transition shapes': [list(s) for s in t_shapes],
            'Fractions': list(np.round(
                torch.softmax(init_frac[0], dim=-1).cpu().numpy(), 3)),
        }
        print(params)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegmentModel(nn.Module):
    """
    Full segment model wrapping Initial_layer_constraints → Custom_RNN_layer
    → Final_layer.

    forward() returns log-likelihoods (shape: batch × nb_sequences).
    Pass return_all=True to also get All_states, All_coefs, All_biases, All_LPs.
    """

    def __init__(self, track_len, nb_states, params, initial_params,
                 transition_rates, transition_shapes, initial_fractions,
                 batch_size, reference_dt, nb_dims=2, sequence_length=3,
                 max_linking_distance=3, estimated_density=0.001,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None, vary_transition_shapes=None,
                 vary_transition_rates=None, nb_LocErr_dims=1,
                 LocErr_type='Linear',
                 init_layer_class=None,
                 transition_param_fn=None):
        super().__init__()

        self.track_len       = track_len
        self.nb_dims         = nb_dims
        self.sequence_length = sequence_length
        self.reference_dt    = reference_dt

        nb_obs_vars    = 1
        nb_hidden_vars = 2
        nb_gaussians   = nb_obs_vars + nb_hidden_vars

        if init_layer_class is None:
            init_layer_class = Initial_layer_constraints
        if transition_param_fn is None:
            transition_param_fn = _default_tpf

        self.init_layer = init_layer_class(
            nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
            params, initial_params, initial_fractions,
            max_linking_distance, constraint_function,
            reference_dt=reference_dt,
            vary_params=vary_params,
            vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            sequence_length=sequence_length,
            carryover=True,
            LocErr_type=LocErr_type)

        self.isfirst_mask = IsfirstMaskLayer()

        self.rnn_layer = Custom_RNN_layer(
            batch_size, transition_shapes, transition_rates,
            estimated_density, nb_states,
            self.init_layer.recurrent_sequence_phase_1,
            self.init_layer.recurrent_sequence_phase_2,
            self.init_layer.transition_sequence,
            transition_param_fn,
            sequence_length=sequence_length,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates,
            carryover=True)

        self.final_layer = Final_layer(
            self.init_layer.final_sequence_phase_1,
            nb_dims=nb_dims,
            sequence_length=sequence_length)

    def forward(self, inputs, input_LocErrs, input_dts, input_mask,
                input_isfirst, return_all=False):
        device = next(self.parameters()).device
        inputs         = inputs.to(device)
        input_LocErrs  = input_LocErrs.to(device)
        input_dts      = input_dts.to(device)
        input_mask     = input_mask.to(device)
        input_isfirst  = input_isfirst.to(device)

        # Reshape: (batch, track_len, nb_dims) → (track_len, 1, batch, 1, 1, nb_dims)
        reshaped   = inputs[:, None, :, None, None, :]
        transposed = reshaped.permute(2, 1, 0, 3, 4, 5)

        transposed, initial_states = self.init_layer(
            transposed, input_LocErrs, input_dts)

        (Prev_coefs, Prev_biases, LP,
         Log_factors, transition_Log_factors,
         reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
         reccurent_next_hidden_var_coefs, reccurent_biases,
         transition_hidden_var_coefs, transition_biases) = initial_states

        softmax_inv_Fractions = self.init_layer.initial_fractions
        log_ds            = self.init_layer.param_vars[:, 1]
        anomalous_factors = self.init_layer.param_vars[:, 2]
        isdir             = self.init_layer.param_vars[:, 4]

        Prev_coefs  = self.isfirst_mask(
            Prev_coefs,  self.init_layer.carryout_coefs,
            input_isfirst[None, :, None, None])
        Prev_biases = self.isfirst_mask(
            Prev_biases, self.init_layer.carryout_biases,
            input_isfirst[None, :, None, None])
        LP          = self.isfirst_mask(
            LP,          self.init_layer.carryout_LP,
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
            reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
            reccurent_next_hidden_var_coefs, reccurent_biases,
            transition_hidden_var_coefs, transition_biases,
            log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
            isfirst=input_isfirst)

        states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]
        outputs, All_states = self.final_layer(states)

        # Carryover assignments (detached — gradients must not flow through state)
        if self.init_layer.carryover_initialized:
            self.init_layer.carryout_coefs.data.copy_(Prev_coefs.detach())
            self.init_layer.carryout_biases.data.copy_(Prev_biases.detach())
            self.init_layer.carryout_LP.data.copy_(LP.detach())
        if self.rnn_layer.carryover:
            self.rnn_layer.carryout_segment_len.data.copy_(segment_len.detach())
            self.rnn_layer.carryout_gamma_dist_mean.data.copy_(
                gamma_dist_mean.detach())
            self.rnn_layer.carryout_gamma_dist_var.data.copy_(
                gamma_dist_var.detach())

        if return_all:
            return outputs, All_states, All_coefs, All_biases, All_LPs
        return outputs


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_segment_model(track_len, nb_states, params, initial_params,
                        transition_rates, transition_shapes, initial_fractions,
                        batch_size, reference_dt, nb_dims=2, sequence_length=3,
                        max_linking_distance=3, estimated_density=0.001,
                        vary_params=None, vary_initial_params=None,
                        vary_initial_fractions=None,
                        vary_transition_shapes=None, vary_transition_rates=None,
                        nb_LocErr_dims=1, LocErr_type='Linear'):

    model = SegmentModel(
        track_len, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt,
        nb_dims=nb_dims, sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        vary_params=vary_params, vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        nb_LocErr_dims=nb_LocErr_dims, LocErr_type=LocErr_type)
    return model, model


def build_abrupt_directed_motion_changes_model(
        segment_length, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt, nb_dims=2, sequence_length=3,
        max_linking_distance=3, estimated_density=0.001,
        abrupt_change_state=1,
        vary_params=None, vary_initial_params=None,
        vary_initial_fractions=None,
        vary_transition_shapes=None, vary_transition_rates=None,
        LocErr_type='Linear'):

    class Initial_layer_constraints_abrupt_change(Initial_layer_constraints):
        def duplicate_states(self, param_vars, initial_param_vars,
                             initial_fractions):
            param_vars = torch.cat(
                (param_vars[:abrupt_change_state + 1],
                 param_vars[abrupt_change_state:]), dim=0)
            initial_param_vars = torch.cat(
                (initial_param_vars[:abrupt_change_state + 1],
                 initial_param_vars[abrupt_change_state:]), dim=0)
            initial_fractions = torch.cat(
                (initial_fractions[:, :abrupt_change_state + 1],
                 torch.tensor([[1e-10]], dtype=dtype),
                 initial_fractions[:, abrupt_change_state + 1:]), dim=1)
            return param_vars, initial_param_vars, initial_fractions

    def _tpf(transition_shapes, transition_rates, density, Fs, effective_ds,
             dts, reference_dt, dtype):
        print('transition_shapes', transition_shapes)
        nb_states       = transition_shapes.shape[0]
        nb_time_points, nb_tracks = dts.shape

        dir_dir_shape = torch.exp(
            transition_shapes[abrupt_change_state, abrupt_change_state])
        dir_dir_rate  = (
            transition_rates[abrupt_change_state, abrupt_change_state] - 4)

        ts_exp = torch.exp(transition_shapes)

        _dev = ts_exp.device
        new_ts = torch.cat(
            (ts_exp, torch.ones(1, nb_states, dtype=dtype, device=_dev)), dim=0)
        new_ts = torch.cat(
            (new_ts, torch.ones(nb_states + 1, 1, dtype=dtype, device=_dev)), dim=1)

        mis_dwell = torch.tensor(
            [0.9 / nb_states] * nb_states, dtype=dtype, device=_dev)
        mis_dwell = torch.cat(
            (mis_dwell, torch.tensor([0.1], dtype=dtype, device=_dev)))[None]

        mis_rates = torch.log(
            1 - torch.exp(
                -0.5 * density
                * torch.sum(
                    Fs[None] * (effective_ds[:, None] ** 2
                                + effective_ds[None] ** 2) ** 0.5,
                    dim=0)[:, None]))

        new_tr = torch.cat((transition_rates, mis_rates), dim=1)
        new_tr = torch.cat((new_tr, mis_dwell), dim=0)

        on_rates = torch.stack(
            [torch.tensor(-10.0, dtype=dtype)] * abrupt_change_state
            + [dir_dir_rate]
            + [torch.tensor(-10.0, dtype=dtype)] * (nb_states - abrupt_change_state)
        )[:, None]
        new_new_tr = torch.cat(
            (new_tr[:, :abrupt_change_state + 1],
             on_rates,
             new_tr[:, abrupt_change_state + 1:]), dim=1)

        off_rates = torch.cat(
            [new_new_tr[abrupt_change_state, :abrupt_change_state],
             dir_dir_rate.unsqueeze(0),
             new_new_tr[abrupt_change_state,
                        abrupt_change_state:abrupt_change_state + 1],
             new_new_tr[abrupt_change_state,
                        abrupt_change_state + 2:]], dim=0)
        new_new_tr = torch.cat(
            (new_new_tr[:abrupt_change_state + 1],
             off_rates[None],
             new_new_tr[abrupt_change_state + 1:]), dim=0)

        on_shapes = torch.stack(
            [torch.tensor(1.0, dtype=dtype)] * abrupt_change_state
            + [dir_dir_shape]
            + [torch.tensor(1.0, dtype=dtype)] * (nb_states - abrupt_change_state)
        )
        new_new_ts = torch.cat(
            (new_ts[:, :abrupt_change_state + 1],
             on_shapes[:, None],
             new_ts[:, abrupt_change_state + 1:]), dim=1)
        off_shapes = torch.cat(
            [new_new_ts[abrupt_change_state, :abrupt_change_state],
             dir_dir_shape.unsqueeze(0),
             new_new_ts[abrupt_change_state, abrupt_change_state + 1:]], dim=0)
        new_new_ts = torch.cat(
            (new_new_ts[:abrupt_change_state + 1],
             off_shapes[None],
             new_new_ts[abrupt_change_state + 1:]), dim=0)

        new_new_tr = (torch.softmax(new_new_tr, dim=1)
                      * new_new_ts / reference_dt)
        new_new_tr = new_new_tr[None, None].expand(
            nb_time_points, nb_tracks,
            new_new_tr.shape[0], new_new_tr.shape[1])
        new_new_tr = new_new_tr * dts[..., None, None]

        return new_new_ts, new_new_tr

    nb_states_model = nb_states + 1
    model = SegmentModel(
        segment_length, nb_states_model, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt,
        nb_dims=nb_dims, sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        vary_params=vary_params, vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        LocErr_type=LocErr_type,
        init_layer_class=Initial_layer_constraints_abrupt_change,
        transition_param_fn=_tpf)
    return model, model


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def get_model_params(model, track_segmentation=False):
    param_vars = model.init_layer.param_vars.detach()
    init_frac  = model.init_layer.initial_fractions.detach()
    ts_raw     = model.rnn_layer.transition_shapes.detach()
    tr_raw     = model.rnn_layer.transition_rates.detach()

    t_shapes = torch.exp(ts_raw).cpu().numpy()
    t_rates  = (torch.softmax(tr_raw, dim=1)
                * torch.exp(ts_raw)).cpu().numpy()

    model_types     = param_vars[:, -1].cpu().numpy().astype(int)
    model_types_str = np.array(['Confined motion', 'Directed motion'])[model_types]

    ano    = param_vars[:, 2]
    is_dir = param_vars[:, 4]
    anomalous_factors = (
        torch.sigmoid(ano) * (1 - is_dir)
        + 2 ** 0.5 * torch.exp(ano) * is_dir).cpu().numpy()

    return {
        'Model types':         model_types_str,
        'anomalous factors':   anomalous_factors,
        'Localization errors': np.exp(param_vars[:, 0].cpu().numpy()),
        'd':                   np.exp(param_vars[:, 1].cpu().numpy()),
        'q':                   np.exp(param_vars[:, 3].cpu().numpy()),
        'transition rates':    t_rates,
        'transition shapes':   t_shapes,
        'Fractions':           torch.softmax(init_frac[0], dim=-1).cpu().numpy(),
    }


def get_model_raw_params(model, track_segmentation=True, return_dict=False):
    params            = model.init_layer.param_vars.detach().cpu().numpy().copy()
    initial_params    = model.init_layer.initial_param_vars.detach().cpu().numpy().copy()
    initial_fractions = model.init_layer.initial_fractions.detach().cpu().numpy().copy()
    transition_shapes = model.rnn_layer.transition_shapes.detach().cpu().numpy().copy()
    transition_rates  = model.rnn_layer.transition_rates.detach().cpu().numpy().copy()
    if return_dict:
        return {
            'params':            params,
            'initial_params':    initial_params,
            'initial_fractions': initial_fractions,
            'transition_shapes': transition_shapes,
            'transition_rates':  transition_rates,
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


def model_to_DataFrame(model, dt):
    param_vars = model.init_layer.param_vars.detach()
    init_frac  = model.init_layer.initial_fractions.detach()
    ts_raw     = model.rnn_layer.transition_shapes.detach()
    tr_raw     = model.rnn_layer.transition_rates.detach()

    nb_states = param_vars.shape[0]
    ano    = param_vars[:, 2]
    is_dir = param_vars[:, 4]
    anomalous_factors = (
        torch.sigmoid(ano) * (1 - is_dir)
        + torch.exp(ano) * is_dir).cpu().numpy()

    params = {
        'anomalous factors':   anomalous_factors,
        'Localization errors': np.exp(param_vars[:, 0].cpu().numpy()),
        'd':                   np.exp(param_vars[:, 1].cpu().numpy()),
        'transition rates':    torch.softmax(tr_raw, dim=1).cpu().numpy(),
        'transition shapes':   torch.exp(ts_raw).cpu().numpy(),
        'Fractions':           torch.softmax(init_frac[0], dim=-1).cpu().numpy(),
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
        data.append(['Confined', 'directed'][int(param_vars[state, 4])])
    for state in range(nb_states):
        colnames.append(f'Localization error {state}')
        data.append(params['Localization errors'][state])
    return pd.DataFrame([data], columns=colnames)
