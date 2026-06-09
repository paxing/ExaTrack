# -*- coding: utf-8 -*-
"""
layers.py
---------
All nn.Module classes for the ExaTrack model.

Layers
------
transpose_layer           : utility permute layer
IsfirstMaskLayer          : selects between fresh-init and carry-over state
CarryoverAssignLayer      : saves hidden state to buffers between batches
Initial_layer_constraints : first time step — initialises parameters, runs t=0
Custom_RNN_layer          : forward algorithm loop over all subsequent time steps
Final_layer               : integrates remaining hidden variables, outputs likelihood

PyTorch conversion notes
------------------------
- tf.keras.layers.Layer → torch.nn.Module
- build() method merged into __init__() (PyTorch builds on first forward)
- call() → forward()
- tf.Variable(x, trainable=True) → nn.Parameter(torch.tensor(x))
- tf.Variable(x, trainable=False) → register_buffer('name', torch.tensor(x))
- tf.stop_gradient(x) → x.detach()
- tf.while_loop + tf.TensorArray → Python for loop + list accumulation
- tf.control_dependencies → no-op (eager mode; assign happens immediately)
- tf.repeat(x, n, axis=k) → x.repeat_interleave(n, dim=k)
- tf.one_hot(idx, depth) → F.one_hot(idx.long(), depth).to(dtype)
- tf.einsum → torch.einsum  (same notation)
- tf.clip_by_value → torch.clamp
- Gamma distribution: torch.distributions.Gamma + torch.igamma for CDF
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class transpose_layer(nn.Module):
    def forward(self, x, perm):
        """Permute tensor dimensions."""
        return x.permute(*perm)


class IsfirstMaskLayer(nn.Module):
    """Element-wise   init_val * isfirst + prev_val * (1 - isfirst)"""
    def forward(self, init_val, prev_val, isfirst):
        return init_val * isfirst + prev_val * (1 - isfirst)


class CarryoverAssignLayer(nn.Module):
    """
    Updates carryover buffer variables with new states.
    In PyTorch (eager mode), assignments happen immediately without
    control dependencies. Values are detached before assignment so
    gradients don't flow back through the carryover path.
    """
    def __init__(self, carryout_variable_names, parent_module, **kwargs):
        super().__init__(**kwargs)
        self.carryout_variable_names = carryout_variable_names
        self.parent_module = parent_module

    def forward(self, output, new_states):
        for name, state in zip(self.carryout_variable_names, new_states):
            buf = self.parent_module.get_buffer(name) if hasattr(
                self.parent_module, name) else None
            if buf is not None:
                buf.data.copy_(state.detach())
        return output


# ---------------------------------------------------------------------------
# RNN cell (one time step of the forward algorithm)
# ---------------------------------------------------------------------------

def RNN_cell(input_i, Prev_coefs, Prev_biases, LP, segment_len,
             reshaped_Log_factors, reshaped_transition_Log_factors,
             reccurent_obs_var_coefs, reccurent_hidden_var_coefs,
             reccurent_next_hidden_var_coefs, reccurent_biases,
             transition_hidden_var_coefs, transition_biases,
             sequence_phase_1, sequence_phase_2,
             transition_mask, transition_sequence,
             transition_mean, transition_var,
             gamma_dist_mean, gamma_dist_var, states, dt_ratios):

    nb_dims    = input_i.shape[-1]
    nb_tracks  = LP.shape[0]
    nb_states  = reccurent_hidden_var_coefs.shape[2]
    sequence_length = LP.shape[1] // nb_states

    # ---- 1. replicate each hypothesis for all possible next states ----------
    Prev_coefs2  = Prev_coefs.repeat_interleave(nb_states, dim=2)
    Prev_biases2 = Prev_biases.repeat_interleave(nb_states, dim=2)
    LP2          = LP.repeat_interleave(nb_states, dim=1)
    segment_len  = segment_len.repeat_interleave(nb_states, dim=1)

    # ---- 2. transition integration -----------------------------------------
    alternative_Prev_coefs  = torch.cat((Prev_coefs2, transition_hidden_var_coefs), dim=0)
    alternative_Prev_biases = torch.cat((Prev_biases2, transition_biases), dim=0)

    transition_Prev_coefs, transition_Prev_biases, LC = \
        transition_RNN_reccurence_formula(
            current_hidden_var_coefs=alternative_Prev_coefs,
            next_hidden_var_coefs=torch.zeros_like(alternative_Prev_coefs),
            biases=alternative_Prev_biases,
            transition_sequence=transition_sequence,
            nb_dims=nb_dims,
            dtype=dtype)

    LP2 = LP2 + LC * transition_mask + reshaped_Log_factors

    # ---- 3. Gamma dwell-time transition probabilities ----------------------
    current_shapes = gamma_dist_mean ** 2 / gamma_dist_var
    current_rates  = gamma_dist_mean / gamma_dist_var

    all_Prev_coefs = (  transition_Prev_coefs  * transition_mask[None, :, :, None]
                      + Prev_coefs2            * (1 - transition_mask[None, :, :, None]))
    all_prev_biases = (  transition_Prev_biases * transition_mask[None, :, :, None]
                       + Prev_biases2           * (1 - transition_mask[None, :, :, None]))

    x_gamma = segment_len + 0.5
    gamma_dist = torch.distributions.Gamma(
        concentration=current_shapes.clamp(min=1e-10),
        rate=current_rates.clamp(min=1e-10))
    gamma_probs = gamma_dist.log_prob(x_gamma.clamp(min=1e-10)).exp()
    # torch.igamma has no backward for its first arg (shape 'a') in most
    # PyTorch versions; detach only that arg.  The second arg (x) does have a
    # backward (PyTorch >= 1.10), so gradient flows through transition_rates.
    gamma_cdfs  = torch.igamma(current_shapes.clamp(min=1e-10).detach(),
                                (current_rates * x_gamma).clamp(min=0))

    transition_probas = torch.clamp(
        (gamma_probs + 1e-14) / (1 - gamma_cdfs + 1e-12),
        min=-19.0, max=1 - 1e-10)

    non_transition_probas = (1 - torch.clamp(
        torch.sum(
            (transition_probas * transition_mask).reshape(
                nb_tracks, nb_states * sequence_length, nb_states),
            dim=2),
        min=-19.0, max=1 - 1e-10)).repeat_interleave(nb_states, dim=1)

    transition_probas = (transition_probas * transition_mask
                         + non_transition_probas * (1 - transition_mask))
    all_LP = LP2 + torch.log(transition_probas)

    # ---- 4. fold observation into biases and run CGP recurrence ------------
    current_reccurent_obs_var_coefs         = reccurent_obs_var_coefs.repeat(1, 1, sequence_length * nb_states, 1)
    current_reccurent_hidden_var_coefs      = reccurent_hidden_var_coefs.repeat(1, 1, sequence_length * nb_states, 1)
    current_reccurent_next_hidden_var_coefs = reccurent_next_hidden_var_coefs.repeat(1, 1, sequence_length * nb_states, 1)
    current_reccurent_biases                = reccurent_biases.repeat(1, 1, sequence_length * nb_states, 1)

    current_hidden_var_coefs = torch.cat(
        (all_Prev_coefs, current_reccurent_hidden_var_coefs), dim=0)
    zero_tensor = torch.zeros_like(all_Prev_coefs)
    next_hidden_var_coefs = torch.cat(
        (zero_tensor, current_reccurent_next_hidden_var_coefs), dim=0)

    current_biases  = current_reccurent_biases.clone()
    current_biases  = current_biases + torch.sum(
        current_reccurent_obs_var_coefs[:, :, :, :, None] * input_i, dim=-2)
    biases = torch.cat((all_prev_biases, current_biases), dim=0)

    Next_coefs, Next_biases, LC = RNN_reccurence_formula(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        sequence_phase_1, sequence_phase_2,
        nb_dims=nb_dims, dtype=dtype)

    all_LP = all_LP + LC

    # ---- 5. reduce transition hypotheses back to sequence_length per state -
    reshaped_Next_coefs = Next_coefs.reshape(
        Next_coefs.shape[:2]
        + torch.Size([sequence_length * nb_states, nb_states])
        + Next_coefs.shape[-1:])

    transition_LPs = (
        (all_LP - 200 * (1 - transition_mask)).reshape(
            nb_tracks, sequence_length * nb_states, nb_states)
        - nb_dims * torch.log(
            torch.abs(reshaped_Next_coefs[0, :, :, :, 0]
                      * reshaped_Next_coefs[1, :, :, :, 1]) + 1e-20))

    max_transition_LPs = transition_LPs.max(dim=1, keepdim=True).values
    transition_Ps      = torch.exp(transition_LPs - max_transition_LPs)
    transition_weights = transition_Ps / transition_Ps.sum(dim=1, keepdim=True)

    transition_states = torch.sum(
        states[:, :, None] * transition_weights[:, :, :, None, None], dim=1)

    transition_Next_coefs = Next_coefs.reshape(
        Next_coefs.shape[:2]
        + torch.Size([sequence_length * nb_states, nb_states])
        + Next_coefs.shape[-1:])
    transition_Next_coefs = torch.sum(
        transition_Next_coefs * transition_weights[None, :, :, :, None], dim=2)

    transition_Next_biases = Next_biases.reshape(
        Next_biases.shape[:2]
        + torch.Size([sequence_length * nb_states, nb_states, nb_dims]))
    transition_Next_biases = torch.sum(
        transition_Next_biases * transition_weights[None, :, :, :, None], dim=2)

    transition_LPs = (
        torch.log(transition_Ps.sum(dim=1))
        + max_transition_LPs[:, 0]
        + nb_dims * torch.log(
            torch.abs(transition_Next_coefs[0, :, :, 0]
                      * transition_Next_coefs[1, :, :, 1]) + 1e-20))

    stable_weights = (1 - transition_mask).reshape(
        sequence_length * nb_states, nb_states)[None]
    stable_LPs = torch.sum(
        all_LP.reshape(nb_tracks, sequence_length * nb_states, nb_states)
        * stable_weights, dim=2)

    stable_states = torch.sum(
        states[:, :, None] * stable_weights[:, :, :, None, None], dim=2)

    stable_Next_coefs = torch.sum(
        Next_coefs.reshape(
            Next_coefs.shape[:2]
            + torch.Size([sequence_length * nb_states, nb_states])
            + Next_coefs.shape[-1:])
        * stable_weights[None, :, :, :, None], dim=3)

    stable_Next_biases = torch.sum(
        Next_biases.reshape(
            Next_biases.shape[:2]
            + torch.Size([sequence_length * nb_states, nb_states, nb_dims]))
        * stable_weights[None, :, :, :, None], dim=3)

    stable_segment_len = torch.sum(
        segment_len.reshape(nb_tracks, sequence_length * nb_states, nb_states)
        * stable_weights, dim=2)

    current_gamma_dist_mean = torch.cat([transition_mean, gamma_dist_mean], dim=1)
    current_gamma_dist_var  = torch.cat([transition_var,  gamma_dist_var],  dim=1)

    Next_coefs  = torch.cat([transition_Next_coefs,  stable_Next_coefs],  dim=2)
    Next_biases = torch.cat([transition_Next_biases, stable_Next_biases], dim=2)
    new_LP          = torch.cat([transition_LPs, stable_LPs], dim=1)
    current_segment_len = torch.cat(
        [torch.ones((nb_tracks, nb_states), dtype=dtype, device=LP.device),
         stable_segment_len + dt_ratios[:, None]], dim=1)
    Next_states = torch.cat([transition_states, stable_states], dim=1)

    # ---- 6. merge oldest slab back into the buffer -------------------------
    saved_Next_coefs  = Next_coefs[:,  :, :-nb_states * 2]
    saved_Next_biases = Next_biases[:, :, :-nb_states * 2]
    saved_LP          = new_LP[:, :-nb_states * 2]
    saved_segment_len = current_segment_len[:, :-nb_states * 2]
    saved_gamma_dist_mean = current_gamma_dist_mean[:, :-nb_states ** 2 * 2]
    saved_gamma_dist_var  = current_gamma_dist_var[:,  :-nb_states ** 2 * 2]
    saved_states      = Next_states[:, :-nb_states * 2]

    nb_prev_gaussians = Next_coefs.shape[0]

    last_Next_coefs = Next_coefs[:, :, -nb_states * 2:].reshape(
        nb_prev_gaussians, nb_tracks, 2, nb_states, Next_coefs.shape[-1])
    last_Next_biases = Next_biases[:, :, -nb_states * 2:].reshape(
        nb_prev_gaussians, nb_tracks, 2, nb_states, nb_dims)
    last_LP = (
        new_LP[:, -nb_states * 2:].reshape(nb_tracks, 2, nb_states)
        - nb_dims * torch.log(
            torch.abs(last_Next_coefs[0, :, :, :, 0]
                      * last_Next_coefs[1, :, :, :, 1]) + 1e-20))
    last_segment_len      = current_segment_len[:, -nb_states * 2:].reshape(
        nb_tracks, 2, nb_states)
    last_gamma_dist_mean  = current_gamma_dist_mean[:, -nb_states ** 2 * 2:].reshape(
        nb_tracks, 2, nb_states, nb_states)
    last_gamma_dist_var   = current_gamma_dist_var[:,  -nb_states ** 2 * 2:].reshape(
        nb_tracks, 2, nb_states, nb_states)
    last_states = Next_states[:, -nb_states * 2:].reshape(
        nb_tracks, 2, nb_states, sequence_length, nb_states)

    last_LP_max = last_LP.max(dim=1, keepdim=True).values
    last_P      = torch.exp(last_LP - last_LP_max)
    sum_last_P  = last_P.sum(dim=1, keepdim=True)

    weight_last_P  = torch.exp(last_LP - last_LP.max(dim=1, keepdim=True).values)
    last_weights   = weight_last_P / weight_last_P.sum(dim=1, keepdim=True)

    reduced_last_Next_coefs  = torch.sum(
        last_Next_coefs  * last_weights[None, :, :, :, None], dim=2)
    reduced_last_Next_biases = torch.sum(
        last_Next_biases * last_weights[None, :, :, :, None], dim=2)
    reduced_last_LPs = (
        torch.log(sum_last_P + 1e-100) + last_LP_max)[:, 0] + nb_dims * torch.log(
        torch.abs(reduced_last_Next_coefs[0, :, :, 0]
                  * reduced_last_Next_coefs[1, :, :, 1]) + 1e-20)
    reduced_last_segment_len     = torch.sum(last_segment_len * last_weights, dim=1)
    reduced_last_gamma_dist_mean = torch.sum(
        last_gamma_dist_mean * last_weights[:, :, :, None], dim=1)
    reduced_last_gamma_dist_var  = torch.sum(
        (last_gamma_dist_var
         + (last_gamma_dist_mean - reduced_last_gamma_dist_mean[:, None]) ** 2)
        * last_weights[:, :, :, None], dim=1)
    reduced_last_gamma_dist_mean = reduced_last_gamma_dist_mean.reshape(
        nb_tracks, nb_states ** 2)
    reduced_last_gamma_dist_var  = reduced_last_gamma_dist_var.reshape(
        nb_tracks, nb_states ** 2)
    reduced_last_states = torch.sum(
        last_states * last_weights[:, :, :, None, None], dim=1)

    new_Next_coefs  = torch.cat((saved_Next_coefs,  reduced_last_Next_coefs),  dim=2)
    new_Next_biases = torch.cat((saved_Next_biases, reduced_last_Next_biases), dim=2)
    new_LPs         = torch.cat((saved_LP,          reduced_last_LPs),         dim=1)
    new_segment_len = torch.cat((saved_segment_len, reduced_last_segment_len), dim=1)
    new_gamma_dist_mean = torch.cat((saved_gamma_dist_mean, reduced_last_gamma_dist_mean), dim=1)
    new_gamma_dist_var  = torch.cat((saved_gamma_dist_var,  reduced_last_gamma_dist_var),  dim=1)
    new_states = torch.cat((saved_states, reduced_last_states), dim=1)

    current_states = states[:, :, -1:]
    new_states = torch.cat((new_states, current_states), dim=2)[:, :, 1:]

    return (new_Next_coefs, new_Next_biases, new_LPs, new_segment_len,
            new_gamma_dist_mean, new_gamma_dist_var, new_states)


# ---------------------------------------------------------------------------
# Initial layer
# ---------------------------------------------------------------------------

class Initial_layer_constraints(nn.Module):
    def __init__(self, nb_states, nb_gaussians, nb_obs_vars, nb_hidden_vars,
                 params, initial_params, initial_fractions,
                 max_linking_distance, constraint_function, reference_dt,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 sequence_length=3, carryover=True,
                 LocErr_type='Linear', **kwargs):
        super().__init__()

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
        self.constraint_function = constraint_function
        self.sequence_length = sequence_length
        self.max_linking_distance_val = max_linking_distance
        self.register_buffer('vary_params', torch.tensor(vary_params, dtype=dtype))
        self.register_buffer('vary_initial_params', torch.tensor(vary_initial_params, dtype=dtype))
        self.register_buffer('vary_initial_fractions', torch.tensor(vary_initial_fractions, dtype=dtype))
        self.reference_dt = reference_dt
        self.carryover = carryover
        self.LocErr_type = LocErr_type

        # Trainable parameters
        self.param_vars = nn.Parameter(
            torch.tensor(params, dtype=dtype))
        self.initial_param_vars = nn.Parameter(
            torch.tensor(initial_params, dtype=dtype))
        self.initial_fractions = nn.Parameter(
            torch.tensor(initial_fractions, dtype=dtype))
        self.register_buffer('max_linking_distance_param',
                             torch.tensor(max_linking_distance, dtype=dtype))

        # Carryover buffers (non-trainable, persistent state)
        # Shapes are set in forward() on first call when input shapes are known;
        # pre-registered here with placeholder size 1.
        self.carryover_initialized = False

        # LocErr function
        if LocErr_type == 'Identity':
            self.LocErr_function = lambda LocErrs, LocErr_param: LocErrs
        elif LocErr_type == 'Linear':
            self.LocErr_function = lambda LocErrs, LocErr_param: LocErrs * LocErr_param
        elif LocErr_type == 'Photon':
            self.LocErr_function = lambda LocErrs, LocErr_param: LocErrs ** 0.5 * LocErr_param
        elif LocErr_type == 'Constant':
            self.LocErr_function = lambda LocErrs, LocErr_param: LocErrs * 0 + LocErr_param
        else:
            raise ValueError(
                "Wrong LocErr_type, can be 'Identity', 'Linear', 'Photon' or 'Constant'.")

        # Precompute integration sequences
        (self.initial_sequence_phase_1,
         self.initial_sequence_phase_2,
         self.recurrent_sequence_phase_1,
         self.recurrent_sequence_phase_2,
         self.final_sequence_phase_1,
         self.transition_sequence) = get_sequences(
            torch.tensor(params, dtype=dtype),
            torch.tensor(initial_params, dtype=dtype),
            constraint_function,
            nb_gaussians, nb_hidden_vars, dtype)

    def _init_carryover_buffers(self, nb_tracks, nb_hidden_vars_out, device):
        """Lazily register carryover buffers once we know the track batch size."""
        nb_sequences = self.sequence_length * (self.nb_states + 1)
        self.register_buffer('carryout_coefs',
            torch.zeros(nb_hidden_vars_out, nb_tracks, nb_sequences,
                        nb_hidden_vars_out, dtype=dtype, device=device))
        self.register_buffer('carryout_biases',
            torch.zeros(nb_hidden_vars_out, nb_tracks, nb_sequences,
                        nb_hidden_vars_out, dtype=dtype, device=device))
        self.register_buffer('carryout_LP',
            torch.zeros(nb_tracks, nb_sequences, dtype=dtype, device=device))
        self.carryover_initialized = True

    def _apply_constraint(self, param):
        """Clamp parameters to be at least log(minval)."""
        return torch.where(param >= float(np.log(minval)),
                           param,
                           torch.full_like(param, float(np.log(minval))))

    def forward(self, inputs, input_LocErrs, input_dts):
        nb_tracks = inputs.shape[2]
        nb_hidden_vars = self.nb_hidden_vars
        constraint_function = self.constraint_function
        reference_dt = self.reference_dt

        param_vars         = self._apply_constraint(self.param_vars)
        initial_param_vars = self._apply_constraint(self.initial_param_vars)
        initial_fractions  = torch.softmax(self.initial_fractions, dim=-1)

        vary_params          = self.vary_params
        vary_initial_params  = self.vary_initial_params
        vary_initial_fractions = self.vary_initial_fractions
        LocErr_function      = self.LocErr_function
        nb_dims              = inputs.shape[-1]
        nb_states            = self.nb_states

        param_vars = (vary_params * param_vars
                      + (1 - vary_params) * param_vars.detach())
        initial_param_vars = (vary_initial_params * initial_param_vars
                              + (1 - vary_initial_params)
                              * initial_param_vars.detach())
        initial_fractions = (vary_initial_fractions * initial_fractions
                             + (1 - vary_initial_fractions)
                             * initial_fractions.detach())

        param_vars, initial_param_vars, initial_fractions = self.duplicate_states(
            param_vars, initial_param_vars, initial_fractions)

        # Add mislinking state
        max_linking_distance = self.max_linking_distance_param
        _dev = param_vars.device
        param_vars = torch.cat(
            (param_vars,
             torch.stack([
                 param_vars[-1][0],
                 torch.log(max_linking_distance.to(dtype)),
                 torch.tensor(-15., dtype=dtype, device=_dev),
                 torch.log(torch.tensor(0.00001, dtype=dtype, device=_dev)),
                 torch.tensor(0., dtype=dtype, device=_dev),
             ]).unsqueeze(0)),
            dim=0)
        initial_param_vars = torch.cat(
            (initial_param_vars, initial_param_vars[-1:]), dim=0)
        nb_states = nb_states + 1

        (hidden_var_coefs, obs_var_coefs, Gaussian_stds, biases,
         initial_hidden_var_coefs, initial_obs_var_coefs,
         initial_Gaussian_stds, initial_biases,
         transition_hidden_var_coefs, transition_Gaussian_stds,
         transition_biases, integration_variable_index,
         Log_factors, initial_Log_factors,
         transition_Log_factors) = constraint_function(
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
            (nb_hidden_vars, nb_tracks, nb_states, nb_hidden_vars), dtype=dtype,
            device=inputs.device)

        transition_hidden_var_coefs = transition_hidden_var_coefs / transition_Gaussian_stds
        transition_biases           = transition_biases           / transition_Gaussian_stds

        sequence_length = self.sequence_length
        transition_hidden_var_coefs = transition_hidden_var_coefs.repeat(1, 1, 1, sequence_length * nb_states, 1)
        transition_biases = transition_biases.repeat(1, 1, 1, nb_states * sequence_length, 1)

        # First time step (t=0)
        biases_t0                   = reccurent_biases[0]
        obs_var_coefs_t0            = reccurent_obs_var_coefs[0]
        current_hidden_var_coefs_t0 = reccurent_hidden_var_coefs[0]
        next_hidden_var_coefs_t0    = reccurent_next_hidden_var_coefs[0]

        biases_t0        = biases_t0 + torch.sum(obs_var_coefs_t0[..., None] * inputs[0], dim=-2)
        initial_biases   = initial_biases + torch.sum(initial_obs_var_coefs[..., None] * inputs[0], dim=-2)

        current_hidden_var_coefs_t0 = torch.cat(
            (current_initial_hidden_var_coefs, current_hidden_var_coefs_t0), dim=0)
        next_hidden_var_coefs_t0 = torch.cat(
            (next_initial_hidden_var_coefs, next_hidden_var_coefs_t0), dim=0)
        biases_t0 = torch.cat((initial_biases, biases_t0), dim=0)

        current_hidden_var_coefs_t0 = current_hidden_var_coefs_t0.repeat(1, 1, sequence_length, 1)
        next_hidden_var_coefs_t0 = next_hidden_var_coefs_t0.repeat(1, 1, sequence_length, 1)
        biases_t0 = biases_t0.repeat(1, 1, sequence_length, 1)

        sequence_phase_1 = self.initial_sequence_phase_1
        sequence_phase_2 = self.initial_sequence_phase_2

        Next_coefs, Next_biases, LC = RNN_reccurence_formula(
            current_hidden_var_coefs_t0,
            next_hidden_var_coefs_t0,
            biases_t0,
            sequence_phase_1,
            sequence_phase_2,
            nb_dims,
            dtype=dtype)

        init_log_fractions = initial_fractions.log().repeat(1, sequence_length)
        init_log_factors = (nb_dims * initial_Log_factors).repeat(1, sequence_length)

        LP = (LC + init_log_factors + init_log_fractions
              + float(np.log(1 / sequence_length)))

        Log_factors            = nb_dims * Log_factors
        transition_Log_factors = nb_dims * transition_Log_factors

        # Initialize carryover buffers on first forward pass
        if self.carryover and not self.carryover_initialized:
            self._init_carryover_buffers(nb_tracks, Next_coefs.shape[0],
                                         device=inputs.device)

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

class Custom_RNN_layer(nn.Module):
    def __init__(self, nb_tracks, transition_shapes, transition_rates,
                 density, nb_states,
                 sequence_phase_1, sequence_phase_2, transition_sequence,
                 transition_param_function,
                 sequence_length=3,
                 vary_transition_shapes=None, vary_transition_rates=None,
                 carryover=False, **kwargs):
        super().__init__()

        nb_states_with_mislinking = nb_states + 1

        self.sequence_phase_1       = sequence_phase_1
        self.sequence_phase_2       = sequence_phase_2
        self.transition_sequence    = transition_sequence
        self.nb_states              = nb_states_with_mislinking
        self.sequence_length        = sequence_length
        self.nb_tracks              = nb_tracks
        self.transition_param_function = transition_param_function
        self.density                = density
        self.carryover              = carryover

        if vary_transition_rates is None:
            vary_transition_rates = torch.ones(
                torch.tensor(transition_rates).shape, dtype=dtype)
        if vary_transition_shapes is None:
            vary_transition_shapes = torch.ones(
                torch.tensor(transition_shapes).shape, dtype=dtype)

        self.register_buffer('vary_transition_shapes',
            vary_transition_shapes if isinstance(vary_transition_shapes, torch.Tensor)
            else torch.tensor(vary_transition_shapes, dtype=dtype))
        self.register_buffer('vary_transition_rates',
            vary_transition_rates if isinstance(vary_transition_rates, torch.Tensor)
            else torch.tensor(vary_transition_rates, dtype=dtype))

        self.transition_rates = nn.Parameter(
            torch.tensor(transition_rates, dtype=dtype))
        self.transition_shapes = nn.Parameter(
            torch.tensor(transition_shapes, dtype=dtype))

        # Transition mask: 1 where source_state != target_state
        indices_row = torch.tensor(
            list(np.arange(nb_states_with_mislinking)) * sequence_length,
            dtype=torch.int64).repeat_interleave(nb_states_with_mislinking)
        indices_col = torch.arange(nb_states_with_mislinking,
                                   dtype=torch.int64).repeat(
                                       nb_states_with_mislinking * sequence_length)
        transition_mask = ((indices_row - indices_col) != 0).to(dtype)[None]
        self.register_buffer('transition_mask', transition_mask)
        self.register_buffer('indices_row', indices_row)
        self.register_buffer('indices_col', indices_col)

        if carryover:
            self.register_buffer('carryout_segment_len',
                torch.zeros(nb_tracks, sequence_length * nb_states_with_mislinking, dtype=dtype))
            self.register_buffer('carryout_gamma_dist_mean',
                torch.zeros(nb_tracks, sequence_length * nb_states_with_mislinking ** 2, dtype=dtype))
            self.register_buffer('carryout_gamma_dist_var',
                torch.zeros(nb_tracks, sequence_length * nb_states_with_mislinking ** 2, dtype=dtype))

    def _apply_constraint(self, param):
        return torch.where(param >= float(np.log(minval)),
                           param,
                           torch.full_like(param, float(np.log(minval))))

    def forward(self, inputs, input_dts, reference_dt, mask,
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
        indices_row            = self.indices_row
        indices_col            = self.indices_col
        sequence_length        = self.sequence_length
        density                = self.density

        transition_rates  = self._apply_constraint(self.transition_rates)
        transition_shapes = self.transition_shapes

        transition_shapes = (self.vary_transition_shapes * transition_shapes
                             + (1 - self.vary_transition_shapes)
                             * transition_shapes.detach())
        transition_rates  = (self.vary_transition_rates  * transition_rates
                             + (1 - self.vary_transition_rates)
                             * transition_rates.detach())

        ds           = torch.exp(log_ds)
        Fs           = torch.softmax(softmax_inv_Fractions[0, :-1], dim=0)
        effective_ds = ds + 2 * torch.exp(anomalous_factors) * isdir

        dts_TN = input_dts.permute(1, 0)
        transition_shapes_full, transition_rates_full = self.transition_param_function(
            transition_shapes, transition_rates, density,
            Fs, effective_ds, dts_TN, reference_dt, dtype)

        oh_row = F.one_hot(indices_row, nb_states).to(dtype)
        oh_col = F.one_hot(indices_col, nb_states).to(dtype)
        oh_src = oh_col

        flat_Log_full       = torch.einsum('tns,ps->tnp', Log_factors,            oh_row)
        flat_trans_Log_full = torch.einsum('tns,ps->tnp', transition_Log_factors, oh_src)
        flat_Log_full = (flat_trans_Log_full * transition_mask
                         + flat_Log_full     * (1 - transition_mask))

        transition_rates_flat_full = torch.einsum(
            'tnij,pi,pj->tnp', transition_rates_full,  oh_row, oh_col)
        transition_shapes_flat = torch.einsum(
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

        flat_Log_seq        = flat_Log_full[1:]
        flat_trans_Log_seq  = flat_trans_Log_full[1:]
        transition_mean_seq = transition_mean_full[1:, :, :nb_states ** 2]
        transition_var_seq  = transition_var_full[1:,  :, :nb_states ** 2]

        segment_len     = torch.ones((nb_tracks, sequence_length * nb_states), dtype=dtype,
                                     device=inputs.device)
        gamma_dist_mean = transition_mean_full[0]
        gamma_dist_var  = transition_var_full[0]

        if self.carryover and isfirst is not None:
            br_isfirst_1 = isfirst[:, None].expand_as(segment_len)
            segment_len  = (br_isfirst_1 * segment_len
                            + (1 - br_isfirst_1) * self.carryout_segment_len)
            br_isfirst_2 = isfirst[:, None].expand_as(gamma_dist_mean)
            gamma_dist_mean = (br_isfirst_2 * gamma_dist_mean
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_mean)
            gamma_dist_var  = (br_isfirst_2 * gamma_dist_var
                               + (1 - br_isfirst_2) * self.carryout_gamma_dist_var)

        _dev = inputs.device
        states_indices = torch.arange(nb_states * sequence_length, dtype=torch.int64, device=_dev) % nb_states
        states_indices = states_indices.unsqueeze(1).expand(-1, sequence_length)
        states = F.one_hot(states_indices, num_classes=nb_states).to(dtype)
        states = states.unsqueeze(0).expand(nb_tracks, -1, -1, -1)

        nb_dims   = reccurent_biases.shape[4]
        num_steps = inputs.shape[0]

        All_states_list = []
        All_coefs_list  = []
        All_biases_list = []
        All_LP_list     = []

        for i in range(num_steps):
            log_w = LP - nb_dims * torch.log(
                torch.abs(Prev_coefs[0, :, :, 0]
                          * Prev_coefs[1, :, :, 1]) + 1e-20)
            max_log_w = log_w.max(dim=1, keepdim=True).values
            w = torch.exp(log_w - max_log_w)
            w = w / w.sum(dim=1, keepdim=True)
            pred_states = (w[:, :, None] * states[:, :, 0]).sum(dim=1, keepdim=True)

            All_states_list.append(pred_states)
            All_coefs_list.append(Prev_coefs)
            All_biases_list.append(Prev_biases)
            All_LP_list.append(LP)

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

        All_states = torch.stack(All_states_list, dim=0).permute(1, 0, 2, 3)[:, :, 0, :]
        All_coefs  = torch.stack(All_coefs_list,  dim=0).permute(2, 0, 3, 1, 4)
        All_biases = torch.stack(All_biases_list, dim=0).permute(2, 0, 3, 1, 4)
        All_LPs    = torch.stack(All_LP_list,     dim=0).permute(1, 0, 2)
        All_states = All_states[:, sequence_length - 1:]

        return (Prev_coefs, Prev_biases, LP, segment_len,
                gamma_dist_mean, gamma_dist_var,
                All_states, All_coefs, All_biases, All_LPs, states)


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------

class Final_layer(nn.Module):
    def __init__(self, sequence_phase_1, nb_dims, sequence_length, **kwargs):
        super().__init__()
        self.sequence_phase_1 = sequence_phase_1
        self.nb_dims          = nb_dims
        self.sequence_length  = sequence_length

    def forward(self, states):
        nb_dims = self.nb_dims
        Prev_coefs, Prev_biases, LP, All_states, last_states = states

        if Prev_coefs.shape[0] > 0:
            current_hidden_var_coefs = Prev_coefs
            zero_tensor = torch.zeros_like(Prev_coefs)
            next_hidden_var_coefs = zero_tensor
            biases = Prev_biases

            Next_coefs, Next_biases, LC = RNN_reccurence_formula(
                current_hidden_var_coefs,
                next_hidden_var_coefs,
                biases,
                self.sequence_phase_1,
                [[], []],
                nb_dims=nb_dims,
                dtype=dtype)
            LP = LP + LC

        log_weights     = LP
        max_log_weights = log_weights.max(dim=1, keepdim=True).values
        weights         = torch.exp(log_weights - max_log_weights)
        weights         = weights / weights.sum(dim=1, keepdim=True)
        pred_states     = torch.sum(weights[:, :, None, None] * last_states, dim=1)
        All_states      = torch.cat((All_states, pred_states), dim=1)
        output          = LP

        return output, All_states
