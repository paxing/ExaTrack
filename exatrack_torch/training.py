# -*- coding: utf-8 -*-
"""
training.py
-----------
High-level training and model selection utilities.

PyTorch conversion notes
------------------------
- tf.keras.optimizers.schedules.LearningRateSchedule → plain Python class
- tf.keras.optimizers.Adam(clipvalue=v) → torch.optim.Adam +
  torch.nn.utils.clip_grad_value_(model.parameters(), v)
- model.compile / model.fit → explicit training loop
- tf.device('/GPU:0') → torch.device('cuda:0')
- model.weights[i].assign(x) → model.init_layer.param_vars.data.copy_(...)
- model.predict(seq) → _model_predict(model, seq, device)
- history.history['loss'][-1] → tracked manually
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import dtype
from .models import (
    build_segment_model,
    MLE_loss,
    get_parameters,
    get_model_params,
    get_model_raw_params,
)
from .io import TrackSegmentSequence


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _tf_device_to_torch(device_str):
    """Convert TF device string ('/GPU:0', '/CPU:0') to torch.device."""
    s = device_str.upper()
    if 'GPU' in s:
        try:
            idx = int(device_str.split(':')[-1])
            return torch.device(f'cuda:{idx}')
        except Exception:
            return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

class WarmupLearningRateSchedule:
    """
    Linear warmup followed by exponential decay.

        lr(step) = peak_lr * (1 - exp(-step/warmup_steps))
                   * exp(-decay_rate * max(step - decay_start, 0))

    Parameters
    ----------
    warmup_steps : number of steps over which lr ramps up
    peak_lr      : maximum learning rate
    decay_rate   : exponential decay rate after decay_start
    decay_start  : step at which decay begins
    """

    def __init__(self, warmup_steps, peak_lr, decay_rate, decay_start):
        self.warmup_steps = warmup_steps
        self.peak_lr      = peak_lr
        self.decay_rate   = decay_rate
        self.decay_start  = decay_start

    def __call__(self, step):
        decay_step = max(step - self.decay_start, 0)
        return (self.peak_lr
                * (1 - math.exp(-step / max(self.warmup_steps, 1e-8)))
                * math.exp(-self.decay_rate * decay_step))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_model_params(model, params_fit, initial_params_fit,
                      initial_fractions_fit,
                      transition_rates_fit, transition_shapes_fit):
    """Write numpy arrays back into model parameter tensors."""
    model.init_layer.param_vars.data.copy_(
        torch.tensor(params_fit, dtype=dtype))
    model.init_layer.initial_param_vars.data.copy_(
        torch.tensor(initial_params_fit, dtype=dtype))
    model.init_layer.initial_fractions.data.copy_(
        torch.tensor(initial_fractions_fit, dtype=dtype))
    model.rnn_layer.transition_rates.data.copy_(
        torch.tensor(transition_rates_fit, dtype=dtype))
    model.rnn_layer.transition_shapes.data.copy_(
        torch.tensor(transition_shapes_fit, dtype=dtype))


def _model_predict(model, seq, device, return_all=False):
    """Run the model on every batch in seq and collect outputs."""
    model.eval()
    all_outputs = []
    extras = [[], [], [], []]
    with torch.no_grad():
        for i in range(len(seq)):
            batch = seq[i]
            tracks, LocErrs, dts, masks, isfirsts = [t.to(device) for t in batch]
            out = model(tracks, LocErrs, dts, masks, isfirsts,
                        return_all=return_all)
            if return_all:
                lp, states, coefs, biases, lps = out
                all_outputs.append(lp.cpu())
                for k, x in enumerate([states, coefs, biases, lps]):
                    extras[k].append(x.cpu())
            else:
                all_outputs.append(out.cpu())
    outputs = torch.cat(all_outputs, dim=0)
    if return_all:
        return outputs, *[torch.cat(e, dim=0) for e in extras]
    return outputs


def _train_one_pass(model, seq, lr_schedule, beta_1, beta_2,
                    clip_value, epochs, device,
                    param_callback=None, verbose=1):
    """
    Run a standard PyTorch training loop over seq for `epochs` epochs.

    Returns
    -------
    loss_history : list of per-epoch mean losses
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_schedule(0),
        betas=(beta_1, beta_2), eps=1e-7)
    model.train()
    model.to(device)
    loss_history = []
    global_step  = 0

    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx in range(len(seq)):
            batch = seq[batch_idx]
            tracks, LocErrs, dts, masks, isfirsts = [t.to(device) for t in batch]

            # Update lr
            new_lr = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            optimizer.zero_grad()
            outputs = model(tracks, LocErrs, dts, masks, isfirsts)
            loss = MLE_loss(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        epoch_loss = float(np.mean(epoch_losses))
        loss_history.append(epoch_loss)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}")
        if param_callback is not None:
            param_callback.on_epoch_end(epoch)

    return loss_history


# ---------------------------------------------------------------------------
# Model_finder
# ---------------------------------------------------------------------------

def Model_finder(track_list, reference_dt, sequence_length,
                 nb_states, params, initial_params, initial_fractions,
                 transition_shapes, transition_rates,
                 max_linking_distance, estimated_density,
                 epochs, batch_size,
                 LocErr_list=None, dt_list=None,
                 segment_length=10, learning_rate=1/500,
                 decay_threshold=500, decay_rate=0.01,
                 device='/GPU:0', verbose=1, shuffle=False,
                 vary_params=None, vary_initial_params=None,
                 vary_initial_fractions=None,
                 vary_transition_shapes=None, vary_transition_rates=None,
                 track_segmentation=False, LocErr_type='Linear'):
    """
    Fit an ExaTrack model with automatic handling of motion-type local optima.

    The motion type (confined vs directed) is a near-discrete parameter that
    gradient descent can get stuck on. This function trains the model once
    with the initial parameter values, then for each state tries flipping the
    motion type and retraining. The model with the highest log-likelihood is
    kept.
    """
    nb_states = params.shape[0]
    torch_device = _tf_device_to_torch(device)

    if LocErr_list is None:
        LocErr_list = [np.ones(len(track)) for track in track_list]
        nb_LocErr_dims = 0
    elif LocErr_list[0].ndim == 2:
        nb_LocErr_dims = LocErr_list[0].shape[1]
    else:
        nb_LocErr_dims = 0

    if dt_list is None:
        dt_list = [np.ones(len(track)) for track in track_list]

    seq = TrackSegmentSequence(
        track_list, LocErr_list, dt_list,
        batch_size=batch_size, segment_length=segment_length,
        min_segment_length=4, cutoff_batch_treshhold=0.5,
        shuffle=shuffle)

    nb_batches = len(seq)
    nb_dims = track_list[0].shape[-1]
    initial_anomalous_factors = params[:, 2].copy()

    model, pred_model = build_segment_model(
        segment_length, nb_states, params, initial_params,
        transition_rates, transition_shapes, initial_fractions,
        batch_size, reference_dt, nb_dims=nb_dims,
        sequence_length=sequence_length,
        max_linking_distance=max_linking_distance,
        estimated_density=estimated_density,
        vary_params=vary_params, vary_initial_params=vary_initial_params,
        vary_initial_fractions=vary_initial_fractions,
        vary_transition_shapes=vary_transition_shapes,
        vary_transition_rates=vary_transition_rates,
        nb_LocErr_dims=nb_LocErr_dims, LocErr_type=LocErr_type)

    beta_1 = max(1 - 5 / nb_batches, 0.8)
    beta_2 = 1 - 0.1 / nb_batches
    lr_schedule = WarmupLearningRateSchedule(20, 0.01, decay_rate, decay_threshold)
    callback = get_parameters(model, track_segmentation=True)

    loss_history = _train_one_pass(
        model, seq, lr_schedule, beta_1, beta_2,
        clip_value=0.01, epochs=epochs,
        device=torch_device,
        param_callback=callback, verbose=verbose)

    All_models = {}
    params_fit, initial_params_fit, initial_fractions_fit, \
        transition_shapes_fit, transition_rates_fit = get_model_raw_params(
            model, track_segmentation=True)

    LogLikelihood = -loss_history[-1]
    All_models['Model 0'] = {
        'params': params_fit,
        'initial_params': initial_params_fit,
        'initial_fractions': initial_fractions_fit,
        'transition_shapes': transition_shapes_fit,
        'transition_rates': transition_rates_fit,
        'LogLikelihood': LogLikelihood,
        'loss_history': loss_history,
    }
    best_LogLikelihood = LogLikelihood
    best_model = 'Model 0'

    for i in range(nb_states):
        _set_model_params(model,
                          params_fit, initial_params_fit, initial_fractions_fit,
                          transition_rates_fit, transition_shapes_fit)
        model.init_layer.param_vars.data[i, 4] = (
            1 - model.init_layer.param_vars.data[i, 4])
        model.init_layer.param_vars.data[i, 2] = initial_anomalous_factors[i]

        lr_schedule2 = WarmupLearningRateSchedule(
            10, 0.01, decay_rate, decay_threshold)
        loss_history2 = _train_one_pass(
            model, seq, lr_schedule2, 0.9, 0.99,
            clip_value=1.0, epochs=epochs,
            device=torch_device,
            param_callback=callback, verbose=verbose)

        p, ip, iff, ts, tr = get_model_raw_params(model, track_segmentation=True)
        LogLikelihood = -loss_history2[-1]
        model_ID = len(All_models)
        All_models[f'Model {model_ID}'] = {
            'params': p, 'initial_params': ip,
            'initial_fractions': iff,
            'transition_shapes': ts, 'transition_rates': tr,
            'LogLikelihood': LogLikelihood, 'loss_history': loss_history2,
        }
        print('Log Likelihood', LogLikelihood)
        print('params', p)
        if LogLikelihood > best_LogLikelihood:
            best_model = f'Model {model_ID}'
            best_LogLikelihood = LogLikelihood

        best = All_models[best_model]
        params_fit         = best['params']
        initial_params_fit = best['initial_params']
        initial_fractions_fit = best['initial_fractions']
        transition_shapes_fit = best['transition_shapes']
        transition_rates_fit  = best['transition_rates']

    _set_model_params(model,
                      params_fit, initial_params_fit, initial_fractions_fit,
                      transition_rates_fit, transition_shapes_fit)

    return model, pred_model


# ---------------------------------------------------------------------------
# get_number_of_states
# ---------------------------------------------------------------------------

def get_number_of_states(track_list,
                          params,
                          initial_params,
                          transition_shapes,
                          transition_rates,
                          initial_fractions,
                          reference_dt,
                          dt_list=None,
                          LocErr_list=None,
                          nb_dims=2,
                          sequence_length=10,
                          max_linking_distance=0.4,
                          estimated_density=0.001,
                          epochs=50,
                          epoch_decay=40,
                          learning_rate=0.02,
                          decay_rate=0.005,
                          batch_size=100,
                          vary_params=True,
                          vary_initial_params=True,
                          vary_initial_fractions=True,
                          vary_transition_shapes=False,
                          vary_transition_rates=True,
                          device='/GPU:0',
                          track_segmentation=True,
                          segment_length=10,
                          LocErr_type='Linear'):
    """Automatically determine the number of motion states using top-down selection.

    Trains models starting from params.shape[0] states down to 1, progressively
    removing the state with the smallest influence on the likelihood at each step.
    Reports AIC and BIC for each model size.

    Parameters
    ----------
    vary_*            : True/False or a float array of the appropriate shape.
    epoch_decay       : epoch at which learning-rate decay begins.
    track_segmentation: if False, segment_length is set to the longest track.

    Returns
    -------
    model_results : dict keyed by nb_states, each containing:
        log_likelihood, aic, bic, num_params, loss_history,
        parameters, raw_parameters, model, pred_model
    """
    nb_states = params.shape[0]
    torch_device = _tf_device_to_torch(device)

    if not track_segmentation:
        segment_length = int(np.max([len(t) for t in track_list]))

    if vary_params is True:
        vary_params = np.ones((nb_states, 5))
    elif vary_params is False:
        vary_params = np.zeros((nb_states, 5))

    if vary_initial_params is True:
        vary_initial_params = np.ones((nb_states, 1))
    elif vary_initial_params is False:
        vary_initial_params = np.zeros((nb_states, 1))

    if vary_initial_fractions is True:
        vary_initial_fractions = np.ones((1, nb_states + 1))
    elif vary_initial_fractions is False:
        vary_initial_fractions = np.zeros((1, nb_states + 1))

    if vary_transition_shapes is True:
        vary_transition_shapes = np.ones((nb_states, nb_states))
    elif vary_transition_shapes is False:
        vary_transition_shapes = np.zeros((nb_states, nb_states))

    if vary_transition_rates is True:
        vary_transition_rates = np.ones((nb_states, nb_states))
    elif vary_transition_rates is False:
        vary_transition_rates = np.zeros((nb_states, nb_states))

    if LocErr_list is None:
        LocErr_list = [np.ones(len(t)) for t in track_list]
    if dt_list is None:
        dt_list = [np.ones(len(t)) for t in track_list]

    nb_LocErr_dims = (LocErr_list[0].shape[1] if LocErr_list[0].ndim == 2 else 0)

    seq = TrackSegmentSequence(
        track_list, LocErr_list, dt_list,
        batch_size=batch_size, segment_length=segment_length,
        min_segment_length=4, cutoff_batch_treshhold=0.5)

    nb_batches     = len(seq)
    nb_tracks      = len(track_list)
    mask_array     = np.concatenate([seq[i][3].numpy() for i in range(nb_batches)], axis=0)
    nb_data_points = int(np.sum(mask_array[:, 1:]))
    decay_step     = epoch_decay * nb_batches

    model_results     = {}
    current_nb_states = nb_states
    current_params              = params.copy()
    current_initial_params      = initial_params.copy()
    current_initial_fractions   = initial_fractions.copy()
    current_transition_rates    = transition_rates.copy()
    current_transition_shapes   = transition_shapes.copy()

    while current_nb_states >= 1:
        print(f"\n{'='*60}")
        print(f"Training model with {current_nb_states} states")
        print(f"{'='*60}")

        model, pred_model = build_segment_model(
            segment_length, current_nb_states,
            current_params, current_initial_params,
            current_transition_rates, current_transition_shapes,
            current_initial_fractions, batch_size, reference_dt,
            nb_dims=nb_dims, sequence_length=sequence_length,
            max_linking_distance=max_linking_distance,
            estimated_density=estimated_density,
            vary_params=vary_params, vary_initial_params=vary_initial_params,
            vary_initial_fractions=vary_initial_fractions,
            vary_transition_shapes=vary_transition_shapes,
            vary_transition_rates=vary_transition_rates,
            nb_LocErr_dims=nb_LocErr_dims, LocErr_type=LocErr_type)

        model.to(torch_device)
        preds_init = _model_predict(model, seq, torch_device)
        print('Initial predictions:', MLE_loss(preds_init))

        cur_epochs     = 2 * epochs    if nb_states == current_nb_states else epochs
        lr_decay_start = decay_step * 2 if nb_states == current_nb_states else decay_step
        beta_1 = max(1 - 5 / nb_batches, 0.8)
        beta_2 = 1 - 0.2 / nb_batches
        lr_schedule = WarmupLearningRateSchedule(
            10, learning_rate, decay_rate, lr_decay_start)
        callback = get_parameters(model, track_segmentation=track_segmentation)

        loss_history = _train_one_pass(
            model, seq, lr_schedule, beta_1, beta_2,
            clip_value=1.0, epochs=cur_epochs,
            device=torch_device,
            param_callback=callback, verbose=1)

        final_preds    = _model_predict(model, seq, torch_device)
        log_likelihood = float(-MLE_loss(final_preds) * nb_tracks)

        num_params = (current_nb_states * 5
                      + current_nb_states * 1
                      + current_nb_states
                      + current_nb_states ** 2 * 2)

        aic = 2 * num_params - 2 * log_likelihood
        bic = np.log(nb_data_points) * num_params - 2 * log_likelihood

        (fitted_params, fitted_initial_params, fitted_initial_fractions,
         fitted_transition_shapes, fitted_transition_rates) = get_model_raw_params(
            model, track_segmentation=track_segmentation, return_dict=False)

        parameters     = get_model_params(model, track_segmentation)
        raw_parameters = get_model_raw_params(
            model, track_segmentation=track_segmentation, return_dict=True)

        model_results[current_nb_states] = {
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'num_params': num_params,
            'loss_history': loss_history,
            'parameters': parameters,
            'raw_parameters': raw_parameters,
            'model': model,
            'pred_model': pred_model,
        }
        print(f"Log-likelihood: {log_likelihood:.2f}")
        print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

        if current_nb_states <= 1:
            break

        state_influences = []
        for state_to_remove in range(current_nb_states):
            print(f"\nTesting removal of state {state_to_remove}")
            keep_states = [i for i in range(current_nb_states)
                           if i != state_to_remove]

            reduced_params         = fitted_params[keep_states]
            reduced_initial_params = fitted_initial_params[keep_states]

            fracs_softmax = torch.softmax(
                torch.tensor(fitted_initial_fractions[0], dtype=dtype), dim=0).numpy()
            reduced_fracs = fracs_softmax[keep_states + [current_nb_states]]
            reduced_fracs /= reduced_fracs.sum()
            reduced_initial_fractions = np.log(
                reduced_fracs / (1 - reduced_fracs)).reshape(1, -1)

            reduced_transition_rates  = fitted_transition_rates[
                np.ix_(keep_states, keep_states)]
            reduced_transition_shapes = fitted_transition_shapes[
                np.ix_(keep_states, keep_states)]

            test_model, _ = build_segment_model(
                segment_length, current_nb_states - 1,
                reduced_params, reduced_initial_params,
                reduced_transition_rates, reduced_transition_shapes,
                reduced_initial_fractions, batch_size, reference_dt,
                nb_dims=nb_dims, sequence_length=sequence_length,
                max_linking_distance=max_linking_distance,
                estimated_density=estimated_density,
                nb_LocErr_dims=nb_LocErr_dims, LocErr_type=LocErr_type)
            test_model.to(torch_device)
            test_preds      = _model_predict(test_model, seq, torch_device)
            test_likelihood = float(MLE_loss(test_preds))
            state_influences.append((state_to_remove, test_likelihood))
            print(f"  Likelihood without state {state_to_remove}: {test_likelihood:.2f}")

        state_influences.sort(key=lambda x: x[1])
        state_to_remove = state_influences[0][0]
        print(f"\n→ Removing state {state_to_remove} "
              f"(least influence: {state_influences[0][1]:.2f})")

        keep_states = [i for i in range(current_nb_states)
                       if i != state_to_remove]
        current_params         = fitted_params[keep_states]
        current_initial_params = fitted_initial_params[keep_states]

        fracs_softmax = torch.softmax(
            torch.tensor(fitted_initial_fractions[0], dtype=dtype), dim=0).numpy()
        reduced_fracs = fracs_softmax[keep_states + [current_nb_states]]
        reduced_fracs /= reduced_fracs.sum()
        current_initial_fractions = np.log(
            reduced_fracs / (1 - reduced_fracs)).reshape(1, -1)

        current_transition_rates  = fitted_transition_rates[
            np.ix_(keep_states, keep_states)]
        current_transition_shapes = fitted_transition_shapes[
            np.ix_(keep_states, keep_states)]

        vary_params            = vary_params[keep_states]
        vary_initial_params    = vary_initial_params[keep_states]
        vary_initial_fractions = vary_initial_fractions[
            :, keep_states + [current_nb_states]]
        vary_transition_shapes = vary_transition_shapes[
            np.ix_(keep_states, keep_states)]
        vary_transition_rates  = vary_transition_rates[
            np.ix_(keep_states, keep_states)]

        current_nb_states -= 1

    print(f"\n{'='*60}")
    print("Model Selection Results:")
    print(f"{'='*60}")
    for n_states in sorted(model_results.keys(), reverse=True):
        r = model_results[n_states]
        print(f"{n_states} states: LL={r['log_likelihood']:.1f}, "
              f"AIC={r['aic']:.1f}, BIC={r['bic']:.1f}")

    return model_results
