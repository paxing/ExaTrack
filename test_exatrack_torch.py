# -*- coding: utf-8 -*-
"""
test_exatrack_torch.py
----------------------
TensorFlow (exatrack) vs PyTorch (exatrack_torch) comparison.
Structure mirrors test_exatrack_paxing.py.

API differences vs. the TensorFlow version
-------------------------------------------
- model.compile / model.fit  →  explicit PyTorch training loop
- model.get_weights()        →  get_model_raw_params(model)
- pred_model.predict(seq)    →  predict_all(model, seq, device)  [local helper]
- seq[i][0][k]  (TF Keras (X,y) wrapper)  →  seq[i][k].numpy()
"""

import sys
import os

# Windows: register conda-env DLL directories before torch is imported
# (no-op on Linux/macOS or when already on PATH via conda activate).
_env_root = os.path.dirname(sys.executable)
if hasattr(os, 'add_dll_directory'):
    for _d in [
        _env_root,
        os.path.join(_env_root, 'Library', 'bin'),
        os.path.join(_env_root, 'Library', 'mingw-w64', 'bin'),
        os.path.join(_env_root, 'Library', 'usr', 'bin'),
        os.path.join(_env_root, 'Scripts'),
        os.path.join(_env_root, 'lib', 'site-packages', 'torch', 'lib'),
    ]:
        if os.path.isdir(_d):
            os.add_dll_directory(_d)

import numpy as np
import torch
import tensorflow as tf
#import matplotlib
#matplotlib.use('Agg')   # non-interactive; script uses savefig, not show
import matplotlib.pyplot as plt
from matplotlib import cm
import random

import exatrack
import exatrack_torch

# ---------------------------------------------------------------------------
# Simulate tracks  (shared between both implementations)
# ---------------------------------------------------------------------------

track_len    = 500
nb_tracks    = 500
reference_dt = 0.02
LocErr       = 0.02
nb_dims      = 2

pu       = 0.02
pb       = 0.1
Ds       = np.array([0.0, 0.25])
dt       = 0.02
ds       = (2 * Ds * dt) ** 0.5
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),
    angular_Ds=np.array([0.0, 0.0]),
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.02],
                                [0.05, 0.00]]),
    shape_matrix=np.array([[0, 1],
                           [1, 0]]),
    LocErr_std=0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std=0.01,
    nb_sub_steps=10,
    nb_burning_steps=0,
    bleaching_rate=0.02)

# Plot simulated tracks
plt.figure(figsize=(15, 15))
lim     = 1
nb_rows = 4
IDs     = random.sample(list(np.arange(len(tracks))), nb_rows ** 2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID    = i * nb_rows + j
        track = tracks[ID]
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1],
                    c=cm.jet(np.linspace(0, 1, len(track))), s=8, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('simulated_tracks_torch.png')

track_list  = [tracks[i, all_masks[i].astype(bool)] for i in range(len(tracks))]
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)] for i in range(len(tracks))]
dt_list     = [all_dts[i, all_masks[i].astype(bool)] for i in range(len(tracks))]

# ---------------------------------------------------------------------------
# Model parameters  (shared initial values)
# ---------------------------------------------------------------------------

batch_size = 100

params = np.array([[np.log(0.2), np.log(0.01), np.log(0.01), np.log(0.0002), 1],
                   [np.log(0.2), np.log(0.1),  np.log(0.1),  np.log(0.001),  0]])
nb_states = len(params)

initial_params     = np.array([[np.log(60)]] * nb_states, dtype='float64')
initial_fractions  = np.array([[0] * nb_states + [-5.0]], dtype='float64')

transition_rates   = 3 * np.eye(nb_states, dtype='float64')
transition_rates[0, 0] = 2
transition_shapes  = np.zeros((nb_states, nb_states), dtype='float64')

print('softmax of transition_rates:',
      tf.math.softmax(transition_rates, 1).numpy())

vary_params             = np.ones(params.shape)
vary_params[:, -1]   = 0   # motion type is a design choice, not a free parameter

vary_initial_params     = True
vary_initial_fractions  = True
vary_transition_shapes  = False
vary_transition_rates   = np.ones(transition_rates.shape)

print('softmax of transition_rates (axis=0):',
      tf.math.softmax(transition_rates).numpy())

device_tf    = '/CPU:0'
device_torch = torch.device('cpu')
#device_tf    = '/GPU:0'
#device_torch = torch.device('cuda')
estimated_density    = 0.00001
nb_dims              = 2
sequence_length      = 5
max_linking_distance = 1
segment_length       = 10

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

seq_tf = exatrack.TrackSegmentSequence(
    track_list,
    LocErr_list=LocErr_list,
    dt_list=dt_list,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)

seq_torch = exatrack_torch.TrackSegmentSequence(
    track_list,
    LocErr_list=LocErr_list,
    dt_list=dt_list,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)

nb_batches = len(seq_tf)

# ---------------------------------------------------------------------------
# Build training models
# ---------------------------------------------------------------------------

model_tf, pred_model_tf = exatrack.build_segment_model(
    segment_length, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    batch_size, reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    vary_params=vary_params, vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates,
    nb_LocErr_dims=nb_dims, LocErr_type='Linear')

model_torch, pred_model_torch = exatrack_torch.build_segment_model(
    segment_length, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    batch_size, reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    vary_params=vary_params, vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates,
    nb_LocErr_dims=nb_dims, LocErr_type='Linear')

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

learning_rate   = 0.005
epochs          = 30
epoch_decay     = 50
decay_threshold = epoch_decay * nb_batches
decay_rate      = 0.005

print('Final learning rate:',
      learning_rate * np.exp(-max(0, epochs - epoch_decay) * decay_rate * nb_batches))

# ---- TensorFlow training ---------------------------------------------------
lr_tf        = exatrack.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold)
optimizer_tf = tf.keras.optimizers.Adam(learning_rate=lr_tf, beta_1=0.99, beta_2=0.999, clipvalue=1.0)
model_tf.compile(loss=exatrack.MLE_loss, optimizer=optimizer_tf, jit_compile=False)

with tf.device(device_tf):
    history_tf = model_tf.fit(
        seq_tf, epochs=epochs,
        callbacks=[exatrack.get_parameters(track_segmentation=True)],
        shuffle=False, verbose=1)

# ---- PyTorch training ------------------------------------------------------
lr_torch       = exatrack_torch.WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold)
callback_torch = exatrack_torch.get_parameters(model_torch, track_segmentation=True)
optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=lr_torch(0),
                                   betas=(0.99, 0.999), eps=1e-7)
model_torch.to(device_torch)
loss_history_torch = []
global_step = 0
for epoch in range(epochs):
    model_torch.train()
    epoch_losses = []
    for batch_idx in range(len(seq_torch)):
        tr, le, dd, ma, fi = [t.to(device_torch) for t in seq_torch[batch_idx]]
        for pg in optimizer_torch.param_groups:
            pg['lr'] = lr_torch(global_step)
        optimizer_torch.zero_grad()
        out  = model_torch(tr, le, dd, ma, fi)
        loss = exatrack_torch.MLE_loss(out)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model_torch.parameters(), 1.0)
        optimizer_torch.step()
        epoch_losses.append(loss.item())
        global_step += 1
    epoch_loss = float(np.mean(epoch_losses))
    loss_history_torch.append(epoch_loss)
    print(f'Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}')
    callback_torch.on_epoch_end(epoch)


plt.figure()
plt.plot( history_tf.history.get('loss', []))
plt.plot(loss_history_torch)
plt.legend(['TF','PyTorch'])
plt.savefig('loss.png')
# ---------------------------------------------------------------------------
# Extract fitted parameters
# TF:     model.get_weights() → indexed list
# PyTorch: get_model_raw_params(model) → named tuple
# ---------------------------------------------------------------------------

weights_tf = model_tf.get_weights()

(params_fit_torch, initial_params_fit_torch, initial_fractions_fit_torch,
 transition_shapes_fit_torch, transition_rates_fit_torch) = exatrack_torch.get_model_raw_params(
    model_torch, track_segmentation=True)

# ---------------------------------------------------------------------------
# Rebuild models for full-sequence prediction
# ---------------------------------------------------------------------------

max_track_len = np.max([len(track) for track in track_list])

seq_full_tf = exatrack.TrackSegmentSequence(
    track_list, LocErr_list=LocErr_list, dt_list=dt_list,
    batch_size=batch_size, segment_length=max_track_len,
    min_segment_length=4, cutoff_batch_treshhold=0.5)

seq_full_torch = exatrack_torch.TrackSegmentSequence(
    track_list, LocErr_list=LocErr_list, dt_list=dt_list,
    batch_size=batch_size, segment_length=max_track_len,
    min_segment_length=4, cutoff_batch_treshhold=0.5)

model_tf, pred_model_tf = exatrack.build_segment_model(
    max_track_len, nb_states,
    params=weights_tf[0], initial_params=weights_tf[1],
    transition_rates=weights_tf[7], transition_shapes=weights_tf[8],
    initial_fractions=weights_tf[2],
    batch_size=batch_size, reference_dt=reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    vary_params=vary_params, vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates,
    nb_LocErr_dims=nb_dims, LocErr_type='Linear')

model_torch, pred_model_torch = exatrack_torch.build_segment_model(
    max_track_len, nb_states,
    params=params_fit_torch, initial_params=initial_params_fit_torch,
    transition_rates=transition_rates_fit_torch,
    transition_shapes=transition_shapes_fit_torch,
    initial_fractions=initial_fractions_fit_torch,
    batch_size=batch_size, reference_dt=reference_dt,
    nb_dims=nb_dims, sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    vary_params=vary_params, vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates,
    nb_LocErr_dims=nb_dims, LocErr_type='Linear')

exatrack.get_model_params(model_tf, track_segmentation=True)
exatrack_torch.get_model_params(model_torch, track_segmentation=True)

# ---------------------------------------------------------------------------
# Prediction helper  (replaces pred_model.predict() for PyTorch)
# ---------------------------------------------------------------------------

def predict_all(model, seq, device):
    """Run model(return_all=True) over all batches; return concatenated numpy arrays."""
    model.eval()
    LPs_l, preds_l, coefs_l, biases_l, lps_l = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(seq)):
            tr, le, dd, ma, fi = [t.to(device) for t in seq[i]]
            lp, st, co, bi, lps = model(tr, le, dd, ma, fi, return_all=True)
            LPs_l.append(lp.cpu().numpy())
            preds_l.append(st.cpu().numpy())
            coefs_l.append(co.cpu().numpy())
            biases_l.append(bi.cpu().numpy())
            lps_l.append(lps.cpu().numpy())
    return (np.concatenate(LPs_l,    axis=0),
            np.concatenate(preds_l,  axis=0),
            np.concatenate(coefs_l,  axis=0),
            np.concatenate(biases_l, axis=0),
            np.concatenate(lps_l,    axis=0))


# TF prediction
LPs_tf, preds_tf, All_coefs_tf, All_biases_tf, All_LPs_tf = \
    pred_model_tf.predict(seq_full_tf)

# PyTorch prediction
LPs_torch, preds_torch, All_coefs_torch, All_biases_torch, All_LPs_torch = \
    predict_all(pred_model_torch, seq_full_torch, device_torch)

# ---------------------------------------------------------------------------
# Collect raw arrays from sequences
# TF:     seq[i][0][k]      (Keras (X, y) wrapper — X is a 5-tuple)
# PyTorch: seq[i][k].numpy() (direct 5-tuple, no (X,y) wrapper)
# ---------------------------------------------------------------------------

tracks_tf     = np.concatenate([seq_full_tf[i][0][0] for i in range(len(seq_full_tf))], 0)
LocErrs_tf    = np.concatenate([seq_full_tf[i][0][1] for i in range(len(seq_full_tf))], 0)
time_steps_tf = np.concatenate([seq_full_tf[i][0][2] for i in range(len(seq_full_tf))], 0)
masks_tf      = np.concatenate([seq_full_tf[i][0][3] for i in range(len(seq_full_tf))], 0)

tracks_torch     = np.concatenate([seq_full_torch[i][0].numpy() for i in range(len(seq_full_torch))], 0)
LocErrs_torch    = np.concatenate([seq_full_torch[i][1].numpy() for i in range(len(seq_full_torch))], 0)
time_steps_torch = np.concatenate([seq_full_torch[i][2].numpy() for i in range(len(seq_full_torch))], 0)
masks_torch      = np.concatenate([seq_full_torch[i][3].numpy() for i in range(len(seq_full_torch))], 0)

# ---------------------------------------------------------------------------
# Plot state predictions
# ---------------------------------------------------------------------------

colors = np.array([[1, 0, 0],
                   [0, 0, 1]])

plt.figure(figsize=(15, 15))
plt.title('ExaTrack (TF) state predictions')
lim     = 0.8
nb_rows = 6
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_tf[ID]
        track = tracks_tf[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = preds_tf[ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=3, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('state_predictions_tf.png')

plt.figure(figsize=(15, 15))
plt.title('ExaTrack (PyTorch) state predictions')
lim     = 0.8
nb_rows = 6
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_torch[ID]
        track = tracks_torch[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = preds_torch[ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=3, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('state_predictions_torch.png')

# ---------------------------------------------------------------------------
# Forward-backward smoother
# ---------------------------------------------------------------------------

motion_types = list(params[:, 4]) + [0]

position_mean_tf, position_std_tf, anomalous_mean_tf, anomalous_std_tf, mean_preds_tf = \
    exatrack.extract_smooth_hidden_variables(
        tracks_tf, LocErrs_tf, time_steps_tf, masks_tf,
        pred_model_tf, batch_size, sequence_length, motion_types, reference_dt)

position_mean_torch, position_std_torch, anomalous_mean_torch, \
    anomalous_std_torch, mean_preds_torch = \
    exatrack_torch.extract_smooth_hidden_variables(
        tracks_torch, LocErrs_torch, time_steps_torch, masks_torch,
        pred_model_torch, batch_size, sequence_length, motion_types,
        reference_dt, device=str(device_torch))

# ---------------------------------------------------------------------------
# Plot refined positions
# ---------------------------------------------------------------------------

for track_ID in range(3):
    plt.figure()
    mask = masks_tf[track_ID, 1:-1].astype(bool)
    plt.title('Refined positions (TF), track %s' % track_ID)
    plt.errorbar(np.arange(len(position_mean_tf[track_ID, mask, 0])),
                 position_mean_tf[track_ID, mask, 0]
                 - np.mean(position_mean_tf[track_ID, mask, 0]),
                 yerr=position_std_tf[track_ID, mask, 0])
    plt.errorbar(np.arange(len(position_mean_tf[track_ID, mask, 1])),
                 position_mean_tf[track_ID, mask, 1]
                 - np.mean(position_mean_tf[track_ID, mask, 1]),
                 yerr=position_std_tf[track_ID, mask, 1])
    plt.xlim([-1, np.sum(mask)])
    plt.ylabel('Position')
    plt.xlabel('Time point')
    plt.legend(['x', 'y'])
    plt.savefig('refined_positions_tf_track_%s.png' % track_ID)

for track_ID in range(3):
    plt.figure()
    mask = masks_torch[track_ID, 1:-1].astype(bool)
    plt.title('Refined positions (PyTorch), track %s' % track_ID)
    plt.errorbar(np.arange(len(position_mean_torch[track_ID, mask, 0])),
                 position_mean_torch[track_ID, mask, 0]
                 - np.mean(position_mean_torch[track_ID, mask, 0]),
                 yerr=position_std_torch[track_ID, mask, 0])
    plt.errorbar(np.arange(len(position_mean_torch[track_ID, mask, 1])),
                 position_mean_torch[track_ID, mask, 1]
                 - np.mean(position_mean_torch[track_ID, mask, 1]),
                 yerr=position_std_torch[track_ID, mask, 1])
    plt.xlim([-1, np.sum(mask)])
    plt.ylabel('Position')
    plt.xlabel('Time point')
    plt.legend(['x', 'y'])
    plt.savefig('refined_positions_torch_track_%s.png' % track_ID)

# ---------------------------------------------------------------------------
# Velocity plot  (directed-state anomalous variable)
# ---------------------------------------------------------------------------

track_ID          = 0
directed_state_ID = 0

mask = masks_tf[track_ID, 1:-1].astype(bool)
plt.figure()
plt.title('Velocity assuming directed state (TF), track %s' % track_ID)
plt.plot((anomalous_mean_tf[track_ID, mask, directed_state_ID, 0] ** 2
          + anomalous_mean_tf[track_ID, mask, directed_state_ID, 1] ** 2) ** 0.5)
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')
plt.savefig('estimated_velocity_tf_track_%s.png' % track_ID)

mask = masks_torch[track_ID, 1:-1].astype(bool)
plt.figure()
plt.title('Velocity assuming directed state (PyTorch), track %s' % track_ID)
plt.plot((anomalous_mean_torch[track_ID, mask, directed_state_ID, 0] ** 2
          + anomalous_mean_torch[track_ID, mask, directed_state_ID, 1] ** 2) ** 0.5)
plt.xlabel('Time step')
plt.ylabel('Estimated velocity (um/time step)')
plt.savefig('estimated_velocity_torch_track_%s.png' % track_ID)

# ---------------------------------------------------------------------------
# State probabilities
# ---------------------------------------------------------------------------

track_ID = 0

mask = masks_tf[track_ID].astype(bool)
plt.figure()
plt.title('State probabilities (TF), track %s' % track_ID)
plt.plot(preds_tf[track_ID, mask])
plt.xlabel('Time step')
plt.ylabel('State probability')
plt.savefig('state_probabilities_tf_track_%s.png' % track_ID)

mask = masks_torch[track_ID].astype(bool)
plt.figure()
plt.title('State probabilities (PyTorch), track %s' % track_ID)
plt.plot(preds_torch[track_ID, mask])
plt.xlabel('Time step')
plt.ylabel('State probability')
plt.savefig('state_probabilities_torch_track_%s.png' % track_ID)

# ---------------------------------------------------------------------------
# Mean-preds state prediction grid
# ---------------------------------------------------------------------------

plt.figure(figsize=(15, 15))
plt.title('State predictions with mean_preds (TF)')
lim     = 2
nb_rows = 6
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_tf[ID]
        track = tracks_tf[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = mean_preds_tf[ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=8, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('state_predictions_mean_preds_tf.png')

plt.figure(figsize=(15, 15))
plt.title('State predictions with mean_preds (PyTorch)')
lim     = 2
nb_rows = 6
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_torch[ID]
        track = tracks_torch[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = mean_preds_torch[ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=8, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('state_predictions_mean_preds_torch.png')

# ---------------------------------------------------------------------------
# Refined positions grid
# ---------------------------------------------------------------------------

plt.figure(figsize=(15, 15))
plt.title('Refined particle positions (TF)')
lim     = 2
nb_rows = 6
IDs     = random.sample(list(np.arange(len(tracks_tf))), nb_rows ** 2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_tf[ID, 1:-1]
        track = position_mean_tf[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = mean_preds_tf[:, 1:-1][ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=8, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('refined_positions_tf.png')

plt.figure(figsize=(15, 15))
plt.title('Refined particle positions (PyTorch)')
lim     = 2
nb_rows = 6
IDs     = random.sample(list(np.arange(len(tracks_torch))), nb_rows ** 2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID   = i * nb_rows + j
        mask = masks_torch[ID, 1:-1]
        track = position_mean_torch[ID, mask.astype(bool)]
        print(len(track))
        track = track - np.mean(track, 0, keepdims=True) + [[lim * i, lim * j]]
        p = mean_preds_torch[:, 1:-1][ID, mask.astype(bool)][:, :-1]
        plt.plot(track[:, 0], track[:, 1], ':k', alpha=0.5)
        plt.scatter(track[:, 0], track[:, 1], c=p @ colors, s=7)
        plt.scatter(track[0, 0], track[0, 1], c='k', s=8, marker='x')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('refined_positions_torch.png')


param_phys_torch=exatrack_torch.get_model_params(model_torch)

param_phys_tf=exatrack.get_model_params(model_tf, track_segmentation=True)

 