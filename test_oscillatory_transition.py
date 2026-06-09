# -*- coding: utf-8 -*-
"""
test_oscillatory_transition.py
-------------------------------
Test the oscillatory_constraint_function on tracks that switch between
confined, directed, and oscillatory motion states.

Mirrors test_exatrack_torch.py structure:
  - Simulate with anomalous_diff_transition_osc
  - Train with build_oscillatory_model
  - Predict with return_all=True (per-step state probabilities)
  - Visualise: colour-coded tracks, state probability traces, confusion matrix

State indices:
    0 = directed
    1 = confined
    2 = oscillatory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torch
from torch.utils.data import DataLoader
import random

from exatrack_torch.oscillatory_constraints import build_oscillatory_model
from exatrack_torch.models import MLE_loss
from exatrack_torch.training import WarmupLearningRateSchedule
from exatrack_torch.simulate_oscillatory import anomalous_diff_transition_osc

np.random.seed(42)
torch.manual_seed(42)

# ===========================================================================
# 1. Simulation parameters
# ===========================================================================
reference_dt  = 0.02
max_track_len = 200
nb_tracks     = 300
nb_dims       = 2
LocErr        = 0.02
sigma         = LocErr

# omega in rad/s — simulate_oscillatory converts to rad/sub-step internally via omega*(dt/nb_sub_steps)
# period = 30 obs-steps at reference_dt=0.02 → period_s = 0.6s → omega = 2pi/0.6
omega    = 2*np.pi / (15 * reference_dt)    # rad/s  (passed to anomalous_diff_transition_osc)
omega_obs = omega * reference_dt            # rad/obs-step (used for model initial params)
A_osc = 0.8        # excursion ±0.09; comparable scale to confined blobs
# d_osc: small position noise that slightly deforms the orbit each step.
# q_osc_noise in simulation must be 0: adding noise to velocity inflates the
# orbit radius unboundedly (harmonic oscillator energy grows). The q parameter
# in the constraint function is a model uncertainty term, not a physical noise.
d_osc = 0.05       # small orbit shape perturbation (~1% of A per step)
q_osc = A_osc * omega_obs**2   # model initial velocity spread (constraint function)
q_osc_sim = 0.001     # simulation velocity noise — must be 0 to keep orbit stable

velocity   = 0.04   # drift speed per obs-step
D_con      = 0.5      # confined diffusion coefficient
conf_force = 0.5       # OU confinement force per obs-step
D_dir      = 0.0       # directed diffusion (pure drift)

# State 0 = directed, 1 = confined, 2 = oscillatory
all_tracks, all_LocErrs, all_dts, all_states, all_masks = \
    anomalous_diff_transition_osc(
        max_track_len    = max_track_len,
        nb_tracks        = nb_tracks,
        LocErr           = LocErr,
        LocErr_std       = 0.002,
        Fs               = np.array([0.33, 0.34, 0.33]),
        velocity         = velocity,
        angular_D        = 0.04,
        D_dir            = D_dir,
        D_con            = D_con,
        conf_force       = conf_force,
        conf_D           = 0.0,
        omega            = omega,
        A_osc            = A_osc,
        d_osc_noise      = d_osc,
        q_osc_noise      = q_osc_sim,
        # Rates in s^-1: _sample_transitions draws exponential waits in seconds.
        # rate=2.0 s^-1 → mean dwell = 0.5s = 25 obs-steps → ~8 transitions/200-step track
        transition_matrix = 0.66*np.array([[0.00, 1.00, 1.00],
                                       [1.00, 0.00, 1.00],
                                       [1.00, 1.00, 0.00]]),
        shape_matrix      = np.ones((3, 3)) - np.eye(3),
        field_of_view    = np.array([-5.0, 5.0]),
        nb_dims          = nb_dims,
        dt               = reference_dt,
        dt_std           = 0.002,
        nb_sub_steps     = 10,
        nb_burning_steps = 0,
        bleaching_rate   = 0.01,
        reference_dt     = reference_dt)

# Build variable-length track lists (mask out padding).
# Discard tracks shorter than 2 steps — the RNN needs at least one transition
# step (sliced_inputs = transposed[1:] must be non-empty).
_min_len = 2
track_list  = [all_tracks[i,  all_masks[i].astype(bool)]   for i in range(nb_tracks)
               if all_masks[i].sum() >= _min_len]
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]   for i in range(nb_tracks)
               if all_masks[i].sum() >= _min_len]
dt_list     = [all_dts[i,     all_masks[i].astype(bool)]   for i in range(nb_tracks)
               if all_masks[i].sum() >= _min_len]
state_list  = [all_states[i,  all_masks[i].astype(bool)]   for i in range(nb_tracks)
               if all_masks[i].sum() >= _min_len]
nb_tracks   = len(track_list)
print(f"Kept {nb_tracks} tracks with length >= {_min_len}")

track_lens = np.array([len(t) for t in track_list])
print(f"Simulated {nb_tracks} tracks, length {track_lens.min()}–{track_lens.max()} "
      f"(mean {track_lens.mean():.0f})")

# Overall state fraction
all_st = np.concatenate(state_list)
for k, name in enumerate(['Directed', 'Confined', 'Oscillatory']):
    frac = (all_st == k).mean()
    print(f"  {name}: {frac:.1%}")

# ===========================================================================
# 2. Visualise simulated tracks — colour by ground-truth state
# ===========================================================================
state_colors = np.array([[0.84, 0.15, 0.16],   # 0 directed   red
                          [0.12, 0.47, 0.71],   # 1 confined   blue
                          [0.17, 0.63, 0.17]])  # 2 oscillatory green

lim     = 3.0
nb_rows = 4
IDs     = random.sample(range(nb_tracks), nb_rows**2)

fig_sim, ax_sim = plt.subplots(figsize=(12, 12))
ax_sim.set_title('Simulated tracks — colour = ground-truth state\n'
                 '(red=directed, blue=confined, green=oscillatory)', fontsize=11)
for ii, ID in enumerate(IDs):
    row, col = divmod(ii, nb_rows)
    t = track_list[ID]
    s = state_list[ID]
    t_plot = t - t.mean(0) + np.array([lim*col, lim*row])
    for step in range(len(t)-1):
        ax_sim.plot(t_plot[step:step+2, 0], t_plot[step:step+2, 1],
                    color=state_colors[s[step]], lw=0.9, alpha=0.8)
    ax_sim.scatter(t_plot[0, 0], t_plot[0, 1], c='k', s=8, zorder=5)
ax_sim.set_aspect('equal'); ax_sim.axis('off')
plt.tight_layout()
plt.savefig('osc_transition_simulated.png', dpi=120, bbox_inches='tight')
plt.show()

# ===========================================================================
# 3. Build model
# ===========================================================================
nb_states  = 3
state_names = ['Directed', 'Confined', 'Oscillatory']
batch_size = 50
segment_length = 10

# params cols: [log_sigma, log_d, motion_param, log_q, is_dir, is_osc]
# State order: 0=directed, 1=confined, 2=oscillatory
params = np.array([
    [np.log(sigma*2), np.log(0.02),    np.log(0.02*0.5), np.log(0.001),      1.0, 0.0],
    [np.log(sigma*2), np.log(0.10*2),  0.0,               np.log(0.001*3),    0.0, 0.0],
    [np.log(sigma*2), np.log(d_osc*2), np.log(omega*1.5), np.log(q_osc*0.5),  0.0, 1.0],   # osc
], dtype='float64')

initial_params    = np.array([[np.log(5.0)]] * nb_states, dtype='float64')
initial_fractions = np.zeros((1, nb_states+1), dtype='float64')
initial_fractions[0, -1] = -5.0

transition_rates  = 3.0 * np.eye(nb_states, dtype='float64')
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')

vary_params = np.ones((nb_states, 6), dtype='float64')
#vary_params[0, 2] = 0.0   # fix log_v for directed
#vary_params[1, 2] = 0.0   # fix logit_l for confined
#vary_params[2, 2] = 0.0   # fix log_omega for oscillatory

vary_transition_rates  = np.ones((nb_states, nb_states), dtype='float64')
vary_transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')

model, pred_model = build_oscillatory_model(
    segment_length, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    batch_size, reference_dt,
    nb_dims=nb_dims, sequence_length=5,
    max_linking_distance=1, estimated_density=1e-5,
    vary_params=vary_params,
    vary_initial_params=np.ones((nb_states, 1)),
    vary_initial_fractions=np.ones((1, nb_states+1)),
    vary_transition_rates=vary_transition_rates,
    vary_transition_shapes=vary_transition_shapes,
    LocErr_type='Constant')

print(f"Model trainable params: {sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# 4. DataLoader — use TrackSegmentSequence
# ===========================================================================
from exatrack_torch import TrackSegmentSequence

seq = TrackSegmentSequence(
    track_list,
    LocErr_list=LocErr_list,
    dt_list=dt_list,
    batch_size=batch_size,
    segment_length=segment_length,
    min_segment_length=4,
    cutoff_batch_treshhold=0.5)

print(f"DataLoader: {len(seq)} batches × {batch_size} tracks")

# ===========================================================================
# 5. Training
# ===========================================================================
nb_epochs     = 30
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.train(); model.to(device)

nb_batches      = len(seq)
learning_rate   = 0.002
epoch_decay     = 5
decay_threshold = epoch_decay * nb_batches
decay_rate      = 0.01

lr_schedule = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold)
optimizer   = torch.optim.Adam(model.parameters(), lr=lr_schedule(0),
                                betas=(0.99, 0.999), eps=1e-7)

loss_history = []; best_loss = float('inf'); best_state_dict = None
global_step  = 0

print(f"\n{'Epoch':>6}  {'Loss':>10}  {'Best':>10}  {'LR':>10}")
print("-" * 44)

for epoch in range(nb_epochs):
    epoch_losses = []
    for batch in seq:
        sig_b, le_b, dt_b, mask_b, first_b = [
            torch.tensor(x, dtype=torch.float64).to(device) for x in batch]
        for pg in optimizer.param_groups:
            pg['lr'] = lr_schedule(global_step)
        optimizer.zero_grad()
        lp = model(sig_b, le_b, dt_b, mask_b, first_b)
        if torch.isnan(lp).any() or torch.isinf(lp).any():
            global_step += 1; continue
        loss = MLE_loss(lp)
        if torch.isnan(loss) or torch.isinf(loss):
            global_step += 1; continue
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
        global_step += 1

    epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
    loss_history.append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss       = epoch_loss
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    lr = optimizer.param_groups[0]['lr']
    print(f"{epoch+1:>6}  {epoch_loss:>10.4f}  {best_loss:>10.4f}  {lr:>10.2e}")

model.load_state_dict(best_state_dict, strict=False)
print(f"\nRestored best model (loss={best_loss:.4f})")

# ===========================================================================
# 6. Learned parameters — compared against simulation ground truth
# ===========================================================================
p = model.init_layer.param_vars.detach().cpu().numpy()
 
# Reference values derived from simulation variables
ref_values = {
    'Directed':    dict(sigma=sigma,
                        d=0.02,
                        motion=('v',     velocity),
                        q=0.001),
    'Confined':    dict(sigma=sigma,
                        d=np.sqrt(2 * D_con * reference_dt),
                        motion=('l',     conf_force),
                        q=0.001),
    'Oscillatory': dict(sigma=sigma,
                        d=d_osc,
                        motion=('omega', omega),
                        q=q_osc),
}
 
def _ref_motion_str(kind, val):
    if kind == 'v':
        return f'v={val:.5f}'
    elif kind == 'omega':
        return f'ω={val:.5f} (T={2*np.pi/val:.1f} steps)'
    else:
        return f'l={val:.5f}'
 
def _learned_motion_str(p_row, is_dir, is_osc):
    if is_dir:
        return f'v={np.exp(p_row[2]):.5f}'
    elif is_osc:
        val = np.exp(p_row[2])   # rad/obs-step
        return f'ω={val:.5f} (T={2*np.pi/val:.1f} steps)'
    else:
        l = 1 / (1 + np.exp(-p_row[2]))
        return f'l={l:.5f}'
 
# ── Reference table ────────────────────────────────────────────────────────
print('\nReference (simulation) parameters:')
print(f"  {'State':<13} {'σ':>9} {'d':>9} {'motion':>28} {'q':>12}")
print('  ' + '-' * 75)
for name, ref in ref_values.items():
    ms = _ref_motion_str(ref['motion'][0], ref['motion'][1])
    print(f"  {name:<13} {ref['sigma']:>9.5f} {ref['d']:>9.5f} {ms:>28} {ref['q']:>12.5f}")
 
# ── Learned table ──────────────────────────────────────────────────────────
print('\nLearned parameters:')
print(f"  {'State':<13} {'σ':>9} {'d':>9} {'motion':>28} {'q':>12}")
print('  ' + '-' * 75)
for i, name in enumerate(state_names):
    is_dir = params[i, 4] >= 0.5
    is_osc = params[i, 5] >= 0.5
    sigma_l = np.exp(p[i, 0])
    d_l     = np.exp(p[i, 1])
    q_l     = np.exp(p[i, 3])
    ms      = _learned_motion_str(p[i], is_dir, is_osc)
    print(f"  {name:<13} {sigma_l:>9.5f} {d_l:>9.5f} {ms:>28} {q_l:>12.5f}")
 
# ===========================================================================
# 7. Inference — predict_all via full-length TrackSegmentSequence
# ===========================================================================
# Mirror the working pattern from test_sULM.py:
# build a seq with segment_length = max_track_len so each track is one segment,
# run model(return_all=True) over all batches, concatenate outputs.
# TrackSegmentSequence may reorder tracks into batches, so we reconstruct the
# mapping back to the original track_list indices via the track data itself.
 
_max_track_len = max(len(t) for t in track_list)
 
seq_full = TrackSegmentSequence(
    track_list,
    LocErr_list=LocErr_list,
    dt_list=dt_list,
    batch_size=batch_size,
    segment_length=_max_track_len,
    min_segment_length=2,
    cutoff_batch_treshhold=0.5)
 
model.eval(); model.to(device)
 
LPs_l, preds_l = [], []
tracks_seq_l, masks_seq_l = [], []
with torch.no_grad():
    for i in range(len(seq_full)):
        tr, le, dd, ma, fi = [t.to(device) for t in seq_full[i]]
        lp, st, co, bi, lps = model(tr, le, dd, ma, fi, return_all=True)
        LPs_l.append(lp.cpu().numpy())
        preds_l.append(st.cpu().numpy())
        tracks_seq_l.append(seq_full[i][0].numpy())
        masks_seq_l.append(seq_full[i][3].numpy())
 
# Concatenated arrays from seq_full (may be reordered / fewer tracks than nb_tracks)
preds_seq   = np.concatenate(preds_l,      axis=0)   # (N_seq, max_len, nb_states+1)
tracks_seq  = np.concatenate(tracks_seq_l, axis=0)   # (N_seq, max_len, nb_dims)
masks_seq   = np.concatenate(masks_seq_l,  axis=0)   # (N_seq, max_len)
 
# Build index mapping: for each seq row, find the matching track_list entry
# by comparing the first valid position (unique enough for matching).
# Tracks not in seq_full get preds=None (will be skipped in visualisation).
preds_np = [None] * nb_tracks
for seq_idx in range(len(preds_seq)):
    seq_first = tracks_seq[seq_idx, 0]          # first position in seq row
    for orig_idx in range(nb_tracks):
        if len(track_list[orig_idx]) > 0:
            if np.allclose(track_list[orig_idx][0], seq_first, atol=1e-6):
                preds_np[orig_idx] = preds_seq[seq_idx]
                break
 
print(f"Matched {sum(p is not None for p in preds_np)}/{nb_tracks} tracks from seq_full")
 
# ===========================================================================
# 8. Visualise predictions — colour-coded by predicted state
# ===========================================================================
fig_pred, ax_pred = plt.subplots(figsize=(12, 12))
ax_pred.set_title('Predicted state probabilities\n'
                  '(red=directed, blue=confined, green=oscillatory)', fontsize=11)
for ii, ID in enumerate(IDs):
    if preds_np[ID] is None:
        continue
    row, col = divmod(ii, nb_rows)
    T_gt  = len(state_list[ID])
    track = track_list[ID]                         # use track_list directly
    preds = preds_np[ID][:T_gt, :nb_states]
    t_plot = track - track.mean(0) + np.array([lim*col, lim*row])
    for step in range(len(track)-1):
        color = state_colors @ preds[step]
        ax_pred.plot(t_plot[step:step+2, 0], t_plot[step:step+2, 1],
                     color=np.clip(color, 0, 1), lw=0.9, alpha=0.8)
    ax_pred.scatter(t_plot[0, 0], t_plot[0, 1], c='k', s=8, zorder=5)
ax_pred.set_aspect('equal'); ax_pred.axis('off')
plt.tight_layout()
plt.savefig('osc_transition_predicted.png', dpi=120, bbox_inches='tight')
plt.show()
 
# ===========================================================================
# 9. State probability traces — track on left, probs on right, GT marked
# ===========================================================================
nb_trace_tracks = 5
trace_IDs = IDs[:nb_trace_tracks]
 
fig_traces, axes_all = plt.subplots(
    nb_trace_tracks, 2,
    figsize=(14, nb_trace_tracks * 3),
    gridspec_kw={'width_ratios': [1, 3]})
fig_traces.suptitle('Per-track: 2D trajectory (left) and state probabilities (right)\n'
                     'Shading = ground truth  |  Lines = predicted P(state)',
                     fontsize=11)
 
state_palette = ['tomato', 'steelblue', 'seagreen']
 
for idx_plot, ID in enumerate(trace_IDs):
    if preds_np[ID] is None:
        continue
    ax_track = axes_all[idx_plot, 0]
    ax_prob  = axes_all[idx_plot, 1]
 
    T_gt  = len(state_list[ID])
    track = track_list[ID]                         # use track_list directly
    pred  = preds_np[ID][:T_gt, :nb_states]
    gt    = state_list[ID]
 
    # ── Left: 2D trajectory coloured by GROUND TRUTH state ──────────────
    t_plot = track - track.mean(0)
    for step in range(T_gt - 1):
        ax_track.plot(t_plot[step:step+2, 0], t_plot[step:step+2, 1],
                      color=state_colors[gt[step]], lw=1.0, alpha=0.85)
    ax_track.scatter(t_plot[0, 0], t_plot[0, 1], c='k', s=12, zorder=5)
    ax_track.set_aspect('equal'); ax_track.axis('off')
    ax_track.set_title(f'Track {ID}  (T={T_gt})', fontsize=8)
 
    # Small legend patches for the track panel (first row only)
    if idx_plot == 0:
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=n)
                   for c, n in zip(state_palette, state_names)]
        ax_track.legend(handles=handles, fontsize=7, loc='upper left',
                        framealpha=0.7, title='GT state', title_fontsize=7)
 
    # ── Right: predicted probabilities with GT shading ───────────────────
    steps = np.arange(T_gt)
    for k, (name, col) in enumerate(zip(state_names, state_palette)):
        ax_prob.plot(steps, pred[:, k], color=col, lw=1.2, label=name)
 
    # Shade background by ground truth
    for k, col in enumerate(state_palette):
        runs = np.where(np.diff(np.concatenate([[False], gt == k, [False]])))[0]
        for start, end in zip(runs[::2], runs[1::2]):
            ax_prob.axvspan(start, end, alpha=0.13, color=col, linewidth=0)
 
    # Mark state transitions with vertical dashed lines
    transitions = np.where(np.diff(gt) != 0)[0] + 1
    for tr in transitions:
        ax_prob.axvline(tr, color='k', lw=0.6, ls='--', alpha=0.4)
 
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_xlim(0, T_gt - 1)
    ax_prob.set_ylabel('P(state)', fontsize=8)
    ax_prob.tick_params(labelsize=8)
    if idx_plot == nb_trace_tracks - 1:
        ax_prob.set_xlabel('Time step')
    if idx_plot == 0:
        ax_prob.legend(loc='upper right', fontsize=8, ncol=3,
                       title='Predicted', title_fontsize=7)
 
plt.tight_layout()
plt.savefig('osc_transition_state_probs.png', dpi=130, bbox_inches='tight')
plt.show()
 
# ===========================================================================
# 10. Per-step accuracy
# ===========================================================================
# Use len(state_list[ID]) as the authoritative track length.
# seq_full pads all tracks to _max_track_len, so masks_np[ID].sum() may
# disagree with the original track length stored in state_list.
correct = 0; total = 0
for ID in range(nb_tracks):
    if preds_np[ID] is None:
        continue
    T_gt = len(state_list[ID])
    pred = preds_np[ID][:T_gt, :nb_states].argmax(axis=1)
    gt   = state_list[ID]
    correct += (pred == gt).sum()
    total   += T_gt
 
print(f"\nPer-step accuracy: {correct/total:.1%}  ({correct}/{total} steps)")
for k, name in enumerate(state_names):
    steps_k = sum((state_list[i] == k).sum() for i in range(nb_tracks)
                  if preds_np[i] is not None)
    right_k = sum(
        ((preds_np[i][:len(state_list[i]), :nb_states].argmax(1) == state_list[i]) &
         (state_list[i] == k)).sum()
        for i in range(nb_tracks) if preds_np[i] is not None)
    print(f"  {name}: {right_k/max(steps_k,1):.1%}  ({right_k}/{steps_k} steps)")
 
# ===========================================================================
# 11. Training loss
# ===========================================================================
fig_loss, ax_loss = plt.subplots(figsize=(9, 4))
ax_loss.plot(loss_history, lw=1.0, color='navy')
ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('MLE loss')
ax_loss.set_title('Training loss — oscillatory transition model')
plt.tight_layout()
plt.savefig('osc_transition_loss.png', dpi=120, bbox_inches='tight')
plt.show()
 
print("\nSaved: osc_transition_simulated.png, osc_transition_predicted.png,")
print("       osc_transition_state_probs.png, osc_transition_loss.png")