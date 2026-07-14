# -*- coding: utf-8 -*-
"""
train_exatrack_shared.py  —  Model 1 + spectrogram: shared hidden activation

Trains the emg_constraints_joint2.py model: a shared hidden (r, u) pair
observed separately by CH1 and CH2, each through a per-state learned
coupling coefficient (k1, k2) — analogous to donor/acceptor observing one
hidden FRET efficiency (Eq. 9, Simon et al. 2025), generalized here to
per-frequency-bin spectral features rather than a scalar envelope.

Confirmed working on envelope input (85.3% raw accuracy). This version
reintroduces the FFT spectrogram used by earlier models: CH1 and CH2 are
each represented by n_freq_ac log-amplitude bins per time step (nb_dims=
n_freq_ac), rather than a single scalar envelope value (nb_dims=1). The
constraint function and model class support this generically — nb_dims
flows through automatically from the input tensor's last axis.

Usage
─────
  1. Set CSV_PATH and LABEL_SEQ.
  2. Run: python train_exatrack_shared.py
  3. Watch Section 8b sanity-check output before the full loop proceeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import medfilt

from exatrack_torch.emg_constraints_joint2 import build_emg_joint2_model

# ===========================================================================
# CONFIGURATION
# ===========================================================================
CSV_PATH  = 'data/EMG/training_acquisition_26-07-08_19-55-26.csv'
CKPT_PATH = 'emg_spectro_checkpoint_ratio.pt'

#ABEL_SEQ = [0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0]
LABEL_SEQ = [0,0, 1, 0,0, -1, 0,0, 1, 0,0, -1, 0,0, 1, 0,0, -1, 0,0, 1, 0,0, -1, 0,0, 1, 0,0, -1,0,0]

TIMING_OFFSET = 0.0

FS         = 860
FFT_LEN    = 16     # matches the best-performing spectral params found earlier
HOP_LEN    = 8
SEG_STEPS  = 5

NB_EPOCHS         = 100
LEARNING_RATE     = 0.005
SUPERVISED_WEIGHT = 0.0
BATCH_SIZE        = 50
MEDIAN_WINDOW     = 3
GRAD_CLIP_NORM    = 1

USE_DQ_FLOOR    = False
DQ_FLOOR_REST   = np.log(0.01)
DQ_FLOOR_ACTIVE = np.log(0.001)

# ===========================================================================
# 1.  LOAD, Z-SCORE, ENVELOPE  (identical preprocessing to the 78.3% run)
# ===========================================================================
print(f"Loading: {CSV_PATH}")
df        = pd.read_csv(CSV_PATH)
raw       = df[['voltage1 (V)', 'voltage2 (V)']].values.astype('float64')
n_samples = len(raw)
print(f"  {n_samples} samples ({n_samples/FS:.2f}s)")

mean_raw = raw.mean(axis=0, keepdims=True)
std_raw  = raw.std(axis=0,  keepdims=True)
raw_z    = (raw - mean_raw) / (std_raw + 1e-8)

n_freq_ac = FFT_LEN // 2      # AC frequency bins (exclude DC)
window_fn = np.hanning(FFT_LEN)

n_frames = (n_samples - FFT_LEN) // HOP_LEN + 1
n_segs   = n_frames // SEG_STEPS

# Per-channel log-amplitude spectrogram — CH1 and CH2 stay SEPARATE
# (no ratio, no combining) so the model learns the coupling itself via k1/k2.
log_amp = np.zeros((n_frames, 2, n_freq_ac), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*HOP_LEN : i*HOP_LEN + FFT_LEN] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    fft_ac  = fft_out[1:]                              # (n_freq_ac, 2)
    log_amp[i, 0, :] = (np.abs(fft_ac[:, 0]))
    log_amp[i, 1, :] = (np.abs(fft_ac[:, 1]))

print(f"  FFT: fft_len={FFT_LEN}  hop={HOP_LEN}  n_freq_ac={n_freq_ac}")
print(f"  Segment: {SEG_STEPS} frames "
      f"({SEG_STEPS*HOP_LEN/FS*1000:.1f}ms)")
print(f"  n_segs = {n_segs}")

# ===========================================================================
# 2.  DOWNSAMPLE TO SEGMENT GRID — keep CH1/CH2 SEPARATE (no ratio)
# ===========================================================================
label_seq     = LABEL_SEQ
n_events      = len(label_seq)
event_samples = n_samples / n_events

state_names = ['Rest', 'Flexion', 'Extension']
nb_states   = len(state_names)

labels = np.zeros(n_segs, dtype=int)
for s in range(n_segs):
    t_centre = (s * SEG_STEPS + SEG_STEPS / 2) * HOP_LEN / FS
    evt_idx  = int(t_centre / (event_samples / FS) - TIMING_OFFSET)
    evt_idx  = max(0, min(evt_idx, n_events - 1))
    raw_lbl  = label_seq[evt_idx]
    labels[s] = 2 if raw_lbl == -1 else raw_lbl

print(f"\nPer-segment label counts:")
for k, name in enumerate(state_names):
    n = (labels == k).sum()
    print(f"  {name}: {n} ({n/n_segs:.1%})")

# ===========================================================================
# 4.  NORMALISE PER CHANNEL AND BUILD (n_segs, seg_steps, 2, n_freq_ac) INPUT
# ===========================================================================
feat_mean = log_amp.mean(axis=0, keepdims=True)   # (1, 2, n_freq_ac)
feat_std  = log_amp.std(axis=0,  keepdims=True)
log_amp_z = (log_amp - feat_mean) / (feat_std + 1e-8)

ch_segs = log_amp_z[:n_segs*SEG_STEPS].reshape(n_segs, SEG_STEPS, 2, n_freq_ac)
specs_model = ch_segs

print(f"\n=== Per-channel spectral separability (z-scored, mean over freq bins) ===")
for k, name in enumerate(state_names):
    m = labels == k
    print(f"  {name:<12}: CH1 mean={ch_segs[m,:,0,:].mean():>7.3f}  "
          f"CH2 mean={ch_segs[m,:,1,:].mean():>7.3f}  n={m.sum()}")

# ===========================================================================
# 5.  INITIAL PARAMETERS
#     [log_sigma1, log_sigma2, log_d, logit_l, k1, k2]
# ===========================================================================
params = np.zeros((nb_states, 7), dtype='float64')
print(f"\nInferring initial parameters:")
print(f"  {'State':<12} {'d':>8} {'k1':>8} {'k2':>8}")
for k, name in enumerate(state_names):
    mask = labels == k
    c1 = ch_segs[mask, :, 0, :].mean(axis=-1)   # average over freq bins → (n_class_segs, seg_steps)
    c2 = ch_segs[mask, :, 1, :].mean(axis=-1)

    k1_init = float(np.clip(c1.mean(), -3.0, 3.0))
    k2_init = float(np.clip(c2.mean(), -3.0, 3.0))
    if k == 0:
        k1_init *= 0.3
        k2_init *= 0.3

    diff = np.concatenate([c1[:, 1:] - c1[:, :-1], c2[:, 1:] - c2[:, :-1]])
    d    = diff.std()
    logit_l = 0.0 if k == 0 else 1.5
    log_d   = np.log(np.clip(d, 0.01, 10.0))
    log_q   = np.log(0.05)

    params[k] = [np.log(0.3), np.log(0.3), log_d, logit_l, log_q, k1_init, k2_init]
    print(f"  {name:<12} {np.exp(log_d):>8.4f} {k1_init:>8.4f} {k2_init:>8.4f}")

initial_params    = np.array([[np.log(2.0)]] * nb_states, dtype='float64')
initial_fractions = np.array([[0.0, 0.0, 0.0, -5.0]], dtype='float64')

_base = 0.01; _tiny = 1e-4
transition_rates = np.array([
    [_base, _base, _base],
    [_base, _base, _tiny],
    [_base, _tiny, _base],
], dtype='float64')
transition_shapes = 9.0 * np.eye(3, dtype='float64')

vary_params           = np.ones((nb_states, 7), dtype='float64')
vary_params[:, 0]     = 0.0
vary_params[:, 1]     = 0.0
vary_transition_rates  = np.zeros((nb_states, nb_states), dtype='float64')
vary_transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')

# ===========================================================================
# 6.  BUILD MODEL — nb_dims=n_freq_ac (spectral), shared (r,u) hidden pair
# ===========================================================================
reference_dt = HOP_LEN / FS
model, pred_model = build_emg_joint2_model(
    SEG_STEPS, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    BATCH_SIZE, reference_dt,
    sequence_length=3, max_linking_distance=1,
    nb_dims=n_freq_ac,
    vary_params=vary_params,
    vary_initial_params=np.ones((nb_states, 1)),
    vary_transition_rates=vary_transition_rates,
    vary_transition_shapes=vary_transition_shapes)

print(f"\nModel built — trainable params: "
      f"{sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# 7.  LOSS FUNCTIONS
# ===========================================================================
def supervised_loss(lp, labels, sequence_length):
    nb_sequences = lp.shape[1]
    hyp_states   = torch.arange(nb_sequences, device=lp.device) // sequence_length
    label_mask   = (hyp_states[None, :] == labels[:, None]).to(lp.dtype)
    masked_lp    = lp + torch.log(label_mask + 1e-40)
    max_lp       = masked_lp.max(dim=1, keepdim=True).values
    log_p        = torch.log(torch.exp(masked_lp - max_lp).sum(dim=1)) + max_lp[:, 0]
    return -log_p.mean()

def semisupervised_loss(lp, labels, sequence_length, nb_dims, sup_weight=20.0):
    sup    = supervised_loss(lp, labels, sequence_length)
    max_lp = lp.max(dim=1, keepdim=True).values
    uns    = -(torch.log(torch.exp(lp - max_lp).sum(dim=1)) + max_lp[:, 0]).mean()
    return (sup_weight * sup + uns) / nb_dims

# ===========================================================================
# 8.  DATALOADER
# ===========================================================================
sig_t   = torch.tensor(specs_model, dtype=torch.float64)     # (n_segs, seg_steps, 2)
le_t    = torch.ones(n_segs, SEG_STEPS,    dtype=torch.float64)
dt_t    = torch.full((n_segs, SEG_STEPS+1), reference_dt, dtype=torch.float64)
mask_t  = torch.ones(n_segs, SEG_STEPS,    dtype=torch.float64)
first_t = torch.ones(n_segs,               dtype=torch.float64)
label_t = torch.tensor(labels,             dtype=torch.long)

dataset = TensorDataset(sig_t, le_t, dt_t, mask_t, first_t, label_t)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"DataLoader: {len(loader)} batches × {BATCH_SIZE} segments")

# ===========================================================================
# 8b.  MANDATORY SANITY CHECK — single batch, before committing to 200 epochs
#      This is NEW, unverified code. Confirm the forward pass runs, the loss
#      is finite, and gradients actually reach k1/k2 before trusting the run.
# ===========================================================================
print(f"\n{'='*60}")
print("SANITY CHECK — single batch forward + backward pass")
print(f"{'='*60}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

_sig_b, _le_b, _dt_b, _mask_b, _first_b, _labels_b = next(iter(loader))
_sig_b, _le_b, _dt_b = _sig_b.to(device), _le_b.to(device), _dt_b.to(device)
_mask_b, _first_b, _labels_b = _mask_b.to(device), _first_b.to(device), _labels_b.to(device)

try:
    _lp = model(_sig_b, _le_b, _dt_b, _mask_b, _first_b)
    print(f"  Forward pass OK — lp shape: {tuple(_lp.shape)}, "
          f"finite: {torch.isfinite(_lp).all().item()}")

    _loss = semisupervised_loss(_lp, _labels_b, 3, n_freq_ac, SUPERVISED_WEIGHT)
    print(f"  Loss: {_loss.item():.4f}  finite: {torch.isfinite(_loss).item()}")

    model.zero_grad()
    _loss.backward()
    pv_grad = model.init_layer.param_vars.grad
    if pv_grad is None:
        print("  ✗ WARNING: param_vars.grad is None — no gradient reached parameters!")
    else:
        print(f"  Gradient norms per column "
              f"[sigma1, sigma2, log_d, logit_l, log_q, k1, k2]:")
        for col, name in enumerate(['sigma1','sigma2','log_d','logit_l','log_q','k1','k2']):
            g = pv_grad[:, col].abs()
            print(f"    {name:<8}: max={g.max().item():.4e}  mean={g.mean().item():.4e}")
        if pv_grad[:, 5].abs().max().item() == 0 or pv_grad[:, 6].abs().max().item() == 0:
            print("  ✗ WARNING: k1 or k2 gradient is exactly zero — "
                  "coupling coefficients won't learn. Stop and debug before continuing.")
        else:
            print("  ✓ k1/k2 gradients non-zero — proceeding to full training.")
except Exception as e:
    print(f"  ✗ SANITY CHECK FAILED: {type(e).__name__}: {e}")
    print("  Fix emg_constraints_shared.py before running full training.")
    raise

model.zero_grad()
print(f"{'='*60}\n")

# ===========================================================================
# 9.  TRAINING
# ===========================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             betas=(0.9, 0.999), eps=1e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NB_EPOCHS, eta_min=1e-5)

model.train()
print(f"Gradient clip norm: {GRAD_CLIP_NORM}")
print(f"Supervised weight: {SUPERVISED_WEIGHT}  (keep >=10)")

loss_history = []; best_loss = float('inf'); best_state = None
print(f"\nTraining {NB_EPOCHS} epochs  (supervised_weight={SUPERVISED_WEIGHT})")
print(f"{'Epoch':>6}  {'Loss':>10}  {'Best':>10}  {'LR':>10}")
print("-" * 44)

for epoch in range(NB_EPOCHS):
    epoch_losses = []
    for sig_b, le_b, dt_b, mask_b, first_b, labels_b in loader:
        sig_b   = sig_b.to(device);   le_b     = le_b.to(device)
        dt_b    = dt_b.to(device);    mask_b   = mask_b.to(device)
        first_b = first_b.to(device); labels_b = labels_b.to(device)

        optimizer.zero_grad()
        lp   = model(sig_b, le_b, dt_b, mask_b, first_b)
        loss = semisupervised_loss(lp, labels_b, 3, n_freq_ac, SUPERVISED_WEIGHT)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if USE_DQ_FLOOR:
            with torch.no_grad():
                pv = model.init_layer.param_vars
                # col 2 = log_d
                pv[0, 2].clamp_(min=DQ_FLOOR_REST)
                pv[1, 2].clamp_(min=DQ_FLOOR_ACTIVE)
                pv[2, 2].clamp_(min=DQ_FLOOR_ACTIVE)

        epoch_losses.append(loss.item())

    epoch_loss = float(np.mean(epoch_losses))
    loss_history.append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss  = epoch_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"{epoch+1:>6}  {epoch_loss:>10.4f}  {best_loss:>10.4f}  {lr:>10.2e}")

model.load_state_dict(best_state, strict=False)
print(f"\nBest model restored (loss={best_loss:.4f})")

# ===========================================================================
# 10. SAVE CHECKPOINT
# ===========================================================================
checkpoint = {
    'model_type'        : 'shared',
    'state_dict'        : {k: v.detach().cpu().tolist()
                           if isinstance(v, torch.Tensor) else v
                           for k, v in model.state_dict().items()},
    'mean_raw'          : mean_raw.tolist(),
    'std_raw'           : std_raw.tolist(),
    'feat_mean'         : feat_mean.tolist(),
    'feat_std'          : feat_std.tolist(),
    'fs'                : FS,
    'fft_len'           : FFT_LEN,
    'hop_len'           : HOP_LEN,
    'n_freq_ac'         : n_freq_ac,
    'seg_steps'         : SEG_STEPS,
    'nb_states'         : nb_states,
    'state_names'       : state_names,
    'params'            : params.tolist(),
    'initial_params'    : initial_params.tolist(),
    'initial_fractions' : initial_fractions.tolist(),
    'transition_rates'  : transition_rates.tolist(),
    'transition_shapes' : transition_shapes.tolist(),
    'reference_dt'      : reference_dt,
    'batch_size'        : BATCH_SIZE,
}
torch.save(checkpoint, CKPT_PATH)
print(f"Checkpoint saved → {CKPT_PATH}")

# ===========================================================================
# 11. OFFLINE INFERENCE + METRICS
# ===========================================================================
model.eval()
all_states, all_lp = [], []
with torch.no_grad():
    for start in range(0, n_segs, BATCH_SIZE):
        end    = min(start + BATCH_SIZE, n_segs)
        actual = end - start
        def pad(t):
            b = t[start:end]
            if actual < BATCH_SIZE:
                b = torch.cat([b, b[:1].expand(BATCH_SIZE-actual, *b.shape[1:])], 0)
            return b.to(device)
        lp_b, states_b, _, _, _ = pred_model(
            *[pad(t) for t in [sig_t, le_t, dt_t, mask_t, first_t]], return_all=True)
        all_states.append(states_b[:actual].cpu().numpy())
        all_lp.append(lp_b[:actual].cpu().numpy())

state_probs   = np.concatenate(all_states, axis=0)
mean_probs    = state_probs[:, :, :nb_states].mean(axis=1)
predicted_raw = mean_probs.argmax(axis=1)
predicted     = medfilt(predicted_raw.astype(float),
                        kernel_size=MEDIAN_WINDOW).astype(int)

acc_raw  = (predicted_raw == labels).mean()
acc_filt = (predicted     == labels).mean()
print(f"\nTraining-set accuracy (raw):      {acc_raw:.1%}")
print(f"Training-set accuracy (filtered): {acc_filt:.1%}")
print("Per-class (filtered):")
for k, name in enumerate(state_names):
    m = labels == k
    if m.sum():
        print(f"  {name}: {(predicted[m]==k).mean():.1%} ({m.sum()} segs)")

cm = np.zeros((nb_states, nb_states), dtype=int)
for t in range(nb_states):
    for p in range(nb_states):
        cm[t, p] = np.sum((labels == t) & (predicted == p))
print(f"\nConfusion matrix (rows=true, cols=pred):")
print(f"{'':>11} " + " ".join(f"{n:>11}" for n in state_names))
for t, name in enumerate(state_names):
    print(f"{name:>11} " + " ".join(f"{cm[t,p]:>11}" for p in range(nb_states)))

# Print learned k1/k2 per state — the key diagnostic for this model
sd = model.state_dict()
pv_key = [k for k in sd if 'param_vars' in k][0]
pv = sd[pv_key].cpu().numpy() if isinstance(sd[pv_key], torch.Tensor) else np.array(sd[pv_key])
print(f"\n=== Learned channel couplings (the key diagnostic for this model) ===")
print(f"  {'State':<12} {'k1 (CH1)':>10} {'k2 (CH2)':>10}")
for i, name in enumerate(state_names):
    print(f"  {name:<12} {pv[i,5]:>10.4f} {pv[i,6]:>10.4f}")
print("  Expect: Rest≈small both, Flexion k1>>k2, Extension k2>>k1")

# ===========================================================================
# 12. PLOTS
# ===========================================================================
t_raw = np.arange(n_samples) / FS
t_seg = np.arange(n_segs) * (SEG_STEPS * HOP_LEN) / FS
colors = ['steelblue', 'tomato', 'seagreen']

fig = plt.figure(figsize=(14, 12))
fig.suptitle(f'Shared-activation model  raw={acc_raw:.1%}  filtered={acc_filt:.1%}',
             fontsize=13)
gs  = GridSpec(4, 1, figure=fig, hspace=0.5)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

t_frm = np.arange(n_frames) * HOP_LEN / FS
ch1_mean_amp = log_amp[:, 0, :].mean(axis=-1)   # mean over freq bins for display
ch2_mean_amp = log_amp[:, 1, :].mean(axis=-1)
ax0.plot(t_frm, ch1_mean_amp, lw=0.5, color='steelblue', alpha=0.8, label='CH1 (mean log-amp)')
ax0.plot(t_frm, ch2_mean_amp, lw=0.5, color='tomato',    alpha=0.8, label='CH2 (mean log-amp)')
ax0.set_ylabel('Envelope'); ax0.legend(fontsize=8)
ax0.set_title('CH1 / CH2 envelopes (fed separately, no ratio)')

for k in range(nb_states):
    ax1.plot(t_seg, mean_probs[:,k], lw=1.2, color=colors[k], label=state_names[k])
ax1.set_ylabel('P(state)'); ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=8); ax1.set_title('State probabilities')

for s in range(n_segs):
    t0 = t_seg[s]; t1 = t0 + SEG_STEPS * HOP_LEN / FS
    ax2.barh(labels[s],          t1-t0, left=t0, height=0.55,
             color=colors[labels[s]], alpha=0.3)
    ax2.barh(predicted_raw[s]+0.30, t1-t0, left=t0, height=0.25,
             color=colors[predicted_raw[s]], alpha=0.85)
ax2.set_yticks([0,1,2]); ax2.set_yticklabels(state_names)
ax2.set_ylim(-0.7, 2.8); ax2.set_title('Ground truth vs predictions')

ax3.plot(loss_history, lw=1.2, color='navy')
ax3.set_ylabel('Loss'); ax3.set_xlabel('Epoch'); ax3.set_title('Training loss')

plt.savefig('emg_shared_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → emg_shared_results.png")
