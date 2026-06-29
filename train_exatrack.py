# -*- coding: utf-8 -*-
"""
train_exatrack.py  —  Step 2 of 3

Loads a CSV produced by acquire_emg.py, trains an ExaTrack spectrogram model,
and saves a checkpoint that pong_exatrack.py (Step 3) loads directly.

Usage
─────
  1. Set CSV_PATH to the file printed by acquire_emg.py.
  2. Run:  python train_exatrack.py
  3. The checkpoint path is printed at the end — copy it into pong_exatrack.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import torch
import types
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt, medfilt

from exatrack_torch.emg_constraints import build_emg_model
from exatrack_torch.models import MLE_loss

# ===========================================================================
# CONFIGURATION — edit here
# ===========================================================================
CSV_PATH  = 'data/training_acquisition_YY-MM-DD_HH-MM-SS.csv'  # <── set this
CKPT_PATH = 'emg_spectro_checkpoint.pt'

# Must match the LABEL_SEQ used in acquire_emg.py
LABEL_SEQ   = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0, 1, 0,
               0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0]
TIMING_OFFSET = 0.0   # shift label alignment if gestures appear offset in the plot

FS        = 2000   # nominal sampling frequency (Hz) — must match acquisition setup
FFT_LEN   = 16     # FFT window size in samples
HOP_LEN   = 8      # FFT hop size in samples
SEG_STEPS = 3      # frames per segment fed to the model

# Training hyperparameters
NB_EPOCHS         = 100
LEARNING_RATE     = 0.01
SUPERVISED_WEIGHT = 5.0
BATCH_SIZE        = 50
MEDIAN_WINDOW     = 5   # post-processing median filter width (must be odd)

# Model B (USE_DQ_FLOOR=True): balanced ~85% across all classes, smooth
# transitions — best for device control where gesture detection matters most.
# Model A (False): highest overall accuracy but Rest-dominant and jittery.
USE_DQ_FLOOR = True

# ===========================================================================
# 1.  LOAD DATA AND Z-SCORE
# ===========================================================================
print(f"Loading: {CSV_PATH}")
df        = pd.read_csv(CSV_PATH)
raw       = df[['voltage1 (V)', 'voltage2 (V)']].values.astype('float64')
n_samples = len(raw)
print(f"  {n_samples} samples ({n_samples/FS:.2f}s)")

mean_raw = raw.mean(axis=0, keepdims=True)
std_raw  = raw.std(axis=0,  keepdims=True)
raw_z    = (raw - mean_raw) / (std_raw + 1e-8)

# ===========================================================================
# 2.  LABEL ALIGNMENT
# ===========================================================================
label_seq     = LABEL_SEQ
n_events      = len(label_seq)
event_samples = n_samples / n_events

n_freq_ac = FFT_LEN // 2 - 1      # exclude DC: FFT_LEN//2 + 1 bins, drop bin 0 → FFT_LEN//2
n_dims    = 2 * n_freq_ac          # real + imaginary parts
window_fn = np.hanning(FFT_LEN)
freqs     = np.fft.rfftfreq(FFT_LEN, d=1/FS)

n_frames  = (n_samples - FFT_LEN) // HOP_LEN + 1
n_segs    = n_frames // SEG_STEPS

state_names = ['Rest', 'Flexion', 'Extension']
nb_states   = len(state_names)

labels = np.zeros(n_segs, dtype=int)
for s in range(n_segs):
    t_centre = (s * SEG_STEPS + SEG_STEPS / 2) * HOP_LEN / FS
    evt_idx  = int(t_centre / (event_samples / FS) - TIMING_OFFSET)
    evt_idx  = max(0, min(evt_idx, n_events - 1))
    raw_lbl  = label_seq[evt_idx]
    labels[s] = 2 if raw_lbl == -1 else raw_lbl    # -1 → class index 2

print(f"\nPer-segment label counts:")
for k, name in enumerate(state_names):
    n = (labels == k).sum()
    print(f"  {name}: {n} ({n/n_segs:.1%})")

# ===========================================================================
# 3.  FFT FEATURES
# ===========================================================================
features = np.zeros((n_frames, n_dims, 2), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*HOP_LEN : i*HOP_LEN + FFT_LEN] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    fft_ac  = fft_out[1:]                           # exclude DC
    features[i, :n_freq_ac, :] = fft_ac.real
    features[i, n_freq_ac:, :] = fft_ac.imag

feat_mean   = features.mean(axis=0, keepdims=True)
feat_std    = features.std(axis=0,  keepdims=True)
feat_z      = (features - feat_mean) / (feat_std + 1e-8)
feat_segs   = feat_z[:n_segs*SEG_STEPS].reshape(n_segs, SEG_STEPS, n_dims, 2)
specs_model = feat_segs.transpose(0, 1, 3, 2)       # (n_segs, SEG_STEPS, 2, n_dims)

# ===========================================================================
# 4.  INITIAL PARAMETERS FROM LABELLED DATA
# ===========================================================================
log_amp_segs = feat_segs[:, :, :n_freq_ac, :]   # real part as d/q proxy

params = np.zeros((nb_states, 8), dtype='float64')
print("\nInferring initial parameters:")
print(f"  {'State':<12} {'d1':>8} {'d2':>8} {'q1':>8} {'q2':>8}")
for k, name in enumerate(state_names):
    mask    = labels == k
    la      = log_amp_segs[mask]
    diff    = la[:, 1:, :, :] - la[:, :-1, :, :]
    d       = diff.std(axis=(0, 1, 2))
    q       = la.std(axis=1).mean(axis=(0, 1))
    logit_l = 0.0 if k == 0 else 1.5
    log_d   = np.log(np.clip(d, 0.01, 5.0))
    log_q   = np.log(np.clip(q * 2, 0.01, 5.0))
    params[k] = [np.log(0.30), log_d[0], logit_l, log_q[0],
                 np.log(0.30), log_d[1], logit_l, log_q[1]]
    print(f"  {name:<12} {np.exp(log_d[0]):>8.4f} {np.exp(log_d[1]):>8.4f} "
          f"{np.exp(log_q[0]):>8.4f} {np.exp(log_q[1]):>8.4f}")

initial_params    = np.array([[np.log(0.3), np.log(0.3)]] * 3, dtype='float64')
initial_fractions = np.array([[0.0, 0.0, 0.0, -5.0]], dtype='float64')

_base = 0.02; _tiny = 1e-4
transition_rates = np.array([
    [_base, _base, _base],
    [_base, _base, _tiny],   # Flexion → Extension blocked
    [_base, _tiny, _base],   # Extension → Flexion blocked
], dtype='float64')
transition_shapes = 2.0 * np.eye(3, dtype='float64')

vary_params              = np.ones((nb_states, 8),  dtype='float64')
vary_params[:, [0, 4]]  = 0.0   # fix sigma
vary_transition_rates    = np.zeros((nb_states, nb_states), dtype='float64')
vary_transition_shapes   = np.zeros((nb_states, nb_states), dtype='float64')

# ===========================================================================
# 5.  BUILD MODEL
# ===========================================================================
reference_dt = HOP_LEN / FS
model, pred_model = build_emg_model(
    SEG_STEPS, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    BATCH_SIZE, reference_dt,
    sequence_length=3, max_linking_distance=1,
    vary_params=vary_params,
    vary_initial_params=np.ones((nb_states, 2)),
    vary_transition_rates=vary_transition_rates,
    vary_transition_shapes=vary_transition_shapes)

print(f"\nModel built — trainable params: "
      f"{sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# 6.  LOSS FUNCTIONS
# ===========================================================================
def supervised_loss(lp, labels, sequence_length):
    nb_sequences = lp.shape[1]
    hyp_states   = torch.arange(nb_sequences, device=lp.device) // sequence_length
    label_mask   = (hyp_states[None, :] == labels[:, None]).to(lp.dtype)
    masked_lp    = lp + torch.log(label_mask + 1e-40)
    max_lp       = masked_lp.max(dim=1, keepdim=True).values
    log_p        = torch.log(torch.exp(masked_lp - max_lp).sum(dim=1)) + max_lp[:, 0]
    return -log_p.mean()

def semisupervised_loss(lp, labels, sequence_length, nb_dims, sup_weight=5.0):
    sup  = supervised_loss(lp, labels, sequence_length)
    max_lp = lp.max(dim=1, keepdim=True).values
    uns  = -(torch.log(torch.exp(lp - max_lp).sum(dim=1)) + max_lp[:, 0]).mean()
    return (sup_weight * sup + uns) / nb_dims

# ===========================================================================
# 7.  FORWARD PATCH  (same as training and validation scripts)
# ===========================================================================
def spectral_forward(self, inputs, input_LocErrs, input_dts, input_mask,
                     input_isfirst, return_all=False):
    device        = next(self.parameters()).device
    inputs        = inputs.to(device)
    input_LocErrs = input_LocErrs.to(device)
    input_dts     = input_dts.to(device)
    input_mask    = input_mask.to(device)
    input_isfirst = input_isfirst.to(device)
    reshaped      = inputs[:, None, :, None, :, :]
    transposed    = reshaped.permute(2, 1, 0, 3, 4, 5)
    transposed, initial_states = self.init_layer(transposed, input_LocErrs, input_dts)
    (Prev_coefs, Prev_biases, LP,
     Log_factors, transition_Log_factors,
     rec_obs_coefs, rec_hid_coefs, rec_next_hid_coefs, rec_biases,
     trans_hid_coefs, trans_biases) = initial_states
    softmax_inv_Fractions = self.init_layer.initial_fractions
    log_ds            = self.init_layer.param_vars[:, 1]
    anomalous_factors = self.init_layer.param_vars[:, 2]
    isdir             = torch.zeros_like(log_ds)
    Prev_coefs  = self.isfirst_mask(Prev_coefs,  self.init_layer.carryout_coefs,
                                     input_isfirst[None, :, None, None])
    Prev_biases = self.isfirst_mask(Prev_biases, self.init_layer.carryout_biases,
                                     input_isfirst[None, :, None, None])
    LP          = self.isfirst_mask(LP, self.init_layer.carryout_LP,
                                     input_isfirst[:, None])
    sliced_inputs = transposed[1:]
    sliced_mask   = input_mask[:, 1:]
    (Prev_coefs, Prev_biases, LP, segment_len,
     gamma_dist_mean, gamma_dist_var,
     All_motion_states, All_coefs, All_biases, All_LPs,
     motion_states) = self.rnn_layer(
        sliced_inputs, input_dts, self.reference_dt, sliced_mask,
        Prev_coefs, Prev_biases, LP, Log_factors, transition_Log_factors,
        rec_obs_coefs, rec_hid_coefs, rec_next_hid_coefs, rec_biases,
        trans_hid_coefs, trans_biases,
        log_ds, softmax_inv_Fractions, anomalous_factors, isdir,
        isfirst=input_isfirst)
    states = [Prev_coefs, Prev_biases, LP, All_motion_states, motion_states]
    outputs, All_states = self.final_layer(states)
    if self.init_layer.carryover_initialized:
        self.init_layer.carryout_coefs.data.copy_(Prev_coefs.detach())
        self.init_layer.carryout_biases.data.copy_(Prev_biases.detach())
        self.init_layer.carryout_LP.data.copy_(LP.detach())
    if self.rnn_layer.carryover:
        self.rnn_layer.carryout_segment_len.data.copy_(segment_len.detach())
        self.rnn_layer.carryout_gamma_dist_mean.data.copy_(gamma_dist_mean.detach())
        self.rnn_layer.carryout_gamma_dist_var.data.copy_(gamma_dist_var.detach())
    if return_all:
        return outputs, All_states, All_coefs, All_biases, All_LPs
    return outputs

model.forward      = types.MethodType(spectral_forward, model)
pred_model.forward = types.MethodType(spectral_forward, pred_model)

# ===========================================================================
# 8.  DATALOADER
# ===========================================================================
sig_t   = torch.tensor(specs_model, dtype=torch.float64)
le_t    = torch.ones(n_segs, SEG_STEPS,    dtype=torch.float64)
dt_t    = torch.full((n_segs, SEG_STEPS+1), reference_dt, dtype=torch.float64)
mask_t  = torch.ones(n_segs, SEG_STEPS,    dtype=torch.float64)
first_t = torch.ones(n_segs,               dtype=torch.float64)
label_t = torch.tensor(labels,             dtype=torch.long)

dataset = TensorDataset(sig_t, le_t, dt_t, mask_t, first_t, label_t)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"DataLoader: {len(loader)} batches × {BATCH_SIZE} segments")

# ===========================================================================
# 9.  TRAINING
# ===========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             betas=(0.9, 0.999), eps=1e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NB_EPOCHS, eta_min=1e-5)

model.train(); model.to(device)

# Gradient diagnostic to set clip value
diag = [t.to(device) for t in next(iter(loader))]
optimizer.zero_grad()
loss_d = semisupervised_loss(model(*diag[:5]), diag[5], 3, n_dims, SUPERVISED_WEIGHT)
loss_d.backward()
grad_norms = {n: p.grad.abs().max().item()
              for n, p in model.named_parameters() if p.grad is not None}
p25      = float(np.percentile(list(grad_norms.values()), 25))
clip_val = max(p25 * 5, 1e-4)
if max(grad_norms.values()) > 1e3:
    clip_val = min(clip_val, 0.1)
print(f"Gradient clip value: {clip_val:.4e}")
optimizer.zero_grad()

loss_history = []; best_loss = float('inf'); best_state = None
print(f"\nTraining {NB_EPOCHS} epochs  (supervised_weight={SUPERVISED_WEIGHT})")
print(f"{'Epoch':>6}  {'Loss':>10}  {'Best':>10}  {'LR':>10}")
print("-" * 44)

for epoch in range(NB_EPOCHS):
    epoch_losses = []
    for sig_b, le_b, dt_b, mask_b, first_b, labels_b in loader:
        sig_b    = sig_b.to(device);    le_b     = le_b.to(device)
        dt_b     = dt_b.to(device);     mask_b   = mask_b.to(device)
        first_b  = first_b.to(device);  labels_b = labels_b.to(device)

        optimizer.zero_grad()
        lp   = model(sig_b, le_b, dt_b, mask_b, first_b)
        loss = semisupervised_loss(lp, labels_b, 3, n_dims, SUPERVISED_WEIGHT)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_val)
        optimizer.step()

        # d/q floor to keep transitions physiologically plausible
        if USE_DQ_FLOOR:
            with torch.no_grad():
                pv = model.init_layer.param_vars
                for col in (1, 3, 5, 7):              # log_d1, log_q1, log_d2, log_q2
                    pv[:, col].clamp_(min=np.log(0.05))

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
#     Contains everything pong_exatrack.py needs to reconstruct the model
#     and reproduce the exact preprocessing pipeline.
# ===========================================================================
checkpoint = {
    'state_dict'        : {k: v.detach().cpu() for k, v in model.state_dict().items()},
    'mean_raw'          : mean_raw,
    'std_raw'           : std_raw,
    'feat_mean'         : feat_mean,
    'feat_std'          : feat_std,
    'fs'                : FS,
    'fft_len'           : FFT_LEN,
    'hop_len'           : HOP_LEN,
    'seg_steps'         : SEG_STEPS,
    'n_freq_ac'         : n_freq_ac,
    'n_dims'            : n_dims,
    'nb_states'         : nb_states,
    'state_names'       : state_names,
    'params'            : params,
    'initial_params'    : initial_params,
    'initial_fractions' : initial_fractions,
    'transition_rates'  : transition_rates,
    'transition_shapes' : transition_shapes,
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
all_lp_np     = np.concatenate(all_lp,     axis=0)
max_lp        = all_lp_np.max(axis=1, keepdims=True)
ll_per_seg    = np.log(np.exp(all_lp_np - max_lp).sum(axis=1)) + max_lp[:, 0]
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

# ===========================================================================
# 12. PLOTS
# ===========================================================================
t_raw  = np.arange(n_samples) / FS
t_seg  = np.arange(n_segs) * (SEG_STEPS * HOP_LEN) / FS
t_frm  = np.arange(n_frames) * HOP_LEN / FS
colors = ['steelblue', 'tomato', 'seagreen']

# Log-amplitude spectrogram for display
log_amp = np.zeros((n_frames, n_freq_ac, 2), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*HOP_LEN : i*HOP_LEN + FFT_LEN] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    log_amp[i] = np.log1p(np.abs(fft_out[1:]))
la_z = (log_amp - log_amp.mean(0)) / (log_amp.std(0) + 1e-8)

fig = plt.figure(figsize=(14, 16))
fig.suptitle(f'ExaTrack training  raw={acc_raw:.1%}  filtered={acc_filt:.1%}', fontsize=13)
gs = GridSpec(6, 1, figure=fig, hspace=0.5)
ax0 = fig.add_subplot(gs[0])   # raw signal
ax1 = fig.add_subplot(gs[1])   # CH1 spectrogram
ax2 = fig.add_subplot(gs[2])   # CH2 spectrogram
ax3 = fig.add_subplot(gs[3])   # state probabilities
ax4 = fig.add_subplot(gs[4])   # predictions vs ground truth
ax5 = fig.add_subplot(gs[5])   # training loss

for ax in [ax1, ax2, ax3, ax4]:
    ax.sharex(ax0)

ax0.plot(t_raw, raw_z[:, 0], lw=0.3, color='steelblue', alpha=0.8, label='CH1')
ax0.plot(t_raw, raw_z[:, 1], lw=0.3, color='tomato',    alpha=0.8, label='CH2')
ax0.set_ylabel('Amplitude'); ax0.legend(loc='upper right', fontsize=8)
ax0.set_title('Raw z-scored EMG')

ax1.pcolormesh(t_frm, freqs[1:], la_z[:, :, 0].T,
               cmap='RdBu_r', vmin=-2, vmax=4, shading='auto')
ax1.set_ylabel('Freq (Hz)'); ax1.set_title('CH1 log-amplitude spectrogram')

ax2.pcolormesh(t_frm, freqs[1:], la_z[:, :, 1].T,
               cmap='RdBu_r', vmin=-2, vmax=4, shading='auto')
ax2.set_ylabel('Freq (Hz)'); ax2.set_title('CH2 log-amplitude spectrogram')

for k in range(nb_states):
    ax3.plot(t_seg, mean_probs[:, k], lw=1.2, color=colors[k], label=state_names[k])
ax3.set_ylabel('P(state)'); ax3.set_ylim(-0.05, 1.05)
ax3.legend(loc='upper right', fontsize=8); ax3.set_title('State probabilities')

for s in range(n_segs):
    t0 = t_seg[s]; t1 = t0 + SEG_STEPS * HOP_LEN / FS
    ax4.barh(labels[s],          t1-t0, left=t0, height=0.55,
             color=colors[labels[s]], alpha=0.3)
    ax4.barh(predicted_raw[s]+0.30, t1-t0, left=t0, height=0.25,
             color=colors[predicted_raw[s]], alpha=0.85)
    ax4.barh(predicted[s]-0.30,  t1-t0, left=t0, height=0.25,
             color=colors[predicted[s]],     alpha=0.85)
ax4.set_yticks([0, 1, 2]); ax4.set_yticklabels(state_names)
ax4.set_ylim(-0.7, 2.8); ax4.set_ylabel('State')
ax4.legend(handles=[
    Patch(facecolor='gray', alpha=0.3,  label='Ground truth'),
    Patch(facecolor='gray', alpha=0.85, label=f'Raw ({acc_raw:.1%})'),
    Patch(facecolor='gray', alpha=0.85, label=f'Filtered ({acc_filt:.1%})'),
], loc='upper right', fontsize=8)
ax4.set_title('Ground truth vs predictions')

ax5.plot(loss_history, lw=1.2, color='navy')
ax5.set_ylabel('Loss'); ax5.set_xlabel('Epoch'); ax5.set_title('Training loss')

plt.savefig('emg_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → emg_training_results.png")

print(f"\nNext step: open pong_exatrack.py and set")
print(f"  CKPT_PATH = '{CKPT_PATH}'")
print(f"then run:  python pong_exatrack.py")
