# -*- coding: utf-8 -*-
"""
test_emg_supervised.py  —  supervised EMG gesture decoding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import types
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt

from exatrack_torch.emg_constraints import build_emg_model
from exatrack_torch.models import MLE_loss

# ===========================================================================
# 1. Load and z-score
# ===========================================================================
CSV_PATH = 'data/EMG/training_acquisition_23-03-28_20-23-39.csv'
df       = pd.read_csv(CSV_PATH)
raw      = df[['voltage1 (V)', 'voltage2 (V)']].values.astype('float64')
fs       = 2000
n_samples = len(raw)
print(f"Loaded {n_samples} samples ({n_samples/fs:.2f}s)")

mean_raw = raw.mean(axis=0, keepdims=True)
std_raw  = raw.std(axis=0,  keepdims=True)
raw_z    = (raw - mean_raw) / (std_raw + 1e-8)

# ===========================================================================
# 2. Upscale ground-truth labels to per-segment level
# ===========================================================================
# Label sequence: 0=rest, 1=flexion, -1=extension
# Each event occupies n_samples/n_events samples.
# Empirically the gestures occur ~1 event duration after the label onset
# (subject reaction time + movement onset latency).
label_seq     = [0,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,1,0,0,-1,0,0,-1,0,0,1,0,0,-1,0]
n_events      = len(label_seq)
event_samples = n_samples / n_events   # ~166.7 samples per event
timing_offset = 0.0                    # no offset needed for this recording

fft_len   = 32
hop_len   = 16
seg_steps = 5
n_freq    = fft_len // 2 + 1
n_freq_ac = n_freq - 1      # exclude DC: 16 bins
n_dims    = 2 * n_freq_ac    # 32: real + imaginary parts
freqs     = np.fft.rfftfreq(fft_len, d=1/fs)
window_fn = np.hanning(fft_len)

n_frames  = (n_samples - fft_len) // hop_len + 1
n_segs    = n_frames // seg_steps

# Map each segment to a label
labels = np.zeros(n_segs, dtype=int)
for s in range(n_segs):
    t_centre  = (s * seg_steps + seg_steps / 2) * hop_len / fs
    evt_idx   = int(t_centre / (event_samples/fs) - timing_offset)
    evt_idx   = max(0, min(evt_idx, n_events - 1))
    raw_lbl   = label_seq[evt_idx]
    labels[s] = 2 if raw_lbl == -1 else raw_lbl   # -1 → 2 for 0-indexed states

state_names = ['Rest', 'Flexion', 'Extension']
print(f"\nPer-segment labels (timing_offset={timing_offset} events):")
for k, name in enumerate(state_names):
    n = (labels == k).sum()
    print(f"  {name}: {n} ({n/n_segs:.1%})")

# ===========================================================================
# 3. Compute FFT features
# ===========================================================================
features = np.zeros((n_frames, n_dims, 2), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*hop_len : i*hop_len + fft_len] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    fft_ac  = fft_out[1:]                          # exclude DC
    features[i, :n_freq_ac, :]  = fft_ac.real      # (n_freq_ac, 2)
    features[i, n_freq_ac:, :]  = fft_ac.imag      # (n_freq_ac, 2)
feat_mean   = features.mean(axis=0, keepdims=True)
feat_std    = features.std(axis=0,  keepdims=True)
feat_z      = (features - feat_mean) / (feat_std + 1e-8)
feat_segs   = feat_z[:n_segs*seg_steps].reshape(n_segs, seg_steps, n_dims, 2)
specs_model = feat_segs.transpose(0, 1, 3, 2)   # (n_segs, 5, 2, 48)

# ===========================================================================
# 4. Infer initial parameters from labelled data
# ===========================================================================
# Compute per-class FFT feature statistics to initialise params
# Key: per-class mean and std of log-amplitude averaged across freq bins
# Envelope for visualisation only
rect = np.abs(raw_z)
b, a = butter(4, 20/(fs/2), btype='low')
env  = np.stack([filtfilt(b, a, rect[:, k]) for k in range(2)], axis=1)
seg_env = env[:n_segs*seg_steps*hop_len:hop_len, :][
    :n_segs*seg_steps].reshape(n_segs, seg_steps, 2).mean(axis=1)

# Infer initial parameters from labelled spectral data.
#
# For each state k:
#   log_d[k,ch] = log(frame-to-frame std of log-amplitude)
#                 → measures how much the spectrum fluctuates per step
#                 → low for rest (stable spectrum), high for active (variable)
#   log_q[k,ch] = log(2 × within-segment std of log-amplitude)
#                 → measures how fast the spectral baseline drifts
#                 → factor 2: q drives the activation variable u, which
#                   has ~2× the variability of the observed log-amplitude
#   logit_l[k]  = 0.0 for rest (tight confinement: sigmoid(0)=0.5)
#                 1.5 for active (looser: sigmoid(1.5)=0.82)
#                 → rest spectrum stays near its baseline, active fluctuates

log_amp_segs = feat_segs[:, :, :n_freq_ac, :]  # real part as d proxy

nb_states = 3   # rest, flexion, extension
params = np.zeros((nb_states, 8), dtype='float64')
print("\nInferring initial parameters from labelled segments:")
print(f"  {'State':<12} {'d1':>8} {'d2':>8} {'q1':>8} {'q2':>8}")
for k, name in enumerate(state_names):
    mask = labels == k
    la   = log_amp_segs[mask]                          # (n_k, seg_steps, n_freq_ac, 2)

    # d: frame-to-frame difference std, averaged across freq bins and segments
    diff  = la[:, 1:, :, :] - la[:, :-1, :, :]        # (n_k, seg_steps-1, n_freq_ac, 2)
    d     = diff.std(axis=(0, 1, 2))                   # (2,) one value per channel

    # q: within-segment std, averaged across freq bins and segments
    q     = la.std(axis=1).mean(axis=(0, 1))           # (2,)

    logit_l = 0.0 if k == 0 else 1.5                   # rest: tight, active: loose

    log_d = np.log(np.clip(d, 0.01, 5.0))
    log_q = np.log(np.clip(q * 2, 0.01, 5.0))         # factor 2: u has ~2× variability

    params[k] = [np.log(0.30), log_d[0], logit_l, log_q[0],
                 np.log(0.30), log_d[1], logit_l, log_q[1]]

    print(f"  {name:<12} {np.exp(log_d[0]):>8.4f} {np.exp(log_d[1]):>8.4f} "
          f"{np.exp(log_q[0]):>8.4f} {np.exp(log_q[1]):>8.4f}")

initial_params    = np.array([[np.log(0.3), np.log(0.3)]] * 3, dtype='float64')
initial_fractions = np.array([[0.0, 0.0, 0.0, -5.0]], dtype='float64')
# Transition dynamics — enforce realistic gesture dwell times.
#
# In ExaTrack the state lifetime follows a Gamma distribution with:
#   mean  = shape / rate   (in units of reference_dt = 8ms)
#   std   = sqrt(shape) / rate
#
# Target: mean dwell ~500ms = 62.5 steps at 8ms/step
#   shape=2, rate=0.032 → mean=62.5 steps=500ms, std=44 steps=354ms
#
# This prevents the model from switching states faster than ~150ms
# (= mean - std), which matches the minimum physiological gesture duration.
#
# All off-diagonal entries stay 0 — we only control the dwell time,
# not the transition destination (which is handled by initial_fractions).
transition_rates  = 0.032 * np.eye(3, dtype='float64')
transition_shapes = 2.0   * np.eye(3, dtype='float64')

# With true labels: free d, l, q; fix sigma
vary_params = np.ones((3, 8), dtype='float64')
vary_params[:, 0] = 0.0   # fix sigma
vary_params[:, 4] = 0.0   # fix sigma

# Also fix transition dynamics — we set them from physiology,
# don't want the optimizer to learn faster transitions
vary_transition_rates  = np.zeros((3, 3), dtype='float64')
vary_transition_shapes = np.zeros((3, 3), dtype='float64')

# ===========================================================================
# 5. Build model
# ===========================================================================
nb_states     = 3
seg_len_model = seg_steps
batch_size    = 50
reference_dt  = hop_len / fs

model, pred_model = build_emg_model(
    seg_len_model, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    batch_size, reference_dt,
    sequence_length=3, max_linking_distance=1,
    vary_params=vary_params,
    vary_initial_params=np.ones((nb_states, 2)),
    vary_transition_rates=vary_transition_rates,
    vary_transition_shapes=vary_transition_shapes)

print(f"\nModel built — trainable params: "
      f"{sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# 6. Supervised loss
# ===========================================================================
def supervised_MLE_loss(lp, labels, sequence_length):
    """Log-likelihood under correct state hypotheses only."""
    nb_sequences = lp.shape[1]
    hyp_states   = torch.arange(nb_sequences, device=lp.device) // sequence_length
    label_mask   = (hyp_states[None, :] == labels[:, None]).to(lp.dtype)
    masked_lp    = lp + torch.log(label_mask + 1e-40)
    max_lp       = masked_lp.max(dim=1, keepdim=True).values
    log_p        = torch.log(torch.exp(masked_lp - max_lp).sum(dim=1)) + max_lp[:, 0]
    return -log_p.mean()

def semisupervised_loss(lp, labels, sequence_length, nb_dims,
                        supervised_weight=5.0):
    sup_loss = supervised_MLE_loss(lp, labels, sequence_length)
    max_lp   = lp.max(dim=1, keepdim=True).values
    uns_loss = -(torch.log(torch.exp(lp - max_lp).sum(dim=1)) + max_lp[:, 0]).mean()
    return (supervised_weight * sup_loss + uns_loss) / nb_dims

# ===========================================================================
# 7. Forward patch
# ===========================================================================
def spectral_forward(self, inputs, input_LocErrs, input_dts, input_mask,
                     input_isfirst, return_all=False):
    device = next(self.parameters()).device
    inputs        = inputs.to(device)
    input_LocErrs = input_LocErrs.to(device)
    input_dts     = input_dts.to(device)
    input_mask    = input_mask.to(device)
    input_isfirst = input_isfirst.to(device)
    reshaped   = inputs[:, None, :, None, :, :]
    transposed = reshaped.permute(2, 1, 0, 3, 4, 5)
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
# 8. DataLoader
# ===========================================================================
sig_t   = torch.tensor(specs_model,  dtype=torch.float64)
le_t    = torch.ones(n_segs, seg_len_model, dtype=torch.float64)
dt_t    = torch.full((n_segs, seg_len_model+1), reference_dt, dtype=torch.float64)
mask_t  = torch.ones(n_segs, seg_len_model, dtype=torch.float64)
first_t = torch.ones(n_segs, dtype=torch.float64)
label_t = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(sig_t, le_t, dt_t, mask_t, first_t, label_t)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"DataLoader: {len(loader)} batches × {batch_size} segments")

# ===========================================================================
# 9. Training
# ===========================================================================
nb_epochs         = 500
supervised_weight = 5.0
device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate     = 0.005
print(f"\nUsing device: {device}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                              betas=(0.9,0.999), eps=1e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=nb_epochs, eta_min=1e-5)

model.train(); model.to(device)

# Gradient diagnostic
diag = [t.to(device) for t in next(iter(loader))]
optimizer.zero_grad()
lp_d   = model(*diag[:5])
loss_d = semisupervised_loss(lp_d, diag[5], 3, n_dims, supervised_weight)
loss_d.backward()
grad_norms = {n: p.grad.abs().max().item()
              for n, p in model.named_parameters() if p.grad is not None}
p25      = float(np.percentile(list(grad_norms.values()), 25))
clip_val = max(p25 * 5, 1e-4)
if max(grad_norms.values()) > 1e3:
    clip_val = min(clip_val, 0.1)
print(f"clip_value = {clip_val:.4e}")
optimizer.zero_grad()

loss_history = []; best_loss = float('inf'); best_state = None

print(f"\nTraining {nb_epochs} epochs  supervised_weight={supervised_weight}")
print(f"{'Epoch':>6}  {'Loss':>10}  {'Best':>10}  {'LR':>10}")
print("-" * 44)

for epoch in range(nb_epochs):
    epoch_losses = []
    for signals_b, le_b, dt_b, mask_b, first_b, labels_b in loader:
        signals_b = signals_b.to(device); le_b     = le_b.to(device)
        dt_b      = dt_b.to(device);      mask_b   = mask_b.to(device)
        first_b   = first_b.to(device);   labels_b = labels_b.to(device)
        optimizer.zero_grad()
        lp   = model(signals_b, le_b, dt_b, mask_b, first_b)
        loss = semisupervised_loss(lp, labels_b, 3, n_dims, supervised_weight)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_val)
        optimizer.step()
        epoch_losses.append(loss.item())

    epoch_loss = float(np.mean(epoch_losses))
    loss_history.append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss  = epoch_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"{epoch+1:>6}  {epoch_loss:>10.4f}  {best_loss:>10.4f}  {current_lr:>10.2e}")

model.load_state_dict(best_state, strict=False)
print(f"\nRestored best model (loss={best_loss:.4f})")

# ===========================================================================
# 10. Learned parameters
# ===========================================================================
p_learned = model.init_layer.param_vars.detach().cpu().numpy()
print("\nLearned parameters:")
print(f"{'State':<12} {'d1':>8} {'l1':>6} {'q1':>8} {'d2':>8} {'l2':>6} {'q2':>8}")
for i in range(nb_states):
    print(f"{state_names[i]:<12} "
          f"{np.exp(p_learned[i,1]):>8.4f} "
          f"{1/(1+np.exp(-p_learned[i,2])):>6.3f} "
          f"{np.exp(p_learned[i,3]):>8.4f} "
          f"{np.exp(p_learned[i,5]):>8.4f} "
          f"{1/(1+np.exp(-p_learned[i,6])):>6.3f} "
          f"{np.exp(p_learned[i,7]):>8.4f}")

# ===========================================================================
# 11. Inference
# ===========================================================================
model.eval()
all_states_list, all_lp_list = [], []
with torch.no_grad():
    for start in range(0, n_segs, batch_size):
        end    = min(start + batch_size, n_segs)
        actual = end - start
        def pad(t):
            b = t[start:end]
            if actual < batch_size:
                b = torch.cat([b, b[:1].expand(batch_size-actual, *b.shape[1:])], 0)
            return b.to(device)
        lp_b, states_b, _, _, _ = pred_model(
            *[pad(t) for t in [sig_t, le_t, dt_t, mask_t, first_t]],
            return_all=True)
        all_states_list.append(states_b[:actual].cpu().numpy())
        all_lp_list.append(lp_b[:actual].cpu().numpy())

state_probs = np.concatenate(all_states_list, axis=0)
all_lp      = np.concatenate(all_lp_list, axis=0)
max_lp      = all_lp.max(axis=1, keepdims=True)
ll_per_seg  = np.log(np.exp(all_lp - max_lp).sum(axis=1)) + max_lp[:, 0]

mean_probs  = state_probs[:, :, :nb_states].mean(axis=1)
predicted_raw = mean_probs.argmax(axis=1)

# Post-processing: median filter to remove transition boundary errors.
# Each segment is 40ms; window=5 → 200ms smoothing.
# Removes isolated wrong predictions shorter than 100ms at state boundaries.
from scipy.signal import medfilt
median_window = 5   # segments — must be odd
predicted = medfilt(predicted_raw.astype(float), kernel_size=median_window).astype(int)

accuracy_raw      = (predicted_raw == labels).mean()
accuracy_filtered = (predicted     == labels).mean()
print(f"\nAccuracy (raw predictions):      {accuracy_raw:.1%}")
print(f"Accuracy (median filter w={median_window}):  {accuracy_filtered:.1%}")
accuracy = accuracy_filtered

print(f"\nPer-class accuracy (filtered):")
for k, name in enumerate(state_names):
    mask = labels == k
    if mask.sum() > 0:
        print(f"  {name}: {(predicted[mask]==k).mean():.1%} ({mask.sum()} segs)")

# ===========================================================================
# 12. Visualisation
# ===========================================================================
t_raw    = np.arange(n_samples) / fs
t_seg    = np.arange(n_segs) * (seg_steps * hop_len) / fs
t_frames = np.arange(n_frames) * hop_len / fs
colors   = ['steelblue', 'tomato', 'seagreen']

# Recompute log-amplitude spectrogram for display only
# (model uses real+imag, but log-amp is more readable in a spectrogram)
log_amp_display = np.zeros((n_frames, n_freq_ac, 2), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*hop_len : i*hop_len + fft_len] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    log_amp_display[i] = np.log1p(np.abs(fft_out[1:]))
la_mean = log_amp_display.mean(axis=0, keepdims=True)
la_std  = log_amp_display.std(axis=0,  keepdims=True)
la_z    = (log_amp_display - la_mean) / (la_std + 1e-8)
spec_display = la_z[:, :, 0]   # CH1 log-amplitude
spec_ch2     = la_z[:, :, 1]   # CH2 log-amplitude

fig = plt.figure(figsize=(14, 18))
fig.suptitle(f'EMG model — supervised  raw={accuracy_raw:.1%}  filtered={accuracy_filtered:.1%}', fontsize=13)

# Use GridSpec for flexible layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(7, 1, figure=fig, hspace=0.50)
ax0 = fig.add_subplot(gs[0])   # raw signal
ax1 = fig.add_subplot(gs[1])   # CH1 spectrogram
ax2 = fig.add_subplot(gs[2])   # CH2 spectrogram
ax3 = fig.add_subplot(gs[3])   # state probabilities
ax4 = fig.add_subplot(gs[4])   # predicted vs ground truth (bar style)
ax5 = fig.add_subplot(gs[5])   # log-likelihood
ax6 = fig.add_subplot(gs[6])   # training loss

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.sharex(ax0)

# Panel 0: raw signal
ax0.plot(t_raw, raw_z[:,0], lw=0.3, color='steelblue', alpha=0.8, label='CH1')
ax0.plot(t_raw, raw_z[:,1], lw=0.3, color='tomato',    alpha=0.8, label='CH2')
ax0.set_ylabel('Amplitude'); ax0.legend(loc='upper right', fontsize=8)
ax0.set_title('Raw z-scored EMG')

# Panel 1: CH1 spectrogram
ax1.pcolormesh(t_frames, freqs[1:], spec_display.T,
               cmap='RdBu_r', vmin=-2, vmax=4, shading='auto')
ax1.set_ylabel('Freq (Hz)')
ax1.set_title('CH1 log-amplitude spectrogram')



# Panel 2: CH2 spectrogram
ax2.pcolormesh(t_frames, freqs[1:], spec_ch2.T,
               cmap='RdBu_r', vmin=-2, vmax=4, shading='auto')
ax2.set_ylabel('Freq (Hz)')
ax2.set_title('CH2 log-amplitude spectrogram')



# Panel 3: state probabilities
for k in range(nb_states):
    ax3.plot(t_seg, mean_probs[:,k], lw=1.2, color=colors[k], label=state_names[k])
ax3.set_ylabel('P(state)'); ax3.set_ylim(-0.05, 1.05)
ax3.legend(loc='upper right', fontsize=8)
ax3.set_title('State probabilities')

# Panel 4: predicted vs ground truth as filled horizontal bands
# Ground truth: solid thick band; predicted: thinner band below
# Much cleaner than overlapping scatter
band_h = 0.4   # height of each band
for s in range(n_segs):
    t0 = t_seg[s]; t1 = t0 + seg_steps * hop_len / fs
    # Ground truth: wide transparent band
    ax4.barh(labels[s],        t1-t0, left=t0, height=0.55,
             color=colors[labels[s]], alpha=0.3)
    # Raw prediction: narrow opaque band above centre
    ax4.barh(predicted_raw[s]+0.30, t1-t0, left=t0, height=0.25,
             color=colors[predicted_raw[s]], alpha=0.85)
    # Filtered prediction: narrow opaque band below centre
    ax4.barh(predicted[s]-0.30, t1-t0, left=t0, height=0.25,
             color=colors[predicted[s]], alpha=0.85)

ax4.set_yticks([0, 1, 2]); ax4.set_yticklabels(state_names)
ax4.set_ylim(-0.7, 2.8); ax4.set_ylabel('State')
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='gray', alpha=0.3,  label=f'Ground truth'),
    Patch(facecolor='gray', alpha=0.85, label=f'Raw pred ({accuracy_raw:.1%})'),
    Patch(facecolor='gray', alpha=0.85, label=f'Filtered pred ({accuracy_filtered:.1%})'),
]
ax4.legend(handles=legend_elems, loc='upper right', fontsize=8)
ax4.set_title(f'Ground truth / raw prediction / filtered prediction  '
              f'(median w={median_window}, {median_window*seg_steps*hop_len/fs*1000:.0f}ms)')

# Panel 5: log-likelihood
ax5.plot(t_seg, ll_per_seg, lw=0.8, color='gray')
ax5.set_ylabel('Log-lik'); ax5.set_xlabel('Time (s)')
ax5.set_title('Per-segment log-likelihood')

# Panel 6: training loss (independent x-axis)
ax6.plot(loss_history, lw=1.2, color='navy')
ax6.set_ylabel('Loss'); ax6.set_xlabel('Epoch')
ax6.set_title('Training loss')
ax6.set_xlim(0, len(loss_history)-1)

plt.savefig('emg_supervised.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to emg_supervised.png")