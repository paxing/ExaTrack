# -*- coding: utf-8 -*-
"""
validate_emg_spectro.py — evaluate the trained spectrogram ExaTrack model on a
SEPARATE acquisition (unseen during training), emulating real-time inference.

Option A (simple): each segment is classified independently (isfirst=1), exactly
as the training script's offline inference did. Segments are processed one at a
time in arrival order to emulate streaming. If results are promising we can move
to Option B (continuous carryover across segments).

CRITICAL: the validation signal is normalised with the TRAINING statistics
(mean_raw/std_raw, feat_mean/feat_std from the checkpoint), NOT its own — so the
"unseen data" evaluation is not contaminated by re-fitting the normalisation.
"""

import time
import numpy as np
import pandas as pd
import torch
import types
from scipy.signal import medfilt

from exatrack_torch.emg_constraints import build_emg_model

# ===========================================================================
# 0. Config — point these at your files
# ===========================================================================
CKPT_PATH = 'emg_spectro_checkpoint.pt'
VAL_CSV   = 'data/EMG/training_acquisition_23-03-28_20-10-47.csv'   # <-- your unseen file

# Validation label sequence (same gesture protocol). Set to None if the
# validation recording has no labels (then we only show predictions, no accuracy).
# 0=rest, 1=flexion, -1=extension
VAL_LABEL_SEQ = [0,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,1,0,0,-1,0,0,-1,0,0,1,0,0,-1,0]

VAL_TIMING_OFFSET = 0.0

EMULATE_REALTIME_DELAY = False   # if True, sleep per segment to mimic acquisition rate

# ===========================================================================
# 1. Load checkpoint
# ===========================================================================
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
fs         = ckpt['fs']
fft_len    = ckpt['fft_len']
hop_len    = ckpt['hop_len']
seg_steps  = ckpt['seg_steps']
n_freq_ac  = ckpt['n_freq_ac']
n_dims     = ckpt['n_dims']
nb_states  = ckpt['nb_states']
state_names = ckpt['state_names']
reference_dt = ckpt['reference_dt']
batch_size   = ckpt['batch_size']

mean_raw  = ckpt['mean_raw'];  std_raw  = ckpt['std_raw']
feat_mean = ckpt['feat_mean']; feat_std = ckpt['feat_std']
window_fn = np.hanning(fft_len)

print(f"Loaded checkpoint: fft_len={fft_len} hop={hop_len} seg_steps={seg_steps} "
      f"n_dims={n_dims}")

# ===========================================================================
# 2. Load validation signal — normalise with TRAINING stats
# ===========================================================================
df  = pd.read_csv(VAL_CSV)
raw = df[['voltage1 (V)', 'voltage2 (V)']].values.astype('float64')
n_samples = len(raw)
print(f"Validation: {n_samples} samples ({n_samples/fs:.2f}s)")

raw_z = (raw - mean_raw) / (std_raw + 1e-8)    # TRAINING mean/std

n_frames = (n_samples - fft_len) // hop_len + 1
n_segs   = n_frames // seg_steps

# ===========================================================================
# 3. FFT features — normalise with TRAINING feat_mean/feat_std
# ===========================================================================
features = np.zeros((n_frames, n_dims, 2), dtype='float64')
for i in range(n_frames):
    frame   = raw_z[i*hop_len : i*hop_len + fft_len] * window_fn[:, None]
    fft_out = np.fft.rfft(frame, axis=0)
    fft_ac  = fft_out[1:]
    features[i, :n_freq_ac, :] = fft_ac.real
    features[i, n_freq_ac:, :] = fft_ac.imag

feat_z    = (features - feat_mean) / (feat_std + 1e-8)   # TRAINING stats
feat_segs = feat_z[:n_segs*seg_steps].reshape(n_segs, seg_steps, n_dims, 2)
specs     = feat_segs.transpose(0, 1, 3, 2)              # (n_segs, seg_steps, 2, n_dims)

# ===========================================================================
# 4. Labels (optional) — same mapping as training
# ===========================================================================
have_labels = VAL_LABEL_SEQ is not None
if have_labels:
    label_seq     = VAL_LABEL_SEQ
    n_events      = len(label_seq)
    event_samples = n_samples / n_events
    labels = np.zeros(n_segs, dtype=int)
    for s in range(n_segs):
        t_centre = (s * seg_steps + seg_steps/2) * hop_len / fs
        evt_idx  = int(t_centre / (event_samples/fs) - VAL_TIMING_OFFSET)
        evt_idx  = max(0, min(evt_idx, n_events - 1))
        raw_lbl  = label_seq[evt_idx]
        labels[s] = 2 if raw_lbl == -1 else raw_lbl
    print("Validation label distribution:")
    for k, name in enumerate(state_names):
        n = (labels == k).sum()
        print(f"  {name}: {n} ({n/n_segs:.1%})")

# ===========================================================================
# 5. Rebuild the identical model and load trained weights
# ===========================================================================
model, pred_model = build_emg_model(
    seg_steps, nb_states, ckpt['params'], ckpt['initial_params'],
    ckpt['transition_rates'], ckpt['transition_shapes'], ckpt['initial_fractions'],
    batch_size, reference_dt,
    sequence_length=3, max_linking_distance=1,
    vary_params=np.ones((nb_states, 8)),
    vary_initial_params=np.ones((nb_states, 2)),
    vary_transition_rates=np.zeros((nb_states, nb_states)),
    vary_transition_shapes=np.zeros((nb_states, nb_states)))

# Same forward patch as training (spectral input shape).
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
    if return_all:
        return outputs, All_states, All_coefs, All_biases, All_LPs
    return outputs

model.forward      = types.MethodType(spectral_forward, model)
pred_model.forward = types.MethodType(spectral_forward, pred_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
pred_model.load_state_dict(ckpt['state_dict'], strict=False)
model.to(device); pred_model.to(device)
model.eval(); pred_model.eval()
print(f"Model loaded on {device}")

# Confirm the trained parameters came through
p = model.init_layer.param_vars.detach().cpu().numpy()
print("\nLoaded parameters (sanity check):")
print(f"{'State':<12} {'d1':>8} {'l1':>6} {'q1':>8} {'d2':>8} {'l2':>6} {'q2':>8}")
for i in range(nb_states):
    print(f"{state_names[i]:<12} {np.exp(p[i,1]):>8.4f} "
          f"{1/(1+np.exp(-p[i,2])):>6.3f} {np.exp(p[i,3]):>8.4f} "
          f"{np.exp(p[i,5]):>8.4f} {1/(1+np.exp(-p[i,6])):>6.3f} "
          f"{np.exp(p[i,7]):>8.4f}")

# ===========================================================================
# 6. Real-time emulation — Option A: one segment at a time, each independent
# ===========================================================================
# Each segment is decoded with isfirst=1 (no cross-segment carryover), matching
# the training-time offline inference. We feed a full batch where only the first
# row is the current segment (the model needs batch_size rows); we read row 0.
seg_t = torch.tensor(specs, dtype=torch.float64)
le_1  = torch.ones(batch_size, seg_steps, dtype=torch.float64)
dt_1  = torch.full((batch_size, seg_steps+1), reference_dt, dtype=torch.float64)
mask_1 = torch.ones(batch_size, seg_steps, dtype=torch.float64)
first_1 = torch.ones(batch_size, dtype=torch.float64)

predicted_raw = np.zeros(n_segs, dtype=int)
seg_proc_ms   = []

print(f"\nStreaming {n_segs} segments (Option A, independent per-segment)...")
with torch.no_grad():
    for s in range(n_segs):
        t0 = time.perf_counter()
        # build a batch whose first row is the current segment, rest are padding
        sig_b = seg_t[s:s+1].expand(batch_size, *seg_t.shape[1:]).contiguous().to(device)
        lp_b, states_b, _, _, _ = pred_model(
            sig_b, le_1, dt_1, mask_1, first_1, return_all=True)
        # state_probs for this segment: average over the seg_steps then argmax
        sp = states_b[0].cpu().numpy()                 # (seg_steps, nb_seq?) -> states
        # mean over the within-segment steps, take first nb_states
        mean_probs = sp[:, :nb_states].mean(axis=0)
        predicted_raw[s] = int(mean_probs.argmax())
        print( int(mean_probs.argmax()))
        seg_proc_ms.append((time.perf_counter() - t0) * 1000)
        if EMULATE_REALTIME_DELAY:
            time.sleep(max(0, seg_steps*hop_len/fs - (time.perf_counter()-t0)))

seg_dur_ms = seg_steps * hop_len / fs * 1000
print(f"Mean processing time/segment: {np.mean(seg_proc_ms):.2f} ms "
      f"(segment covers {seg_dur_ms:.1f} ms of signal)")
if np.mean(seg_proc_ms) < seg_dur_ms:
    print("  ✓ Faster than real-time (inference < segment duration)")
else:
    print("  ✗ Slower than real-time — would lag in a live system")

# ===========================================================================
# 7. Causal smoothing (real-time-honest) + offline median for comparison
# ===========================================================================
# Causal: median of the trailing w segments only (no peeking into the future).
def causal_median(x, w=5):
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = int(np.median(x[lo:i+1]))
    return out

pred_causal  = causal_median(predicted_raw, w=5)
pred_offline = medfilt(predicted_raw.astype(float), kernel_size=5).astype(int)

# ===========================================================================
# 8. Results
# ===========================================================================
if have_labels:
    def report(pred, tag):
        overall = (pred == labels).mean()
        per = [ (labels==k).sum() and (pred[labels==k]==k).mean() or 0.0
                for k in range(nb_states)]
        bal = np.mean(per)
        print(f"\n[{tag}] overall={overall:.1%}  balanced={bal:.1%}")
        for k, name in enumerate(state_names):
            m = labels == k
            if m.sum():
                print(f"  {name}: {(pred[m]==k).mean():.1%} ({m.sum()} segs)")
        return overall

    report(predicted_raw, "raw (per-segment)")
    report(pred_causal,   "causal median w=5 (real-time-honest)")
    report(pred_offline,  "offline median w=5 (centered, non-causal)")

    # Confusion matrix on the causal (real-time) predictions
    pred = pred_causal
    cm = np.zeros((nb_states, nb_states), dtype=int)
    for t in range(nb_states):
        for pp in range(nb_states):
            cm[t, pp] = np.sum((labels == t) & (pred == pp))
    print("\nConfusion matrix (causal predictions; rows=true, cols=pred):")
    print(f"{'':>10} " + " ".join(f"{n:>10}" for n in state_names))
    for t, name in enumerate(state_names):
        print(f"{name:>10} " + " ".join(f"{cm[t,pp]:>10}" for pp in range(nb_states)))
    fe, ef = cm[1,2], cm[2,1]
    print(f"\nFlexion→Extension: {fe}    Extension→Flexion: {ef}")
else:
    print("\nNo labels provided — predicted gesture stream (causal):")
    print(pred_causal)
