# -*- coding: utf-8 -*-
"""
test_oscillatory.py
-------------------
Test the oscillatory_constraint_function on simulated particle tracks
containing directed, confined, and oscillatory motion.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from exatrack_torch.oscillatory_constraints import (
    oscillatory_constraint_function, build_oscillatory_model)
from exatrack_torch.models import MLE_loss
from exatrack_torch.training import WarmupLearningRateSchedule

# ===========================================================================
# 1. Simulation parameters
# ===========================================================================
np.random.seed(42)
reference_dt = 0.02
dt           = reference_dt
FOV          = 5.0
n_tracks     = 75        # 100 per class
track_len    = 100
nb_dims      = 2

sigma  = 0.02

d_con  = 0.10;  l_con = 0.50;  q_con = 0.001

d_dir  = 0.0;   v_dir = 0.05;  q_dir = 0.0     # pure drift, zero diffusion
angular_D_dir = 0.04                             # angular diffusion for curvature

# omega in rad/s — period = 0.6 s = 30 steps at reference_dt=0.02
omega     = 2*np.pi / (30 * reference_dt)   # rad/s, period=30 steps
omega_obs = omega * reference_dt              # rad/obs-step (used for model params)
A_osc  = 1.5
d_osc  = 0.005
q_osc  = A_osc * omega_obs**2          # initial velocity spread: A*(omega_obs)^2
q_sim_osc = A_osc * omega_obs * 0.1   # ~10% of peak velocity A*omega_obs

# ===========================================================================
# 2. Simulation functions
# ===========================================================================
def simulate_confined(n, T, sigma, d, l, q, nb_dims, fov=1.0):
    """OU motion, well centre starts randomly in FOV."""
    tracks = np.zeros((n, T, nb_dims))
    for dim in range(nb_dims):
        u = np.random.uniform(-fov, fov, n)
        r = u + np.random.randn(n) * d / np.sqrt(2 * (-np.log(1-l)))
        for t in range(T):
            tracks[:, t, dim] = r + np.random.randn(n) * sigma
            r_new = (1-l)*r + l*u + np.random.randn(n) * d
            u    += np.random.randn(n) * q
            r     = r_new
    return tracks

def simulate_directed(n, T, sigma, d, v, q, nb_dims, fov=1.0, angular_D=0.04):
    """Pure drift with angular diffusion for realistic curvature."""
    tracks = np.zeros((n, T, nb_dims))
    angles = np.random.uniform(0, 2*np.pi, n)
    rx = np.random.uniform(-fov, fov, n)
    ry = np.random.uniform(-fov, fov, n)
    for t in range(T):
        tracks[:, t, 0] = rx + np.random.randn(n) * sigma
        tracks[:, t, 1] = ry + np.random.randn(n) * sigma
        angles += np.random.randn(n) * np.sqrt(2 * angular_D)
        rx += v * np.cos(angles)
        ry += v * np.sin(angles)
        if d > 0:
            rx += np.random.randn(n) * d
            ry += np.random.randn(n) * d
    return tracks

def simulate_oscillatory(n, T, sigma, d, omega, q_vel_noise, nb_dims, fov=1.0):
    """Harmonic oscillator with random centre in FOV.
    omega: rad/s. q_vel_noise: per-step velocity noise (NOT q_osc from model)."""
    tracks     = np.zeros((n, T, nb_dims))
    omega_step = omega * reference_dt   # rad/reference_step
    cos_w      = np.cos(omega_step)
    sin_w      = np.sin(omega_step)
    for dim in range(nb_dims):
        centre = np.random.uniform(-fov, fov, n)
        phase  = np.random.uniform(0, 2*np.pi, n)
        amp    = np.abs(np.random.randn(n)) * 0.3 + A_osc * 0.8
        r_rel  = amp * np.cos(phase)                      # initial position relative to centre
        vel    = -amp * omega_step * np.sin(phase)         # initial velocity (pos/ref_step)
        for t in range(T):
            r = centre + r_rel
            tracks[:, t, dim] = r + np.random.randn(n) * sigma
            r_rel_new = cos_w * r_rel + (sin_w/omega_step) * vel + np.random.randn(n)*d
            vel_new   = -omega_step*sin_w*r_rel + cos_w*vel + np.random.randn(n)*q_vel_noise
            r_rel = r_rel_new
            vel   = vel_new
    return tracks
n_each  = n_tracks // 3
n_total = n_each * 3

tracks_con = simulate_confined(   n_each, track_len, sigma, d_con, l_con, q_con,
                                  nb_dims, fov=FOV)
tracks_dir = simulate_directed(   n_each, track_len, sigma, d_dir, v_dir, q_dir,
                                  nb_dims, fov=FOV, angular_D=angular_D_dir)
tracks_osc = simulate_oscillatory(n_each, track_len, sigma, d_osc, omega, q_sim_osc,
                                  nb_dims, fov=FOV)

tracks      = np.concatenate([tracks_con, tracks_dir, tracks_osc], axis=0)
true_labels = np.array([0]*n_each + [1]*n_each + [2]*n_each)
n_tracks    = n_total

print(f"Simulated {n_tracks} tracks × {track_len} steps:")
print(f"  Confined:    {n_each}  d={d_con}, l={l_con}, q={q_con}")
print(f"  Directed:    {n_each}  v={v_dir}, angular_D={angular_D_dir}")
print(f"  Oscillatory: {n_each}  d={d_osc}, ω={omega:.4f}, A={A_osc}, "
      f"period={2*np.pi/(omega*reference_dt):.1f} steps, {track_len*reference_dt*omega/(2*np.pi):.1f} cycles")

# ===========================================================================
# 3. Visualise simulated tracks
# ===========================================================================
colors_sim = ['steelblue', 'tomato', 'seagreen']
titles_sim = [f'Confined  (l={l_con})',
              f'Directed  (v={v_dir})',
              f'Oscillatory  (ω={omega:.1f} rad/s, T={2*np.pi/(omega*reference_dt):.0f} steps)']
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Simulated tracks', fontsize=13)
for ax, data, title, col in zip(axes,
        [tracks_con, tracks_dir, tracks_osc], titles_sim, colors_sim):
    for i in range(min(20, n_each)):
        ax.plot(data[i, :, 0], data[i, :, 1], lw=0.6, alpha=0.5, color=col)
    ax.set_title(title); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('oscillatory_simulated_overlaid.png', dpi=120, bbox_inches='tight')
plt.show()

# ===========================================================================
# 4. Model parameters
# ===========================================================================
nb_states  = 3
batch_size = 50

# All initial values offset from truth to ensure nonzero gradients.
# sigma*2: offset so gradient flows; d*2: overestimate noise; motion params offset.
params = np.array([
    # log_sigma,           log_d,              motion_param,       log_q,             is_dir, is_osc
    [np.log(sigma*2), np.log(d_con*2),    0.0,               np.log(q_con*3),    0.0, 0.0],
    [np.log(sigma*2), np.log(0.02),       np.log(v_dir*0.5), np.log(0.001),      1.0, 0.0],
    [np.log(sigma*2), np.log(d_osc*2),    np.log(omega_obs*1.5), np.log(q_osc*0.5),  0.0, 1.0],  # omega_obs in rad/obs-step
], dtype='float64')

# Note for directed: d_dir=0 cannot be log-transformed.
# Use a small but positive value as initial d (e.g. 0.02) — model will learn small d.

l_c_val = -np.log(1 - l_con)
initial_params = np.array([[np.log(FOV)]] * nb_states, dtype='float64')

initial_fractions = np.array([[0.0, 0.0, 0.0, -5.0]], dtype='float64')
transition_rates  = 1.0 * np.eye(nb_states, dtype='float64')
transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')

vary_params = np.ones((nb_states, 6), dtype='float64')
# Fix motion params to prevent label switching:
#   confined:    logit_l fixed (defines OU structure)
#   directed:    log_v  fixed  (defines drift scale)
#   oscillatory: log_omega fixed (defines period)
vary_params[0, 2] = 0.0
vary_params[1, 2] = 0.0
vary_params[2, 2] = 0.0
# is_dir and is_osc fixed automatically inside build_oscillatory_model

vary_transition_rates  = np.zeros((nb_states, nb_states), dtype='float64')
vary_transition_shapes = np.zeros((nb_states, nb_states), dtype='float64')

model, pred_model = build_oscillatory_model(
    track_len, nb_states, params, initial_params,
    transition_rates, transition_shapes, initial_fractions,
    batch_size, reference_dt,
    nb_dims=nb_dims, sequence_length=3,
    max_linking_distance=3, estimated_density=1e-4,
    vary_params=vary_params,
    vary_initial_params=np.ones((nb_states, 1)),
    vary_initial_fractions=np.ones((1, nb_states+1)),
    vary_transition_rates=vary_transition_rates,
    vary_transition_shapes=vary_transition_shapes,
    LocErr_type='Constant')   # sigma learned from data via exp(log_sigma)

p_init = model.init_layer.param_vars.detach().cpu().numpy()
print(f"Initial sigma: {np.exp(p_init[:,0]).round(4).tolist()}")
print(f"Model trainable params: {sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# 5. DataLoader — interleaved for balanced batches
# ===========================================================================
interleave_idx = np.empty(n_total, dtype=int)
for k in range(3):
    interleave_idx[k::3] = np.arange(k*n_each, (k+1)*n_each)

tracks_int      = tracks[interleave_idx]
true_labels_int = true_labels[interleave_idx]

sig_t   = torch.tensor(tracks_int, dtype=torch.float64)
le_t    = torch.ones(n_total, track_len, dtype=torch.float64) * sigma
dt_t    = torch.full((n_total, track_len+1), reference_dt, dtype=torch.float64)
mask_t  = torch.ones(n_total, track_len, dtype=torch.float64)
first_t = torch.ones(n_total, dtype=torch.float64)

dataset = TensorDataset(sig_t, le_t, dt_t, mask_t, first_t)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
print(f"DataLoader: {len(loader)} batches × {batch_size} tracks")

# ===========================================================================
# 6. Training
# ===========================================================================
nb_epochs     = 100
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

nb_batches      = len(loader)
learning_rate   = 0.005
epoch_decay     = 20
decay_threshold = epoch_decay * nb_batches
decay_rate      = 0.005

lr_schedule = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold)
optimizer   = torch.optim.Adam(model.parameters(), lr=lr_schedule(0),
                                betas=(0.99, 0.999), eps=1e-7)
model.train(); model.to(device)

loss_history = []; best_loss = float('inf'); best_state = None
global_step  = 0

print(f"\n{'Epoch':>6}  {'Loss':>10}  {'Best':>10}  {'LR':>10}")
print("-" * 44)

for epoch in range(nb_epochs):
    epoch_losses = []
    for sig_b, le_b, dt_b, mask_b, first_b in loader:
        sig_b   = sig_b.to(device); le_b   = le_b.to(device)
        dt_b    = dt_b.to(device);  mask_b = mask_b.to(device)
        first_b = first_b.to(device)
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
        best_loss  = epoch_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    lr = optimizer.param_groups[0]['lr']
    print(f"{epoch+1:>6}  {epoch_loss:>10.4f}  {best_loss:>10.4f}  {lr:>10.2e}")

model.load_state_dict(best_state, strict=False)
print(f"\nRestored best model (loss={best_loss:.4f})")

# ===========================================================================
# 7. Learned parameters
# ===========================================================================
p = model.init_layer.param_vars.detach().cpu().numpy()
state_names = ['Confined', 'Directed', 'Oscillatory']
print("\nLearned parameters:")
print(f"{'State':<14} {'σ':>8} {'d':>8} {'motion':>10} {'q':>8}")
for i, name in enumerate(state_names):
    sigma_l = np.exp(p[i, 0])
    d_l     = np.exp(p[i, 1])
    q_l     = np.exp(p[i, 3])
    if params[i, 4] >= 0.5:
        motion_str = f"v={np.exp(p[i,2]):.4f}"
    elif params[i, 5] >= 0.5:
        # p[i,2] = log(omega_obs) in rad/obs-step; period = 2pi/omega_obs obs-steps
        motion_str = f"ω={np.exp(p[i,2]):.4f} rad/obs-step (period={2*np.pi/np.exp(p[i,2]):.1f} steps)"
    else:
        motion_str = f"l={1/(1+np.exp(-p[i,2])):.4f}"
    print(f"  {name:<12} {sigma_l:>8.4f} {d_l:>8.4f} {motion_str:>10} {q_l:>8.4f}")

print(f"\nTrue values:")
print(f"  {'Confined':<12} {sigma:>8.4f} {d_con:>8.4f} {'l='+str(l_con):>10} {q_con:>8.4f}")
print(f"  {'Directed':<12} {sigma:>8.4f} {'~0':>8} {'v='+str(v_dir):>10} {'~0':>8}")
print(f"  {'Oscillatory':<12} {sigma:>8.4f} {d_osc:>8.4f} {'ω_obs='+str(round(omega_obs,4)):>14} {q_osc:>8.4f}")

# ===========================================================================
# 8. Inference
# ===========================================================================
model.eval()
all_lp = []; all_preds = []
with torch.no_grad():
    for start in range(0, n_tracks, batch_size):
        end    = min(start + batch_size, n_tracks)
        actual = end - start
        def pad(t):
            b = t[start:end]
            if actual < batch_size:
                b = torch.cat([b, b[:1].expand(batch_size-actual, *b.shape[1:])], 0)
            return b.to(device)
        lp_b, preds_b, _, _, _ = model(
            *[pad(t) for t in [sig_t, le_t, dt_t, mask_t, first_t]],
            return_all=True)
        all_lp.append(lp_b[:actual].cpu())
        all_preds.append(preds_b[:actual].cpu())

all_lp    = torch.cat(all_lp,    dim=0)
all_preds = torch.cat(all_preds, dim=0)

mean_preds = all_preds[:, :, :nb_states].mean(dim=1)

print("\nMean state probability per true class:")
print(f"{'':>12} {'Con':>10} {'Dir':>10} {'Osc':>10}")
for k, name in enumerate(state_names):
    m  = true_labels_int == k
    mp = mean_preds[m].mean(dim=0)
    print(f"  {name:<10} {mp[0].item():>10.4f} {mp[1].item():>10.4f} {mp[2].item():>10.4f}")

predicted = mean_preds.argmax(dim=1).numpy()
accuracy  = (predicted == true_labels_int).mean()
print(f"\nAccuracy: {accuracy:.1%}")
for k, name in enumerate(state_names):
    m = true_labels_int == k
    print(f"  {name}: {(predicted[m]==k).mean():.1%} ({m.sum()} tracks)")

# ===========================================================================
# 9. Plots
# ===========================================================================
colors = ['steelblue', 'tomato', 'seagreen']

fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(loss_history, lw=1.0, color='navy')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('MLE loss')
ax1.set_title('Training loss — oscillatory constraint function')
plt.tight_layout()
plt.savefig('oscillatory_loss.png', dpi=120, bbox_inches='tight')
plt.show()

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle(f'Predicted motion types  (acc={accuracy:.1%})', fontsize=13)
for k, (name, col) in enumerate(zip(state_names, colors)):
    ax = axes3[k]
    mask_true = true_labels_int == k
    correct   = mask_true & (predicted == k)
    wrong     = mask_true & (predicted != k)
    for i in np.where(correct)[0][:15]:
        ax.plot(tracks_int[i, :, 0], tracks_int[i, :, 1],
                lw=0.7, alpha=0.55, color=col)
    for i in np.where(wrong)[0][:8]:
        ax.plot(tracks_int[i, :, 0], tracks_int[i, :, 1],
                lw=0.6, alpha=0.4, color='gray', linestyle='--')
    acc_k = correct.sum() / mask_true.sum() if mask_true.sum() > 0 else 0
    ax.set_title(f'{name}  (acc={acc_k:.1%})\ncolour=correct  gray=wrong')
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('oscillatory_predicted.png', dpi=120, bbox_inches='tight')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm   = confusion_matrix(true_labels_int, predicted)
fig4, ax4 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=state_names).plot(
    ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(f'Confusion matrix  (acc={accuracy:.1%})')
plt.tight_layout()
plt.savefig('oscillatory_confusion.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nPlots saved.")