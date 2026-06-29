# -*- coding: utf-8 -*-
"""
pong_exatrack.py  —  Step 3 of 3

Loads the checkpoint produced by train_exatrack.py and plays Pong controlled
by live EMG classification using the ExaTrack model.

Usage
─────
  1. Set CKPT_PATH to the checkpoint printed by train_exatrack.py.
  2. Run:  python pong_exatrack.py

Gesture → paddle mapping
  Flexion   → paddle moves UP
  Extension → paddle moves DOWN
  Rest      → paddle stays still
"""

import numpy as np
import torch
import types
import threading
import time
import turtle
from collections import deque

import board
import busio
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

from exatrack_torch.emg_constraints import build_emg_model

# ===========================================================================
# CONFIGURATION — edit here
# ===========================================================================
CKPT_PATH    = 'emg_spectro_checkpoint.pt'   # <── set this

# Game feel
TARGET_FPS   = 30    # render rate (does not affect paddle sensitivity)
BALL_SPEED   = 5     # pixels the ball moves per frame
PADDLE_STEP  = 20    # pixels paddle moves per new EMG classification
PADDLE_LIMIT = 250   # max |y| of paddle centre
WIN_SCORE    = 15    # first to this score wins

# Causal smoothing — median over last SMOOTH_W classifications.
# Increase for a smoother but slightly laggier paddle.
SMOOTH_W     = 3

# ===========================================================================
# 1.  HARDWARE SETUP
# ===========================================================================
i2c  = busio.I2C(board.SCL, board.SDA)
ads  = ADS.ADS1115(i2c, data_rate=860, mode=256)
chan1 = AnalogIn(ads, ADS.P0)
chan2 = AnalogIn(ads, ADS.P1)

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, 0)
buzzer = GPIO.PWM(12, 10)

# ===========================================================================
# 2.  LOAD CHECKPOINT
# ===========================================================================
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)

fs           = ckpt['fs']
fft_len      = ckpt['fft_len']
hop_len      = ckpt['hop_len']
seg_steps    = ckpt['seg_steps']
n_freq_ac    = ckpt['n_freq_ac']
n_dims       = ckpt['n_dims']
nb_states    = ckpt['nb_states']
state_names  = ckpt['state_names']
reference_dt = ckpt['reference_dt']
batch_size   = ckpt['batch_size']
mean_raw     = ckpt['mean_raw']
std_raw      = ckpt['std_raw']
feat_mean    = ckpt['feat_mean']
feat_std     = ckpt['feat_std']
window_fn    = np.hanning(fft_len)

# Samples needed to fill one segment for inference
n_samples_per_seg = (seg_steps - 1) * hop_len + fft_len
print(f"  Segment size : {n_samples_per_seg} samples "
      f"({n_samples_per_seg/fs*1000:.1f} ms)")

# ===========================================================================
# 3.  REBUILD MODEL AND LOAD WEIGHTS
# ===========================================================================
def _spectral_forward(self, inputs, input_LocErrs, input_dts, input_mask,
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
    if return_all:
        return outputs, All_states, All_coefs, All_biases, All_LPs
    return outputs

_, pred_model = build_emg_model(
    seg_steps, nb_states,
    ckpt['params'], ckpt['initial_params'],
    ckpt['transition_rates'], ckpt['transition_shapes'],
    ckpt['initial_fractions'],
    batch_size, reference_dt,
    sequence_length=3, max_linking_distance=1,
    vary_params=np.ones((nb_states, 8)),
    vary_initial_params=np.ones((nb_states, 2)),
    vary_transition_rates=np.zeros((nb_states, nb_states)),
    vary_transition_shapes=np.zeros((nb_states, nb_states)))

pred_model.forward = types.MethodType(_spectral_forward, pred_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pred_model.load_state_dict(ckpt['state_dict'], strict=False)
pred_model.to(device)
pred_model.eval()
print(f"  ExaTrack model ready on {device}")

# Fixed tensors reused every inference call (avoids re-allocation each segment)
_le    = torch.ones(batch_size, seg_steps,    dtype=torch.float64)
_dt    = torch.full((batch_size, seg_steps+1), reference_dt, dtype=torch.float64)
_mask  = torch.ones(batch_size, seg_steps,    dtype=torch.float64)
_first = torch.ones(batch_size,               dtype=torch.float64)

# ===========================================================================
# 4.  PREPROCESSING + INFERENCE
# ===========================================================================
def classify_segment(raw_seg):
    """
    raw_seg : (n_samples_per_seg, 2) float64 — raw voltages from both channels

    Returns ExaTrack class index: 0=Rest, 1=Flexion, 2=Extension
    """
    # Z-score with TRAINING statistics (never refit on live data)
    raw_z = (raw_seg - mean_raw) / (std_raw + 1e-8)

    # FFT features for each frame in the segment
    feats = np.zeros((seg_steps, n_dims, 2), dtype='float64')
    for i in range(seg_steps):
        frame   = raw_z[i*hop_len : i*hop_len + fft_len] * window_fn[:, None]
        fft_out = np.fft.rfft(frame, axis=0)
        fft_ac  = fft_out[1:]
        feats[i, :n_freq_ac, :] = fft_ac.real
        feats[i, n_freq_ac:, :] = fft_ac.imag

    # Normalise with TRAINING feature statistics
    feat_z = (feats - feat_mean) / (feat_std + 1e-8)           # (seg_steps, n_dims, 2)
    spec   = feat_z.transpose(0, 2, 1)[np.newaxis]             # (1, seg_steps, 2, n_dims)

    sig_b = torch.tensor(spec, dtype=torch.float64) \
                 .expand(batch_size, -1, -1, -1).contiguous()

    with torch.no_grad():
        _, states_b, _, _, _ = pred_model(
            sig_b, _le, _dt, _mask, _first, return_all=True)

    mean_probs = states_b[0].cpu().numpy()[:, :nb_states].mean(axis=0)
    return int(mean_probs.argmax())

# ExaTrack label → paddle direction
_TO_DIR = {0: 0, 1: 1, 2: -1}   # Rest→still, Flexion→up, Extension→down

# ===========================================================================
# 5.  SHARED STATE
# ===========================================================================
current_dir   = 0
label_updated = False
label_lock    = threading.Lock()
game_running  = True

# ===========================================================================
# 6.  EMG CLASSIFICATION THREAD
# ===========================================================================
def emg_loop():
    global current_dir, label_updated, game_running

    recent = deque(maxlen=SMOOTH_W)   # causal median buffer

    while game_running:
        # Acquire one segment of raw samples
        buf = np.zeros((n_samples_per_seg, 2), dtype='float64')
        for i in range(n_samples_per_seg):
            buf[i, 0] = chan1.voltage
            buf[i, 1] = chan2.voltage

        label    = classify_segment(buf)
        recent.append(label)
        smoothed = int(np.median(recent))

        print(f"EMG  raw={state_names[label]:<10}  "
              f"smoothed={state_names[smoothed]:<10}  "
              f"dir={_TO_DIR[smoothed]:+d}")

        with label_lock:
            current_dir   = _TO_DIR[smoothed]
            label_updated = True

emg_thread = threading.Thread(target=emg_loop, daemon=True)
emg_thread.start()

# ===========================================================================
# 7.  PONG GAME
# ===========================================================================
wn = turtle.Screen()
wn.title("EMG Pong  |  Flex = UP   Extend = DOWN")
wn.bgcolor("black")
wn.setup(width=800, height=600)
wn.tracer(0)

# Left paddle — EMG controlled
paddle_a = turtle.Turtle()
paddle_a.speed(0); paddle_a.shape("square"); paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup(); paddle_a.goto(-350, 0)

# Right paddle — bot
paddle_b = turtle.Turtle()
paddle_b.speed(0); paddle_b.shape("square"); paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup(); paddle_b.goto(350, 0)

# Ball
ball = turtle.Turtle()
ball.speed(0); ball.shape("circle"); ball.color("white")
ball.penup(); ball.goto(0, 0)
ball.dx = BALL_SPEED; ball.dy = BALL_SPEED

# Score display
score_a = score_b = 0
pen = turtle.Turtle()
pen.speed(0); pen.color("white"); pen.penup(); pen.hideturtle()
pen.goto(0, 260)
pen.write("Human : 0   Bot : 0", align="center",
          font=("courier", 24, "normal"))

def bot():
    if (ball.xcor() > 0 and ball.dx > 0
            and np.abs(ball.ycor() - paddle_b.ycor()) > 50):
        paddle_b.goto(350,
            np.round((0.8 + 0.2*np.random.rand()) * ball.ycor()
                     + np.random.randint(-10, 10)))

PADDLE_HALF = 60   # 50px paddle half-height + 10px ball radius

# ===========================================================================
# 8.  MAIN GAME LOOP
# ===========================================================================
while True:
    time.sleep(1 / TARGET_FPS)
    wn.update()

    # Move paddle only once per fresh EMG classification
    with label_lock:
        direction     = current_dir
        updated       = label_updated
        label_updated = False

    if updated:
        y = paddle_a.ycor()
        if direction == 1 and y < PADDLE_LIMIT:
            paddle_a.sety(y + PADDLE_STEP)
        elif direction == -1 and y > -PADDLE_LIMIT:
            paddle_a.sety(y - PADDLE_STEP)

    # Ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)
    bot()

    # Walls
    if ball.ycor() > 290:
        ball.sety(290);  ball.dy *= -1
    if ball.ycor() < -290:
        ball.sety(-290); ball.dy *= -1

    # Ball out right → human scores
    if ball.xcor() > 390:
        ball.goto(0, 0); ball.dx = -abs(ball.dx)
        score_a += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # Ball out left → bot scores
    if ball.xcor() < -390:
        ball.goto(0, 0); ball.dx = abs(ball.dx)
        score_b += 1
        pen.clear()
        pen.write(f"Human : {score_a}   Bot : {score_b}",
                  align="center", font=("courier", 24, "normal"))

    # Collisions — crossed-line test prevents tunnelling
    if ball.dx > 0 and ball.xcor() >= 340:
        if abs(ball.ycor() - paddle_b.ycor()) < PADDLE_HALF:
            ball.setx(339); ball.dx *= -1

    if ball.dx < 0 and ball.xcor() <= -340:
        if abs(ball.ycor() - paddle_a.ycor()) < PADDLE_HALF:
            ball.setx(-339); ball.dx *= -1

    if score_a >= WIN_SCORE or score_b >= WIN_SCORE:
        game_running = False
        break

buzzer.stop()
turtle.done()
