# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:30:22 2026

@author: Franc


#kjjgfsklgf
#Next to do: adjust the memory of the transition processes so it considers the uneven dts
#Parse the current coefficients, biases, scaling factors
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# Import the ExaTrack module (ensure exatrack.py is in your path)
import sys
import os
try:
    rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
except:
    # add the absolute path if you are running the script line by line
    rootdir = r"C:\Users\Franc\Data\GitHub\ExaTrack"
sys.path.insert(0, rootdir)
import exatrack
#import exatrack as exatrack
from glob import glob

# %%

track_len = 100
nb_tracks = 500
reference_dt = 0.02                 # Time interval between frames (seconds)
LocErr = 0.02             # Localization error (µm)
nb_dims = 2               # Number of spatial dimensions

pu = 0.02
pb = 0.1
Ds = np.array([0.0, 0.25])
dt = 0.02
ds = (2*Ds*dt)**0.5
velocity = 0.005

tracks, all_LocErrs, all_dts, all_states, all_masks = exatrack.anomalous_diff_transition(
    max_track_len=track_len,
    nb_tracks=nb_tracks,
    LocErr=0.02,
    Fs=np.array([0.4, 0.6]),
    Ds=Ds,
    nb_dims=nb_dims,
    velocities=np.array([velocity, 0.0]),      # No directed motion
    angular_Ds=np.array([0.0, 0.0]),      # No rotational diffusion
    conf_forces=np.array([0.0, 0.2]),
    conf_Ds=np.array([0.0, 0.0]),         # No diffusion of confinement center
    conf_dists=np.array([0.0, 0.0]),
    transition_matrix=np.array([[0.00, 0.02],   # State 0 -> State 1
                                [0.05, 0.00]]),
    shape_matrix=np.array([[0, 1],
                           [1, 0]]),
    LocErr_std = 0.004,
    field_of_view=np.array([-1, 1]),
    dt=dt,
    dt_std = 0.01,
    nb_sub_steps=10,  # Sub-steps for accurate simulation
    nb_burning_steps=0,
    bleaching_rate = 0.02)

# Plot tracks
plt.figure(figsize = (15, 15))
lim = 1 # MreB
nb_rows = 4
IDs = random.sample(list(np.arange(len(tracks))), nb_rows**2)
for i in range(nb_rows):
    for j in range(nb_rows):
        ID = i*nb_rows+j #IDs[i*nb_rows+j]
        track = tracks[ID]
        track = track - np.mean(track,0 , keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], ':k', alpha = 0.5)
        plt.scatter(track[:,0], track[:,1] , c = cm.jet(np.linspace(0,1,len(track))), s = 8, marker = 'x')
plt.gca().set_aspect('equal', adjustable='box')
i=1
track_list = [tracks[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list and dt_list can be set to None if they are assumed to be constant
LocErr_list = [all_LocErrs[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]
# LocErr_list = None 
dt_list = [all_dts[i, all_masks[i].astype(bool)]  for i in range(len(tracks))]


# %%
# Define initial parameter guesses for 2 states
nb_states = 2

# Calculate d from diffusion coefficient: d = sqrt(2 * D * dt)
d_values = np.sqrt(2 * Ds * dt)

# Parameter array: [log(LocErr), log(d), anomalous_param, log(q), model_type]
# model_type: 0 = confined, 1 = directed
params = np.array([[np.log(0.025), np.log(0.1), np.log(0.1), np.log(0.01), 1],  # Directed state
                   [np.log(0.025), np.log(0.5), np.log(0.01), np.log(0.01), 1]], dtype='float64')  # Directed state

# Initial parameters (position spread at track start)
initial_params = np.array([[np.log(1.0)],
                           [np.log(1.0)]], dtype='float64')

# Initial fractions (will be optimized, using softmax internally)
# Last element accounts for mislinking probability
initial_fractions = np.array([[0.0, 0.0, -5.0]], dtype='float64')

# Transition rates (log-space, converted via softmax)
transition_rates = np.array([[4.0, 0.0],
                             [0.0, 4.0]], dtype='float64')

# Transition shapes - FIXED to 1 for exponential dwell times
# We use log(1) = 0 and then fix these parameters
transition_shapes = np.array([[0.0, 0.0],
                              [0.0, 0.0]], dtype='float64')

print("Initial Parameters:")
print(f"  params shape: {params.shape}")
print(f"  initial_params shape: {initial_params.shape}")
print(f"  transition_rates shape: {transition_rates.shape}")
print(f"  transition_shapes shape: {transition_shapes.shape}")



# Create vary masks to fix certain parameters
# vary_params: which recurrent parameters to optimize
vary_params = np.ones(params.shape, dtype='float64')
vary_params[:, 4] = 0  # Fix model type (we know it's confined motion)

# vary_transition_shapes: fix shapes to 1 (exponential)
vary_transition_shapes = np.zeros(transition_shapes.shape, dtype='float64')

# Allow other parameters to vary
vary_initial_params = np.ones(initial_params.shape, dtype='float64')
vary_initial_fractions = np.ones(initial_fractions.shape, dtype='float64')
vary_transition_rates = np.ones(transition_rates.shape, dtype='float64')

print("Parameter Variation Masks:")
print(f"  vary_params:\n{vary_params}")
print(f"  vary_transition_shapes (all zeros = fixed):\n{vary_transition_shapes}")


device_tf    = '/CPU:0'


# Model hyperparameters
batch_size = 100
sequence_length = 3  # Number of past states to consider
max_linking_distance = 0.5  # Maximum expected mislinking distance
estimated_density = 0.001  # Track density (for mislinking model)
#track_len = max_track_len

# Training configuration
epochs = 60
learning_rate = 1/30
decay_rate = 0.01
decay_threshold = 30 * nb_tracks // batch_size




model, pred_model = exatrack.Model_finder(
    track_list=track_list,
    reference_dt=reference_dt,
    sequence_length=sequence_length,
    nb_states=nb_states,
    params=params,
    initial_params=initial_params,
    initial_fractions=initial_fractions,
    transition_shapes=transition_shapes,
    transition_rates=transition_rates,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    epochs=epochs,
    batch_size=batch_size,
    LocErr_list=LocErr_list,
    dt_list=dt_list,
    learning_rate=learning_rate,
    decay_threshold=decay_threshold,
    decay_rate=decay_rate,
    device=device_tf,
    shuffle=True,
    verbose=1,
    vary_params=vary_params,
    vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates)

# Analyze the found model
found_params = exatrack.get_model_params(model)
print("\n" + "="*60)
print("MODEL FINDER RESULTS")
print("="*60)

print(f"\nDetected Motion Types: {found_params['Model types']}\n  (True: the first state should be confined and the second can be either)")
print(f"\nFitted Anomalous Factors: {found_params['anomalous factors']}")
print(f"\nDiffusion Parameters (d): {found_params['d']}\nLocalization Errors: {found_params['Localization errors']}")
print(f"\nState Fractions: {found_params['Fractions']}")




# Get state predictions using the prediction model
print("Computing state predictions...")

max_track_len = max(len(t) for t in track_list)
weights_tf = model.get_weights()

seq = exatrack.TrackSegmentSequence(
        track_list, LocErr_list, dt_list,
        batch_size=batch_size, segment_length=max_track_len,
        min_segment_length=4, cutoff_batch_treshhold=0.5)

# Rebuild pred_model with the fitted weights and the full track length.
# build_segment_model always uses carryover=True, so 3 non-trainable carryout
# buffers (coefs/biases/LP) sit at weights[4..6], pushing transition_rates
# and transition_shapes to weights[7] and weights[8].
_, pred_model = exatrack.build_segment_model(
    max_track_len, nb_states,
    params=weights_tf[0],
    initial_params=weights_tf[1],
    transition_rates=weights_tf[7],
    transition_shapes=weights_tf[8],
    initial_fractions=weights_tf[2],
    batch_size=batch_size,
    reference_dt=reference_dt,
    nb_dims=nb_dims,
    sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    vary_params=vary_params,
    vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates,
    LocErr_type='Linear')

with tf.device(device_tf):
    LPs_tf, preds_tf, All_coefs_tf, All_biases_tf, All_LPs_tf = pred_model.predict(seq)

print(f"State predictions shape: {preds_tf.shape}")
print(f"  (n_tracks, track_len, n_states+1)")
print(f"  Last state dimension is mislinking probability")



tracks_tf     = np.concatenate([seq[i][0][0] for i in range(len(seq))], 0)
LocErrs_tf    = np.concatenate([seq[i][0][1] for i in range(len(seq))], 0)
time_steps_tf = np.concatenate([seq[i][0][2] for i in range(len(seq))], 0)
masks_tf      = np.concatenate([seq[i][0][3] for i in range(len(seq))], 0)

#exatrack.correct_state_predictions_padding(preds_tf, masks_tf, sequence_length)
# Get the most likely state for each position
most_likely_states = np.argmax(preds_tf, axis=-1)

print(f"\nMost likely states shape: {most_likely_states.shape}")# Visualize state predictions for example tracks

def plot_state_predictions(tracks_data, true_states, pred_probs, masks, n_examples=4):
    """Plot tracks with true states and predicted state probabilities."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 3*n_examples))
    
    colors = ['#e74c3c', '#3498db', '#95a5a6']  # Confined, Free, Mislinking
    state_names = ['State 0 (Confined)', 'State 1 (Free)', 'Mislinking']
    
    for idx in range(n_examples):
        # Left plot: Track colored by true state
        ax1 = axes[idx, 0]
        track = tracks_data[idx, :, :]  # Extract actual positions
        state = true_states[idx].astype(int)
        mask = masks[idx].astype(bool)
        n_points = int(np.sum(mask))
        
        for i in range(n_points - 1):
            ax1.plot(track[i:i+2, 0], track[i:i+2, 1], 
                    color=colors[state[i]], linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('X (µm)')
        ax1.set_ylabel('Y (µm)')
        ax1.set_title(f'Track {idx+1}: True States')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: State probability over time
        ax2 = axes[idx, 1]
        time = np.arange(n_points) * dt
        
        for s in range(pred_probs.shape[-1]):
            ax2.plot(time, pred_probs[idx, :n_points, s], 
                    color=colors[s] if s < len(colors) else 'gray',
                    label=state_names[s] if s < len(state_names) else f'State {s}',
                    linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Track {idx+1}: State Probabilities')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        if idx == 0:
            ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

fig = plot_state_predictions(tracks, all_states, preds_tf, all_masks)
plt.savefig('state_predictions.png', dpi=150, bbox_inches='tight')
plt.show()



# Calculate state labeling accuracy
def calculate_accuracy(true_states, pred_probs, masks):
    """Calculate accuracy of state predictions."""
    pred_states = np.argmax(pred_probs[:, :, :-1], axis=-1)  # Exclude mislinking
    
    correct = 0
    total = 0
    
    for i in range(len(true_states)):
        n_points = int(np.sum(masks[i]))
        correct += np.sum(true_states[i, :n_points] == pred_states[i, :n_points])
        total += n_points
    
    return correct / total

accuracy = calculate_accuracy(all_states,  preds_tf, all_masks)
print(f"\nState Labeling Accuracy: {accuracy*100:.1f}%")
# Convert predictions to DataFrame format for further analysis
# This creates a table with positions, frames, track IDs, and state probabilities

# Create dummy frame and track ID lists for demonstration
frame_list = [np.arange(int(np.sum(m))) for m in all_masks]
track_ID_list = [np.array([i] * int(np.sum(m))) for i, m in enumerate(all_masks)]
track_list = [tracks[i, :int(np.sum(all_masks[i]))] for i in range(len(tracks))]

# Create DataFrame
df = exatrack.ExaTrack_2_DataFrame(
    track_list=track_list,
    frame_list=frame_list,
    track_ID_list=track_ID_list,
    opt_metrics={},
    state_preds=preds_tf,
    all_masks=all_masks)

print("\nDataFrame Preview:")
print(df.head(10))

# Save to CSV
df.to_csv('state_labeled_tracks.csv', index=False)
print("\nSaved state-labeled tracks to 'state_labeled_tracks.csv'")




# Prepare parameters for a maximum of 4 states
max_states = 4
sequence_length = 4
max_linking_distance = 1
estimated_density = 0.001  # Track density (for mislinking model)
batch_size = 100

# Initialize with generic guesses
params = np.array([[np.log(0.025), np.log(0.01), np.log(0.1), np.log(0.01), 0],
                          [np.log(0.025), np.log(0.05), np.log(0.1), np.log(0.01), 1],
                          [np.log(0.025), np.log(0.2), np.log(0.1), np.log(0.01), 0],
                          [np.log(0.025), np.log(0.8), np.log(0.1), np.log(0.01), 1]], dtype='float64')

initial_params = np.array([[np.log(1.0)],
                                  [np.log(1.0)],
                                  [np.log(1.0)],
                                  [np.log(1.0)]], dtype='float64')

# Equal initial fractions
initial_fractions = np.array([[0.0, 0.0, 0.0, 0.0, -5.0]], dtype='float64')

# Transition matrices
transition_rates = 4 * np.eye(max_states, dtype='float64')
transition_shapes = np.zeros((max_states, max_states), dtype='float64')

# Create vary masks to fix certain parameters
# vary_params: which recurrent parameters to optimize
vary_params = True
# vary_transition_shapes: fix shapes to 1 (exponential)
vary_transition_shapes = False

# Allow other parameters to vary
vary_initial_params = True
vary_initial_fractions = True
vary_transition_rates = True

print("Testing models with 1 to 4 states to find optimal number...")
print("This task can take a few minutes to start...")



# Run state number determination
results = exatrack.get_number_of_states(
    track_list=track_list,
    params=params,
    initial_params=initial_params,
    transition_shapes=transition_shapes,
    transition_rates=transition_rates,
    initial_fractions=initial_fractions,
    nb_dims=nb_dims,
    reference_dt=reference_dt,
    sequence_length=sequence_length,
    max_linking_distance=max_linking_distance,
    estimated_density=estimated_density,
    epochs=60,
    epoch_decay=30,
    batch_size=batch_size,
    device=device_tf,
    vary_params=vary_params,
    vary_initial_params=vary_initial_params,
    vary_initial_fractions=vary_initial_fractions,
    vary_transition_shapes=vary_transition_shapes,
    vary_transition_rates=vary_transition_rates)



# Analyze results
print("\n" + "="*60)
print("MODEL SELECTION RESULTS")
print("="*60)




print(f"\nResults for each model:")
for n_states in sorted(results.keys(), reverse=True):
    r = results[n_states]
    print(f"  {n_states} states: LL={r['log_likelihood']:.1f}, "
          f"AIC={r['aic']:.1f}, BIC={r['bic']:.1f}")
    


# Plot model selection results
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

n_states_list = sorted(results.keys())
ll_values = [results[n]['log_likelihood'] for n in n_states_list]
aic_values = [results[n]['aic'] for n in n_states_list]
bic_values = [results[n]['bic'] for n in n_states_list]

# Log-likelihood
ax = axes[0]
ax.plot(n_states_list, ll_values, 'o-', markersize=10, linewidth=2)
ax.axvline(x=2, color='green', linestyle='--', label='True # states', alpha=0.7)
ax.set_xlabel('Number of States')
ax.set_ylabel('Log-Likelihood')
ax.set_title('Log-Likelihood vs Number of States')
ax.legend()
ax.grid(True, alpha=0.3)

# AIC
ax = axes[1]
ax.plot(n_states_list, aic_values, 'o-', markersize=10, linewidth=2, color='orange')
ax.axvline(x=2, color='green', linestyle='--', label='True # states', alpha=0.7)
#ax.axvline(x=best_nb_states['aic'], color='red', linestyle=':', label='Selected (AIC)', alpha=0.7)
ax.set_xlabel('Number of States')
ax.set_ylabel('AIC')
ax.set_title('AIC vs Number of States')
ax.legend()
ax.grid(True, alpha=0.3)

# BIC
ax = axes[2]
ax.plot(n_states_list, bic_values, 'o-', markersize=10, linewidth=2, color='purple')
ax.axvline(x=2, color='green', linestyle='--', label='True # states', alpha=0.7)
#ax.axvline(x=best_nb_states['bic'], color='red', linestyle=':', label='Selected (BIC)', alpha=0.7)
ax.set_xlabel('Number of States')
ax.set_ylabel('BIC')
ax.set_title('BIC vs Number of States')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_selection.png', dpi=150, bbox_inches='tight')
plt.show()## Summary




