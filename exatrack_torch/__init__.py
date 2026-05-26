# -*- coding: utf-8 -*-
"""
ExaTrack (PyTorch)
==================
A framework for analysing single-molecule dynamics with complex motion types
and non-Markovian state transition kinetics.

Based on the Conditional Gaussian Process (CGP) framework — see:
  - Simon et al. (2025) HAL preprint hal-04692487  [CGP paper]
  - Simon et al. (2026) eLife 99347               [aTrack paper]
  - Simon et al. (2026) biorxiv 2026.01.22.700663 [ExaTrack paper]

Package structure
-----------------
config        : global constants (dtype, pi, minval)
gaussian_ops  : core Gaussian product math (equation 5 of the CGP paper)
integration   : CGP recurrence formula and integration schedule
constraints   : physical model definition (θ → A-matrix coefficients)
simulation    : synthetic track generation
io            : data loading, padding, segmentation
layers        : nn.Module layers (Initial, RNN, Final, carry-over)
models        : model assembly, loss function, parameter utilities
training      : Model_finder, get_number_of_states, learning rate schedule
inference     : forward-backward smoother, hidden variable extraction
uncertainty   : bootstrapping and HMC sampler

Public API
----------
    from exatrack_torch import (
        anomalous_diff_transition,  # simulate tracks
        build_segment_model,        # assemble model
        Model_finder,               # fit model
        get_number_of_states,       # automated state selection
        extract_smooth_hidden_variables,  # denoise tracks
        bootstrapping,              # uncertainty quantification
        run_hmc,                    # Bayesian posterior sampling
        read_table,                 # load data
        padding,                    # prepare data
        segment_tracks,             # batch data
    )
"""

# ---- simulation ----
from .simulation import (
    anomalous_diff_transition,
    anomalous_diff_2D,
    anomalous_diff_3D,
    simulate_3D_rotational_diffusion,
    generate_movie,
)

# ---- data I/O ----
from .io import (
    read_table,
    padding,
    segment_tracks,
    TrackSegmentSequence,
    ExaTrack_2_DataFrame,
    correct_state_predictions_padding,
)

# ---- model assembly ----
from .models import (
    build_segment_model,
    build_abrupt_directed_motion_changes_model,
    MLE_loss,
    get_model_params,
    get_model_raw_params,
    equilibrium_distribution,
    model_to_DataFrame,
    get_parameters,
)

# ---- training ----
from .training import (
    WarmupLearningRateSchedule,
    Model_finder,
    get_number_of_states,
)

# ---- inference ----
from .inference import (
    marginalise_variable,
    extract_hidden_variables,
    extract_smooth_hidden_variables,
)

# ---- uncertainty ----
from .uncertainty import (
    bootstrapping,
    HMCSampler,
    run_hmc,
    effective_sample_size,
    r_hat,
    transform_hmc_samples,
)

# ---- lower-level building blocks (for custom models) ----
from .constraints import constraint_function, transition_param_function
from .gaussian_ops import (
    log_gaussian,
    norm_log_gaussian,
    RNN_gaussian_product,
)
from .integration import (
    RNN_reccurence_formula,
    transition_RNN_reccurence_formula,
    get_sequences,
)
from .layers import (
    Initial_layer_constraints,
    Custom_RNN_layer,
    Final_layer,
    IsfirstMaskLayer,
    CarryoverAssignLayer,
    transpose_layer,
)

__version__ = '0.1.0'
__all__ = [
    # simulation
    'anomalous_diff_transition', 'anomalous_diff_2D', 'anomalous_diff_3D',
    'simulate_3D_rotational_diffusion', 'generate_movie',
    # io
    'read_table', 'padding', 'segment_tracks', 'TrackSegmentSequence',
    'ExaTrack_2_DataFrame', 'correct_state_predictions_padding',
    # models
    'build_segment_model', 'build_abrupt_directed_motion_changes_model',
    'MLE_loss', 'get_model_params', 'get_model_raw_params',
    'equilibrium_distribution', 'model_to_DataFrame', 'get_parameters',
    # training
    'WarmupLearningRateSchedule', 'Model_finder', 'get_number_of_states',
    # inference
    'marginalise_variable', 'extract_hidden_variables',
    'extract_smooth_hidden_variables',
    # uncertainty
    'bootstrapping', 'HMCSampler', 'run_hmc',
    'effective_sample_size', 'r_hat', 'transform_hmc_samples',
    # building blocks
    'constraint_function', 'transition_param_function',
    'log_gaussian', 'norm_log_gaussian', 'RNN_gaussian_product',
    'RNN_reccurence_formula', 'transition_RNN_reccurence_formula',
    'get_sequences',
    'Initial_layer_constraints', 'Custom_RNN_layer', 'Final_layer',
    'IsfirstMaskLayer', 'CarryoverAssignLayer', 'transpose_layer',
]
