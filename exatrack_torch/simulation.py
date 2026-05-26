# -*- coding: utf-8 -*-
"""
simulation.py
-------------
Synthetic data generation for ExaTrack benchmarking and testing.

Functions
---------
anomalous_diff_transition  : generate tracks with state switching (main entry point)
anomalous_diff_2D          : simulate one 2D track segment (Numba JIT)
anomalous_diff_3D          : simulate one 3D track segment
_sample_transitions        : sample Gamma-distributed dwell times
simulate_3D_rotational_diffusion : generate directed displacements in 3D
generate_movie             : generate a fluorescence microscopy movie
emit_photons               : emit photons for one particle position (Numba JIT)

Dependencies: numpy, numba, scipy  (no PyTorch used in this module)
"""

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Low-level 2D/3D diffusion simulators
# ---------------------------------------------------------------------------

@njit
def anomalous_diff_2D(track_len=20, LocErr=0.02, D=0.05,
                      velocity=None, angular_D=0.0,
                      conf_force=None, conf_D=0.0, conf_dist=0.0,
                      dt=None, nb_sub_steps=10,
                      initial_positions=np.array([0., 0.])):
    """
    Simulate one 2D anomalous diffusion track segment using sub-steps.

    Parameters
    ----------
    track_len        : number of observed time points
    LocErr           : localisation error standard deviation (added at the end)
    D                : diffusion coefficient
    velocity         : array of per-sub-step directed displacement magnitudes
    angular_D        : rotational diffusion coefficient for the velocity direction
    conf_force       : array of per-sub-step confinement force factors
    conf_D           : diffusion coefficient of the potential well centre
    conf_dist        : initial std of the potential well centre position
    dt               : array of per-sub-step time durations
    nb_sub_steps     : number of sub-steps per observed time point
    initial_positions: starting position

    Returns
    -------
    final_track : (track_len, 2) array of observed positions
    """
    nb_dims = 2
    n_disps = track_len * nb_sub_steps - 1

    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    positions[0] = initial_positions

    disps = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * D * dt[i])
        for d in range(nb_dims):
            disps[i, d] = np.random.normal(0.0, s)

    anchor_positions = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * conf_D * dt[i])
        for d in range(nb_dims):
            anchor_positions[i, d] = np.random.normal(0.0, s)
    for d in range(nb_dims):
        anchor_positions[0, d] = positions[0, d] + np.random.normal(0.0, conf_dist)
    for i in range(1, n_disps):
        for d in range(nb_dims):
            anchor_positions[i, d] += anchor_positions[i - 1, d]

    angles = np.zeros(n_disps)
    angles[0] = np.random.rand() * 2.0 * np.pi
    for i in range(1, n_disps):
        angles[i] = angles[i - 1] + np.random.normal(0.0, np.sqrt(2.0 * angular_D * dt[i - 1]))

    for i in range(n_disps):
        cos_a, sin_a = np.cos(angles[i]), np.sin(angles[i])
        positions[i + 1, 0] = positions[i, 0] + cos_a * velocity[i] + disps[i, 0]
        positions[i + 1, 1] = positions[i, 1] + sin_a * velocity[i] + disps[i, 1]
        cf = conf_force[i]
        positions[i + 1, 0] = (1.0 - cf) * positions[i + 1, 0] + cf * anchor_positions[i, 0]
        positions[i + 1, 1] = (1.0 - cf) * positions[i + 1, 1] + cf * anchor_positions[i, 1]

    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i * nb_sub_steps]

    if LocErr > 0:
        final_track += np.random.normal(0.0, LocErr, (track_len, nb_dims))
    return final_track


def anomalous_diff_3D(track_len=20, LocErr=0.02, D=0.05,
                      velocity=None, angular_D=0.0,
                      conf_force=None, conf_D=0.0, conf_dist=0.0,
                      dt=None, nb_sub_steps=10,
                      initial_positions=np.array([0., 0., 0.])):
    """Simulate one 3D anomalous diffusion track segment using sub-steps."""
    nb_dims = 3
    n_disps = track_len * nb_sub_steps - 1

    positions = np.zeros((track_len * nb_sub_steps, nb_dims))
    positions[0] = initial_positions

    disps = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * D * dt[i])
        disps[i] = np.random.normal(0.0, s, nb_dims)

    anchor_positions = np.zeros((n_disps, nb_dims))
    for i in range(n_disps):
        s = np.sqrt(2.0 * conf_D * dt[i])
        anchor_positions[i] = np.random.normal(0.0, s, nb_dims)
    anchor_positions[0] = positions[0] + np.random.normal(0.0, conf_dist, nb_dims)
    for i in range(1, n_disps):
        anchor_positions[i] += anchor_positions[i - 1]

    persistent_displacements = simulate_3D_rotational_diffusion(
        n_disps, velocity, angular_D, dt)

    for i in range(n_disps):
        positions[i + 1] = positions[i] + persistent_displacements[i] + disps[i]
        cf = conf_force[i]
        positions[i + 1] = (1.0 - cf) * positions[i + 1] + cf * anchor_positions[i]

    final_track = np.zeros((track_len, nb_dims))
    for i in range(track_len):
        final_track[i] = positions[i * nb_sub_steps]

    if LocErr > 0:
        final_track += np.random.normal(0.0, LocErr, (track_len, nb_dims))
    return final_track


def simulate_3D_rotational_diffusion(nb_steps, velocities, D_r, dts):
    """
    Generate directed displacements for 3D tracks with rotational diffusion.

    Parameters
    ----------
    nb_steps   : number of sub-steps
    velocities : (nb_steps,) per-step displacement magnitudes
    D_r        : rotational diffusion coefficient
    dts        : (nb_steps,) per-step durations

    Returns
    -------
    (nb_steps, 3) array of directed displacement vectors
    """
    theta = 2.0 * np.pi * np.random.rand()
    phi = np.arccos(2.0 * np.random.rand() - 1.0)
    v = np.array([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)])
    v /= np.linalg.norm(v)

    vs = np.zeros((nb_steps, 3))
    vs[0] = v
    for i in range(1, nb_steps):
        sigma_theta = np.sqrt(2.0 * D_r * dts[i - 1])
        dtheta = np.random.normal(0.0, sigma_theta, size=3)
        v = R.from_rotvec(dtheta).apply(v)
        v /= np.linalg.norm(v)
        vs[i] = v

    return vs * velocities[:, None]


# ---------------------------------------------------------------------------
# Transition sampling
# ---------------------------------------------------------------------------

def _sample_transitions(state, current_sub_idx, cum_sub_times,
                        shape_matrix, transition_matrix, nb_states, dt_mean):
    """
    Sample Gamma-distributed dwell times for all possible next states.

    For each candidate target state, samples a lifetime from Gamma(shape, scale)
    in continuous time, then converts it to a sub-step count via cum_sub_times.

    Returns
    -------
    transitions : (nb_states,) array of sub-step counts until transition
    """
    transitions = np.full(nb_states, len(cum_sub_times), dtype=np.int64)
    current_time = cum_sub_times[current_sub_idx]
    for target in range(nb_states):
        if target != state and transition_matrix[state, target] > 0:
            lifetime = np.random.gamma(
                shape_matrix[state, target],
                dt_mean / transition_matrix[state, target])
            end_idx = np.searchsorted(cum_sub_times, current_time + lifetime)
            transitions[target] = max(1, end_idx - current_sub_idx)
    return transitions


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def anomalous_diff_transition(
        max_track_len=100,
        nb_tracks=100,
        LocErr=0.02,
        Fs=np.array([0., 1]),
        Ds=np.array([0.0, 0.25]),
        nb_dims=2,
        velocities=np.array([0.03, 0.0]),
        angular_Ds=np.array([0.0, 0.0]),
        conf_forces=np.array([0.0, 0.2]),
        conf_Ds=np.array([0.0, 0.0]),
        conf_dists=np.array([0.0, 0.0]),
        transition_matrix=np.array([[0.00, 0.1], [0.1, 0.00]]),
        shape_matrix=np.array([[0, 1], [1, 0]]),
        bleaching_rate=1e-10,
        LocErr_std=0.002,
        dt=0.02,
        dt_std=0.002,
        field_of_view=np.array([10, 10]),
        nb_burning_steps=100,
        nb_sub_steps=10,
        return_list=False):
    """
    Simulate tracks of particles switching between anomalous motion states.

    Parameters
    ----------
    max_track_len     : maximum track length in frames
    nb_tracks         : number of tracks to simulate
    LocErr            : base localisation error standard deviation
    Fs                : initial state fractions (must sum to 1)
    Ds                : diffusion coefficients per state
    nb_dims           : number of spatial dimensions (2 or 3)
    velocities        : directed motion speed per state
    angular_Ds        : rotational diffusion per state
    conf_forces       : confinement force per state (0=free, 1=fully confined)
    conf_Ds           : diffusion of potential well centre per state
    conf_dists        : initial spread of potential well per state
    transition_matrix : (nb_states, nb_states) rate matrix
    shape_matrix      : (nb_states, nb_states) Gamma shape parameters
    bleaching_rate    : probability of track ending per sub-step
    LocErr_std        : std of localisation error variation across frames
    dt                : mean frame duration
    dt_std            : std of frame duration variation
    field_of_view     : spatial extent for initial position sampling
    nb_burning_steps  : burn-in steps to reach steady state
    nb_sub_steps      : sub-steps per observed frame (for continuous simulation)

    Returns
    -------
    all_tracks   : (nb_tracks, max_track_len, nb_dims)
    all_LocErrs  : (nb_tracks, max_track_len, nb_dims)
    all_dts      : (nb_tracks, max_track_len)
    all_states   : (nb_tracks, max_track_len) ground-truth state labels
    all_masks    : (nb_tracks, max_track_len) validity mask
    """
    nb_states = len(velocities)
    if not np.all(np.array([len(Fs), len(Ds), len(velocities), len(angular_Ds),
                             len(conf_forces), len(conf_Ds), len(conf_dists),
                             len(transition_matrix)]) == nb_states):
        raise ValueError('All per-state arrays must have the same length.')

    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for s in range(1, nb_states):
        cum_Fs[s] = cum_Fs[s - 1] + Fs[s]

    all_tracks = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_states = np.zeros((nb_tracks, max_track_len))
    all_masks = np.zeros((nb_tracks, max_track_len))
    all_LocErrs = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_dts = np.zeros((nb_tracks, max_track_len))

    LocErr = np.array([LocErr])

    for k in range(nb_tracks):
        # Track length (bleaching)
        if bleaching_rate / nb_sub_steps > 1e-8:
            track_len = min(max_track_len, np.random.geometric(p=bleaching_rate))
        else:
            track_len = max_track_len

        # Variable time steps
        if dt_std > 0:
            dt_scale = dt_std ** 2 / dt
            dt_shape = dt / dt_scale
            dts = np.random.gamma(dt_shape, dt_scale, track_len)
            dts[dts < 0.05 * dt] = 0.05 * dt
            dts[dts > 3 * dt] = 3 * dt
        else:
            dts = np.full(track_len, dt)

        # Cumulative sub-step times
        burn_in_subs_total = nb_burning_steps * nb_sub_steps
        burn_sub_dts = np.full(burn_in_subs_total, dt / nb_sub_steps)
        main_sub_dts = np.repeat(dts / nb_sub_steps, nb_sub_steps)
        all_sub_dts = np.concatenate([burn_sub_dts, main_sub_dts])
        cum_sub_times = np.concatenate([[0.0], np.cumsum(all_sub_dts)])

        # Initial position and state
        initial_positions = np.random.rand(nb_dims) * field_of_view
        track = []
        states = []
        next_state = np.argmin(np.random.rand() > cum_Fs)

        # Burn-in
        n = 0
        while n <= burn_in_subs_total:
            state = next_state
            transitions = _sample_transitions(state, n, cum_sub_times,
                                              shape_matrix, transition_matrix,
                                              nb_states, dt)
            next_state = np.argmin(transitions)
            n += int(np.min(transitions))
        transitions[next_state] = n - burn_in_subs_total
        if transitions[next_state] == 0:
            state = next_state
            transitions = _sample_transitions(state, burn_in_subs_total,
                                              cum_sub_times, shape_matrix,
                                              transition_matrix, nb_states, dt)

        # Main simulation loop
        track_len_subs = track_len * nb_sub_steps
        while len(track) < track_len_subs:
            if len(track) > 0:
                transitions = _sample_transitions(
                    state, burn_in_subs_total + len(track),
                    cum_sub_times, shape_matrix, transition_matrix, nb_states, dt)

            l = min(int(np.min(transitions)), track_len_subs - len(track))
            D, velocity, angular_D, conf_force, conf_D, conf_dist = (
                Ds[state], velocities[state], angular_Ds[state],
                conf_forces[state], conf_Ds[state], conf_dists[state])

            seg_start_sub = len(track)
            sub_indices = np.arange(seg_start_sub, seg_start_sub + l)
            ts_indices = sub_indices // nb_sub_steps
            sub_dts_seg = dts[ts_indices] / nb_sub_steps
            velocity_seg = np.full(l, velocity / nb_sub_steps)
            conf_force_seg = np.full(l, conf_force / nb_sub_steps)

            if nb_dims < 3:
                segment = anomalous_diff_2D(
                    track_len=l + 1, LocErr=0, D=D,
                    velocity=velocity_seg, angular_D=angular_D,
                    conf_force=conf_force_seg, conf_D=conf_D,
                    conf_dist=conf_dist, dt=sub_dts_seg,
                    nb_sub_steps=1,
                    initial_positions=initial_positions)
                segment = segment[:, :nb_dims]
            elif nb_dims == 3:
                segment = anomalous_diff_3D(
                    track_len=l + 1, LocErr=0, D=D,
                    velocity=velocity_seg, angular_D=angular_D,
                    conf_force=conf_force_seg, conf_D=conf_D,
                    conf_dist=conf_dist, dt=sub_dts_seg,
                    nb_sub_steps=1,
                    initial_positions=initial_positions)
            else:
                raise ValueError('nb_dims must be 1, 2, or 3.')

            track += list(segment[:-1])
            states += [state] * l
            initial_positions = segment[-1]
            state = np.argmin(transitions)

        # Localisation error
        if LocErr_std > 0:
            scale = LocErr_std ** 2 / LocErr
            shape = LocErr / scale
            LocErrs = np.random.gamma(shape, scale, (track_len, nb_dims))
        else:
            LocErrs = LocErr

        track = (np.array(track)[::nb_sub_steps]
                 + np.random.normal(0, LocErrs, (track_len, nb_dims)))
        states = np.array(states)[::nb_sub_steps]

        all_tracks[k, :track_len] = track
        all_tracks[k, track_len:] = track[-1]
        all_LocErrs[k, :track_len] = LocErrs
        all_LocErrs[k, track_len:] = LocErr[-1]
        all_dts[k, :track_len] = dts
        all_dts[k, track_len:] = dts[-1]
        all_states[k, :track_len] = states
        all_states[k, track_len:] = states[-1]
        all_masks[k, :track_len - 1] = 1
        all_masks[k, -1] = 1

    return all_tracks, all_LocErrs, all_dts, all_states, all_masks


# ---------------------------------------------------------------------------
# Movie generation
# ---------------------------------------------------------------------------

def generate_movie(track_list, time_list, state_list,
                   average_photon_number, average_background,
                   emission_std, max_time, pixel_dims, pixel_size):
    """Generate a synthetic fluorescence microscopy movie."""
    movie = np.random.poisson(
        average_background, size=[max_time] + list(pixel_dims)).astype('int16')
    nb_counts = 0
    for track, times in zip(track_list, time_list):
        for pos, time in zip(track, times):
            pixel_pos = pos / pixel_size
            nb_photons = np.random.poisson(average_photon_number)
            movie = emit_photons(pixel_pos, nb_photons, movie, time,
                                  emission_std, pixel_dims)
            nb_counts += 1
    return movie, nb_counts


@njit
def emit_photons(pixel_pos, nb_photons, movie, time, emission_std, pixel_dims):
    """Emit photons for one particle at one time point (Numba JIT)."""
    for k in range(nb_photons):
        photon_pos_x = int(np.random.normal(pixel_pos[0], emission_std))
        photon_pos_y = int(np.random.normal(pixel_pos[1], emission_std))
        if (0 <= photon_pos_x < pixel_dims[0]
                and 0 <= photon_pos_y < pixel_dims[1]):
            movie[time, photon_pos_x, photon_pos_y] += 1
    return movie
