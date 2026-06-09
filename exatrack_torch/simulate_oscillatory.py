# -*- coding: utf-8 -*-
"""
simulate_oscillatory.py
-----------------------
Extends exatrack_torch simulation to support oscillatory motion state.

Provides anomalous_diff_transition_osc() — a drop-in replacement for
exatrack_torch.anomalous_diff_transition that adds oscillatory motion
as a third state alongside confined and directed.

Oscillatory state is parameterised by:
    omega   : angular frequency in rad/step (= 2*pi / period_in_steps)
    A       : typical amplitude  [same units as positions]
    d_osc   : position noise per sub-step
    q_osc   : velocity noise per sub-step (keep small — ~0.1 * A * omega)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers copied / adapted from simulation.py
# ---------------------------------------------------------------------------

def _sample_transitions(state, current_sub, cum_sub_times,
                         shape_matrix, transition_matrix, nb_states, dt):
    """Sample next transition times for all target states."""
    transitions = np.full(nb_states, np.inf)
    t0 = cum_sub_times[current_sub]
    for s2 in range(nb_states):
        if s2 == state:
            continue
        rate = transition_matrix[state, s2]
        if rate <= 0:
            continue
        shape = shape_matrix[state, s2]
        if shape <= 0:
            mean_dt = 1.0 / rate
            wait    = np.random.exponential(mean_dt)
        else:
            mean_dt  = 1.0 / rate
            scale_dt = mean_dt / shape
            wait     = np.random.gamma(shape, scale_dt)
        t_trans = t0 + wait
        # Convert to sub-step index
        idx = np.searchsorted(cum_sub_times, t_trans)
        transitions[s2] = max(1, idx - current_sub)
    return transitions


def _sim_directed_2d(l, initial_pos, velocity, angular_D, D, dt_seg, nb_dims):
    """Simulate directed motion sub-segment (pure drift + angular diffusion)."""
    pos   = initial_pos.copy()
    angle = np.random.uniform(0, 2 * np.pi)
    traj  = []
    for i in range(l):
        dt_i   = dt_seg[i] if hasattr(dt_seg, '__len__') else dt_seg
        ds     = np.sqrt(2 * D * dt_i)
        pos   += velocity * np.array([np.cos(angle), np.sin(angle)])[:nb_dims]
        if D > 0:
            pos += np.random.randn(nb_dims) * ds
        angle  += np.random.randn() * np.sqrt(2 * angular_D * dt_i)
        traj.append(pos.copy())
    return traj, pos


def _sim_confined_2d(l, initial_pos, conf_force, D, conf_D, dt_seg, nb_dims):
    """Simulate confined (OU) motion sub-segment."""
    pos    = initial_pos.copy()
    centre = pos.copy()
    traj   = []
    for i in range(l):
        dt_i     = dt_seg[i] if hasattr(dt_seg, '__len__') else dt_seg
        ds       = np.sqrt(2 * D * dt_i)
        ds_conf  = np.sqrt(2 * conf_D * dt_i) if conf_D > 0 else 0.0
        pos      = (1 - conf_force) * pos + conf_force * centre
        if D > 0:
            pos += np.random.randn(nb_dims) * ds
        if conf_D > 0:
            centre += np.random.randn(nb_dims) * ds_conf
        traj.append(pos.copy())
    return traj, pos


def _sim_oscillatory_2d(l, initial_disp, initial_vel, orbit_centre, omega,
                         d_osc, q_osc, dt_seg, nb_dims):
    """Simulate oscillatory motion sub-segment using the 1D harmonic propagator
    applied independently to each dimension.

    The harmonic map oscillates around 0, so we simulate DISPLACEMENT from the
    orbit centre, then add orbit_centre back to get absolute positions.
    Circular orbits are obtained by initialising x and y with a pi/2 phase
    offset so each dim runs the same harmonic map as the constraint function
    (G3 remains valid, q does not inflate).

    initial_disp  : displacement from orbit_centre at start  (nb_dims,)
    initial_vel   : velocity at start  (nb_dims,)
    orbit_centre  : absolute position of orbit centre  (nb_dims,)  — drifts with q_osc
    omega         : phase advance per sub-step (rad/sub-step)
    d_osc         : displacement noise std per sub-step
    q_osc         : orbit-centre drift std per sub-step
    """
    cos_w  = np.cos(omega)
    sin_w  = np.sin(omega)
    disp   = initial_disp.copy()
    vel    = initial_vel.copy()
    centre = orbit_centre.copy()
    traj   = []
    for i in range(l):
        disp_new = (cos_w * disp + (sin_w / (omega + 1e-20)) * vel
                    + np.random.randn(nb_dims) * d_osc)
        vel_new  = (-omega * sin_w * disp + cos_w * vel
                    + np.random.randn(nb_dims) * q_osc)
        centre  += np.random.randn(nb_dims) * q_osc   # slow centre drift
        disp     = disp_new
        vel      = vel_new
        traj.append((centre + disp).copy())
    return traj, centre + disp, centre, vel


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def anomalous_diff_transition_osc(
        max_track_len   = 100,
        nb_tracks       = 200,
        LocErr          = 0.02,
        LocErr_std      = 0.002,
        # State fractions  [directed, confined, oscillatory]
        Fs              = np.array([0.33, 0.34, 0.33]),
        # Directed state
        velocity        = 0.05,     # drift speed per observed step
        angular_D       = 0.04,     # angular diffusion for curvature
        D_dir           = 0.0,      # diffusion coefficient (0 = pure drift)
        # Confined state
        D_con           = 0.25,     # diffusion coefficient
        conf_force      = 0.2,      # OU confinement force per sub-step
        conf_D          = 0.0,      # well-centre diffusion
        # Oscillatory state
        omega           = 2*np.pi/0.6,  # rad/s (period=0.6s = 30 steps at dt=0.02)
        A_osc           = 1.5,         # typical amplitude
        d_osc_noise     = 0.005,       # position noise per sub-step
        q_osc_noise     = None,        # velocity noise per sub-step (default: A*omega*0.1)
        # Transition rates (3×3, diagonal unused)
        transition_matrix = np.array([[0.00, 0.02, 0.02],
                                       [0.05, 0.00, 0.02],
                                       [0.02, 0.02, 0.00]]),
        shape_matrix      = np.ones((3, 3)) - np.eye(3),
        # Geometry
        field_of_view   = np.array([-5.0, 5.0]),
        nb_dims         = 2,
        # Time
        dt              = 0.02,
        dt_std          = 0.002,
        nb_sub_steps    = 10,
        nb_burning_steps = 0,
        bleaching_rate  = 0.02,
        reference_dt    = 0.02):
    """
    Simulate tracks with transitions between directed, confined, and oscillatory
    motion states.

    State indices:
        0 = directed
        1 = confined
        2 = oscillatory

    Returns
    -------
    all_tracks  : (nb_tracks, max_track_len, nb_dims)
    all_LocErrs : (nb_tracks, max_track_len, nb_dims)
    all_dts     : (nb_tracks, max_track_len)
    all_states  : (nb_tracks, max_track_len)  ground-truth state labels
    all_masks   : (nb_tracks, max_track_len)  validity mask (1=valid, 0=padded)
    """
    if q_osc_noise is None:
        q_osc_noise = A_osc * omega * 0.1

    nb_states = 3
    Fs = Fs / Fs.sum()   # normalise
    cum_Fs = np.cumsum(Fs)

    fov_lo, fov_hi = field_of_view[0], field_of_view[1]
    fov_size = fov_hi - fov_lo

    all_tracks   = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_states   = np.zeros((nb_tracks, max_track_len), dtype=int)
    all_masks    = np.zeros((nb_tracks, max_track_len), dtype=float)
    all_LocErrs  = np.zeros((nb_tracks, max_track_len, nb_dims))
    all_dts      = np.zeros((nb_tracks, max_track_len))

    # Per-sub-step parameters
    v_sub       = velocity / nb_sub_steps
    cf_sub      = conf_force / nb_sub_steps
    omega_sub   = omega * (dt / nb_sub_steps)   # rad/sub-step = omega_phys * dt_sub
    d_osc_sub   = d_osc_noise / np.sqrt(nb_sub_steps)
    q_osc_sub   = q_osc_noise / np.sqrt(nb_sub_steps)

    for k in range(nb_tracks):
        # Track length (bleaching)
        if bleaching_rate / nb_sub_steps > 1e-8:
            track_len = min(max_track_len,
                            np.random.geometric(p=bleaching_rate))
        else:
            track_len = max_track_len

        # Variable time steps
        if dt_std > 0:
            dt_scale = dt_std**2 / dt
            dt_shape = dt / dt_scale
            dts = np.random.gamma(dt_shape, dt_scale, track_len)
            dts = np.clip(dts, 0.05 * dt, 3 * dt)
        else:
            dts = np.full(track_len, dt)

        # Sub-step time array
        burn_subs = nb_burning_steps * nb_sub_steps
        burn_dts  = np.full(burn_subs, dt / nb_sub_steps)
        main_dts  = np.repeat(dts / nb_sub_steps, nb_sub_steps)
        all_sub   = np.concatenate([burn_dts, main_dts])
        cum_times = np.concatenate([[0.0], np.cumsum(all_sub)])

        # Initial state and position
        rand_state = np.random.rand()
        cur_state  = int(np.searchsorted(cum_Fs, rand_state))
        pos        = np.random.uniform(fov_lo, fov_hi, nb_dims)

        # Initial oscillatory state: store orbit_centre (absolute) and displacement.
        # Pi/2 phase offset between x and y gives circular orbits while keeping
        # each dimension compatible with the 1D harmonic constraint function.
        #   x: disp=A*cos(phi),      vel=-A*omega*sin(phi)
        #   y: disp=-A*sin(phi),     vel=-A*omega*cos(phi)   [phi_y = phi + pi/2]
        phase        = np.random.uniform(0, 2 * np.pi)
        osc_centre   = pos.copy()          # orbit centre = current position
        osc_disp     = np.zeros(nb_dims)   # displacement from centre
        osc_vel      = np.zeros(nb_dims)
        osc_disp[0]  =  A_osc * np.cos(phase)
        osc_vel[0]   = -A_osc * omega_sub * np.sin(phase)
        if nb_dims >= 2:
            osc_disp[1] = -A_osc * np.sin(phase)
            osc_vel[1]  = -A_osc * omega_sub * np.cos(phase)
        pos = osc_centre + osc_disp        # absolute starting position

        # Burn-in
        n = 0
        transitions = _sample_transitions(cur_state, n, cum_times,
                                           shape_matrix, transition_matrix,
                                           nb_states, dt)
        while n < burn_subs:
            prev_state = cur_state
            cur_state  = np.argmin(transitions)
            l_seg      = int(np.min(transitions))
            dt_seg     = np.full(l_seg, dt / nb_sub_steps)
            # Reinitialise orbit when entering oscillatory from another state
            if cur_state == 2 and prev_state != 2:
                phase       = np.random.uniform(0, 2 * np.pi)
                osc_centre  = pos.copy()
                osc_disp    = np.zeros(nb_dims)
                osc_vel     = np.zeros(nb_dims)
                osc_disp[0] =  A_osc * np.cos(phase)
                osc_vel[0]  = -A_osc * omega_sub * np.sin(phase)
                if nb_dims >= 2:
                    osc_disp[1] = -A_osc * np.sin(phase)
                    osc_vel[1]  = -A_osc * omega_sub * np.cos(phase)
                pos = osc_centre + osc_disp
            if cur_state == 0:   # directed
                _, pos = _sim_directed_2d(l_seg, pos, v_sub, angular_D,
                                          D_dir, dt_seg, nb_dims)
            elif cur_state == 1:  # confined
                _, pos = _sim_confined_2d(l_seg, pos, cf_sub, D_con,
                                          conf_D, dt_seg, nb_dims)
            else:                 # oscillatory
                _, pos, osc_centre, osc_vel = _sim_oscillatory_2d(
                    l_seg, pos - osc_centre, osc_vel, osc_centre,
                    omega_sub, d_osc_sub, q_osc_sub, dt_seg, nb_dims)
            n += l_seg
            if n < burn_subs:
                transitions = _sample_transitions(cur_state, n, cum_times,
                                                   shape_matrix,
                                                   transition_matrix,
                                                   nb_states, dt)

        # Main simulation
        sub_traj   = []
        sub_states = []
        n_main     = 0
        total_subs = track_len * nb_sub_steps
        transitions = _sample_transitions(cur_state,
                                           burn_subs + n_main, cum_times,
                                           shape_matrix, transition_matrix,
                                           nb_states, dt)

        while n_main < total_subs:
            l_seg = min(int(np.min(transitions)), total_subs - n_main)
            sub_idx  = np.arange(n_main, n_main + l_seg)
            ts_idx   = sub_idx // nb_sub_steps
            dt_seg   = dts[ts_idx] / nb_sub_steps

            if cur_state == 0:   # directed
                traj, pos = _sim_directed_2d(l_seg, pos, v_sub, angular_D,
                                              D_dir, dt_seg, nb_dims)
            elif cur_state == 1:  # confined
                traj, pos = _sim_confined_2d(l_seg, pos, cf_sub, D_con,
                                              conf_D, dt_seg, nb_dims)
            else:                 # oscillatory
                traj, pos, osc_centre, osc_vel = _sim_oscillatory_2d(
                    l_seg, pos - osc_centre, osc_vel, osc_centre,
                    omega_sub, d_osc_sub, q_osc_sub, dt_seg, nb_dims)

            sub_traj   += traj
            sub_states += [cur_state] * l_seg
            n_main     += l_seg

            if n_main < total_subs:
                prev_state = cur_state
                cur_state  = np.argmin(transitions)
                # Reinitialise orbit whenever we enter oscillatory from another state
                if cur_state == 2 and prev_state != 2:
                    phase       = np.random.uniform(0, 2 * np.pi)
                    osc_centre  = pos.copy()
                    osc_disp    = np.zeros(nb_dims)
                    osc_vel     = np.zeros(nb_dims)
                    osc_disp[0] =  A_osc * np.cos(phase)
                    osc_vel[0]  = -A_osc * omega_sub * np.sin(phase)
                    if nb_dims >= 2:
                        osc_disp[1] = -A_osc * np.sin(phase)
                        osc_vel[1]  = -A_osc * omega_sub * np.cos(phase)
                    pos = osc_centre + osc_disp
                transitions = _sample_transitions(
                    cur_state, burn_subs + n_main, cum_times,
                    shape_matrix, transition_matrix, nb_states, dt)

        # Subsample: take every nb_sub_steps-th point
        obs_pos    = np.array(sub_traj)[nb_sub_steps-1::nb_sub_steps]
        obs_states = np.array(sub_states)[nb_sub_steps-1::nb_sub_steps]
        obs_len    = min(len(obs_pos), track_len)

        # Add localisation noise
        if LocErr_std > 0:
            scale_le = LocErr_std**2 / LocErr
            shape_le = LocErr / scale_le
            le_vals  = np.random.gamma(shape_le, scale_le, (obs_len, nb_dims))
        else:
            le_vals  = np.full((obs_len, nb_dims), LocErr)

        obs_noisy = obs_pos[:obs_len] + np.random.randn(obs_len, nb_dims) * le_vals

        all_tracks[k, :obs_len, :]  = obs_noisy
        all_states[k, :obs_len]     = obs_states[:obs_len]
        all_LocErrs[k, :obs_len, :] = le_vals
        all_dts[k, :obs_len]        = dts[:obs_len]
        all_masks[k, :obs_len]      = 1.0

    return all_tracks, all_LocErrs, all_dts, all_states, all_masks