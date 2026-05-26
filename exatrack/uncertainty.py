# -*- coding: utf-8 -*-
"""
uncertainty.py
--------------
Uncertainty quantification for fitted ExaTrack models.

Two complementary approaches:

1. **Bootstrapping** (frequentist): resample tracks with replacement and
   refit the model many times. The spread of parameter estimates gives
   confidence intervals without distributional assumptions.

2. **Hamiltonian Monte Carlo** (Bayesian): sample from the full posterior
   distribution P(θ | data) using gradient-based MCMC. Gives complete
   posterior distributions, not just point estimates with error bars.

Classes / Functions
-------------------
bootstrapping        : frequentist uncertainty via bootstrap resampling
HMCSampler           : HMC sampler class with dual-averaging adaptation
run_hmc              : high-level convenience wrapper for HMC
default_log_prior    : weak Gaussian prior on parameters
leapfrog             : Hamiltonian dynamics integrator
effective_sample_size: ESS diagnostic for HMC chains
r_hat                : Gelman-Rubin convergence diagnostic
transform_hmc_samples: convert raw HMC samples to physical parameters
flatten_params       : flatten model weights to a 1D vector
unflatten_params     : write a 1D vector back into model weights
get_trainable_param_indices : identify trainable weight indices

Dependencies: config.py, models.py
"""

import numpy as np
import tensorflow as tf
import scipy
from copy import deepcopy

from .config import dtype
from .models import MLE_loss, get_model_params, get_model_raw_params


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def bootstrapping(model, tracks, masks, bootstrap_number=100,
                  epochs=100, batch_size=65,
                  learning_rate=1/100, decay_threshold=None,
                  decay_rate=None, device='/GPU:0', verbose=1,
                  track_segmentation=False):
    """
    Estimate parameter uncertainty via bootstrap resampling.

    For each bootstrap iteration:
      1. Resample tracks with replacement
      2. Reset model to original fitted weights
      3. Refit model on resampled tracks
      4. Record the fitted parameters

    The spread of parameters across iterations gives bootstrap confidence
    intervals for each physical parameter.

    Parameters
    ----------
    model            : fitted ExaTrack model
    tracks           : (nb_tracks, track_len, nb_dims)
    masks            : (nb_tracks, track_len)
    bootstrap_number : number of resampling iterations
    epochs           : training epochs per bootstrap
    batch_size       : batch size for training
    learning_rate    : peak learning rate
    decay_threshold  : step at which lr decay starts (auto if None)
    decay_rate       : exponential decay rate (auto if None)
    device           : TensorFlow device string
    verbose          : Keras verbosity
    track_segmentation : whether model uses track segmentation

    Returns
    -------
    all_model_parameters : list of parameter dicts, one per bootstrap
    all_likelihoods      : list of final training losses
    """
    from .training import WarmupLearningRateSchedule
    from .models import get_parameters

    nb_tracks = tracks.shape[0]
    nb_batchs = nb_tracks // batch_size

    if decay_threshold is None:
        decay_threshold = int(epochs * nb_batchs * 0.75)
    if decay_rate is None:
        decay_rate = -np.log(0.001) / (0.25 * epochs * nb_batchs)

    lr = WarmupLearningRateSchedule(10, learning_rate, decay_rate, decay_threshold)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.99, clipvalue=1.0)
    model.compile(loss=MLE_loss, optimizer=optimizer, jit_compile=False)
    callbacks = [get_parameters(track_segmentation=track_segmentation)]

    original_weights = [w.numpy() for w in model.weights]
    all_model_parameters = []
    all_likelihoods = []

    for i in range(bootstrap_number):
        # Reset model weights and optimizer state
        for w, ow in zip(model.weights, original_weights):
            w.assign(ow)
        for var in model.optimizer.variables():
            var.assign(tf.zeros_like(var))

        # Resample with replacement
        sampling_indices = np.random.randint(0, nb_tracks, size=nb_tracks)
        sampled_tracks = tracks[sampling_indices]
        sampled_masks = masks[sampling_indices]

        with tf.device(device):
            history = model.fit(
                (sampled_tracks, sampled_masks), sampled_tracks,
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=verbose)

        parameter_dict = get_model_params(model, track_segmentation)
        all_model_parameters.append(parameter_dict)
        all_likelihoods.append(history.history['loss'][-1])

    return all_model_parameters, all_likelihoods


# ---------------------------------------------------------------------------
# HMC utilities
# ---------------------------------------------------------------------------

def get_trainable_param_indices(model):
    """Return indices of trainable model weights."""
    return [i for i, w in enumerate(model.weights) if w.trainable]


def flatten_params(model, indices):
    """Flatten selected model weights into a single 1D tensor."""
    parts = [tf.reshape(tf.cast(model.weights[i], dtype), [-1])
             for i in indices]
    return tf.concat(parts, axis=0)


def unflatten_params(flat, model, indices):
    """Write a flat 1D parameter vector back into model weights."""
    offset = 0
    for i in indices:
        w = model.weights[i]
        size = int(tf.reduce_prod(w.shape))
        chunk = tf.reshape(flat[offset:offset + size], w.shape)
        w.assign(tf.cast(chunk, w.dtype))
        offset += size


def shapes_and_sizes(model, indices):
    """Return shapes and sizes of selected model weights."""
    shapes = [model.weights[i].shape for i in indices]
    sizes = [int(tf.reduce_prod(model.weights[i].shape)) for i in indices]
    return shapes, sizes


def default_log_prior(flat_params):
    """Weak Gaussian prior: log P(θ) ∝ -0.01 * ||θ||²"""
    return -0.01 * tf.reduce_sum(flat_params ** 2)


def leapfrog(q, p, grad_log_posterior_fn, step_size,
             num_steps, mass_inv):
    """
    Leapfrog integrator for Hamiltonian dynamics.

    Integrates the equations of motion:
        dq/dt =  M⁻¹ p
        dp/dt = ∇ log P(q)

    Parameters
    ----------
    q                     : current position (flat parameter vector)
    p                     : current momentum
    grad_log_posterior_fn : function q → (log_prob, gradient)
    step_size             : leapfrog step size ε
    num_steps             : number of leapfrog steps L
    mass_inv              : diagonal of inverse mass matrix

    Returns
    -------
    q_new, p_new, log_prob_new
    """
    log_prob, grad = grad_log_posterior_fn(q)
    p = p + 0.5 * step_size * grad

    for _ in range(num_steps - 1):
        q = q + step_size * mass_inv * p
        log_prob, grad = grad_log_posterior_fn(q)
        p = p + step_size * grad

    q = q + step_size * mass_inv * p
    log_prob, grad = grad_log_posterior_fn(q)
    p = p + 0.5 * step_size * grad

    return q, p, log_prob


# ---------------------------------------------------------------------------
# HMC Sampler
# ---------------------------------------------------------------------------

class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler for ExaTrack model parameters.

    Samples from the posterior P(θ | data) using HMC with:
      - Leapfrog integration for proposals
      - Metropolis-Hastings accept/reject
      - Dual averaging for automatic step-size adaptation (Hoffman & Gelman 2014)
      - Diagonal mass matrix adaptation from warmup samples

    Parameters
    ----------
    model              : fitted ExaTrack likelihood model
    tracks             : (N, 1, T, 1, 1, D) track data
    masks              : (N, T) padding masks
    batch_size         : batch size for likelihood evaluation
    step_size          : initial leapfrog step size
    num_leapfrog_steps : leapfrog steps per HMC iteration
    log_prior_fn       : flat_params → scalar log-prior (default: weak Gaussian)
    param_indices      : which model.weights to sample (default: all trainable)
    mass_diag          : diagonal of mass matrix (default: identity/5)
    target_accept_rate : target Metropolis acceptance rate (default: 0.65)
    fix_model_type     : if True, hold params[:, 4] (motion type flags) fixed
    """

    def __init__(self, model, tracks, masks, batch_size,
                 step_size=1e-3, num_leapfrog_steps=10,
                 log_prior_fn=None, param_indices=None,
                 mass_diag=None, target_accept_rate=0.65,
                 fix_model_type=True):

        self.model = model
        self.tracks = tf.constant(tracks, dtype=dtype)
        self.masks = tf.constant(masks, dtype=dtype)
        self.batch_size = batch_size

        if param_indices is None:
            param_indices = get_trainable_param_indices(model)
        self.param_indices = param_indices
        self._shapes, self._sizes = shapes_and_sizes(model, param_indices)
        self._trainable_weights = [model.weights[i] for i in param_indices]
        self._ndim = sum(self._sizes)

        self.step_size = tf.Variable(step_size, dtype=dtype)
        self.num_leapfrog_steps = num_leapfrog_steps
        self.log_prior_fn = log_prior_fn or default_log_prior

        self.mass_inv = (tf.constant(1.0 / mass_diag, dtype=dtype)
                         if mass_diag is not None
                         else tf.ones(self._ndim, dtype=dtype) / 5)

        # Handle fixed model-type parameters
        self.fix_model_type = fix_model_type
        if fix_model_type:
            free_mask = np.ones(self._ndim, dtype=bool)
            flat_offset = 0
            weight_0_offset = None
            weight_0_shape = None
            for idx, sh, sz in zip(self.param_indices,
                                    self._shapes, self._sizes):
                if idx == 0:
                    weight_0_offset = flat_offset
                    weight_0_shape = sh
                    break
                flat_offset += sz

            if weight_0_offset is not None and weight_0_shape is not None:
                nb_states_model = weight_0_shape[0]
                nb_cols = weight_0_shape[1]
                for s in range(nb_states_model):
                    free_mask[weight_0_offset + s * nb_cols + 4] = False

            self._free_mask = tf.constant(free_mask, dtype=tf.bool)
            self._free_mask_float = tf.cast(self._free_mask, dtype=dtype)
            q_init = flatten_params(self.model, self.param_indices)
            self._fixed_values = tf.where(self._free_mask,
                                           tf.zeros_like(q_init), q_init)
        else:
            self._free_mask = None
            self._free_mask_float = None
            self._fixed_values = None

        # Dual-averaging state
        self.target_accept_rate = target_accept_rate
        self._mu = tf.cast(tf.math.log(10.0 * step_size), dtype=dtype)
        self._log_step_size_bar = tf.Variable(0.0, dtype=dtype)
        self._h_bar = tf.Variable(0.0, dtype=dtype)
        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75

        self.samples = []
        self.log_probs = []
        self.accept_count = 0
        self.total_count = 0

    def _enforce_fixed(self, q):
        """Replace fixed entries in q with their frozen initial values."""
        if self._fixed_values is not None:
            return tf.where(self._free_mask, q, self._fixed_values)
        return q

    def _grad_log_posterior(self, q):
        """Compute log posterior and its gradient at parameter vector q."""
        q = tf.cast(q, dtype)
        q = self._enforce_fixed(q)

        with tf.GradientTape() as tape:
            tape.watch(q)
            offset = 0
            for w, sh, sz in zip(self._trainable_weights,
                                   self._shapes, self._sizes):
                w.assign(tf.reshape(q[offset:offset + sz], sh))
                offset += sz

            y_pred = self.model((self.tracks, self.masks), training=False)
            y_pred = tf.cast(y_pred, dtype)
            max_lp = tf.reduce_max(y_pred, axis=1, keepdims=True)
            per_track_ll = (tf.math.log(
                tf.reduce_sum(tf.exp(y_pred - max_lp), axis=1, keepdims=True))
                + max_lp)
            log_lik = tf.reduce_sum(per_track_ll)
            log_prior = self.log_prior_fn(q)
            log_post = log_lik + log_prior

        grad = tape.gradient(log_post, q)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        if self._free_mask_float is not None:
            grad = grad * self._free_mask_float

        return log_post, grad

    def _hmc_step(self, q_current, log_prob_current):
        """One HMC iteration: sample momentum, leapfrog, Metropolis accept/reject."""
        p_current = tf.random.normal([self._ndim], dtype=dtype) / tf.sqrt(self.mass_inv)
        if self._free_mask_float is not None:
            p_current = p_current * self._free_mask_float

        kinetic_current = 0.5 * tf.reduce_sum(self.mass_inv * p_current ** 2)
        H_current = -log_prob_current + kinetic_current

        q_proposed, p_proposed, log_prob_proposed = leapfrog(
            q_current, p_current,
            self._grad_log_posterior,
            self.step_size, self.num_leapfrog_steps, self.mass_inv)

        q_proposed = self._enforce_fixed(q_proposed)
        kinetic_proposed = 0.5 * tf.reduce_sum(self.mass_inv * p_proposed ** 2)
        H_proposed = -log_prob_proposed + kinetic_proposed

        log_accept_ratio = H_current - H_proposed
        accept_prob = tf.minimum(
            1.0, tf.exp(tf.minimum(log_accept_ratio,
                                    tf.constant(20.0, dtype=dtype))))
        u = tf.random.uniform([], dtype=dtype)
        accepted = u < accept_prob

        if accepted:
            return q_proposed, log_prob_proposed, accept_prob, True
        else:
            offset = 0
            for w, sh, sz in zip(self._trainable_weights,
                                   self._shapes, self._sizes):
                w.assign(tf.reshape(q_current[offset:offset + sz], sh))
                offset += sz
            return q_current, log_prob_current, accept_prob, False

    def _adapt_step_size(self, iteration, accept_prob):
        """Dual averaging step-size adaptation (Hoffman & Gelman, 2014)."""
        m = iteration + 1.0
        w = 1.0 / (m + self._t0)
        self._h_bar.assign(
            (1.0 - w) * self._h_bar
            + w * (self.target_accept_rate - accept_prob))
        log_eps = self._mu - tf.sqrt(m) / self._gamma * self._h_bar
        self.step_size.assign(tf.exp(log_eps))
        m_kappa = m ** (-self._kappa)
        self._log_step_size_bar.assign(
            m_kappa * log_eps
            + (1.0 - m_kappa) * self._log_step_size_bar)

    def _adapt_mass_matrix(self, warmup_samples):
        """Set diagonal mass matrix from empirical variance of warmup samples."""
        if len(warmup_samples) < 20:
            return
        stacked = tf.stack(warmup_samples)
        var = tf.math.reduce_variance(stacked[::-1][:200], axis=0)
        var = tf.maximum(var, tf.constant(1e-8, dtype=dtype))
        self.mass_inv = 1.0 / var

    def sample(self, num_samples=500, num_warmup=200, thin=1,
               adapt_step_size=True, adapt_mass_matrix=True, verbose=True):
        """
        Run the HMC sampler.

        Parameters
        ----------
        num_samples        : number of post-warmup samples to collect
        num_warmup         : warmup (burn-in) iterations with adaptation
        thin               : keep every thin-th sample
        adapt_step_size    : adapt step size during warmup
        adapt_mass_matrix  : adapt mass matrix during warmup
        verbose            : print progress

        Returns
        -------
        samples     : (num_samples//thin, D) flat parameter samples
        log_probs   : (num_samples//thin,) log-posterior at each sample
        accept_rate : overall Metropolis acceptance rate
        """
        q = flatten_params(self.model, self.param_indices)
        q = self._enforce_fixed(q)
        log_prob, _ = self._grad_log_posterior(q)

        warmup_samples = []
        total_iterations = num_warmup + num_samples
        self.samples = []
        self.log_probs = []
        self.accept_count = 0
        self.total_count = 0

        for i in range(total_iterations):
            is_warmup = i < num_warmup
            q, log_prob, accept_prob, accepted = self._hmc_step(q, log_prob)

            self.total_count += 1
            if accepted:
                self.accept_count += 1

            if is_warmup:
                warmup_samples.append(q.numpy().copy())
                if adapt_step_size:
                    self._adapt_step_size(tf.cast(i, dtype=dtype), accept_prob)
                if adapt_mass_matrix and i % 50 == 0:
                    self._adapt_mass_matrix(
                        [tf.constant(s, dtype=dtype) for s in warmup_samples])
                    if verbose:
                        print(f"  [warmup {i}] mass matrix adapted")
                if i == num_warmup - 1:
                    if adapt_step_size:
                        self.step_size.assign(
                            tf.exp(self._log_step_size_bar))
                    if verbose:
                        rate = self.accept_count / max(1, self.total_count)
                        print(f"  Warmup complete. "
                              f"step_size={self.step_size.numpy():.6g}, "
                              f"accept rate={rate:.2%}")
                        self.accept_count = 0
                        self.total_count = 0
            else:
                sample_idx = i - num_warmup
                if sample_idx % thin == 0:
                    self.samples.append(q.numpy().copy())
                    self.log_probs.append(float(log_prob.numpy()))

            if verbose and (i + 1) % max(1, 5) == 0:
                phase = "warmup" if is_warmup else "sampling"
                rate = self.accept_count / max(1, self.total_count)
                print(f"  [{phase} iter {i+1}/{total_iterations}] "
                      f"log_post={float(log_prob.numpy()):.2f} "
                      f"accept={rate:.2%} "
                      f"eps={float(self.step_size.numpy()):.4g}")

        accept_rate = self.accept_count / max(1, self.total_count)
        if verbose:
            print(f"\nSampling done. "
                  f"Collected {len(self.samples)} samples, "
                  f"accept rate={accept_rate:.2%}")

        return (np.array(self.samples),
                np.array(self.log_probs),
                accept_rate)

    def unflatten_samples(self, flat_samples):
        """Convert (N, D) flat samples to a list of per-weight dicts."""
        results = []
        for s in flat_samples:
            d = {}
            offset = 0
            for idx, sh, sz in zip(self.param_indices,
                                    self._shapes, self._sizes):
                d[idx] = np.reshape(s[offset:offset + sz], sh)
                offset += sz
            results.append(d)
        return results

    def get_param_samples(self, flat_samples, weight_index):
        """Extract samples for one model weight from the flat sample array."""
        if weight_index not in self.param_indices:
            raise ValueError(
                f"Weight {weight_index} not in sampled indices "
                f"{self.param_indices}")
        offset = 0
        for idx, sh, sz in zip(self.param_indices,
                                 self._shapes, self._sizes):
            if idx == weight_index:
                return flat_samples[:, offset:offset + sz].reshape((-1,) + tuple(sh))
            offset += sz


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def run_hmc(model, tracks, masks, batch_size,
            num_samples=500, num_warmup=200,
            step_size=1e-3, num_leapfrog_steps=10,
            thin=1, log_prior_fn=None, param_indices=None,
            target_accept_rate=0.65, fix_model_type=True,
            verbose=True):
    """
    High-level wrapper for HMC sampling.

    Parameters
    ----------
    model, tracks, masks, batch_size : standard ExaTrack model/data
    num_samples, num_warmup          : sampling and warmup iterations
    step_size, num_leapfrog_steps    : HMC tuning
    thin                             : keep every thin-th sample
    log_prior_fn                     : custom prior function (default: weak Gaussian)
    param_indices                    : which weights to sample
    target_accept_rate               : dual-averaging target
    fix_model_type                   : keep motion type flags fixed
    verbose                          : print progress

    Returns
    -------
    sampler    : HMCSampler instance (for diagnostics and unflattening)
    samples    : (num_samples//thin, D) flat parameter samples
    log_probs  : log-posterior at each sample
    accept_rate: Metropolis acceptance rate
    """
    sampler = HMCSampler(
        model=model, tracks=tracks, masks=masks, batch_size=batch_size,
        step_size=step_size, num_leapfrog_steps=num_leapfrog_steps,
        log_prior_fn=log_prior_fn, param_indices=param_indices,
        target_accept_rate=target_accept_rate,
        fix_model_type=fix_model_type)

    samples, log_probs, accept_rate = sampler.sample(
        num_samples=num_samples, num_warmup=num_warmup,
        thin=thin, verbose=verbose)

    return sampler, samples, log_probs, accept_rate


# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------

def effective_sample_size(samples):
    """
    Estimate effective sample size (ESS) for each parameter dimension.

    Uses the initial positive sequence estimator (Geyer 1992).

    Parameters
    ----------
    samples : (N, D) array of flat parameter samples

    Returns
    -------
    ess : (D,) effective sample sizes
    """
    n, d = samples.shape
    ess = np.zeros(d)
    for j in range(d):
        x = samples[:, j] - samples[:, j].mean()
        fft_x = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
        acf /= acf[0]
        sum_rho = 0.0
        for t in range(0, n - 1, 2):
            rho_pair = acf[t] + (acf[t + 1] if t + 1 < n else 0.0)
            if rho_pair < 0:
                break
            sum_rho += rho_pair
        tau = -1.0 + 2.0 * sum_rho
        ess[j] = n / max(tau, 1.0)
    return ess


def r_hat(chains):
    """
    Gelman-Rubin R-hat convergence diagnostic for multiple chains.

    R-hat ≈ 1.0 indicates all chains converged to the same distribution.
    R-hat > 1.1 suggests poor mixing or convergence failure.

    Parameters
    ----------
    chains : list of (N, D) arrays — at least 2 chains

    Returns
    -------
    rhat : (D,) R-hat values
    """
    m = len(chains)
    n = chains[0].shape[0]
    chain_means = np.array([c.mean(axis=0) for c in chains])
    grand_mean = chain_means.mean(axis=0)
    B = n / (m - 1.0) * np.sum(
        (chain_means - grand_mean[None, :]) ** 2, axis=0)
    W = np.mean([c.var(axis=0, ddof=1) for c in chains], axis=0)
    var_hat = (n - 1.0) / n * W + B / n
    return np.sqrt(var_hat / (W + 1e-30))


def transform_hmc_samples(flat_samples, sampler):
    """
    Convert raw HMC flat parameter samples to physical parameter arrays.

    Parameters
    ----------
    flat_samples : (N, D) flat parameter samples from HMCSampler.sample()
    sampler      : HMCSampler instance (needed for unflattening)

    Returns
    -------
    dict with keys:
        'Model types'         : (N, nb_states) int array
        'anomalous factors'   : (N, nb_states)
        'Localization errors' : (N, nb_states)
        'd'                   : (N, nb_states)
        'Fractions'           : (N, nb_fractions)
        'transition shapes'   : (N, nb_states, nb_states)
        'transition rates'    : (N, nb_states, nb_states)
    """
    unflattened = sampler.unflatten_samples(flat_samples)
    N = len(unflattened)

    s0 = unflattened[0]
    nb_states = s0[0].shape[0]
    nb_fractions = s0[2].shape[1]

    model_types = np.zeros((N, nb_states), dtype=int)
    anomalous_factors = np.zeros((N, nb_states))
    localization_errors = np.zeros((N, nb_states))
    d_values = np.zeros((N, nb_states))
    fractions = np.zeros((N, nb_fractions))
    tr_shapes = np.zeros((N, nb_states, nb_states))
    tr_rates = np.zeros((N, nb_states, nb_states))

    for i, sample_dict in enumerate(unflattened):
        params = sample_dict[0]
        initial_fractions = sample_dict[2]
        transition_rates_raw = sample_dict[4]
        transition_shapes_raw = sample_dict[5]

        is_dir = params[:, 4]
        model_types[i] = (is_dir > 0.5).astype(int)
        localization_errors[i] = np.exp(params[:, 0])
        d_values[i] = np.exp(params[:, 1])

        a = params[:, 2]
        anomalous_factors[i] = (scipy.special.expit(a) * (1.0 - is_dir)
                                 + np.sqrt(2) * np.exp(a) * is_dir)

        fractions[i] = scipy.special.softmax(initial_fractions[0])
        tr_shapes[i] = np.exp(transition_shapes_raw)
        tr_rates[i] = (scipy.special.softmax(transition_rates_raw, axis=1)
                       * tr_shapes[i])

    return {
        'Model types': model_types,
        'anomalous factors': anomalous_factors,
        'Localization errors': localization_errors,
        'd': d_values,
        'Fractions': fractions,
        'transition shapes': tr_shapes,
        'transition rates': tr_rates,
    }
