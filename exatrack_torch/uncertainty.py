# -*- coding: utf-8 -*-
"""
uncertainty.py
--------------
Uncertainty quantification for fitted ExaTrack models.

PyTorch conversion notes
------------------------
- tf.GradientTape → .backward() with requires_grad on model parameters
- tf.Variable (step_size, h_bar, …) → plain Python floats / torch scalars
- model.weights[i] → list(model.parameters()) accessed by iteration
- flatten_params / unflatten_params → work on model.parameters()
- tf.random.normal → torch.randn
- tf.random.uniform → torch.rand
- tf.where → torch.where
- tf.math.reduce_variance → torch.var
- w.assign(x) → p.data.copy_(x)
- model.compile / model.fit → explicit PyTorch training loop
"""

import numpy as np
import torch
import scipy
from copy import deepcopy

from .config import dtype
from .models import MLE_loss, get_model_params, get_model_raw_params


# ---------------------------------------------------------------------------
# Parameter flatten / unflatten helpers
# ---------------------------------------------------------------------------

def get_trainable_param_indices(model):
    """Return list of (name, param) for all trainable parameters."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flatten_params(model):
    """Flatten all trainable model parameters into a single 1-D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()
                      if p.requires_grad]).to(dtype)


def unflatten_params(flat, model):
    """Write a flat 1-D parameter tensor back into model parameters."""
    offset = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        size = p.numel()
        p.data.copy_(flat[offset:offset + size].view(p.shape).to(p.dtype))
        offset += size


def shapes_and_sizes(model):
    """Return (shapes, sizes) of all trainable model parameters."""
    shapes = [p.shape for p in model.parameters() if p.requires_grad]
    sizes  = [p.numel() for p in model.parameters() if p.requires_grad]
    return shapes, sizes


# ---------------------------------------------------------------------------
# Default prior
# ---------------------------------------------------------------------------

def default_log_prior(flat_params):
    """Weak Gaussian prior: log P(θ) ∝ -0.01 * ||θ||²"""
    return -0.01 * (flat_params ** 2).sum()


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------

def leapfrog(q, p, grad_log_posterior_fn, step_size, num_steps, mass_inv):
    """
    Leapfrog integrator for Hamiltonian dynamics.

    Parameters
    ----------
    q                     : current flat parameter tensor
    p                     : current momentum tensor
    grad_log_posterior_fn : q → (log_prob scalar, gradient tensor)
    step_size             : leapfrog step size ε (float)
    num_steps             : number of leapfrog steps L (int)
    mass_inv              : diagonal of inverse mass matrix (tensor)
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
      - Dual averaging for automatic step-size adaptation
      - Diagonal mass matrix adaptation from warmup samples

    Parameters
    ----------
    model              : fitted ExaTrack SegmentModel
    tracks             : (N, T, D) position array
    LocErrs            : (N, T) or (N, T, d) localisation error array
    dts                : (N, T+1) frame-duration array
    masks              : (N, T) padding mask
    isfirsts           : (N,) first-segment flags
    batch_size         : tracks per evaluation batch
    step_size          : initial leapfrog step size
    num_leapfrog_steps : leapfrog steps per HMC iteration
    log_prior_fn       : flat_params → scalar log-prior
    mass_diag          : diagonal of mass matrix (default: identity/5)
    target_accept_rate : target Metropolis acceptance rate
    fix_model_type     : if True, hold param_vars[:, 4] (isdir) fixed
    device             : torch.device for model and data
    """

    def __init__(self, model, tracks, LocErrs, dts, masks, isfirsts,
                 batch_size,
                 step_size=1e-3, num_leapfrog_steps=10,
                 log_prior_fn=None, mass_diag=None,
                 target_accept_rate=0.65,
                 fix_model_type=True,
                 device='cpu'):

        self.model      = model
        self.batch_size = batch_size
        self.device     = torch.device(device)

        # Store data as torch tensors on device
        self.tracks   = torch.tensor(tracks,   dtype=dtype, device=self.device)
        self.LocErrs  = torch.tensor(LocErrs,  dtype=dtype, device=self.device)
        self.dts      = torch.tensor(dts,      dtype=dtype, device=self.device)
        self.masks    = torch.tensor(masks,    dtype=dtype, device=self.device)
        self.isfirsts = torch.tensor(isfirsts, dtype=dtype, device=self.device)

        self._shapes, self._sizes = shapes_and_sizes(model)
        self._ndim = sum(self._sizes)

        self.step_size         = float(step_size)
        self.num_leapfrog_steps = num_leapfrog_steps
        self.log_prior_fn      = log_prior_fn or default_log_prior

        self.mass_inv = (torch.tensor(1.0 / mass_diag, dtype=dtype)
                         if mass_diag is not None
                         else torch.ones(self._ndim, dtype=dtype) / 5)

        # Handle fixed motion-type parameters (param_vars[:, 4])
        self.fix_model_type = fix_model_type
        if fix_model_type and hasattr(model, 'init_layer'):
            free_mask = torch.ones(self._ndim, dtype=torch.bool)
            offset = 0
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name == 'init_layer.param_vars':
                    nb_states = p.shape[0]
                    nb_cols   = p.shape[1]
                    for s in range(nb_states):
                        free_mask[offset + s * nb_cols + 4] = False
                    break
                offset += p.numel()
            self._free_mask       = free_mask
            self._free_mask_float = free_mask.to(dtype)
            q_init = flatten_params(model)
            self._fixed_values = torch.where(
                self._free_mask, torch.zeros_like(q_init), q_init)
        else:
            self._free_mask       = None
            self._free_mask_float = None
            self._fixed_values    = None

        # Dual-averaging state (plain Python scalars / tensors)
        self.target_accept_rate   = target_accept_rate
        self._mu                  = float(torch.log(
            torch.tensor(10.0 * step_size, dtype=dtype)))
        self._log_step_size_bar   = 0.0
        self._h_bar               = 0.0
        self._gamma               = 0.05
        self._t0                  = 10.0
        self._kappa               = 0.75

        self.samples      = []
        self.log_probs    = []
        self.accept_count = 0
        self.total_count  = 0

    def _enforce_fixed(self, q):
        if self._fixed_values is not None:
            return torch.where(self._free_mask, q, self._fixed_values)
        return q

    def _grad_log_posterior(self, q):
        """Compute log posterior and its gradient at flat parameter vector q."""
        q = q.detach().to(dtype)
        q = self._enforce_fixed(q)

        # Write q into model parameters (no gradient flow here)
        unflatten_params(q, self.model)

        # Enable gradients on model parameters
        for p in self.model.parameters():
            if p.requires_grad:
                p.requires_grad_(True)
                if p.grad is not None:
                    p.grad.zero_()

        # Forward pass — compute log-likelihood
        nb_tracks = self.tracks.shape[0]
        per_track_ll_list = []

        self.model.train()
        for start in range(0, nb_tracks, self.batch_size):
            end = min(start + self.batch_size, nb_tracks)
            out = self.model(
                self.tracks[start:end], self.LocErrs[start:end],
                self.dts[start:end],   self.masks[start:end],
                self.isfirsts[start:end])
            out = out.to(dtype)
            max_lp = out.max(dim=1, keepdim=True).values
            per_track_ll_list.append(
                torch.log(torch.exp(out - max_lp).sum(dim=1, keepdim=True))
                + max_lp)

        log_lik = torch.cat(per_track_ll_list, dim=0).sum()

        # Log prior through the flat param vector assembled from model params
        q_flat = torch.cat([p.view(-1) for p in self.model.parameters()
                             if p.requires_grad])
        log_prior = self.log_prior_fn(q_flat)

        log_post = log_lik + log_prior
        log_post.backward()

        # Collect flattened gradient
        flat_grad = torch.cat([
            (p.grad.view(-1) if p.grad is not None
             else torch.zeros(p.numel(), dtype=dtype))
            for p in self.model.parameters() if p.requires_grad
        ])

        flat_grad = torch.where(
            torch.isfinite(flat_grad), flat_grad, torch.zeros_like(flat_grad))
        if self._free_mask_float is not None:
            flat_grad = flat_grad * self._free_mask_float

        return log_post.detach(), flat_grad.detach()

    def _hmc_step(self, q_current, log_prob_current):
        p_current = torch.randn(self._ndim, dtype=dtype) / self.mass_inv.sqrt()
        if self._free_mask_float is not None:
            p_current = p_current * self._free_mask_float

        kinetic_current = 0.5 * (self.mass_inv * p_current ** 2).sum()
        H_current = -log_prob_current + kinetic_current

        q_proposed, p_proposed, log_prob_proposed = leapfrog(
            q_current, p_current,
            self._grad_log_posterior,
            self.step_size, self.num_leapfrog_steps, self.mass_inv)

        q_proposed = self._enforce_fixed(q_proposed)
        kinetic_proposed = 0.5 * (self.mass_inv * p_proposed ** 2).sum()
        H_proposed = -log_prob_proposed + kinetic_proposed

        log_accept_ratio = H_current - H_proposed
        accept_prob = float(torch.minimum(
            torch.tensor(1.0, dtype=dtype),
            torch.exp(torch.clamp(log_accept_ratio,
                                  max=torch.tensor(20.0, dtype=dtype)))))
        u = float(torch.rand(1, dtype=dtype))
        accepted = u < accept_prob

        if accepted:
            return q_proposed, log_prob_proposed, accept_prob, True
        else:
            unflatten_params(q_current, self.model)
            return q_current, log_prob_current, accept_prob, False

    def _adapt_step_size(self, iteration, accept_prob):
        m   = float(iteration) + 1.0
        w   = 1.0 / (m + self._t0)
        self._h_bar = (1.0 - w) * self._h_bar + w * (
            self.target_accept_rate - accept_prob)
        log_eps = self._mu - (m ** 0.5) / self._gamma * self._h_bar
        self.step_size = float(torch.exp(torch.tensor(log_eps, dtype=dtype)))
        m_kappa = m ** (-self._kappa)
        self._log_step_size_bar = (m_kappa * log_eps
                                   + (1.0 - m_kappa) * self._log_step_size_bar)

    def _adapt_mass_matrix(self, warmup_samples):
        if len(warmup_samples) < 20:
            return
        stacked = torch.stack(warmup_samples)
        var = torch.var(stacked.flip(0)[:200], dim=0)
        var = torch.clamp(var, min=1e-8)
        self.mass_inv = 1.0 / var

    def sample(self, num_samples=500, num_warmup=200, thin=1,
               adapt_step_size=True, adapt_mass_matrix=True, verbose=True):
        """
        Run the HMC sampler.

        Returns
        -------
        samples     : (num_samples//thin, D) flat parameter samples (numpy)
        log_probs   : (num_samples//thin,) log-posterior values (numpy)
        accept_rate : overall Metropolis acceptance rate
        """
        q = flatten_params(self.model)
        q = self._enforce_fixed(q)
        log_prob, _ = self._grad_log_posterior(q)

        warmup_samples = []
        total_iterations = num_warmup + num_samples
        self.samples      = []
        self.log_probs    = []
        self.accept_count = 0
        self.total_count  = 0

        for i in range(total_iterations):
            is_warmup = i < num_warmup
            q, log_prob, accept_prob, accepted = self._hmc_step(q, log_prob)

            self.total_count += 1
            if accepted:
                self.accept_count += 1

            if is_warmup:
                warmup_samples.append(q.detach().clone())
                if adapt_step_size:
                    self._adapt_step_size(i, accept_prob)
                if adapt_mass_matrix and i % 50 == 0:
                    self._adapt_mass_matrix(warmup_samples)
                    if verbose:
                        print(f"  [warmup {i}] mass matrix adapted")
                if i == num_warmup - 1:
                    if adapt_step_size:
                        self.step_size = float(torch.exp(
                            torch.tensor(self._log_step_size_bar, dtype=dtype)))
                    if verbose:
                        rate = self.accept_count / max(1, self.total_count)
                        print(f"  Warmup complete. "
                              f"step_size={self.step_size:.6g}, "
                              f"accept rate={rate:.2%}")
                        self.accept_count = 0
                        self.total_count  = 0
            else:
                sample_idx = i - num_warmup
                if sample_idx % thin == 0:
                    self.samples.append(q.detach().cpu().numpy().copy())
                    self.log_probs.append(float(log_prob))

            if verbose and (i + 1) % max(1, 5) == 0:
                phase = "warmup" if is_warmup else "sampling"
                rate  = self.accept_count / max(1, self.total_count)
                print(f"  [{phase} iter {i+1}/{total_iterations}] "
                      f"log_post={float(log_prob):.2f} "
                      f"accept={rate:.2%} "
                      f"eps={self.step_size:.4g}")

        accept_rate = self.accept_count / max(1, self.total_count)
        if verbose:
            print(f"\nSampling done. "
                  f"Collected {len(self.samples)} samples, "
                  f"accept rate={accept_rate:.2%}")

        return (np.array(self.samples),
                np.array(self.log_probs),
                accept_rate)

    def unflatten_samples(self, flat_samples):
        """Convert (N, D) flat samples to a list of per-parameter-name dicts."""
        results = []
        for s in flat_samples:
            d = {}
            offset = 0
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                size = p.numel()
                d[name] = s[offset:offset + size].reshape(p.shape)
                offset += size
            results.append(d)
        return results

    def get_param_samples(self, flat_samples, param_name):
        """Extract samples for one named parameter from the flat sample array."""
        offset = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name == param_name:
                sz = p.numel()
                return flat_samples[:, offset:offset + sz].reshape(
                    (-1,) + tuple(p.shape))
            offset += p.numel()
        raise ValueError(f"Parameter '{param_name}' not found in model.")


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def run_hmc(model, tracks, LocErrs, dts, masks, isfirsts, batch_size,
            num_samples=500, num_warmup=200,
            step_size=1e-3, num_leapfrog_steps=10,
            thin=1, log_prior_fn=None,
            target_accept_rate=0.65, fix_model_type=True,
            verbose=True, device='cpu'):
    """
    High-level wrapper for HMC sampling.

    Returns
    -------
    sampler    : HMCSampler instance (for diagnostics and unflattening)
    samples    : (num_samples//thin, D) flat parameter samples (numpy)
    log_probs  : log-posterior at each sample (numpy)
    accept_rate: Metropolis acceptance rate
    """
    sampler = HMCSampler(
        model=model, tracks=tracks, LocErrs=LocErrs, dts=dts,
        masks=masks, isfirsts=isfirsts, batch_size=batch_size,
        step_size=step_size, num_leapfrog_steps=num_leapfrog_steps,
        log_prior_fn=log_prior_fn,
        target_accept_rate=target_accept_rate,
        fix_model_type=fix_model_type,
        device=device)

    samples, log_probs, accept_rate = sampler.sample(
        num_samples=num_samples, num_warmup=num_warmup,
        thin=thin, verbose=verbose)

    return sampler, samples, log_probs, accept_rate


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def bootstrapping(model, tracks, LocErrs, dts, masks, isfirsts,
                  bootstrap_number=100, epochs=100, batch_size=65,
                  learning_rate=1/100, decay_threshold=None,
                  decay_rate=None, verbose=1,
                  track_segmentation=False, device='cpu'):
    """
    Estimate parameter uncertainty via bootstrap resampling.

    For each bootstrap iteration:
      1. Resample tracks with replacement
      2. Reset model to original fitted weights
      3. Refit model on resampled tracks
      4. Record the fitted parameters

    Parameters
    ----------
    model         : fitted ExaTrack SegmentModel
    tracks        : (N, T, D)
    LocErrs       : (N, T) or (N, T, d)
    dts           : (N, T+1)
    masks         : (N, T)
    isfirsts      : (N,)
    """
    import math
    from .training import WarmupLearningRateSchedule
    from .models import get_parameters

    torch_device = torch.device(device)
    nb_tracks    = tracks.shape[0]
    nb_batches   = nb_tracks // batch_size

    if decay_threshold is None:
        decay_threshold = int(epochs * nb_batches * 0.75)
    if decay_rate is None:
        decay_rate = -math.log(0.001) / (0.25 * epochs * nb_batches)

    # Snapshot original parameters
    original_state = deepcopy(model.state_dict())

    all_model_parameters = []
    all_likelihoods      = []

    for boot_i in range(bootstrap_number):
        # Restore model weights
        model.load_state_dict(original_state)
        model.to(torch_device)

        # Resample with replacement
        idx             = np.random.randint(0, nb_tracks, size=nb_tracks)
        s_tracks        = tracks[idx]
        s_LocErrs       = LocErrs[idx]
        s_dts           = dts[idx]
        s_masks         = masks[idx]
        s_isfirsts      = isfirsts[idx]

        lr_schedule = WarmupLearningRateSchedule(
            10, learning_rate, decay_rate, decay_threshold)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr_schedule(0),
            betas=(0.9, 0.99))

        model.train()
        loss_history = []
        global_step  = 0

        for epoch in range(epochs):
            epoch_losses = []
            for start in range(0, nb_tracks, batch_size):
                end = min(start + batch_size, nb_tracks)
                bt  = torch.tensor(s_tracks[start:end],   dtype=dtype, device=torch_device)
                bl  = torch.tensor(s_LocErrs[start:end],  dtype=dtype, device=torch_device)
                bd  = torch.tensor(s_dts[start:end],      dtype=dtype, device=torch_device)
                bm  = torch.tensor(s_masks[start:end],    dtype=dtype, device=torch_device)
                bi  = torch.tensor(s_isfirsts[start:end], dtype=dtype, device=torch_device)

                new_lr = lr_schedule(global_step)
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr

                optimizer.zero_grad()
                outputs = model(bt, bl, bd, bm, bi)
                loss = MLE_loss(outputs)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

            epoch_loss = float(np.mean(epoch_losses))
            loss_history.append(epoch_loss)

        parameter_dict = get_model_params(model, track_segmentation)
        all_model_parameters.append(parameter_dict)
        all_likelihoods.append(loss_history[-1])
        if verbose:
            print(f"  Bootstrap {boot_i+1}/{bootstrap_number}  "
                  f"loss={loss_history[-1]:.4f}")

    # Restore original weights
    model.load_state_dict(original_state)
    return all_model_parameters, all_likelihoods


# ---------------------------------------------------------------------------
# MCMC diagnostics (pure numpy — unchanged from TF version)
# ---------------------------------------------------------------------------

def effective_sample_size(samples):
    """
    Estimate effective sample size (ESS) for each parameter dimension
    using the initial positive sequence estimator (Geyer 1992).
    """
    n, d = samples.shape
    ess  = np.zeros(d)
    for j in range(d):
        x     = samples[:, j] - samples[:, j].mean()
        fft_x = np.fft.fft(x, n=2 * n)
        acf   = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
        acf  /= acf[0]
        sum_rho = 0.0
        for t in range(0, n - 1, 2):
            rho_pair = acf[t] + (acf[t + 1] if t + 1 < n else 0.0)
            if rho_pair < 0:
                break
            sum_rho += rho_pair
        tau    = -1.0 + 2.0 * sum_rho
        ess[j] = n / max(tau, 1.0)
    return ess


def r_hat(chains):
    """
    Gelman-Rubin R-hat convergence diagnostic for multiple chains.
    R-hat ≈ 1.0 indicates convergence; R-hat > 1.1 suggests poor mixing.
    """
    m = len(chains)
    n = chains[0].shape[0]
    chain_means = np.array([c.mean(axis=0) for c in chains])
    grand_mean  = chain_means.mean(axis=0)
    B = n / (m - 1.0) * np.sum(
        (chain_means - grand_mean[None, :]) ** 2, axis=0)
    W = np.mean([c.var(axis=0, ddof=1) for c in chains], axis=0)
    var_hat = (n - 1.0) / n * W + B / n
    return np.sqrt(var_hat / (W + 1e-30))


def transform_hmc_samples(flat_samples, sampler):
    """
    Convert raw HMC flat parameter samples to physical parameter arrays.

    Returns dict with keys:
        'Model types', 'anomalous factors', 'Localization errors',
        'd', 'Fractions', 'transition shapes', 'transition rates'
    """
    unflattened = sampler.unflatten_samples(flat_samples)
    N = len(unflattened)

    s0        = unflattened[0]
    pv_key    = 'init_layer.param_vars'
    frac_key  = 'init_layer.initial_fractions'
    tr_key    = 'rnn_layer.transition_rates'
    ts_key    = 'rnn_layer.transition_shapes'

    nb_states     = s0[pv_key].shape[0]
    nb_fractions  = s0[frac_key].shape[1]

    model_types         = np.zeros((N, nb_states), dtype=int)
    anomalous_factors   = np.zeros((N, nb_states))
    localization_errors = np.zeros((N, nb_states))
    d_values            = np.zeros((N, nb_states))
    fractions           = np.zeros((N, nb_fractions))
    tr_shapes           = np.zeros((N, nb_states, nb_states))
    tr_rates            = np.zeros((N, nb_states, nb_states))

    for i, sample_dict in enumerate(unflattened):
        params              = sample_dict[pv_key]
        initial_fractions   = sample_dict[frac_key]
        transition_rates_r  = sample_dict[tr_key]
        transition_shapes_r = sample_dict[ts_key]

        is_dir = params[:, 4]
        model_types[i]         = (is_dir > 0.5).astype(int)
        localization_errors[i] = np.exp(params[:, 0])
        d_values[i]            = np.exp(params[:, 1])

        a = params[:, 2]
        anomalous_factors[i] = (scipy.special.expit(a) * (1.0 - is_dir)
                                 + np.sqrt(2) * np.exp(a) * is_dir)

        fractions[i]  = scipy.special.softmax(initial_fractions[0])
        tr_shapes[i]  = np.exp(transition_shapes_r)
        tr_rates[i]   = (scipy.special.softmax(transition_rates_r, axis=1)
                         * tr_shapes[i])

    return {
        'Model types':         model_types,
        'anomalous factors':   anomalous_factors,
        'Localization errors':  localization_errors,
        'd':                   d_values,
        'Fractions':           fractions,
        'transition shapes':   tr_shapes,
        'transition rates':    tr_rates,
    }
