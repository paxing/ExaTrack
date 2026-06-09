# -*- coding: utf-8 -*-
"""
gaussian_ops.py
---------------
Core mathematical operations on univariate Gaussians.

This module is the mathematical heart of the CGP framework. It contains:
  - Log-Gaussian evaluations
  - The fundamental Gaussian product operation (equation 5 of the CGP paper)
  - The step functions used by the integration scheduler

All functions here are pure math — no nn.Module, no physical model
assumptions. They can be tested and reasoned about in isolation.

Dependency: config.py only.

PyTorch conversion notes
------------------------
- @tf.function decorators removed (PyTorch is always eager)
- tf.math.* → torch.*
- tf.unstack(x) → list(x.unbind(0))  [unbind reduces rank by 1, same as unstack]
- tf.stack(list) → torch.stack(list, dim=0)
- tf.identity(x) → x  (no-op in eager PyTorch)
- tf.math.divide_no_nan(a,b) → divide_no_nan(a, b) helper
- tf.random.normal([1,1,1], 0, std) → torch.randn(1,1,1, dtype=dtype) * std
"""

import torch
import numpy as np
from .config import dtype, pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def divide_no_nan(a, b):
    """Element-wise a/b, returning 0 where b == 0 (matches tf.math.divide_no_nan)."""
    mask = (b == 0)
    safe_b = torch.where(mask, torch.ones_like(b), b)
    return torch.where(mask, torch.zeros_like(a), a / safe_b)


# ---------------------------------------------------------------------------
# Log-Gaussian evaluations
# ---------------------------------------------------------------------------

def log_gaussian(top, variance=None):
    """
    Log of a Gaussian with arbitrary variance evaluated at `top`.

        log f(z) = -0.5 * log(2π σ²) - z² / (2σ²)

    Parameters
    ----------
    top      : tensor — the argument z
    variance : tensor — σ², defaults to 1 (standard normal)
    """
    if variance is None:
        variance = 1.0
    return -0.5 * (np.log(2 * pi) + torch.log(torch.as_tensor(variance, dtype=dtype, device=top.device))) - top ** 2 / (2 * variance)


def norm_log_gaussian(top):
    """
    Log of a *standard* normal Gaussian (mean=0, variance=1) evaluated at `top`.

        log f(z) = -0.5 * (log(2π) + z²)
    """
    return -0.5 * (np.log(2 * pi) + top ** 2)


# ---------------------------------------------------------------------------
# Gaussian product: the atomic CGP operation
# ---------------------------------------------------------------------------

def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2,
                         next_hidden_var_coefs_1, next_hidden_var_coefs_2,
                         biases_1, biases_2, coef_index, nb_dims=1):
    """
    Multiply two Gaussians that each depend on a common hidden variable,
    then integrate out that variable (equation 5 of the CGP paper).

    Parameters
    ----------
    current_hidden_var_coefs_1/2 : (nb_tracks, nb_states, nb_hidden_vars)
    next_hidden_var_coefs_1/2    : same shape
    biases_1/2                   : (nb_tracks, nb_states, nb_dims)
    coef_index : int
    nb_dims    : int

    Returns
    -------
    LogConstant, current_coefs3, current_coefs4,
    next_coefs3, next_coefs4, biases3, biases4
    """
    C1 = (current_hidden_var_coefs_1[:, :, coef_index:coef_index + 1] + 1e-20)
    C2 = (current_hidden_var_coefs_2[:, :, coef_index:coef_index + 1] + 1e-20)

    current_coefs1 = divide_no_nan(current_hidden_var_coefs_1, C1)
    current_coefs2 = divide_no_nan(current_hidden_var_coefs_2, C2)
    next_coefs1 = divide_no_nan(next_hidden_var_coefs_1, C1)
    next_coefs2 = divide_no_nan(next_hidden_var_coefs_2, C2)
    biases1 = divide_no_nan(biases_1, C1)
    biases2 = divide_no_nan(biases_2, C2)

    var1 = 1. / (C1 ** 2 + 1e-100)
    var2 = 1. / (C2 ** 2 + 1e-100)

    var3 = var1 + var2
    std3 = var3 ** 0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    biases3 = (biases1 - biases2) / std3

    var4 = var1 * var2 / var3
    std4 = var4 ** 0.5
    current_coefs4 = (current_coefs1 * var2 + current_coefs2 * var1) / (var3 * std4)
    next_coefs4 = (next_coefs1 * var2 + next_coefs2 * var1) / (var3 * std4)
    biases4 = (biases1 * var2 + biases2 * var1) / (var3 * std4)

    LogConstant = -nb_dims * torch.log(torch.abs(C1 * C2 * std4 * std3))[:, :, 0]

    return (LogConstant,
            current_coefs3, current_coefs4,
            next_coefs3, next_coefs4,
            biases3, biases4)


def simple_RNN_gaussian_product(C1, C2,
                                current_hidden_var_coefs_1, current_hidden_var_coefs_2,
                                next_hidden_var_coefs_1, next_hidden_var_coefs_2):
    """
    Numpy-only simplification of RNN_gaussian_product used by get_sequences()
    during the one-time precomputation of integration schedules.

    No bias handling, no numerical noise, no torch overhead — just the coefficient
    arithmetic needed to trace which Gaussians depend on which variables.
    """
    current_coefs1 = current_hidden_var_coefs_1 / C1
    current_coefs2 = current_hidden_var_coefs_2 / C2
    next_coefs1 = next_hidden_var_coefs_1 / C1
    next_coefs2 = next_hidden_var_coefs_2 / C2

    var1 = 1. / C1 ** 2
    var2 = 1. / C2 ** 2
    var3 = var1 + var2
    std3 = var3 ** 0.5

    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3

    var4 = var1 * var2 / var3
    std4 = var4 ** 0.5
    current_coefs4 = (current_coefs1 * var2 + current_coefs2 * var1) / (var3 * std4)
    next_coefs4 = (next_coefs1 * var2 + next_coefs2 * var1) / (var3 * std4)

    return current_coefs3, current_coefs4, next_coefs3, next_coefs4


# ---------------------------------------------------------------------------
# Step functions used by the integration scheduler
# ---------------------------------------------------------------------------
# Each function below corresponds to one step in the pre-computed integration
# schedule returned by get_sequences(). The schedule tells which function to
# call and which Gaussian indices to operate on.
#
# Phase 1 functions: integrate over *current* hidden variables.
# Phase 2 functions: rearrange remaining *next* hidden variables.
#
# Naming convention:
#   intermediate_*  — there are still more Gaussians depending on this variable
#   final_*         — this is the last Gaussian depending on this variable
#   no_*            — only one Gaussian depends on this variable (trivial case)
# ---------------------------------------------------------------------------

def intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs,
                              biases, coef_index, ID_1, ID_2,
                              nb_hidden_variables, LC, nb_gaussians,
                              kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Intermediate phase-1 step: multiply Gaussians ID_1 and ID_2 together,
    replacing them with the phi and eta outputs of RNN_gaussian_product.
    """
    current_hidden_var_coefs_cp = list(current_hidden_var_coefs.unbind(0))
    next_hidden_var_coefs_cp = list(next_hidden_var_coefs.unbind(0))
    biases_cp = list(biases.unbind(0))

    (current_hidden_var_coefs_1, current_hidden_var_coefs_2,
     next_hidden_var_coefs_1, next_hidden_var_coefs_2,
     biases_1, biases_2) = (current_hidden_var_coefs_cp[ID_1],
                             current_hidden_var_coefs_cp[ID_2],
                             next_hidden_var_coefs_cp[ID_1],
                             next_hidden_var_coefs_cp[ID_2],
                             biases_cp[ID_1],
                             biases_cp[ID_2])

    (LogConstant,
     current_coefs3, current_coefs4,
     next_coefs3, next_coefs4,
     biases3, biases4) = RNN_gaussian_product(
        current_hidden_var_coefs_1, current_hidden_var_coefs_2,
        next_hidden_var_coefs_1, next_hidden_var_coefs_2,
        biases_1, biases_2, coef_index, nb_dims)

    current_hidden_var_coefs_cp[ID_1] = current_coefs3
    current_hidden_var_coefs_cp[ID_2] = current_coefs4
    next_hidden_var_coefs_cp[ID_1] = next_coefs3
    next_hidden_var_coefs_cp[ID_2] = next_coefs4
    biases_cp[ID_1] = biases3
    biases_cp[ID_2] = biases4
    LC = LC + LogConstant

    return (torch.stack(current_hidden_var_coefs_cp),
            torch.stack(next_hidden_var_coefs_cp),
            torch.stack(biases_cp),
            LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


def final_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs,
                               biases, coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Final phase-1 step: after multiplying the last pair of Gaussians that
    depend on coef_index, remove Gaussian ID_2 and account for normalisation.
    """
    (current_hidden_var_coefs_cp,
     next_hidden_var_coefs_cp,
     biases_cp, LC, nb_gaussians,
     kept_next_hidden_var_coefs,
     kept_biases) = intermediate_RNN_function(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
        kept_next_hidden_var_coefs, kept_biases, nb_dims)

    current_hidden_var_coefs_cp = list(current_hidden_var_coefs_cp.unbind(0))
    next_hidden_var_coefs_cp = list(next_hidden_var_coefs_cp.unbind(0))
    biases_cp = list(biases_cp.unbind(0))

    LC = LC + (-nb_dims
               * torch.log(torch.abs(current_hidden_var_coefs_cp[ID_2][:, :, coef_index])))

    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    nb_gaussians -= 1

    biases_cp = torch.stack(biases_cp) if biases_cp else biases[:0]

    return (torch.stack(current_hidden_var_coefs_cp),
            torch.stack(next_hidden_var_coefs_cp),
            biases_cp, LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


def no_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs,
                            biases, coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Trivial phase-1 step: only one Gaussian depends on coef_index,
    so no product is needed — just remove it and account for normalisation.
    """
    current_hidden_var_coefs_cp = list(current_hidden_var_coefs.unbind(0))
    next_hidden_var_coefs_cp = list(next_hidden_var_coefs.unbind(0))
    biases_cp = list(biases.unbind(0))

    LC = LC + (-nb_dims
               * torch.log(torch.abs(current_hidden_var_coefs_cp[ID_2][:, :, coef_index])))

    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    nb_gaussians -= 1

    biases_cp = torch.stack(biases_cp) if biases_cp else biases[:0]

    stacked_c = (torch.stack(current_hidden_var_coefs_cp)
                 if current_hidden_var_coefs_cp else current_hidden_var_coefs[:0])
    stacked_n = (torch.stack(next_hidden_var_coefs_cp)
                 if next_hidden_var_coefs_cp else next_hidden_var_coefs[:0])

    return (stacked_c, stacked_n,
            biases_cp, LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


def final_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs,
                               biases, coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Final phase-2 step: after rearranging the last Gaussian that depends on
    coef_index in the next-variable axis, move it to the kept buffer.
    """
    (next_hidden_var_coefs_cp,
     current_hidden_var_coefs_cp,
     biases_cp, LC, nb_gaussians,
     kept_next_hidden_var_coefs,
     kept_biases) = intermediate_RNN_function(
        next_hidden_var_coefs, current_hidden_var_coefs, biases,
        coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
        kept_next_hidden_var_coefs, kept_biases, nb_dims)

    current_hidden_var_coefs_cp = list(current_hidden_var_coefs_cp.unbind(0))
    next_hidden_var_coefs_cp = list(next_hidden_var_coefs_cp.unbind(0))
    biases_cp = list(biases_cp.unbind(0))

    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)

    kept_next_hidden_var_coefs_cp = (list(kept_next_hidden_var_coefs)
                                     if isinstance(kept_next_hidden_var_coefs, list)
                                     else list(kept_next_hidden_var_coefs.unbind(0)))
    kept_biases_cp = (list(kept_biases)
                      if isinstance(kept_biases, list)
                      else list(kept_biases.unbind(0)))
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    nb_gaussians -= 1

    return (torch.stack(next_hidden_var_coefs_cp),
            torch.stack(current_hidden_var_coefs_cp),
            torch.stack(biases_cp), LC, nb_gaussians,
            torch.stack(kept_next_hidden_var_coefs_cp),
            torch.stack(kept_biases_cp))


def no_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs,
                            biases, coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Trivial phase-2 step: only one Gaussian depends on coef_index in the
    next-variable axis — move it directly to the kept buffer.
    """
    next_hidden_var_coefs_cp = list(next_hidden_var_coefs.unbind(0))
    biases_cp = list(biases.unbind(0))

    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)

    kept_next_hidden_var_coefs_cp = (list(kept_next_hidden_var_coefs)
                                     if isinstance(kept_next_hidden_var_coefs, list)
                                     else list(kept_next_hidden_var_coefs.unbind(0)))
    kept_biases_cp = (list(kept_biases)
                      if isinstance(kept_biases, list)
                      else list(kept_biases.unbind(0)))
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    nb_gaussians -= 1

    biases_cp = torch.stack(biases_cp) if biases_cp else biases[:0]

    return (torch.stack(next_hidden_var_coefs_cp),
            current_hidden_var_coefs,
            biases_cp, LC, nb_gaussians,
            torch.stack(kept_next_hidden_var_coefs_cp),
            torch.stack(kept_biases_cp))
