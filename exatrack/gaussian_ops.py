# -*- coding: utf-8 -*-
"""
gaussian_ops.py
---------------
Core mathematical operations on univariate Gaussians.

This module is the mathematical heart of the CGP framework. It contains:
  - Log-Gaussian evaluations
  - The fundamental Gaussian product operation (equation 5 of the CGP paper)
  - The step functions used by the integration scheduler

All functions here are pure math — no Keras layers, no physical model
assumptions. They can be tested and reasoned about in isolation.

Dependency: config.py only.
"""

import tensorflow as tf
import numpy as np
from .config import dtype, pi, jit_compile


# ---------------------------------------------------------------------------
# Log-Gaussian evaluations
# ---------------------------------------------------------------------------

@tf.function(jit_compile=jit_compile)
def log_gaussian(top, variance=tf.constant(1, dtype=dtype)):
    """
    Log of a Gaussian with arbitrary variance evaluated at `top`.

        log f(z) = -0.5 * log(2π σ²) - z² / (2σ²)

    Parameters
    ----------
    top      : tensor — the argument z (already a linear combination of variables)
    variance : tensor — σ², defaults to 1 (standard normal)
    """
    return -0.5 * tf.math.log(2 * pi * variance) - top ** 2 / (2 * variance)


@tf.function(jit_compile=jit_compile)
def norm_log_gaussian(top):
    """
    Log of a *standard* normal Gaussian (mean=0, variance=1) evaluated at `top`.

        log f(z) = -0.5 * (log(2π) + z²)

    This is the fast, unit-variance specialisation used throughout the
    recurrence after each Gaussian has been normalised by its aij coefficient
    so that the effective variance equals 1.
    """
    return -0.5 * (tf.math.log(2 * pi) + top ** 2)


# ---------------------------------------------------------------------------
# Gaussian product: the atomic CGP operation
# ---------------------------------------------------------------------------

def RNN_gaussian_product(current_hidden_var_coefs_1, current_hidden_var_coefs_2,
                         next_hidden_var_coefs_1, next_hidden_var_coefs_2,
                         biases_1, biases_2, coef_index, nb_dims=1):
    """
    Multiply two Gaussians that each depend on a common hidden variable,
    then integrate out that variable (equation 5 of the CGP paper).

    Each Gaussian is represented by its coefficient vector over the hidden
    variables and a bias vector. The argument of Gaussian i is:

        z_i = sum_j current_coefs_i[j] * y_j
              + sum_j next_coefs_i[j] * y_{j, next}
              - bias_i

    This function:
      1. Normalises both Gaussians by their coefficient at `coef_index`
         (equivalent to completing the square in that variable).
      2. Computes the product, yielding:
           - phi  (coefs3, biases3): the scaling Gaussian whose integral gives
             the log-likelihood contribution LogConstant.
           - eta  (coefs4, biases4): the new Gaussian passed forward to the
             next time step.

    Parameters
    ----------
    current_hidden_var_coefs_1/2 : (nb_tracks, nb_states, nb_hidden_vars)
        Coefficients on current hidden variables y_k for each Gaussian.
    next_hidden_var_coefs_1/2    : same shape
        Coefficients on next hidden variables y_{k+1}.
    biases_1/2                   : (nb_tracks, nb_states, nb_dims)
        Bias vectors.
    coef_index : int
        Index of the hidden variable to integrate out.
    nb_dims    : int
        Number of independent spatial dimensions (multiplies LogConstant).

    Returns
    -------
    LogConstant   : (nb_tracks, nb_states)  — log of the scaling factor φ.
    current_coefs3, current_coefs4 : updated current-step coefficient vectors.
    next_coefs3,    next_coefs4    : updated next-step coefficient vectors.
    biases3,        biases4        : updated bias vectors.
    """
    # Extract the pivot coefficient for the variable being integrated out
    C1 = (current_hidden_var_coefs_1[:, :, coef_index:coef_index + 1]
          + tf.random.normal([1, 1, 1], 0, 1e-20, dtype=dtype))
    C2 = (current_hidden_var_coefs_2[:, :, coef_index:coef_index + 1]
          + tf.random.normal([1, 1, 1], 0, 1e-20, dtype=dtype))

    # Normalise: divide all coefficients and biases by the pivot
    # After normalisation the pivot coefficient equals 1, which prepares
    # the completing-the-square integration step.
    current_coefs1 = tf.math.divide_no_nan(current_hidden_var_coefs_1, C1)
    current_coefs2 = tf.math.divide_no_nan(current_hidden_var_coefs_2, C2)
    next_coefs1 = tf.math.divide_no_nan(next_hidden_var_coefs_1, C1)
    next_coefs2 = tf.math.divide_no_nan(next_hidden_var_coefs_2, C2)
    biases1 = tf.math.divide_no_nan(biases_1, C1[:, :])
    biases2 = tf.math.divide_no_nan(biases_2, C2[:, :])

    # Effective variances after normalisation: σ² = 1/C²
    var1 = 1. / (C1 ** 2 + tf.random.normal([1, 1, 1], 0, 1e-100, dtype=dtype))
    var2 = 1. / (C2 ** 2 + tf.random.normal([1, 1, 1], 0, 1e-100, dtype=dtype))

    # phi — the scaling Gaussian (log-likelihood contribution)
    # Its variance is var1 + var2; its coefficients encode the distance
    # between the two Gaussian means.
    var3 = var1 + var2
    std3 = var3 ** 0.5
    current_coefs3 = (current_coefs1 - current_coefs2) / std3
    next_coefs3 = (next_coefs1 - next_coefs2) / std3
    biases3 = (biases1 - biases2) / std3[:, :]

    # eta — the new Gaussian passed forward to the next time step
    # Its variance is the harmonic mean var1*var2/(var1+var2).
    var4 = var1 * var2 / var3
    std4 = var4 ** 0.5
    current_coefs4 = (current_coefs1 * var2 + current_coefs2 * var1) / (var3 * std4)
    next_coefs4 = (next_coefs1 * var2 + next_coefs2 * var1) / (var3 * std4)
    biases4 = (biases1 * var2[:, :] + biases2 * var1[:, :]) / (var3 * std4)[:, :]

    # Log of the scaling factor (accumulates into total log-likelihood)
    LogConstant = -nb_dims * tf.math.log(tf.math.abs(C1 * C2 * std4 * std3))[:, :, 0]

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

    No bias handling, no numerical noise, no TF overhead — just the coefficient
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

@tf.function(jit_compile=jit_compile)
def intermediate_RNN_function(current_hidden_var_coefs, next_hidden_var_coefs,
                              biases, coef_index, ID_1, ID_2,
                              nb_hidden_variables, LC, nb_gaussians,
                              kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Intermediate phase-1 step: multiply Gaussians ID_1 and ID_2 together,
    replacing them with the phi and eta outputs of RNN_gaussian_product.
    The variable at coef_index is NOT yet removed — more Gaussians still
    depend on it and will be folded in by subsequent steps.
    """
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)

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

    current_hidden_var_coefs_cp[ID_1] = tf.identity(current_coefs3)
    current_hidden_var_coefs_cp[ID_2] = tf.identity(current_coefs4)
    next_hidden_var_coefs_cp[ID_1] = tf.identity(next_coefs3)
    next_hidden_var_coefs_cp[ID_2] = tf.identity(next_coefs4)
    biases_cp[ID_1] = tf.identity(biases3)
    biases_cp[ID_2] = tf.identity(biases4)
    LC += LogConstant

    return (tf.stack(current_hidden_var_coefs_cp),
            tf.stack(next_hidden_var_coefs_cp),
            tf.stack(biases_cp),
            LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


@tf.function(jit_compile=jit_compile)
def final_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs,
                               biases, coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Final phase-1 step: after multiplying the last pair of Gaussians that
    depend on coef_index, remove Gaussian ID_2 (it has been fully absorbed)
    and account for the normalisation factor.
    """
    (current_hidden_var_coefs_cp,
     next_hidden_var_coefs_cp,
     biases_cp, LC, nb_gaussians,
     kept_next_hidden_var_coefs,
     kept_biases) = intermediate_RNN_function(
        current_hidden_var_coefs, next_hidden_var_coefs, biases,
        coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
        kept_next_hidden_var_coefs, kept_biases, nb_dims)

    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)

    # Account for the normalisation: log|a_ij| is subtracted because
    # log_gaussian(z * a, 1) = log_gaussian(z, 1/a²) - log|a|
    LC += (-nb_dims
           * tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:, :, coef_index])))

    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    nb_gaussians -= 1

    biases_cp = tf.cast(
        tf.reshape(tf.stack(biases_cp), [len(biases_cp)] + biases.shape[1:]),
        dtype=dtype)

    return (tf.stack(current_hidden_var_coefs_cp),
            tf.stack(next_hidden_var_coefs_cp),
            biases_cp, LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


@tf.function(jit_compile=jit_compile)
def no_RNN_function_phase_1(current_hidden_var_coefs, next_hidden_var_coefs,
                            biases, coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Trivial phase-1 step: only one Gaussian depends on coef_index,
    so no product is needed — just remove it and account for normalisation.
    """
    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)

    LC += (-nb_dims
           * tf.math.log(tf.abs(current_hidden_var_coefs_cp[ID_2][:, :, coef_index])))

    current_hidden_var_coefs_cp.pop(ID_2)
    next_hidden_var_coefs_cp.pop(ID_2)
    biases_cp.pop(ID_2)
    nb_gaussians -= 1

    biases_cp = tf.cast(
        tf.reshape(tf.stack(biases_cp), [len(biases_cp)] + biases.shape[1:]),
        dtype=dtype)

    return (tf.stack(current_hidden_var_coefs_cp),
            tf.stack(next_hidden_var_coefs_cp),
            biases_cp, LC, nb_gaussians,
            kept_next_hidden_var_coefs, kept_biases)


@tf.function(jit_compile=jit_compile)
def final_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs,
                               biases, coef_index, ID_1, ID_2,
                               nb_hidden_variables, LC, nb_gaussians,
                               kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Final phase-2 step: after rearranging the last Gaussian that depends on
    coef_index in the *next* variable axis, move it to the kept buffer so
    it can be returned as Next_coefs at the end of the recurrence.
    """
    (next_hidden_var_coefs_cp,
     current_hidden_var_coefs_cp,
     biases_cp, LC, nb_gaussians,
     kept_next_hidden_var_coefs,
     kept_biases) = intermediate_RNN_function(
        next_hidden_var_coefs, current_hidden_var_coefs, biases,
        coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
        kept_next_hidden_var_coefs, kept_biases, nb_dims)

    current_hidden_var_coefs_cp = tf.unstack(current_hidden_var_coefs_cp)
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs_cp)
    biases_cp = tf.unstack(biases_cp)

    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)

    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    nb_gaussians -= 1

    return (tf.stack(next_hidden_var_coefs_cp),
            tf.stack(current_hidden_var_coefs_cp),
            tf.stack(biases_cp), LC, nb_gaussians,
            tf.stack(kept_next_hidden_var_coefs_cp),
            tf.stack(kept_biases_cp))


@tf.function(jit_compile=jit_compile)
def no_RNN_function_phase_2(next_hidden_var_coefs, current_hidden_var_coefs,
                            biases, coef_index, ID_1, ID_2,
                            nb_hidden_variables, LC, nb_gaussians,
                            kept_next_hidden_var_coefs, kept_biases, nb_dims):
    """
    Trivial phase-2 step: only one Gaussian depends on coef_index in the
    next-variable axis — move it directly to the kept buffer.
    """
    next_hidden_var_coefs_cp = tf.unstack(next_hidden_var_coefs)
    biases_cp = tf.unstack(biases)

    new_next_hidden_var_coefs_cp = next_hidden_var_coefs_cp.pop(ID_2)
    new_biases_cp = biases_cp.pop(ID_2)

    kept_next_hidden_var_coefs_cp = tf.unstack(kept_next_hidden_var_coefs)
    kept_biases_cp = tf.unstack(kept_biases)
    kept_next_hidden_var_coefs_cp.append(new_next_hidden_var_coefs_cp)
    kept_biases_cp.append(new_biases_cp)
    nb_gaussians -= 1

    biases_cp = tf.cast(
        tf.reshape(tf.stack(biases_cp), [len(biases_cp)] + biases.shape[1:]),
        dtype=dtype)

    return (tf.stack(next_hidden_var_coefs_cp),
            current_hidden_var_coefs,
            biases_cp, LC, nb_gaussians,
            tf.stack(kept_next_hidden_var_coefs_cp),
            tf.stack(kept_biases_cp))
