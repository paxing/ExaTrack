# -*- coding: utf-8 -*-
"""
integration.py
--------------
CGP recurrence formulas and integration schedule precomputation.

This module contains:
  - simple_RNN_gaussian_product(): numpy-only version used by get_sequences
  - get_sequences(): one-time precomputation of the optimal integration order
  - RNN_reccurence_formula(): executes the integration schedule at each time step
  - transition_RNN_reccurence_formula(): variant used at state transitions

PyTorch conversion notes
------------------------
- @tf.function decorators removed
- tf.identity(x) → x  (no-op; PyTorch tensors are eagerly evaluated)
- tf.constant(0, shape=s, dtype=d) → torch.zeros(s, dtype=d)
- tf.reduce_sum → torch.sum / .sum()
- tf.stack(list) → torch.stack(list, dim=0)
- get_sequences uses numpy internally; the constraint_function call returns torch
  tensors which are immediately converted to numpy for the structural precomputation
"""

import numpy as np
import torch
from .config import dtype
from .gaussian_ops import (
    norm_log_gaussian,
    intermediate_RNN_function,
    final_RNN_function_phase_1,
    final_RNN_function_phase_2,
    no_RNN_function_phase_1,
    no_RNN_function_phase_2,
    simple_RNN_gaussian_product,
)


def get_sequences(params, initial_params, constraint_function,
                  nb_gaussians, nb_hidden_vars, dtype):
    """
    Function that gets the sequences of integration of the hidden variables.

    The integration process for one time step is composed of 2 phases:
      phase 1: integration over the current hidden variables
      phase 2: rearrangement of the remaining next hidden variables

    Returns 6 lists:
      [initial_functions_phase_1,   initial_sequence_phase_1]
      [initial_functions_phase_2,   initial_sequence_phase_2]
      [recurrent_functions_phase_1, recurrent_sequence_phase_1]
      [recurrent_functions_phase_2, recurrent_sequence_phase_2]
      [final_functions_phase_1,     final_sequence_phase_1]
      [transition_functions,        transition_sequence]
    """
    from .constraints import constraint_function as cf
    nb_dims = 1
    LocErrs = torch.ones((1, 1), dtype=dtype)

    def LocErr_function(LocErrs, LocErr_param):
        return LocErrs

    dts = torch.ones((1, 2), dtype=dtype)
    (hidden_var_coefs, _, _, _,
     initial_hidden_var_coefs, _, _, _,
     transition_hidden_var_coefs, _, _,
     integration_variable_index, _, _, _) = constraint_function(
        params, initial_params, LocErrs, dts, nb_dims, 1., LocErr_function, dtype)

    # Convert to numpy for the structural precomputation
    hidden_var_coefs = hidden_var_coefs[0].detach().cpu().numpy()
    transition_hidden_var_coefs = transition_hidden_var_coefs[0].detach().cpu().numpy()
    integration_variable_index = int(integration_variable_index)
    initial_hidden_var_coefs_np = initial_hidden_var_coefs.detach().cpu().numpy()

    recurrent_current_hidden_var_coefs = np.copy(hidden_var_coefs[:, 0, 0, :nb_hidden_vars])
    recurrent_next_hidden_var_coefs = np.copy(hidden_var_coefs[:, 0, 0, nb_hidden_vars:])

    current_hidden_var_coefs = hidden_var_coefs[:, 0, 0, :nb_hidden_vars]
    next_hidden_var_coefs = hidden_var_coefs[:, 0, 0, nb_hidden_vars:]

    current_initial_hidden_var_coefs = initial_hidden_var_coefs_np[:, 0, 0, :nb_hidden_vars]
    next_initial_hidden_var_coefs = np.zeros((nb_hidden_vars, nb_hidden_vars))

    current_hidden_var_coefs = np.concatenate(
        (current_initial_hidden_var_coefs, current_hidden_var_coefs), axis=0)
    next_hidden_var_coefs = np.concatenate(
        (next_initial_hidden_var_coefs, next_hidden_var_coefs), axis=0)

    current_nb_gaussians = len(current_hidden_var_coefs)

    # ------------------------------------------------------------------
    # Initial step phase 1
    # ------------------------------------------------------------------
    initial_sequence_phase_1 = []
    initial_functions_phase_1 = []

    for coef_index in np.arange(nb_hidden_vars - 1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                current_hidden_var_coefs[ID_1], current_hidden_var_coefs[ID_2],
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2])

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            initial_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            initial_sequence_phase_1.append([coef_index, ID_1, ID_2])
            initial_functions_phase_1.append(no_RNN_function_phase_1)

        if len(non_zero_gaussian_IDs) >= 1:
            current_hidden_var_coefs = np.delete(
                current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1

    # ------------------------------------------------------------------
    # Initial step phase 2
    # ------------------------------------------------------------------
    initial_sequence_phase_2 = []
    initial_functions_phase_2 = []

    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))

    for coef_index in np.arange(nb_hidden_vars - 1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(intermediate_RNN_function)

            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                next_hidden_var_coefs[ID_1] * 0, next_hidden_var_coefs[ID_2] * 0,
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2])

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            initial_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            initial_sequence_phase_2.append([coef_index, ID_1, ID_2])
            initial_functions_phase_2.append(no_RNN_function_phase_2)

        if len(non_zero_gaussian_IDs) >= 1:
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1

    initial_saved_Gaussians = saved_Gaussians

    # ------------------------------------------------------------------
    # Recurrent step phase 1
    # ------------------------------------------------------------------
    current_hidden_var_coefs = np.concatenate(
        (saved_Gaussians, recurrent_current_hidden_var_coefs), 0)
    next_hidden_var_coefs = np.concatenate(
        (saved_Gaussians * 0, recurrent_next_hidden_var_coefs), 0)

    current_nb_gaussians = len(current_hidden_var_coefs)

    recurrent_sequence_phase_1 = []
    recurrent_functions_phase_1 = []

    for coef_index in np.arange(nb_hidden_vars - 1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                current_hidden_var_coefs[ID_1], current_hidden_var_coefs[ID_2],
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2])

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            recurrent_sequence_phase_1.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_1.append(no_RNN_function_phase_1)

        if len(non_zero_gaussian_IDs) >= 1:
            current_hidden_var_coefs = np.delete(
                current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1

    # ------------------------------------------------------------------
    # Recurrent step phase 2
    # ------------------------------------------------------------------
    recurrent_sequence_phase_2 = []
    recurrent_functions_phase_2 = []

    saved_Gaussians = np.zeros((nb_hidden_vars, nb_hidden_vars))

    for coef_index in np.arange(nb_hidden_vars - 1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = next_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(intermediate_RNN_function)

            C1 = next_hidden_var_coefs[ID_1, coef_index]
            C2 = next_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2],
                next_hidden_var_coefs[ID_1] * 0, next_hidden_var_coefs[ID_2] * 0)

            next_hidden_var_coefs[ID_1] = current_coefs3
            next_hidden_var_coefs[ID_2] = current_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            recurrent_functions_phase_2[-1] = final_RNN_function_phase_2
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            recurrent_sequence_phase_2.append([coef_index, ID_1, ID_2])
            recurrent_functions_phase_2.append(no_RNN_function_phase_2)

        if len(non_zero_gaussian_IDs) >= 1:
            saved_Gaussians[coef_index] = next_hidden_var_coefs[ID_2]
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, ID_2, 0)
            current_nb_gaussians += -1

    print('Checking that the recurrent next Gaussians have the same form than '
          'the initial next gaussians:',
          np.all((initial_saved_Gaussians == 0) == (saved_Gaussians == 0)))

    # ------------------------------------------------------------------
    # Transition step
    # ------------------------------------------------------------------
    current_hidden_var_coefs = saved_Gaussians
    next_hidden_var_coefs = saved_Gaussians * 0
    current_nb_gaussians = len(current_hidden_var_coefs)

    transition_sequence = []
    transition_functions = []

    transition_integration_variables = np.arange(
        integration_variable_index, nb_hidden_vars)[::-1]

    for coef_index in transition_integration_variables:
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            transition_sequence.append([coef_index, ID_1, ID_2])
            transition_functions.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                current_hidden_var_coefs[ID_1], current_hidden_var_coefs[ID_2],
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2])

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            transition_functions[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            transition_sequence.append([coef_index, ID_1, ID_2])
            transition_functions.append(no_RNN_function_phase_1)

        if len(non_zero_gaussian_IDs) >= 1:
            current_hidden_var_coefs = np.delete(
                current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1

    current_hidden_var_coefs = np.concatenate(
        (current_hidden_var_coefs, transition_hidden_var_coefs[:, 0, 0]), 0)
    next_hidden_var_coefs = np.concatenate(
        (next_hidden_var_coefs, transition_hidden_var_coefs[:, 0, 0] * 0), 0)
    current_nb_gaussians = current_hidden_var_coefs.shape[0]

    saved_Gaussians = current_hidden_var_coefs

    # ------------------------------------------------------------------
    # Final step phase 1
    # ------------------------------------------------------------------
    current_hidden_var_coefs = saved_Gaussians
    current_nb_gaussians = len(current_hidden_var_coefs)
    next_hidden_var_coefs = np.zeros(current_hidden_var_coefs.shape)

    final_sequence_phase_1 = []
    final_functions_phase_1 = []

    for coef_index in np.arange(nb_hidden_vars - 1, -1, -1):
        non_zero_gaussian_IDs = []
        for Gaussian_ID in range(current_nb_gaussians):
            Coef = current_hidden_var_coefs[Gaussian_ID, coef_index]
            if Coef != 0:
                non_zero_gaussian_IDs.append(Gaussian_ID)

        for i in range(len(non_zero_gaussian_IDs) - 1):
            ID_1 = non_zero_gaussian_IDs[i]
            ID_2 = non_zero_gaussian_IDs[i + 1]

            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(intermediate_RNN_function)

            C1 = current_hidden_var_coefs[ID_1, coef_index]
            C2 = current_hidden_var_coefs[ID_2, coef_index]
            (current_coefs3, current_coefs4,
             next_coefs3, next_coefs4) = simple_RNN_gaussian_product(
                C1, C2,
                current_hidden_var_coefs[ID_1], current_hidden_var_coefs[ID_2],
                next_hidden_var_coefs[ID_1], next_hidden_var_coefs[ID_2])

            current_hidden_var_coefs[ID_1] = current_coefs3
            current_hidden_var_coefs[ID_2] = current_coefs4
            next_hidden_var_coefs[ID_1] = next_coefs3
            next_hidden_var_coefs[ID_2] = next_coefs4

        if len(non_zero_gaussian_IDs) > 1:
            recurrent_functions_phase_1[-1] = final_RNN_function_phase_1
        elif len(non_zero_gaussian_IDs) == 1:
            ID_1 = 0
            ID_2 = non_zero_gaussian_IDs[0]
            final_sequence_phase_1.append([coef_index, ID_1, ID_2])
            final_functions_phase_1.append(no_RNN_function_phase_1)

        if len(non_zero_gaussian_IDs) >= 1:
            current_hidden_var_coefs = np.delete(
                current_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            next_hidden_var_coefs = np.delete(
                next_hidden_var_coefs, non_zero_gaussian_IDs[-1], 0)
            current_nb_gaussians += -1

    return (
        [initial_functions_phase_1,   initial_sequence_phase_1],
        [initial_functions_phase_2,   initial_sequence_phase_2],
        [recurrent_functions_phase_1, recurrent_sequence_phase_1],
        [recurrent_functions_phase_2, recurrent_sequence_phase_2],
        [final_functions_phase_1,     final_sequence_phase_1],
        [transition_functions,        transition_sequence],
    )


# ---------------------------------------------------------------------------
# Core recurrence formulas
# ---------------------------------------------------------------------------

def RNN_reccurence_formula(current_hidden_var_coefs,
                           next_hidden_var_coefs,
                           biases,
                           sequence_phase_1,
                           sequence_phase_2,
                           nb_dims,
                           dtype=dtype):
    """
    RNN_reccurence_formula organizes and executes the integration steps.

    Phase 1: integrate over the current hidden variables.
    Phase 2: rearrange the remaining next hidden variables (posterior).
    """
    current_hidden_var_coefs_cp = current_hidden_var_coefs
    next_hidden_var_coefs_cp = next_hidden_var_coefs
    biases_cp = biases

    kept_next_hidden_var_coefs_cp = []
    kept_biases_cp = []

    nb_gaussians = len(biases_cp)
    nb_hidden_variables = current_hidden_var_coefs_cp[0].shape[-1]
    LC = torch.zeros(current_hidden_var_coefs_cp[0].shape[:2], dtype=dtype,
                     device=current_hidden_var_coefs_cp[0].device)

    for f, s in zip(sequence_phase_1[0], sequence_phase_1[1]):
        coef_index, ID_1, ID_2 = s
        (current_hidden_var_coefs_cp,
         next_hidden_var_coefs_cp,
         biases_cp, LC, nb_gaussians,
         kept_next_hidden_var_coefs_cp,
         kept_biases_cp) = f(
            current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp,
            coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
            kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    for f, s in zip(sequence_phase_2[0][:], sequence_phase_2[1][:]):
        coef_index, ID_1, ID_2 = s
        (next_hidden_var_coefs_cp,
         current_hidden_var_coefs_cp,
         biases_cp, LC, nb_gaussians,
         kept_next_hidden_var_coefs_cp,
         kept_biases_cp) = f(
            next_hidden_var_coefs_cp, current_hidden_var_coefs_cp, biases_cp,
            coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
            kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    new_LCs = torch.sum(norm_log_gaussian(biases_cp), dim=3)
    LC = LC + torch.sum(new_LCs, dim=0)

    # phase_2 functions return torch.stack(...); unbind back to list for [::-1]
    if not isinstance(kept_next_hidden_var_coefs_cp, list):
        kept_next_hidden_var_coefs_cp = list(kept_next_hidden_var_coefs_cp.unbind(0))
        kept_biases_cp = list(kept_biases_cp.unbind(0))

    if kept_next_hidden_var_coefs_cp:
        Next_coefs = torch.stack(kept_next_hidden_var_coefs_cp[::-1])
        Next_biases = torch.stack(kept_biases_cp[::-1])
    else:
        # Final_layer: phase_2 is empty; caller only uses LC
        Next_coefs = current_hidden_var_coefs_cp
        Next_biases = biases_cp

    return Next_coefs, Next_biases, LC


def transition_RNN_reccurence_formula(current_hidden_var_coefs,
                                      next_hidden_var_coefs,
                                      biases,
                                      transition_sequence,
                                      nb_dims,
                                      dtype=dtype):
    """
    Adaptation of RNN_reccurence_formula for transitions between states i to j
    with i != j. Integrates over the previous hidden variable that disappears
    during transitions.
    """
    current_hidden_var_coefs_cp = current_hidden_var_coefs
    next_hidden_var_coefs_cp = next_hidden_var_coefs
    biases_cp = biases

    kept_next_hidden_var_coefs_cp = []
    kept_biases_cp = []

    nb_gaussians = len(biases_cp)
    nb_hidden_variables = current_hidden_var_coefs_cp[0].shape[-1]
    LC = torch.zeros(current_hidden_var_coefs_cp[0].shape[:2], dtype=dtype,
                     device=current_hidden_var_coefs_cp[0].device)

    for f, s in zip(transition_sequence[0], transition_sequence[1]):
        coef_index, ID_1, ID_2 = s
        (current_hidden_var_coefs_cp,
         next_hidden_var_coefs_cp,
         biases_cp, LC, nb_gaussians,
         kept_next_hidden_var_coefs_cp,
         kept_biases_cp) = f(
            current_hidden_var_coefs_cp, next_hidden_var_coefs_cp, biases_cp,
            coef_index, ID_1, ID_2, nb_hidden_variables, LC, nb_gaussians,
            kept_next_hidden_var_coefs_cp, kept_biases_cp, nb_dims)

    Next_coefs = current_hidden_var_coefs_cp
    Next_biases = biases_cp

    return Next_coefs, Next_biases, LC
