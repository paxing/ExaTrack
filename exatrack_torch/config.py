# -*- coding: utf-8 -*-
"""
Global constants shared across all ExaTrack modules.
Import from here rather than redefining in each file.
"""

import numpy as np
import torch

dtype = torch.float64
pi = np.pi
minval = np.array(1e-14)
jit_compile = False  # no-op in PyTorch, kept for structural parity
