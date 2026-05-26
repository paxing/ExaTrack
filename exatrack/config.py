# -*- coding: utf-8 -*-
"""
Global constants shared across all ExaTrack modules.
Import from here rather than redefining in each file.
"""

import numpy as np
import tensorflow as tf

dtype = 'float64'
pi = tf.constant(np.pi, dtype=dtype)
minval = np.array(1e-14)
jit_compile = False
