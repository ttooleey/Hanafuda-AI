#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch compatibility layer for loading models saved with older PyTorch versions.
This module adds support for _LinearWithBias which was removed in PyTorch 1.9,
and redirects old module names to the new koikoi package structure.
"""

import sys
import torch
import torch.nn as nn

# Add compatibility for _LinearWithBias (removed in PyTorch 1.9)
# This class was used internally by nn.MultiheadAttention in older PyTorch versions
class _LinearWithBias(nn.Linear):
    """Compatibility class for models saved with PyTorch < 1.9"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

# Monkey-patch the module to add backward compatibility
if not hasattr(nn.modules.linear, '_LinearWithBias'):
    nn.modules.linear._LinearWithBias = _LinearWithBias
    torch.nn.modules.linear._LinearWithBias = _LinearWithBias

# Redirect old module names to new package structure
# This allows loading models that were pickled with the old module names
from koikoi.ai import models as koikoinet2L
sys.modules['koikoinet2L'] = koikoinet2L
