"""Hahino MP — the core multi-path neural network.

Public API:
    MPBrain      multi-path brain with the self-structuring growth ladder
    PoolAdam     flat-pool Adam with moment carry across pool rebuilds
    path_forward / path_params / describe_path   path (descriptor) utils
"""
from .core import (MPBrain, PoolAdam, path_forward, path_params,
                   describe_path)

__version__ = '1.0.0'
__all__ = ['MPBrain', 'PoolAdam', 'path_forward', 'path_params',
           'describe_path']
