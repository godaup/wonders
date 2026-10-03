"""Multiprocessing helpers shared by the benchmark, vector envs and data collection.

Forking a process after PyTorch has run its intra-op thread pool deadlocks the child
the moment it uses torch again (inherited OpenMP state). So once torch is loaded in
this process, worker pools default to ``forkserver`` (or ``spawn``) instead of the
platform default. Workers also pin torch to one thread: they run batch-1 inference,
and N workers each spawning N threads would oversubscribe the CPU.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from multiprocessing.context import BaseContext
from typing import Optional


def get_context(method: Optional[str] = None) -> BaseContext:
    """The multiprocessing context to use; see the module docstring for the default."""
    if method is None and 'torch' in sys.modules:
        method = 'forkserver' if 'forkserver' in mp.get_all_start_methods() else 'spawn'
    return mp.get_context(method)


def init_worker() -> None:
    """Pool/process initializer: keep each worker single-threaded for torch / BLAS."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(1)
