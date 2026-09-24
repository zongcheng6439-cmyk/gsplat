"""Standalone access to gsplat's chunked kNN WITHOUT importing the gsplat package.

``from gsplat.strategy.gtlr import knn_indices`` executes ``gsplat/__init__``,
which triggers the CUDA JIT build ("Setting up CUDA with MAX_JOBS=...") when
no compiled ``gsplat/csrc`` is present. The offline preprocessing only needs
the pure-torch kNN, so the leaf module ``gsplat/strategy/_knn.py`` is loaded
by file path. Falls back to the package import (e.g. after
``python setup.py install`` the compiled extension is used and no JIT
happens anyway).
"""

from __future__ import annotations

import importlib.util
import os

_impl = None


def _load():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "gsplat",
        "strategy",
        "_knn.py",
    )
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("gsplat_strategy_knn", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.knn_indices
    from gsplat.strategy.gtlr import knn_indices

    return knn_indices


def knn_indices(*args, **kwargs):
    global _impl
    if _impl is None:
        _impl = _load()
    return _impl(*args, **kwargs)
