"""Analysis facades.

The facade submodules are resolved lazily (PEP 562) so that importing a pure
data container -- ``lib.data.param_est`` holds ParamEst and friends, which the
plot layer and the database pipeline both need -- does not drag in statsmodels
by way of this package's __init__. ``from lib.data import vecm`` still works.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names for the type checker only; no runtime import
    from lib.data.impl import adf, arima, stats, bm, fbm, ou, var, ecm, vecm
    from lib.data.param_est import OLSResult
    from lib.data.hyp_test import HypothesisTestType

__all__ = ["adf", "arima", "stats", "bm", "fbm", "ou", "var", "ecm", "vecm",
           "OLSResult", "HypothesisTestType"]

_FACADES = {"adf", "arima", "stats", "bm", "fbm", "ou", "var", "ecm", "vecm"}
_NAMES = {"OLSResult": "lib.data.param_est", "HypothesisTestType": "lib.data.hyp_test"}


def __getattr__(name):
    if name in _FACADES:
        module = importlib.import_module(f"lib.data.impl.{name}")
        globals()[name] = module
        return module
    if name in _NAMES:
        value = getattr(importlib.import_module(_NAMES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
