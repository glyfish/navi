"""navi shared library.

Subpackages are resolved lazily (PEP 562). The plot layer imports the analysis
facades, which import statsmodels, so an eager __init__ meant that holding a
ParamEst -- a pure data container the plot layer and the database pipeline both
need -- cost the whole of statsmodels. ``from lib import config`` and
``import lib.plots`` behave exactly as before.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names for the type checker only; no runtime import
    from lib import clients, config, env, logger, plots, utils, models

__all__ = ["clients", "config", "env", "logger", "plots", "utils", "models"]

# optional dependency: absent or unsupported installs get None, as before
try:
    from . import mcp_client  # noqa: F401
except Exception:  # pragma: no cover - best effort import
    mcp_client = None
else:
    __all__.append("mcp_client")


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"lib.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
