"""Utilities for loading secret API keys shared across projects."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Allow overriding the location of the dotenv file but default to the navi root.
_DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_ENV_PATH = Path(os.getenv("NAVI_ENV_FILE") or _DEFAULT_ENV_PATH)

DEFAULT_FRED_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2"
DEFAULT_MCP_URL = "http://localhost:8080/sse"
DEFAULT_TIINGO_BASE_URL = "https://api.tiingo.com/tiingo"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    # Developers might opt to provide keys via the shell environment instead.
    pass


@lru_cache
def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Populate {_ENV_PATH} or export the variable.")
    return value


def get_fred_api_key() -> str:
    """Return the configured FRED API key."""
    return _get_env_var("FRED_API_KEY")


def get_bls_api_key(required: bool = True) -> str | None:
    """Return the configured Bureau of Labor Statistics API key.

    BLS (unlike FRED/Tiingo) works without a key at reduced limits, so callers
    may pass ``required=False`` to get ``None`` instead of an error when unset.
    """
    if required:
        return _get_env_var("BLS_API_KEY")
    return os.getenv("BLS_API_KEY") or None


def get_fred_base_url() -> str:
    """Return the base URL for FRED requests."""
    return os.getenv("FRED_BASE_URL", DEFAULT_FRED_BASE_URL)


def get_bls_base_url() -> str:
    """Return the base URL for BLS requests."""
    return os.getenv("BLS_BASE_URL", DEFAULT_BLS_BASE_URL)


def get_mcp_url() -> str:
    """Return the base URL for MCP requests."""
    return os.getenv("MCP_URL", DEFAULT_MCP_URL)


def get_tiingo_api_key() -> str:
    """Return the configured Tiingo API key."""
    # NOTE: env var is spelled TINGO_API_KEY (no second "i") to match .env.
    return _get_env_var("TINGO_API_KEY")


def get_tiingo_base_url() -> str:
    """Return the base URL for Tiingo requests."""
    return os.getenv("TINGO_BASE_URL", DEFAULT_TIINGO_BASE_URL)