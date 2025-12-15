"""HTTP clients for external data providers."""

from .fred import FredClient, FredAPIError

__all__ = ["FredClient", "FredAPIError"]
