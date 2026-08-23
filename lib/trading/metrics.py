"""
metrics.py

Metrics in used in financial analysis.

"""

import numpy
from numpy.typing import NDArray

from typing import Any, Tuple

from lib.utils import verify_condition


def compute_zscore(time: NDArray[numpy.floating[Any]], data: NDArray[numpy.floating[Any]], window: int) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Calculate the z-score for a time series using a rolling window. The order of the
    time series is assumed oldest data to most recent data.

    Parameters
    ----------
    data : NDArray[numpy.floating[Any]]
        The time series.
    time : NDArray[numpy.floating[Any]]
        The time series time.
    window : int
        The lookback window.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        The z-score series.
    """

    npts = len(data) - window + 1
    zscores = [zscore(data[i:i + window]) for i in range(npts)]
    return time[window - 1:], numpy.array(zscores)


def compute_std(time: NDArray[numpy.floating[Any]], data: NDArray[numpy.floating[Any]], window: int) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Calculate the standard deviation for a time series using a rolling window. The order of the
    time series is assumed oldest data to most recent data.

    Parameters
    ----------
    data : NDArray[numpy.floating[Any]]
        The time series.
    time : NDArray[numpy.floating[Any]]
        The time series time.
    window : int
        The lookback window.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        The standard deviation series.
    """

    npts = len(data) - window + 1
    stds = [std(numpy.flip(data[i:i + window])) for i in range(npts)]
    return time[window - 1:], numpy.array(stds)


def zscore(samples: NDArray[numpy.floating[Any]]) -> float:
    """
    Calculate the z-score using samples to compute the mean and standard deviation
    and use the last value in samples as the test value.

    Parameters
    ----------
    samples : numpy.ndarray
        The time series.

    Returns
    -------
    float
        The z-score.
    """

    verify_condition(samples, len(samples) > 0, "No samples to compute z-score")

    mean = numpy.mean(samples)
    std = numpy.std(samples)
    val = samples[-1]

    return (val - mean) / std if std > 0 else 0.0


def std(samples: NDArray[numpy.floating[Any]]) -> float:    
    """
    Calculate the standard deviation of the samples.

    Parameters
    ----------
    samples : NDArray[numpy.floating[Any]]
        The time series.

    Returns
    -------
    float
        The standard deviation.
    """

    return float(numpy.std(samples))


