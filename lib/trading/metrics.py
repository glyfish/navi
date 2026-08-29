"""
metrics.py

Metrics in used in financial analysis.

"""

import numpy
from pandas import Series
from numpy.typing import NDArray

from typing import Any, Tuple

from lib.utils import verify_condition


def compute_zscore(time: NDArray[Any], data: NDArray[numpy.number[Any]] | Series, window: int) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
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

    verify_condition(time, len(time) == len(data), "len(time) == len(data)")
    verify_condition(window, window > 0, "window > 0")

    npts = len(data) - window + 1
    zscores = [zscore(data[i:i + window]) for i in range(npts)]
    return time[window - 1:], numpy.array(zscores)


def compute_std(time: NDArray[Any], data: NDArray[numpy.number[Any]] | Series, window: int) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
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

    verify_condition(time, len(time) == len(data), "len(time) == len(data)")
    verify_condition(window, window > 0, "window > 0")

    npts = len(data) - window + 1
    stds = [std(numpy.flip(data[i:i + window])) for i in range(npts)]
    return time[window - 1:], numpy.array(stds)


def zscore(samples: NDArray[numpy.number[Any]] | Series) -> float:
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

    # asarray so a pandas Series indexes positionally: pandas>=3 reads
    # samples[-1] as a label lookup and raises KeyError on a RangeIndex.
    samples = numpy.asarray(samples)

    mean = numpy.mean(samples)
    std = numpy.std(samples)
    val = samples[-1]

    # A window holding inf or NaN makes std NaN, and `NaN > 0` is False -- the
    # old `else 0.0` therefore reported "exactly on the rolling mean" for
    # corrupt data. Only a genuinely flat window is 0.0; NaN propagates.
    if std > 0:
        return float((val - mean) / std)
    return 0.0 if std == 0 else float("nan")


def std(samples: NDArray[numpy.number[Any]] | Series) -> float:    
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


