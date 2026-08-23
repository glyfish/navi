from typing import Any, Tuple
import numpy
from numpy.typing import NDArray
import statsmodels.api as sm
from pandas import DataFrame

from lib.data.impl import ou, bm, ecm, adf, arima, fbm, stats, var, vecm
from lib.data.param_est import (ParamEst, OLSResult, OLSTransform, OLSParamType)


def compute_mean_reversion_halflife(data: NDArray[numpy.floating[Any]]) -> Tuple[float, sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
    """
    Compute mean reversion half-life.

    Parameters
    ----------
    data : NDArray[numpy.floating[Any]]
        Time series data.
    """

    mean = numpy.full(len(data), numpy.mean(data))
    xt = data - mean
    result, result_model = ou.compute_mean_half_life_estimate(xt)
    assert result_model.param_transforms is not None
    return result_model.param_transforms[0].param.est, result, result_model
