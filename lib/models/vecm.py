
from typing import Any, Tuple, cast
import numpy
from numpy.typing import NDArray
from statsmodels.tsa.vector_ar.var_model import LagOrderResults
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order, JohansenTestResult, VECMResults

from lib import stats

def fit(endog: NDArray[numpy.floating[Any]], maxlags: int, rank: int, trend: str="co") -> VECMResults:
    """
    Estimate the parameters for and assumed VECM(n) model.

    Parameters
    ----------
    endog: DataFrame
        VAR(n) process endogenous variable samples.
    maxlags: int
        Maximum number of time lags tried. (default is 12)
     rank: int
        Cointegration rank.
    trend: str
        "n" - no deterministic terms
        "co" - constant outside the cointegration relation
        "ci" - constant within the cointegration relation
        "lo" - linear trend outside the cointegration relation
        "li" - linear trend within the cointegration relation

    Returns
    -------
    VECMResults
        Analysis results.
    """

    return __vecm_model(endog, maxlags=maxlags, rank=rank, trend=trend).fit()

def order_estimate(samples: NDArray[numpy.floating[Any]], maxlags: int=12, deterministic='n') -> LagOrderResults:
    """
    Estimate order of VECM samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples analyzed.    
    maxlags: int
        Maximum number of lags.
    deterministic: str
        Assumed trend (default 'n'). 
        Values 'n' -no deterministic terms, 'co' -constant outside the cointegration relation, 'ci' -constant within the cointegration relation, 
        'lo' -linear trend outside the cointegration relation, 'li' -linear trend within the cointegration relation.

    Returns
    -------
    LagOrderResults
        Order results.
    """

    return select_order(samples, maxlags=maxlags, deterministic=deterministic)


def vecm1(λ: numpy.matrix, β: numpy.matrix, a: NDArray[numpy.floating[Any]], 
          Ω: NDArray[numpy.floating[Any]], nsamp: int) -> NDArray[numpy.floating[Any]]:
    """
    Simulate a first order Vector Error Correction Model (VECM) process with the specified parameters.

    Parameters
    ----------
    λ: NDArray[numpy.floating[Any]]
        Damping matrix.
    β: NDArray[numpy.floating[Any]]
        Transpose of cointegration vector.
    a: NDArray[numpy.floating[Any]]
        Coefficient matrix.
    Ω: NDArray[numpy.floating[Any]]
        Noise covariance matrix.
    nsamp: int
        Number of samples generated.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsamp)))
    εt = numpy.matrix(stats.multivariate_normal_samples(numpy.zeros(n), Ω, nsamp))
    for i in range(2, nsamp):
        Δxt1 = xt[:,i-1] - xt[:,i-2]
        Δxt = λ*β*xt[:,i-1] + a*Δxt1 + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return xt


def vecm(λ: numpy.matrix, β: numpy.matrix, a: NDArray[numpy.floating[Any]], Ω: NDArray[numpy.floating[Any]], nsamp: int) -> NDArray[numpy.floating[Any]]:
    """
    Simulate a first order Vector Error Correction Model (VECM) process with the specified parameters.

    Parameters
    ----------
    λ: NDArray[numpy.floating[Any]]
        Damping matrix.
    β: NDArray[numpy.floating[Any]]
        Transpose of cointegration vector.
    a: NDArray[numpy.floating[Any]]
        Coefficient matrices.
    Ω: NDArray[numpy.floating[Any]]
        Noise covariance matrix.
    nsamp: int
        Number of samples generated.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    m, n, _ = a.shape
    xt = numpy.matrix(numpy.zeros((n, nsamp)))
    εt = numpy.matrix(stats.multivariate_normal_samples(numpy.zeros(n), Ω, nsamp))
    a_matrix = [numpy.matrix(a[i]) for i in range(m)]
    for i in range(m + 1, nsamp):
        lag_terms = 0.0
        for j in range(m):
            lag_terms += a_matrix[j]*(xt[:,i-j-1] - xt[:,i-j-2])
        Δxt = λ*β*xt[:,i-1] + lag_terms + εt[i].T
        xt[:,i] = Δxt + xt[:,i-1]
    return xt


def johansen_test_coint(samples, max_lags, trend: int=0) -> JohansenTestResult:
    """
    Perform Johansen's cointegration test.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples analyzed.
    max_lags: int
        maximum number of lags.
    trend: int
        Trend to include in cointegration test.
            -1 - no trend
            0 - constant
            1 - linear trend,.

    Returns
    -------
    JohansenTestResult
        Johansen cointegration test result.
    """

    return coint_johansen(samples, trend, max_lags)


def predict(vecm: VECMResults, steps: int, alpha: float=0.05) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Predict values for the specified number of steps.

    Parameters
    ----------
    vecm: VECMResults
        VECM model.
    steps: int
        Number of steps to predict.
    alpha: float
        Confidence interval (default 0.5).

    Returns
    -------
    Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Predicted values.
    """

    # alpha is always supplied, so statsmodels returns the (forecast, lower, upper) tuple
    return cast(Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]],
                vecm.predict(steps, alpha=alpha))

def __vecm_model(endog: NDArray[numpy.floating[Any]], maxlags: int, rank: int, trend: str = "n") -> VECM:
    """
    Estimate the parameters for and assumed VAR(n) model.

    Parameters
    ----------
    endog: DataFrame
        VAR(n) process endogenous variable samples.
    lags: int
        Maximum number of time lags tried.
    rank: int
        Cointegration rank.
    trend: str
        "n" - no deterministic terms
        "co" - constant outside the cointegration relation
        "ci" - constant within the cointegration relation
        "lo" - linear trend outside the cointegration relation
        "li" - linear trend within the cointegration relation

    Returns
    -------
    VECM
        VECM model.
    """

    return VECM(endog, k_ar_diff=maxlags, coint_rank=rank, deterministic=trend)


