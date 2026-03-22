"""
data.models.arima.py

Simulations and analysis of ARIMA(p,d,q) random process.
"""

from typing import Any
from astroid.nodes import Unknown
import numpy
from numpy.typing import NDArray

import statsmodels.api as sm
import statsmodels.tsa as tsa
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

def maq_sigma(θ: list[float], σ: float=1) -> float:
    """
    Compute MA(q) standard deviation.

    Parameters
    ----------
    θ: list[float]
        List of MA(q) coefficients.
    σ: float
        Standard deviation of noise term.
    
    Returns
    -------
    float
        Standard deviation.
    """

    v = 0
    for u in θ:
        v += u**2
    return σ * numpy.sqrt(v + 1)

def maq_cov(θ: list[float], σ: float=1) -> NDArray[numpy.floating[Any]]:
    """
    Compute MA(q) auto covariance.

    Parameters
    ----------
    θ: list[float]
        List of MA(q) coefficients.
    σ: float
        Standard deviation of noise term.
    
    Returns
    -------
    NDArray[numpy.floating[Any]]
        Autocovariance.
    """

    q = len(θ)
    c = numpy.zeros(q)
    s = numpy.zeros(q)
    for n in range(1,q):
        for i in range(q-n):
            c[n] += θ[i]*θ[i+n]
    for n in range(q):
        s[n] = θ[n]
    return σ**2 * (c + s)

def maq_acf(θ: list[float], σ: float=1, max_lag: int=15) -> NDArray[numpy.floating[Any]]:
    """
    Compute MA(q) auto correlation function.

    Parameters
    ----------
    θ: list[float]
        List of MA(q) coefficients.
    σ: float
        Standard deviation of noise term.
    max_lag: float
        Maximum lag setting time lag limit used in calculation.
    
    Returns
    -------
    NDArray[numpy.floating[Any]]
        Autocorrelation function.
    """

    ac = maq_cov(θ, σ) / maq_sigma(θ, σ)**2
    ac_eq = numpy.zeros(max_lag + 1)
    ac_eq[0] = 1
    for i in range(len(ac)):
        ac_eq[i + 1] = ac[i]
    return ac_eq

def ar1_sigma(φ: float, σ: float=1) -> float:
    """
    Compute AR(1) standard deviation.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    σ: float
        Standard deviation of noise term.
        
    Returns
    -------
    float
        Standard deviation.
    """

    return numpy.sqrt(σ**2/(1.0-φ**2))

def ar1_acf(φ: float, nvals: int) -> list[float]:
    """
    Compute AR(1) standard deviation.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    nvals: int
        Number of terms computed.
    
    Returns
    -------
    list[float]
        Autocorrelation function.
    """

    return [φ**n for n in range(nvals)]

def ar1_offset_mean(φ: float, μ: float) -> float:
    """
    Compute AR(1) with offset mean.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    μ: float
        Constant offset.

    Returns
    -------
    float
        Mean value.
    """

    return μ / (1.0 - φ)

def ar1_offset_sigma(φ: float, σ: float) -> float:
    """
    Compute AR(1) with offset standard deviation.

    Parameters
    ----------
    φ: float
        AR(1) coefficient.
    μ: float
        Constant offset.

    Returns
    -------
    float
        Standard deviation.
    """

    return σ / numpy.sqrt(1.0 - φ**2)

def noise(n: int) -> NDArray[numpy.floating[Any]]:
    """
    Generate gaussian noise with unit variance.

    Parameters
    ----------
    n: int
        Number of values generated.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    return numpy.random.normal(0.0, 1.0, n)

def ar(φ: list[float], x0: list[float], n: int, σ: float) -> NDArray[numpy.floating[Any]]:
    """
    Generate an AR(p) process using specified parameters.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    x0: list[float]
        List of initial values.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
   """
    
    p = len(φ)
    samples = numpy.zeros(n)
    for i in range(0, p):
        samples[i] = x0[i]
    ε = σ*noise(n)
    for i in range(p, n):
        samples[i] = ε[i]
        for j in range(0, p):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ou(λ: float, μ: float, n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate the Ornstein-Uhlenbeck process using an AR(1) with offset simulation
    using the specified parameters.
    
    Parameters
    ----------
    λ: float
        Ornstein-Uhlenbeck parameter (0 < λ < 2).
    μ: float
        Mean value.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    φ = 1.0 - λ
    m = μ*λ
    return arp_offset([φ], m, n, σ)

def arp_offset(φ: list[float], μ: float, n: int, σ: float) -> NDArray[numpy.floating[Any]]:
    """
    Generate AR(p) with a constant offset using the specified parameters.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    u: float
        Offset.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    return arp_drift(φ, μ, 0.0, n, σ)

def arp_drift(φ: list[float], μ: float, γ: float, n: int, σ: float) -> NDArray[numpy.floating[Any]]:
    """
    Generate AR(p) with drift using the specified parameters.

    Parameters
    ----------
    φ: list[float]
        AR(p) parameters.
    u: float
        Offset.
    γ: float
        Drift parameter.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    p = len(φ)
    samples = numpy.zeros(n)
    ε = σ*noise(n)
    for i in range(p, n):
        samples[i] = ε[i] + γ*i + μ
        for j in range(0, p):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ar1(φ: float, n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate AR(1) using specified parameters.

    Parameters
    ----------
    φ: float
        AR(2) parameter.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    return arp(numpy.array([φ]), n, σ)

def arp(φ: NDArray[numpy.floating[Any]], n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate AR(p) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        AR(p) parameters.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    φ_sim = numpy.r_[1, -φ]
    δ_sim = numpy.array([1.0])
    # 
    return sm.tsa.arma_generate_sample(φ_sim, δ_sim, n, σ) # type: ignore[arg-type]


def maq(θ: NDArray[numpy.floating[Any]], n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate MA(q) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    θ: NDArray[numpy.floating[Any]]
        MA(q) parameters.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    φ_sim = numpy.array([1.0])
    θ_sim = numpy.r_[1, θ]
    return sm.tsa.arma_generate_sample(φ_sim, θ_sim, n, σ) # type: ignore[arg-type]


def arma(φ: NDArray[numpy.floating[Any]], θ: NDArray[numpy.floating[Any]], n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate ARMA(p, q) using specified parameters and the statsmodels.tas simulator.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        AR(p) parameters.
    θ: NDArray[numpy.floating[Any]]
        MA(q) parameters.
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    φ_sim = numpy.r_[1, -φ]
    θ_sim = numpy.r_[1, θ]
    return sm.tsa.arma_generate_sample(φ_sim, θ_sim, n, σ) # type: ignore[arg-type]

def arima(φ: NDArray[numpy.floating[Any]], δ: NDArray[numpy.floating[Any]], d: int, n: int, σ: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Generate ARIMA(p,d,q) using specified parameters and the statsmodels.tas simulator arma
    and integrate the result d times to obtain the ARIMA process.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        AR(p) parameters.
    δ: NDArray[numpy.floating[Any]]
        MA(q) parameters.
    d: int
        Number of integrations to perform (d = 1 or 2).
    n: int
        Number of steps in simulation.
    σ: float
        Standard deviation of noise term.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.

    Raises
    ______
    Exception
        d < 1 or d > 2
    """

    assert d <= 2, "d must equal 1 or 2"
    assert d >= 1, "d must equal 1 or 2"
    samples = arma(φ, δ, n, σ)
    if d == 1:
        return numpy.cumsum(samples)
    else:
        for i in range(2, n):
            samples[i] = samples[i] + 2.0*samples[i-1] - samples[i-2]
        return samples


def arima_from_arma(samples: NDArray[numpy.floating[Any]], d: int) -> NDArray[numpy.floating[Any]]:
    """
    Generate ARIMA(p,d,q) using the samples from a ARMA(p,q) process
    by integrating d times,.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        ARMA(p,q) processes samples
    d: int
        Number of integrations to perform (d = 1 or 2).

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.

    Raises
    ______
    Exception
        d < 1 or d > 2
    """

    assert d <= 2, "d must equal 1 or 2"
    assert d >= 1, "d must equal 1 or 2"
    n = len(samples)
    if d == 1:
        return numpy.cumsum(samples)
    else:
        result = numpy.zeros(n)
        result[0] = samples[0]
        result[1] = samples[1]
        for i in range(2, n):
            result[i] = samples[i] + 2.0*result[i-1] - result[i-2]
        return result


def yw(samples: NDArray[numpy.floating[Any]], order: int) -> NDArray[numpy.floating[Any]]:
    """
    Compute the coefficients of an AR(p) processes by solving the Yule-Walker equations.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        The assumed order of the AR9p) process.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Estimate of AR(p) coefficients.
    """

    pacf = sm.regression.yule_walker(samples, order=order, method='mle')
    return pacf[0]

pacf_return_type = tuple[Unknown | NDArray[numpy.floating[Any]] | NDArray[Any], NDArray[Any]] | Unknown | NDArray[numpy.floating[Any]] | NDArray[Any]
def pacf(samples: NDArray[numpy.floating[Any]], nlags: int) -> pacf_return_type:
    """
    Compute the partial auto-correlation function using statsmodels.tsa.stattools.pacf.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    nlags: int
        The number of lags desired for the PACF.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        The of AR(p) partial autocorrelation function.
    """

    return sm.tsa.stattools.pacf(samples, nlags=nlags)


def ___ar_model(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMA:
    """
    Create an AR(p) of the specified order with the specified samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMA
        The of AR(p) model.
    """

    return ARIMA(samples, order=(order, 0, 0))


def ar_fit(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMAResults:
    """
    Estimate the parameters for the assumed AR(p) model from the samples
    assuming the specified order.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMAResults
        Contains the AR(p) estimation results.
    """

    return ___ar_model(samples, order).fit()


def __ar_offset_model(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMA:
    """
    Create and AR(p) with an offset of the specified order with the specified samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMA
        The of AR(p) with offset model.
    """

    return ARIMA(samples, order=(order, 0, 0), trend='c')


def ar_offset_fit(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMAResults:
    """
    Estimate the parameters for the assumed AR(p) with offset model from the samples
    assuming the specified order.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMAResults
        Contains the AR(p) estimation results.
    """

    return __ar_offset_model(samples, order).fit()


def __ma_model(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMA:
    """
    Create a MA(p) model of the specified order with the specified samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        MA(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMA
        The of MA(p) model.
    """

    return ARIMA(samples, order=(0, 0, order))


def ma_fit(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMAResults:
    """
    Estimate the parameters for the assumed MA(q) model from the samples
    assuming the specified order.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMAResults
        Contains the AR(p) estimation results.
    """

    return __ma_model(samples, order).fit()


def __ma_offset_model(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMA:
    """
    Create a MA(p) model with offset of the specified order with the specified samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMA
        The of MA(q) with offset model.
    """

    return ARIMA(samples, order=(0, 0, order), trend='c')

def ma_offset_fit(samples: NDArray[numpy.floating[Any]], order: int) -> ARIMAResults:
    """
    Estimate the parameters for the assumed MA(q) model from the samples
    assuming the specified order.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        AR(p) processes samples
    order: int
        Model order.

    Returns
    -------
    tsa.arima.model.ARIMAResults
        Contains the AR(p) estimation results.
    """

    return __ma_offset_model(samples, order).fit()

