"""
stats.py

Useful statistical functions.

"""

import numpy
from numpy.typing import NDArray
from pandas import DataFrame

import statsmodels.api as sm
from scipy.stats import multivariate_normal
from typing import Any, Sequence
from statsmodels.tsa.stattools import grangercausalitytests

# An ensemble is nsim realizations of npts samples. create_ensemble() builds one
# as a list of 1-D arrays, while the ensemble_* functions below index it in two
# dimensions, so they accept either form and normalize with numpy.asarray.
EnsembleSamples = NDArray[numpy.floating[Any]] | Sequence[NDArray[numpy.floating[Any]]]


def to_noise(samples: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Difference the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Differenced data
    """

    return diff(samples)


def from_noise(dB: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Integrate the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Integrate data.
    """

    B = numpy.zeros(len(dB))
    for i in range(1, len(dB)):
        B[i] = B[i-1] + dB[i]
    return B


def to_geometric(samples: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Take the exponential of the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Exponential of sampled data.
    """

    return numpy.exp(samples)


def from_geometric(samples: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Take the log of the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Logarithm of sampled data.
    """

    return numpy.log(samples)


def ndiff(samples: NDArray[numpy.floating[Any]], ndiff: int) -> NDArray[numpy.floating[Any]]:
    """
    Take the specified number of differences of the samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    ndiff : int
        Number of differences taken.
        
    Returns
    -------
    NDArray[numpy.floating[Any]]
        Samples differenced n times.
    """

    return numpy.diff(samples, ndiff)


def diff(samples: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Difference the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Differenced data
    """

    return numpy.diff(samples)


def ensemble_mean(samples: EnsembleSamples) -> NDArray[numpy.floating[Any]]:
    """
    Compute the time varying mean of the sampled ensemble.

    Parameters
    ----------
    samples: NDArray[tuple[int, int], float] or list[NDArray[float]]
        Ensemble of sampled data, as returned by create_ensemble().

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    samples = numpy.asarray(samples)
    if len(samples) == 0:
        raise Exception(f"no data")
    if len(samples.shape) != 2:
        raise Exception(f"Input must be a two dimensional array.")

    nsim = len(samples)
    npts = len(samples[0])
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j][i] / float(nsim)
    return mean


def ensemble_var(samples: EnsembleSamples, Δt: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Compute the time varying variance of the sampled ensemble.

    Parameters
    ----------
    samples: NDArray[tuple[int, int], float]
        Ensemble of sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble average variance oas a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    samples = numpy.asarray(samples)
    if len(samples) == 0:
        raise Exception(f"no data")
    if len(samples.shape) != 2:
        raise Exception(f"Input must be a two dimensional array.")

    nsim = len(samples)
    mean = ensemble_mean(samples)
    npts = len(samples[0])
    var = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            var[i] += (samples[j][i] - mean[i])**2 / float(nsim)
    
    return var/Δt


def ensemble_sd(samples: EnsembleSamples, Δt: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Compute the time varying standard deviation of the sampled ensemble.

    Parameters
    ----------
    samples: NDArray[tuple[int, int], float]
        Ensemble of sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble average standard deviation oas a function of time.

    Raises
    ______
    Exception
        Ensemble averaged
    """

    return numpy.sqrt(ensemble_var(samples, Δt))


def ensemble_acf(samples: EnsembleSamples, nlags: int | None=None) -> NDArray[numpy.floating[Any]]:
    """
    Compute the ensemble averaged autocorrelation function of the sampled ensemble.

    Parameters
    ----------
    samples: NDArray[tuple[int, int], float]
        Sampled data.
    nlags: int
        Number of lags (default len(sample))

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    samples = numpy.asarray(samples)
    if len(samples) == 0:
        raise Exception(f"no data")

    if len(samples.shape) != 2:
        raise Exception(f"Input must be a two dimensional array.")

    nsim = len(samples)
    if nlags is None or nlags > len(samples):
        nlags = len(samples[0])
    ac_avg = numpy.zeros(nlags)

    for j in range(nsim):
        ac = acf(samples[j], nlags).real
        for i in range(nlags):
            ac_avg[i] += ac[i]
    return ac_avg / float(nsim)


def ensemble_cov(x: EnsembleSamples, y: EnsembleSamples) -> NDArray[numpy.floating[Any]]:
    """
    Compute the ensemble averaged covariance of the sampled ensemble.

    Parameters
    ----------
    x: NDArray[tuple[int, int], float]
        x data samples.
    y: NDArray[tuple[int, int], float]
        y data samples.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble averaged auto correlation function.
    """

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    x_nsim, x_npts = x.shape
    y_nsim, y_npts = y.shape
    npts = min(x_npts, y_npts)
    nsim = min(x_nsim, y_nsim)

    cov = numpy.zeros(npts)
    mean_x = ensemble_mean(x)
    mean_y = ensemble_mean(y)
    for i in range(npts):
        for j in range(nsim):
            cov[i] += (x[j,i] - mean_x[i])*(y[j,i] - mean_y[i])
    return cov / float(nsim)


def ensemble_correlation_coefficient(x: EnsembleSamples, y: EnsembleSamples) -> NDArray[numpy.floating[Any]]:
    """
    Compute the ensemble averaged correlation coefficient of the sampled ensemble.

    Parameters
    ----------
    x: NDArray[tuple[int, int], float]
        x data samples.
    y: NDArray[tuple[int, int], float]
        y data samples.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    cov = ensemble_cov(x, y)
    std_x = ensemble_sd(x)
    std_y = ensemble_sd(y)
    for i in range(1,len(cov)):
        cov[i] = cov[i] / (std_x[i]*std_y[i])
    return cov


def cumu_mean(samples: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
   Cumulative mean of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Cumulative mean of samples as a function of time.
    """

    npts = numpy.arange(1, len(samples) + 1)
    csum = numpy.cumsum(samples)

    return csum / npts


def cumu_var(samples: NDArray[numpy.floating[Any]], Δt: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Cumulative variance of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Cumulative variance of samples as a function of time.
    """

    npts = numpy.arange(1, len(samples) + 1)
    csum = numpy.cumsum(samples)
    mean = csum / npts
    csum2 = numpy.cumsum(samples**2)

    return (csum2 / npts - mean**2) / Δt


def cumu_sd(samples: NDArray[numpy.floating[Any]], Δt: float=1.0) -> NDArray[numpy.floating[Any]]:
    """
    Cumulative standard deviation of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Cumulative variance of samples as a function of time.
    """

    return numpy.sqrt(cumu_var(samples, Δt))


def moving_avg(samples: NDArray[numpy.floating[Any]], window: int) -> NDArray[numpy.floating[Any]]:
    """
    Moving average of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    window: int
        Window size.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Moving average of samples as a function of time.
    """

    result = numpy.cumsum(samples, dtype=float)
    result[window:] = result[window:] - result[:-window]
    return result[window - 1:] / window


def moving_var(samples: NDArray[numpy.floating[Any]], window: int) -> NDArray[numpy.floating[Any]]:
    """
    Moving variance of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    window: int
        Window size.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Moving variance of samples as a function of time.
    """

    result = numpy.cumsum(samples, dtype=float)
    result[window:] = result[window:] - result[:-window]
    result2 = numpy.cumsum(samples**2, dtype=float)
    result2[window:] = result2[window:] - result2[:-window]
    return (result2[window - 1:] - result[window - 1:]**2 / window) / window


def moving_std(samples: NDArray[numpy.floating[Any]], window: int) -> NDArray[numpy.floating[Any]]:
    """
    Moving standard deviation of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    window: int
        Window size.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Moving variance of samples as a function of time.
    """

    return numpy.sqrt(moving_var(samples, window))


def cumu_cov(x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Cumulative covariance of the samples.

    Parameters
    ----------
    x: NDArray[numpy.floating[Any]]
        Sampled data.
    y: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Cumulative covariance of samples as a function of time.
    """

    nsample = min(len(x), len(y))
    npts = numpy.arange(nsample) + 1
    meanx = cumu_mean(x)
    meany = cumu_mean(y)
    cxy = numpy.cumsum(x * y) / npts

    return cxy - meanx * meany


def cov(x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]]) -> numpy.floating[Any]:
    """
    Covariance of samples computed using brute force summation.

    Parameters
    ----------
    x: NDArray[numpy.floating[Any]]
        Sampled data.
    y: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    float
        Covariance of samples.
    """

    nsample = len(x)
    meanx = numpy.mean(x)
    meany = numpy.mean(y)
    c = 0.0

    for i in range(nsample):
        c += x[i] * y[i]

    return c / nsample - meanx * meany


def cov_fft(x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Covariance of samples computed using FFT with isotropic truncation summation.

    Parameters
    ----------
    x: NDArray[numpy.floating[Any]]
        Sampled data.
    y: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Covariance of samples as a function of time.
    """

    n = len(x)
    x_shifted = x - x.mean()
    y_shifted = y - y.mean()

    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    y_padded = numpy.concatenate((y_shifted, numpy.zeros(n-1)))

    x_fft = numpy.fft.fft(x_padded)
    y_fft = numpy.fft.fft(y_padded)
    h_fft = numpy.conj(x_fft) * y_fft
    cc = numpy.fft.ifft(h_fft)

    return (cc[0:n] / float(n)).real


def acf(samples: NDArray[numpy.floating[Any]], nlags: int) -> NDArray[numpy.floating[Any]]:
    """
    Autocorrelation function of samples computed using sm.tsa.stattools.acf.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Covariance of samples as a function of lag.
    """

    return numpy.asarray(sm.tsa.stattools.acf(samples, nlags=nlags, fft=True, missing="drop"))


def pspec(x: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Power spectrum computed using FFT methods.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Covariance of samples as a function of lag.
    """

    n = len(x)
    μ = x.mean()
    x_shifted = x - μ
    energy = numpy.sum(x_shifted**2)
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    
    x_fft = numpy.fft.fft(x_padded)
    power = numpy.conj(x_fft) * x_fft

    return power[1:n].real/(n*energy)


def pdf_hist(samples: NDArray[numpy.floating[Any]], range: tuple[float, float] | None=None, nbins: int=50) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute PDF histogram of provided samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    range: tuple[float, float]
        Value range
    nbins: int
        Number of bind used in calculation.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        PDF histogram.
    """
    
    if range is None:
        range = (float(numpy.min(samples)), float(numpy.max(samples)))
        
    return numpy.histogram(samples, bins=nbins, range=range, density=True)


def cdf_hist(x: NDArray[numpy.floating[Any]], pdf: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Compute CDF histogram from x values and PDF histogram using integration

    Parameters
    ----------
    x: NDArray[numpy.floating[Any]]
        CDF x values.
    pdf: NDArray[numpy.floating[Any]]
        Value range

    Returns
    -------
    NDArray[numpy.floating[Any]]
        CDF histogram.
    """

    npoints = len(x)
    cdf = numpy.zeros(npoints)
    dx = x[1] - x[0]
    for i in range(npoints):
        cdf[i] = numpy.sum(pdf[:i]) * dx
    return cdf


def agg(samples: NDArray[numpy.floating[Any]], m: int) -> NDArray[numpy.floating[Any]]:
    """
    Aggregate sample averages of m elements into len(samples)/m bins. 

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    m : int
        Number of aggregates

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Aggreated sample average.
    """

    n = len(samples)
    d = int(n/m)
    agg = numpy.zeros(d)

    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k] / m

    return agg


def agg_var(samples: NDArray[numpy.floating[Any]], m_vals: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Compute the aggregated variance using the specified bin sizes.. 

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples used in calculation.
    m_vals: list[int]
        Desired bin sizes.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Array of length m containing results.
    """

    npts = len(m_vals)
    var = numpy.zeros(npts)

    for i in range(npts):
        m = int(m_vals[i])
        vals = agg(samples, m)
        mean = numpy.mean(vals)
        d = len(vals)
        for k in range(d):
            var[i] += (vals[k] - mean)**2 / (d - 1)

    return var


def agg_time(x: NDArray[numpy.floating[Any]], m: int) -> NDArray[numpy.floating[Any]]:
    """
    Compute aggregated time values 

    Parameters
    ----------
    x: NDArray[numpy.floating[Any]]
        Unaggregated time values.
    m: int
        Number of points to aggregate.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Aggregated time values.
    """

    n = len(x)
    d = int(n/m)
    return numpy.linspace(x[0], x[n-1], d)


def lag_var(samples: NDArray[numpy.floating[Any]], s: int) -> float:
    """
    Compute lagged variance with specified lag from provided samples using method
    from Lo and Mackinlay, 1988, "Stock market Prices do not Follow Random Walks".

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples
    s: int
       Lag value.

    Returns
    -------
    float
        Lagged variance.
    """

    t = len(samples) - 1
    μ = (samples[t] - samples[0]) / t
    m = (t - s + 1.0)*(1.0 - s/t)
    σ = 0.0

    for i in range(int(s), t+1):
        σ += (samples[i] - samples[i-s] - μ*s)**2

    return σ / m


def lag_var_scan(samples: NDArray[numpy.floating[Any]], s_vals: list[int]) -> NDArray[numpy.floating[Any]]:
    """
    Compute lagged variance for a specified range of values.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Unaggregated time values.
    m: int
        Number of points to aggregate.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        lagged variance for specified lag values.
    """

    return numpy.array([lag_var(samples, s) for s in s_vals])


def multivariate_normal_pdf(vals: NDArray, 
                            μ: NDArray[numpy.floating[Any]], 
                            Ω: NDArray[numpy.floating[Any]]) -> NDArray[numpy.floating[Any]]:
    """
    Return multivariate normal PDF with the specified parameters.

    Parameters
    ----------
    vals: numpy.array
        PDF coordinate values..
    μ: NDArray[numpy.floating[Any]]
        Distribution mean values contains m elements
    Ω: NDArray[numpy.floating[Any]]
        Distribution correlation matrix contains mxm elements.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        PDF values for given coordinates.
    """

    return multivariate_normal.pdf(vals, μ, Ω) # type: ignore[arg-type]

 
def multivariate_normal_samples(μ: NDArray[numpy.floating[Any]], Ω: NDArray[numpy.floating[Any]], n: int) -> NDArray[numpy.floating[Any]]:
    """
    Return multivariate normal samples with the specified parameters.

    Parameters
    ----------
    μ: NDArray[numpy.floating[Any]]
        Distribution mean values contains m elements
    Ω: NDArray[numpy.floating[Any]]
        Distribution correlation matrix contains mxm elements.
    n: int
        Number of samples.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Samples for multivariate normal distribution.
    """

    return numpy.random.multivariate_normal(μ, Ω, n)


def causality_matrix(samples: NDArray[numpy.floating[Any]], nlags: int, add_const: bool=False, critical_value: float=0.05) -> DataFrame:
    """
    Compute Granger causality matrix for the given samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples used in calculation.
    nlags: int
        Maximum number of lags.
    add_const: bool
        Add constant term to model (default False).
    critical_value: float
        Critical value for causality F-test (default 0.05)

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Causality matrix.
    """

    n, _ = samples.shape
    results = []

    for i in range(n):
        for j in range(n):
            test_result = grangercausalitytests(numpy.array([samples[i], samples[j]]).T, nlags)
            pval = min([round(test_result[k][0]['ssr_ftest'][1], 4) for k in range(1, nlags+1)])
            results.append({'pvalue': pval, 
                            'critical_value': critical_value,
                            'result': pval <= critical_value,
                            'dependent_var': i + 1,
                            'causal_var': j + 1})
           
    return DataFrame.from_records(numpy.array(results), index=range(1, len(results) + 1)).sort_values(by=['dependent_var'])


def bias(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> numpy.floating[Any]:
    """
    Compute bias of prediction relative to target.

    Parameters
    ----------
    pred: NDArray[numpy.floating[Any]]
        Predicted values.
    obs: NDArray[numpy.floating[Any]]
        Observed values.

    Returns
    -------
    float
        Bias of prediction relative to target.
    """

    return numpy.mean(pred - obs)


def mae(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> numpy.floating[Any]:
    """
    Compute mean absolute error of prediction relative to target.

    Parameters
    ----------
    pred: NDArray[numpy.floating[Any]]
        Predicted values.
    obs: NDArray[numpy.floating[Any]]
        Observed values.

    Returns
    -------
    float
        Mean absolute error of prediction relative to target.
    """

    return numpy.mean(numpy.abs(pred - obs))


def rmse(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> float:
    """
    Compute root mean squared error of prediction relative to target.

    Parameters
    ----------
    pred: NDArray[numpy.floating[Any]]
        Predicted values.
    obs: NDArray[numpy.floating[Any]]
        Observed values.

    Returns
    -------
    float
        Root mean squared error of prediction relative to target.
    """

    return numpy.sqrt(numpy.mean((pred - obs)**2))


def zscore(samples: NDArray[numpy.floating[Any]], window: int) -> NDArray[numpy.floating[Any]]:
    """
    Compute z-score of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples.
    window: int
        Averaging window.

    Returns
    -------
    float
        Z-score of samples.
    """

    return (samples[window - 1:] - moving_avg(samples, window)) / moving_std(samples, window)
