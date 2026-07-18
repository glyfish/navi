"""
stats.py

Compute generic statistics functions.

"""

import numpy

from numpy.typing import NDArray
from typing import Any
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from enum import Enum
import statsmodels.api as sm
import uuid
from pandas import DataFrame
import statsmodels.formula.api as smf

from lib import stats
from lib.data.param_est import (ParamEst, OLSResult, OLSParamType)
from lib.data.hyp_test import (GrangerCausalityTestResult, GrangerCausalityTestReport)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing, get_s_vals, 
                       create_logspace, create_space)

def compute_pspec(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Power spectrum computed using FFT methods.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Frequency and power spectrum.
    """

    return time[1:], stats.pspec(data)


def compute_acf(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Autocorrelation function of samples computed using sm.tsa.stattools.acf.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    nlags: int
        max number of lags computed.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time lags and autocovariance of samples as a function of lag.
    """

    nlags = get_param_throw_if_missing("nlags", **kwargs)

    return time[:nlags + 1], stats.acf(data, nlags)


def compute_ndiff(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Take the specified number of differences of the samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    ndiff : int
        Number of differences taken (default 1).

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and samples differenced n times.
    """

    ndiff = get_param_default_if_missing("ndiff", 1, **kwargs)
    return time[:-ndiff], stats.ndiff(data, ndiff)


def compute_diff(time: NDArray[numpy.floating[Any]], data: NDArray[numpy.floating[Any]]) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Com[ute the sample difference].

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and samples differenced n times.
    """

    return time[:-1], stats.diff(data)


def compute_cumu_mean(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Cumulative mean of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative mean of samples as a function of time.
    """

    return time, stats.cumu_mean(data)


def compute_cumu_sd(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Cumulative standard deviation of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative standard deviation of samples as a function of time.
    """
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.cumu_sd(data, Δt)


def compute_cumu_var(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Cumulative variance of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative variance of samples as a function of time.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.cumu_var(data, Δt)


def compute_moving_avg(time: numpy.ndarray, samples: NDArray[numpy.floating[Any]], window: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Moving average of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    window: int
        Window size.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Moving average of samples as a function of time.
    """

    return time[window - 1:], stats.moving_avg(samples, window)


def compute_moving_var(time: numpy.ndarray, samples: NDArray[numpy.floating[Any]], window: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Moving variance of samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    time: numpy.ndarray
        Time
    window: int
        Window size.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Moving variance of samples as a function of time.
    """

    return time[window - 1:], stats.moving_var(samples, window)

def compute_cumu_cov(time: numpy.ndarray, x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Cumulative covariance of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    x: NDArray[numpy.floating[Any]]
        Sampled data.
    y: NDArray[numpy.floating[Any]]
        Sampled data.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative covariance of samples as a function of time.
    """

    return time, stats.cumu_cov(x, y)


def compute_agg_var(data: numpy.ndarray, **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the aggregated variance using the specified bin sizes.. 

    Parameters
    ----------
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    npts : int
        Number of aggregation steps
    m_max: int
        Maximum lags.
    m_min: int
        Minimum lag. (default 1)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Lags and lagged variance for each value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    m_max = get_param_throw_if_missing("m_max", **kwargs)
    m_min = get_param_default_if_missing("m_min", 1, **kwargs)

    m_vals = create_space(npts=npts, xmax=m_max, xmin=m_min)
    return m_vals, stats.agg_var(data, m_vals)


def compute_agg(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Aggregate sample averages of m elements into len(samples)/m bins. 

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    m : int
        Number of aggregates

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Aggreated sample average.
    """

    m = get_param_throw_if_missing("m", **kwargs)
    return stats.agg_time(time, m), stats.agg(data, m)


def compute_lag_var(data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[list[int], NDArray[numpy.floating[Any]]]:
    """
    Compute lagged variance for a specified range of values.

    Parameters
    ----------
    data: NDArray[numpy.floating[Any]]
        Unaggregated time values.
    s_max : int
        Maximum s-value.
    s_min : int
        Minimum s value.
    npts : int
        Number of s-values to create
    s_vals : list[int]
        List if s-values to use

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        lagged variance for specified lag values.
    """

    s_vals =  [int(s) for s in get_s_vals(**kwargs)]
    return s_vals, stats.lag_var_scan(data, s_vals)


def compute_ensemble_mean(time: NDArray[numpy.floating[Any]], data: list[NDArray[numpy.floating[Any]]]) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the time varying mean of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[NDArray[numpy.floating[Any]]]
        Sampled data.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    return time, stats.ensemble_mean(data)


def compute_ensemble_sd(time: numpy.ndarray, data: list[NDArray[numpy.floating[Any]]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the time varying standard deviation of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[NDArray[numpy.floating[Any]]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.ensemble_sd(data, Δt)


def compute_ensemble_var(time: NDArray[numpy.floating[Any]], data: list[NDArray[numpy.floating[Any]]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the time varying variance of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: list[NDArray[numpy.floating[Any]]]
        Sampled data.
    Δt: float
        Time delta (default 1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble average mean as a function of time.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    return time, stats.ensemble_var(data, Δt)


def compute_ensemble_acf(time: numpy.ndarray, data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the ensemble averaged autocorrelation function of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    data: NDArray[numpy.floating[Any]]
        Sampled data.
    nlags: int
        Number of lags (default len(sample))

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    nlags = get_param_default_if_missing("nlags", None, **kwargs)

    return time[:nlags], stats.ensemble_acf(data, nlags)


def compute_ensemble_cov(time: NDArray[numpy.floating[Any]], x: list[NDArray[numpy.floating[Any]]], y: list[NDArray[numpy.floating[Any]]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the ensemble averaged covariance function of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    x: list[NDArray[numpy.floating[Any]]]
        x data samples.
    y: list[NDArray[numpy.floating[Any]]]
        y data samples.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    return time, stats.ensemble_cov(x, y)


def compute_ensemble_correlation_coefficient(time: NDArray[numpy.floating[Any]], x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the ensemble averaged covariance function of the sampled ensemble.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    x: numpy.ndarray[tuple[int, int], float]
        x data samples.
    y: numpy.ndarray[tuple[int, int], float]
        y data samples.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Ensemble averaged auto correlation function.

    Raises
    ______
    Exception
        Samples are not a two dimensional array.
    """

    return time, stats.ensemble_correlation_coefficient(x, y)


def compute_pdf_hist(data: NDArray[numpy.floating[Any]], **kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Create a PDF histogram for the provided data.

    Parameters
    ----------
    data : NDArray[numpy.floating[Any]]
        Sampled data.
    xmin : float
        Minimum x value (required).
    xmax : float
        maximum x value (required).
    nbins : int
        Number of bins. (default 50)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        PDF histogram and bin values

    """

    nbins = get_param_default_if_missing("nbins", 50, **kwargs)
    pdf, bin_edges = stats.pdf_hist(data, None, nbins)

    return pdf, bin_edges[:-1] + (bin_edges[:-1] - bin_edges[1:]) / 2


def compute_cdf_hist(x: NDArray[numpy.floating[Any]], pdf: NDArray[numpy.floating[Any]]) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Create a CDF histogram from the given PDF histogram.

    x : NDArray[numpy.floating[Any]]
        Random variable values.
    pdf : NDArray[numpy.floating[Any]]
        PDF.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        CDF histogram.
    """

    return x, stats.cdf_hist(x, pdf)


def compute_multivariate_normal_pdf(μ: NDArray[numpy.floating[Any]], Ω: NDArray[numpy.floating[Any]], n: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Return multivariate normal PDF with the specified parameters.

    Parameters
    ----------
    μ: NDArray[numpy.floating[Any]]
        Distribution mean values contains m elements
    Ω: NDArray[numpy.floating[Any]]
        Distribution correlation matrix contains mxm elements.
    n: int
        Number of points along an axis.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        coordinates and generated samples. Generated samples.

    Raises
    ------
        Exception invalid array dimensions.
    """

    nvars = len(μ)
    if nvars == 1 or nvars > 3:
        raise Exception("Number of variables must be between 2 or 3")

    σ = min(numpy.diag(Ω))
    δ = 6.0*σ / (n - 1)

    x1 = -3.0*σ + μ[0]
    x2 = 3.0*σ + δ + μ[0]
    y1 = -3.0*σ + μ[1]
    y2 = 3.0*σ + δ + μ[1]

    if nvars == 2:
        vals = numpy.mgrid[x1:x2:δ, y1:y2:δ]
        coords = numpy.transpose(vals)[:,:,::-1]
    else:
        z1 = -3.0*σ + μ[3]
        z2 = 3.0*σ + δ + μ[3]
        vals =  numpy.mgrid[x1:x2:δ, y1:y2:δ, z1:z2:δ]
        coords = numpy.transpose(vals)[:,:,:,::-1]

    return vals, stats.multivariate_normal_pdf(coords, μ, Ω)


def compute_causality_matrix(samples: NDArray[numpy.floating[Any]], **kwargs):
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
        Critical value for causality test (default 0.05)
        
    Returns
    -------
    NDArray[numpy.floating[Any]]
        Causality matrix.
    """

    nlags = get_param_default_if_missing("nlags", 12, **kwargs)
    add_const = get_param_default_if_missing("add_const", False, **kwargs)
    critical_value = get_param_default_if_missing("critical_value", 0.05, **kwargs)

    result = stats.causality_matrix(samples, nlags, add_const, critical_value)
    return result, __granger_causality_model_from_result(result)


def compute_bias(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> numpy.floating[Any]:
    """
    Compute bias between predicted and observed values.

    Parameters
    ----------
    pred: NDArray[numpy.floating[Any]]
        Predicted values.
    obs: NDArray[numpy.floating[Any]]
        Observed values.

    Returns
    -------
    float
        Bias.
    """

    return stats.bias(pred, obs)


def compute_mae(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> numpy.floating[Any]:    
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
        MSE.
    """

    return stats.mae(pred, obs)


def compute_rmse(pred: NDArray[numpy.floating[Any]], obs: NDArray[numpy.floating[Any]]) -> float:    
    """
    Compute root mean squared error between predicted and observed values.

    Parameters
    ----------
    pred: NDArray[numpy.floating[Any]]
        Predicted values.
    obs: NDArray[numpy.floating[Any]]
        Observed values.

    Returns
    -------
    float
        RMSE.
    """

    return stats.rmse(pred, obs)


def create_multivariate_normal_samples_source(μ: NDArray[numpy.floating[Any]], Ω: NDArray[numpy.floating[Any]], n: int) -> NDArray[numpy.floating[Any]]:
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
        Generated samples.
    """

    return numpy.transpose(stats.multivariate_normal_samples(μ, Ω, n))


class OLS(Enum):
    """
    Specify regression model to use for linear ols repression models.

    Values
    ------
    LINEAR : str
        Assume a linear relation between regression variables.
            y = a*x + b
        where a and b be are regression constants.
    LOG : str
        Assume power law relation between the regression variables.
            y = b*x**a
        where a and b be are regression constants.
    XLOG : str
        Assume an exponential relationship between the regression variables.
            y = b*exp(a*x)
        where a and b be are regression constants.
    YLOG : str
        Assume a logarithmic relation between the regression variables.
            y = b*ln(a*x)
        where a and b be are regression constants.
    """

    LINEAR = "LINEAR"
    LOG = "LOG"
    XLOG = "XLOG"
    YLOG = "YLOG"


    def single_variable_estimate(self, y: NDArray[numpy.floating[Any]], x: NDArray[numpy.floating[Any]]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
        """
        Perform single variable OLS regression on the provided data.

        Parameters
        ----------
        y: NDArray[numpy.floating[Any]]
            Dependent variable
        x: NDArray[numpy.floating[Any]]
            Variable
 
        Return
        ------
        tuple[sm.regression.linear_model.RegressionResults, OLSResult]
            OLS report and result model.
        """

        result = self.__OLS_fit(y, x)
        return result, self.__ols_estimate_from_result(result)


    def two_variable_estimate(self, y: NDArray[numpy.floating[Any]], x1: NDArray[numpy.floating[Any]], x2: NDArray[numpy.floating[Any]]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
        """
        Perform single variable OLS regression on the provided data.

        Parameters
        ----------
        y: NDArray[numpy.floating[Any]]
            Dependent variable
        x1: NDArray[numpy.floating[Any]]
            Independent variable one
        x2: NDArray[numpy.floating[Any]]
            Independent variable two
 
        Return
        ------
        tuple[sm.regression.linear_model.RegressionResults, OLSResult]
            OLS report and result model.
        """

        result = self.__OLS_fit(y, numpy.transpose(numpy.array([x1, x2])))
        return result, self.__ols_estimate_from_result(result)
    

    def multi_variable_estimate(self, y: NDArray[numpy.floating[Any]], x: NDArray[numpy.floating[Any]]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
        """
        Perform multi variable OLS regression on the provided data.

        Parameters
        ----------
        y: NDArray[numpy.floating[Any]]
            Dependent variable
        x: NDArray[numpy.floating[Any]]
            Independent variables
 
        Return
        ------
        tuple[sm.regression.linear_model.RegressionResults, OLSResult]
            OLS report and result model.
        """

        x_cols = [f"x{i+1}" for i in range(len(x))]
        cols = ['y'] + x_cols
        data = numpy.concatenate((numpy.reshape(y, (1,len(y))), x), axis=0)
        formula = "y ~ " + " + ".join(x_cols)
        result = self.__OLS_formula_fit(DataFrame(data.T, columns=cols), formula)
        return result, self.__ols_estimate_from_result(result)


    def __OLS_fit(self, y: NDArray[numpy.floating[Any]], x: NDArray[numpy.floating[Any]]) -> sm.regression.linear_model.RegressionResultsWrapper:
        """ 
        Create statsmodels OLS object using specified samples assuming a single dependent variable.

        Parameters
        ----------
        y: NDArray[numpy.floating[Any]]
            Dependent variable
        x: NDArray[numpy.floating[Any]]
            Variable

        Returns
        -------
        sm.OLS
            OLS object
        """

        if self.value == OLS.LOG.value:
            x = numpy.log10(x)
            y = numpy.log10(y)

        x = sm.add_constant(x)
        return sm.OLS(y, x, missing='drop').fit()
    

    def __OLS_formula_fit(self, data: DataFrame, formula: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        """ 
        Create statsmodels OLS object using specified samples assuming a single dependent variable.

        Parameters
        ----------
        data: DataFrame
            Sample data.
        formula: str
            OLS formula.

        Returns
        -------
        sm.regression.linear_model.RegressionResults
            OLS regression results.
        """

        return smf.ols(formula=formula, data=data).fit()


    def __ols_estimate_from_result(self, result: sm.regression.linear_model.RegressionResultsWrapper) -> OLSResult:
        """
        Create an OLS result model from the ols single variable returned report.

        Parameters
        ----------
        report: sm.regression.linear_model.RegressionResults
            OLS results report.
 
        Return
        ------
        OLSResult
            OLS result model.
        """
        
        est_id = str(uuid.uuid4())
        const = ParamEst.from_dict({"est": result.params[0], 
                                    "err": result.bse[0],
                                    "est_label": f"$\\beta$",
                                    "err_label": f"$\\sigma_{{\\beta}}$",
                                    "est_id": est_id,
                                    "param_type": OLSParamType.OLS_CONST.value})

        nparams = len(result.params)
        params = []
        for i in range(1, nparams):
            params.append(ParamEst.from_dict({"est": result.params[i], 
                                              "err": result.bse[i],
                                              "est_label": f"$\\alpha_{i}$",
                                              "err_label": f"$\\sigma_{{\\alpha_{i}}}$",
                                              "column": i,
                                              "est_id": est_id,
                                              "param_type": OLSParamType.OLS_PARAM.value}))

        r2 = result.rsquared
        return OLSResult(est_id, const, params, r2)
    

def __granger_causality_model_from_result(result: DataFrame) -> GrangerCausalityTestReport:
    results = result.to_dict(orient='records')
    est_id = str(uuid.uuid4())

    causality_result = result['result'].to_numpy()
    dep_var = result['dependent_var'].to_numpy()
    rank = len(numpy.unique(dep_var[causality_result]))

    return GrangerCausalityTestReport(est_id, rank, [GrangerCausalityTestResult.from_dict(r, est_id) for r in results])


def compute_zscore(time: numpy.ndarray, samples: NDArray[numpy.floating[Any]], window: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute z-score of samples.

    Parameters
    ----------
    time: numpy.ndarray
        Time
    samples: NDArray[numpy.floating[Any]]
        Sampled data.
    window: int
        Averaging window.

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative variance of samples as a function of time.
    """

    return time[window - 1:], stats.zscore(samples, window)


def compute_moving_std(time: numpy.ndarray, samples: NDArray[numpy.floating[Any]], window: int) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
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
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and cumulative variance of samples as a function of time.
    """

    return time[window - 1:], stats.moving_std(samples, window)
