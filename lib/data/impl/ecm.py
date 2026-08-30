"""
data.impl.ecm.py

Interface to models.ecm.py
"""

import numpy

from numpy.typing import NDArray
from typing import Any, cast

from lib.models import ecm
import statsmodels.tsa as tsa
import statsmodels.api as sm

from lib.data.param_est import (ParamEst, OLSResult, OLSTransform, OLSParamType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_condition, create_space)
from lib.data.impl.stats import OLS
from lib.stats import diff


def compute_xt_mean(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the ARIMA process mean value.

    Parameters
    ----------
    npts: int
        Number of points to evaluate
    Δt: float
        Width of time step. (default 1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and mean value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)

    # xmax must not be supplied: create_space ignores Δx whenever it has one
    return create_space(npts=npts, Δx=Δt), cast(NDArray[numpy.floating[Any]], numpy.full(npts, 0.0))


def compute_yt_mean(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the Error Correction Model (ECM) process mean value.

    Parameters
    ----------
    npts: int
        Number of points to evaluate
    Δt: float
        Width of time step. (default 1.0)
    δ: float
        Error correction drift. (default 0.0)
    α: float
        Cointegration relation offset. (default 0.0)
    λ: float
        Error correction rate, must be non-zero. (default -1.0)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and mean value.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    δ = get_param_default_if_missing("δ", 0.0, **kwargs)
    α = get_param_default_if_missing("α", 0.0, **kwargs)
    λ = get_param_default_if_missing("λ", -1.0, **kwargs)

    verify_condition("λ", λ != 0.0, "should be non-zero")

    # E[y_t] = α - δ/λ, the fixed point the error correction term relaxes to.
    # The δ = α = 0 defaults reproduce the zero curve.
    μ = α - δ/λ
    # xmax must not be supplied: create_space ignores Δx whenever it has one
    return create_space(npts=npts, Δx=Δt), cast(NDArray[numpy.floating[Any]], numpy.full(npts, μ))


def compute_xt_var(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the ARIMA process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    σ: float
        Residual variance.
    tmax: int
        Maximum time. (default None)
    Δt: float
        Width of time step. (default 1.0)
    npts: int
        Number of points to evaluate

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    t_vals = create_space(xmin=0, npts=npts, xmax=tmax, Δx=Δt)
    return t_vals, ecm.xt_var(φ, σ, t_vals)


def compute_yt_var(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the Error Correction Model (ECM) process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    β: float
        ECM correlation parameter.
    σ: float
        Residual variance.
    tmax: int
        Maximum time. (default None)
    Δt: float
        Width of time step. (default 1.0)
    npts: int
        Number of points to evaluate

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    t_vals = create_space(xmin=0, npts=npts, xmax=tmax, Δx=Δt)
    return t_vals, ecm.yt_var(φ, σ, β, t_vals)


def compute_cov(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Compute the Error Correction Model (ECM) process variance value.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    σ: float
        Residual variance.
    β: float
        ECM correlation parameter.
    npts: int
        Number of points to evaluate

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Time and mean value.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    Δt = get_param_default_if_missing("Δt", 1.0, **kwargs)
    tmax = get_param_default_if_missing("tmax", None, **kwargs)
    npts = get_param_throw_if_missing("npts", **kwargs)

    t_vals = create_space(xmin=0, npts=npts, xmax=tmax, Δx=Δt)
    return t_vals, ecm.cov(φ, σ, β, t_vals)


def compute_beta_estimate(yt: NDArray[numpy.floating[Any]], xt: NDArray[numpy.floating[Any]]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
    """
    Compute OLS estimate of Error Correction Model (ECM) β parameter.

    Parameters
    ----------
    xt: NDArray[numpy.floating[Any]]
        ECM variable.
    yt: NDArray[numpy.floating[Any]]
        ECM variable.

    Returns
    -------
    tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]
        Ols report and result model.
    """

    report, result = OLS.LINEAR.single_variable_estimate(yt, xt)
    __add_beta_transform(result)
    return report, result


def compute_gamma_lambda_estimate(yt: NDArray[numpy.floating[Any]], xt: NDArray[numpy.floating[Any]], est_beta: float) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]:
    """
    Compute OLS estimate of Error Correction Model (ECM) β parameter.

    Parameters
    ----------
    xt: NDArray[numpy.floating[Any]]
        ECM variable.
    yt: NDArray[numpy.floating[Any]]
        ECM variable.
    est_beta: float
        Estimated beta.

    Returns
    -------
    tuple[sm.regression.linear_model.RegressionResultsWrapper, OLSResult]
        Ols report and result model.
    """

    εt = yt - est_beta * xt
    dxt = diff(xt)
    dyt = diff(yt)

    report, result = OLS.LINEAR.two_variable_estimate(dyt, dxt, εt[:-1])
    __add_gamma_lambda_transform(result)
    return report, result


def create_source(**kwargs) -> tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Generate an Error Correction Model (ECM) time series from an AR(1) process using the specified parameters.

    Parameters
    ----------
    φ: float
        AR(1) parameter satisfying |φ| < 1.
    δ: float
        ECM term parameter. (default 0.0)
    α: float
        ECM term offset parameter. (default 0.0)
    β: float
        ECM correlation parameter.
    γ: float
        ECM X(t) scale parameter.
    λ: float
        ECM relaxation rate.
    σ: float
        Residual variance. (default 1.0)
    npts: int
        Number of samples generated. (default 1000)

    Returns
    -------
    tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]
        Generated x(t) and y(t) ECM time series.
    """

    φ = get_param_throw_if_missing("φ", **kwargs)
    β = get_param_throw_if_missing("β", **kwargs)
    γ = get_param_throw_if_missing("γ", **kwargs)
    λ = get_param_throw_if_missing("λ", **kwargs)
    δ = get_param_default_if_missing("δ", 0.0, **kwargs)
    α = get_param_default_if_missing("α", 0.0, **kwargs)
    σ = get_param_default_if_missing("σ", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    xt, yt = ecm.ecm(φ, δ, α, β, γ, λ, npts, σ)

    return create_space(xmax=npts - 1, npts=npts), numpy.array([xt, yt])


def __add_beta_transform(result: OLSResult):
    """
    Add transformation OLS beta parameter estimate.

    Parameters
    ----------
    result: OLSResult
        OLS analysis results.
    """

    model = r"$\hat{\alpha} + \hat{\beta} x_t$"

    param = ParamEst(est_id=result.est_id,
                     est=result.params[0].est,
                     err=result.params[0].err,
                     est_label=r"$\hat{\beta}$",
                     err_label=r"$\sigma_{\hat{\beta}}$",
                     order=1,
                     row=0,
                     column=0,
                     param_type=OLSParamType.TRANS_PARAM.value)

    const = ParamEst(est_id=result.est_id,
                     est=result.const.est,
                     err=result.const.err,
                     est_label=r"$\hat{\alpha}$",
                     err_label=r"$\sigma_{\hat{\alpha}}$",
                     order=1,
                     row=0,
                     column=0,
                     param_type=OLSParamType.TRANS_CONST.value)
    
    result.set_transforms(model, [OLSTransform(param)], OLSTransform(const))


def __add_gamma_lambda_transform(result: OLSResult):
    """
    Add transformation OLS beta parameter estimate.

    Parameters
    ----------
    result: OLSResult
        OLS analysis results.
    """

    # the Δy regression, not the β regression this string was copied from
    model = r"$\Delta y_t = \hat{\delta} - \hat{\lambda}\hat{\alpha} + \hat{\gamma} \Delta x_t + \hat{\lambda} \varepsilon_{t-1}$"

    # One transform per fitted parameter, in the regression's own column order:
    # column 0 is γ̂ (on Δx) and column 1 is λ̂ (on ε_{t-1}). Degenerate input can
    # leave the fit with fewer columns than that, so only label what is there.
    labels = [(r"$\hat{\gamma}$", r"$\sigma_{\hat{\gamma}}$"),
              (r"$\hat{\lambda}$", r"$\sigma_{\hat{\lambda}}$")]
    param_transforms = [
        OLSTransform(ParamEst(est_id=result.est_id,
                              est=result.params[i].est,
                              err=result.params[i].err,
                              est_label=est_label,
                              err_label=err_label,
                              order=1,
                              row=0,
                              column=0,
                              param_type=OLSParamType.TRANS_PARAM.value))
        for i, (est_label, err_label) in enumerate(labels) if i < len(result.params)
    ]

    const = ParamEst(est_id=result.est_id,
                     est=result.const.est,
                     err=result.const.err,
                     # the Δy regression's constant estimates δ - λα, not λ
                     est_label=r"$\hat{\delta} - \hat{\lambda}\hat{\alpha}$",
                     err_label=r"$\sigma_{\hat{\delta} - \hat{\lambda}\hat{\alpha}}$",
                     order=1,
                     row=0,
                     column=0,
                     param_type=OLSParamType.TRANS_CONST.value)
    
    result.set_transforms(model, param_transforms, OLSTransform(const))
