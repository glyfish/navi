import numpy
from numpy.typing import NDArray
from typing import Any, Tuple, cast
import uuid

from statsmodels.tsa.vector_ar.var_model import LagOrderResults
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult, VECMResults

from lib.models import vecm
from lib.utils import get_param_throw_if_missing, get_param_default_if_missing, verify_type, create_space
from lib.data.hyp_test import VAROrderTestReport, __var_order_test_report_from_result
from lib.data.hyp_test import JohansenCointTestReport, JohansenCointTestStatistic, JohansenCointTestRank, JohansenCointTestEigenVector
from lib.data.reports import JohansenTestReport
from lib.data.param_est import VECMEst, ParamEst, VECMParamType

def compute_estimate(samples: NDArray[numpy.floating[Any]], **kwargs) -> Tuple[VECMResults, VECMEst]:
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
    
    maxlags = get_param_default_if_missing("maxlags", 12, **kwargs)
    trend = get_param_default_if_missing("trend", 'co', **kwargs)
    rank = get_param_default_if_missing("rank", 1, **kwargs)

    result = vecm.fit(samples.T, maxlags=maxlags, trend=trend, rank=rank)

    return result, __vecm_estimate_from_result(result)

def compute_lag_order(samples: NDArray[numpy.floating[Any]], **kwargs) -> Tuple[LagOrderResults, VAROrderTestReport]:
    """
    Determine the lag order of a VAR process using the AIC criterion.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples analyzed.    
    maxlags: int
        Maximum number of lags.
    trend: str
        "n" - no deterministic terms
        "co" - constant outside the cointegration relation
        "ci" - constant within the cointegration relation
        "lo" - linear trend outside the cointegration relation
        "li" - linear trend within the cointegration relation

    Returns
    -------
    LagOrderResults
        Lag order results.
    """

    maxlags = get_param_default_if_missing("maxlags", 12, **kwargs)
    trend = get_param_default_if_missing("trend", 'co', **kwargs)

    result = vecm.lag_order_estimate(samples.T, maxlags, trend)
    return result, __var_order_test_report_from_result(result)


def compute_johansen_coint_test(samples: NDArray[numpy.floating[Any]], max_lags: int, **kwargs) -> Tuple[JohansenTestReport, JohansenCointTestReport, JohansenTestResult]:
    """
    Compute the Johansen cointegration test.

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
             1 - linear trend.
        default is no trend.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Eigenvalues.
    NDArray[numpy.floating[Any]]
        Eigenvectors.
    NDArray[numpy.floating[Any]]
        Trace statistic.
    """

    trend = get_param_default_if_missing("trend", 0, **kwargs)
    result = vecm.johansen_test_coint(samples.T, max_lags, trend)

    return JohansenTestReport(result), __vecm_johansen_coint_test_report_from_result(result), result


def compute_prediction(vecm_result: VECMResults, steps: int, **kwargs) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
    """
    Predict values for the specified number of steps.

    Parameters
    ----------
    vecm_result: VECMResults
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

    alpha = get_param_default_if_missing("alpha", 0.05, **kwargs)
    # alpha is always supplied, so statsmodels returns the (forecast, lower, upper) tuple
    return cast(Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]],
                vecm_result.predict(steps, alpha=alpha))


def create_vecm1_source(λ: numpy.matrix, β: numpy.matrix, a: NDArray[numpy.floating[Any]], **kwargs) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
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
        Noise covariance matrix. (default identity matrix)
    npts: int
        Number of samples generated (default 1000).

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    n, _ = a.shape
    Ω_default = numpy.matrix(numpy.eye(n))
    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    return create_space(npts=npts), numpy.array(vecm.vecm1(λ, β, a, Ω, npts))


def create_vecm_source(λ: numpy.matrix, β: numpy.matrix, a: NDArray[numpy.floating[Any]], **kwargs) -> Tuple[NDArray[numpy.floating[Any]], NDArray[numpy.floating[Any]]]:
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
        Noise covariance matrix. (default identity matrix)
    npts: int
        Number of samples generated (default 1000).

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Simulation results.
    """

    _, n, _ = a.shape
    Ω_default = numpy.matrix(numpy.eye(n))
    Ω = get_param_default_if_missing("Ω", Ω_default, **kwargs)
    npts = get_param_default_if_missing("npts", 1000, **kwargs)

    return create_space(npts=npts), numpy.array(vecm.vecm(λ, β, a, Ω, npts))


def __vecm_johansen_coint_test_report_from_result(result: JohansenTestResult) -> JohansenCointTestReport:
    """
    Create a Johansen test report from a Johansen test result.

    Parameters
    ----------
    result: JohansenTestResult
        Johansen test result.

    Returns
    -------
    JohansenTestReport
        Johansen test report.
    """

    eigen_values = result.eig
    eigen_vectors = result.evec
    trace_critical_vals = result.cvt
    trace_statistic = result.lr1
    eigen_value_critical_values = result.cvm
    eigen_value_statistic = result.lr2

    def compute_rank():
        # Same sequential rule as JohansenTestReport.compute_rank: one rank
        # per significance level column, stopping at the first null not
        # rejected. Plain ints -- the ranks are serialized through
        # json.dumps(default=lambda o: o.__dict__), which cannot encode
        # numpy integer scalars.
        rejections = numpy.asarray(trace_statistic)[:, None] > numpy.asarray(trace_critical_vals)
        ranks = []
        for level in range(rejections.shape[1]):
            rank = 0
            for rejected in rejections[:, level]:
                if not rejected:
                    break
                rank += 1
            ranks.append(int(rank))
        return ranks

    ranks = compute_rank()
    n = len(eigen_values)
    test_id = str(uuid.uuid4())

    trace_statistic_report = [JohansenCointTestStatistic(test_id, i, trace_statistic[i], trace_critical_vals[i]) for i in range(n)]
    # The maximum eigenvalue test's null is r = i (against r = i + 1), not
    # the trace test's r <= i.
    eigen_value_statistic_report = [JohansenCointTestStatistic(test_id, i, eigen_value_statistic[i], eigen_value_critical_values[i], null_hypothesis=f"r={i}") for i in range(n)]
    rank_report = JohansenCointTestRank(test_id, ranks)
    # statsmodels returns the eigenvectors as COLUMNS of evec, so vector i is
    # evec[:, i]; evec[i] is a row and is not a cointegrating vector.
    # JohansenTestReport.summary (reports.py) already reads the columns.
    eigen_value_report = [JohansenCointTestEigenVector(test_id, eigen_values[i], eigen_vectors[:, i]) for i in range(n)]

    return JohansenCointTestReport(test_id, trace_statistic_report, eigen_value_statistic_report, rank_report, eigen_value_report)


def __vecm_estimate_from_result(result: VECMResults) -> VECMEst:
    rank = result.coint_rank
    order = result.k_ar - 1
    neq, _ = result.alpha.shape

    # statsmodels 0.15 exposes these as cache_readonly descriptors; read each into
    # an array once so the 2-D indexing below is on an array, not a descriptor
    const_est = numpy.asarray(result.det_coef)
    const_err = numpy.asarray(result.stderr_det_coef)
    a_est = numpy.asarray(result.gamma)
    a_err = numpy.asarray(result.stderr_gamma)
    beta_est = numpy.asarray(result.beta)
    beta_err = numpy.asarray(result.stderr_beta)
    lambda_est = numpy.asarray(result.alpha)
    lambda_err = numpy.asarray(result.stderr_alpha)
    omega_est = numpy.asarray(result.sigma_u)

    est_id = str(uuid.uuid4())
    lambda_result = []
    beta_result = []
    omega_result = []
    const_result = []
    a_result = []

    for j in range(rank):
        for i in range(neq):
            lambda_result.append(ParamEst(est_id=est_id, 
                                 est=lambda_est[i,j], 
                                 err=lambda_err[i,j], 
                                 est_label=f"$\\hat{{\\lambda}}$", 
                                 err_label=f"$\\sigma_{{\\lambda}}$", 
                                 order=0,
                                 row=i,
                                 column=j,                     
                                 param_type=VECMParamType.VECM_LAMBDA.value))
            beta_result.append(ParamEst(est_id=est_id, 
                               est=beta_est[i,j], 
                               err=beta_err[i,j], 
                               est_label=f"$\\hat{{\\beta}}$", 
                               err_label=f"$\\sigma_{{\\beta}}$", 
                               order=0,
                               row=i,
                               column=j,                     
                               param_type=VECMParamType.VECM_BETA.value))
            
    # det_coef is empty for trends with no deterministic term outside the
    # cointegration relation ('n') or with it inside ('ci', 'li'), where it lives
    # in det_coef_coint instead. For 'lo' the column is the LINEAR TREND slope,
    # not the model constant, so it gets its own label -- otherwise a persisted
    # estimate cannot tell the two apart.
    is_linear_trend = str(getattr(result, "deterministic", "co")).startswith("l")
    const_est_label = "$\\hat{D}$" if is_linear_trend else "$\\hat{M}$"
    const_err_label = "$\\sigma_{D}$" if is_linear_trend else "$\\sigma_{M}$"

    if numpy.asarray(const_est).shape[1] > 0:
        for i in range(neq):
            const_result.append(ParamEst(est_id=est_id,
                                est=const_est[i,0],
                                err=const_err[i,0],
                                est_label=const_est_label,
                                err_label=const_err_label,
                                order=0,
                                row=i,
                                column=0,
                                param_type=VECMParamType.VECM_CONST.value))
            
    
            
    for i in range(neq):
        for j in range(neq):
            omega_result.append(ParamEst(est_id=est_id, 
                                est=omega_est[i,j],
                                err=0.0,
                                est_label=f"$\\hat{{\\Omega}}$", 
                                err_label=f"$\\sigma_{{\\Omega}}$", 
                                order=0,
                                row=i,
                                column=j,
                                param_type=VECMParamType.VECM_OMEGA.value))

    # gamma is (neq, neq*order): lag k occupies columns [k*neq, (k+1)*neq).
    # Indexing a_est[i, j] read the lag-1 block for every lag.
    for k in range(order):
        for j in range(neq):
            for i in range(neq):
                a_result.append(ParamEst(est_id=est_id,
                                         est=a_est[i, k*neq + j],
                                         err=a_err[i, k*neq + j],
                                         est_label=f"$\\hat{{A}}$", 
                                         err_label=f"$\\sigma_A$", 
                                         order=k + 1,
                                         row=i,
                                         column=j,
                                         param_type=VECMParamType.VECM_ALPHA.value))

    return VECMEst(rank=rank, order=order, const=const_result, lambda_est=lambda_result, beta_est=beta_result, a_est=a_result, omega=omega_result)
