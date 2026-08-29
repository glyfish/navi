
from typing import Any
import numpy
from numpy.typing import NDArray
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults, VARResultsWrapper, LagOrderResults

from lib.stats import multivariate_normal_samples

def mean(φ: NDArray[numpy.floating[Any]], μ: NDArray[numpy.floating[Any]]) -> numpy.matrix:
    """
    Compute the stationary mean matrix for a VAR(n) process with the given parameters.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        VAR(n) process coefficient matrix.
    μ: NDArray[numpy.floating[Any]]
        VAR(n) process offset matrix.

    Returns
    -------
    numpy.matrix
        Stationary mean matrix.
    """

    n, m, _ = φ.shape
    Φ = phi_comp(φ)
    Μ = mean_comp(μ, n)
    tmp = numpy.matrix(numpy.eye(n * m)) - Φ
    # numpy.linalg.inv preserves the matrix subclass at runtime, but its stub is
    # typed as returning a plain ndarray. asmatrix is a no-op view here and makes
    # the declared matrix return checkable.
    return numpy.asmatrix(numpy.linalg.inv(tmp)*Μ)


def cov(φ: NDArray[numpy.floating[Any]], ω: NDArray[numpy.floating[Any]]) -> numpy.matrix:
    """
    Compute the stationary covariance matrix for the given VAR(n) process
    parameters.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        VAR(n) process coefficient matrix.
    ω: NDArray[numpy.floating[Any]]
        VAR(n) process gaussian noise autocovariance matrix.

    Returns
    -------
    numpy.matrix
        Stationary covariance matrix.
    """

    n, m, _ = φ.shape

    Ω = omega_comp(ω, n)
    Φ = phi_comp(φ)
    eye = numpy.matrix(numpy.eye(m**2 * n**2))
    tmp = eye - numpy.kron(Φ, Φ)
    inv_tmp = numpy.linalg.inv(tmp)
    vec_var = inv_tmp * vec(Ω)
    return unvec(vec_var)


def acov(φ: NDArray[numpy.floating[Any]], ω: NDArray[numpy.floating[Any]], n: int) -> NDArray[numpy.floating[Any]]:
    """
    Compute the stationary auto covariance matrix for the given VAR(n)
    parameters.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        VAR(n) process coefficient matrix.
    ω: NDArray[numpy.floating[Any]]
        VAR(n) process gaussian noise autocovariance matrix.
    n: int
        Maximum lag.

    Returns
    -------
    numpy.matrix
        Stationary mean matrix.
    """

    Φ = phi_comp(φ)
    Σ = cov(φ, ω)
    l, _ = Φ.shape
    γ = numpy.zeros((n, l, l))
    γ[0] = numpy.matrix(numpy.eye(l))
    for i in range(1,n):
        γ[i] = γ[i-1]*Φ
    for i in range(n):
        γ[i] = Σ*γ[i].T
    return γ


def eig(φ: NDArray[numpy.floating[Any]]) -> NDArray[numpy.complexfloating[Any, Any]]:
    """
    Compute eigen values of VAR(n) parameter matrix transformed to VAR(1) companion form. 
    Stationarity requires that |λ| < 1.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
       VAR(n) coefficient matrix in companion form.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Array of eigen values.
    """

    Φ = phi_comp(φ)
    λ, _ = numpy.linalg.eig(Φ)
    return λ


def is_stationary(φ: NDArray[numpy.floating[Any]]) -> bool:
    """
    Return True if the VAR(n) parameter matrix is stationary.

    Parameters
    ----------
    φ: NDArray[numpy.floating[Any]]
        VAR(n) covariance matrix.

    Returns
    -------
    bool
        True if VAR(n) process is stationary.
    """

    for λ in eig(φ):
        if abs(λ) >= 1:
            return False
    return True


def phi_comp(φ: NDArray[numpy.floating[Any]]) -> numpy.matrix:
    """
    Convert the VAR(n) coefficient matrix to the VAR(1) companion form used for calculations. 

    Parameters
    ----------
   φ: NDArray[numpy.floating[Any]]
         VAR(n) coefficient matrix

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Companion form of noise covariance matrix.
    """

    n, m, _ = φ.shape
    p = φ[0]
    for i in range(1, n):
        p = numpy.concatenate((p, φ[i]), axis=1)
    for i in range(n-1):
        if i == 0:
            r = numpy.eye(m)
        else:
            r = numpy.zeros((m, m))
        for j in range(1, n):
            if j == i:
                r = numpy.concatenate((r, numpy.eye(m)), axis=1)
            else:
                r = numpy.concatenate((r, numpy.zeros((m, m))), axis=1)
        p = numpy.concatenate((p, r), axis=0)

    return numpy.matrix(p)


def mean_comp(Μ: NDArray[numpy.floating[Any]], n: int) -> numpy.matrix:
    """
    Convert the VAR(n) offset matrix the VAR(1) companion form used for calculations.

    Parameters
    ----------
    Μ: NDArray[numpy.floating[Any]]
        VAR(n) offset matrix.
    n: int
        Order of VAR process.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Companion form of VAR(n) offset matrix.
    """

    m = len(Μ)
    p = numpy.zeros(m * n)
    p[:m] = Μ
    return numpy.matrix([p]).T


def omega_comp(ω: NDArray[numpy.floating[Any]], n: int) -> numpy.matrix:
    """
    Convert VAR(n) gaussian noise covariance matrix to companion form.

    Parameters
    ----------
    ω: NDArray[numpy.floating[Any]]
        VAR(n) noise covariance matrix.
    n: int
        Order of VAR process.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Companion form of noise covariance matrix.
    """

    m, _ = ω.shape

    p = numpy.zeros((m * n, m * n))
    p[:m, :m] = ω

    return numpy.matrix(p)


def vec(m: NDArray[numpy.floating[Any]] | numpy.matrix) -> numpy.matrix:
    """
    Apply the vec operator to the given matrix. The vec operation 
    applied to the matrix,

    A = [[a11, a12],
         [a21, a22]]

    gives,

    vec(A) = [[a11],
              [a21],
              [a12],
              [a22]]

    Parameters
    ----------
    m: NDArray[numpy.floating[Any]]
        Matrix to be converted to vec form.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Input vector converted to vec form.
    """

    # m[:, i] must come back as an (n, 1) column to fill the column slice below;
    # a plain ndarray gives (n,) and fails to broadcast. asmatrix is a view when
    # m is already a matrix, so this accepts both at no cost.
    m = numpy.asmatrix(m)

    _, n = m.shape
    v = numpy.matrix(numpy.zeros(n**2)).T
    for i in range(n):
        d = i*n
        v[d:d+n] = m[:,i]
    return v


def unvec(v: NDArray[numpy.floating[Any]] | numpy.matrix) -> numpy.matrix:
    """
    Apply the inverse of the vec operation to the given matrix. For the following
    matrix in vec form,

    A = [[a11],
         [a21],
         [a12],
         [a22]]

    apply unvec gives,

    unvec(A) = [[a11, a12],
                [a21, a22]]

    Parameters
    ----------
    m: NDArray[numpy.floating[Any]]
        Matrix to be converted to unvec form.

    Returns
    -------
    NDArray[numpy.floating[Any]]
        Input vector in unvec form.
    """

    n2, _ = v.shape
    n = int(numpy.sqrt(n2))
    m = numpy.matrix(numpy.zeros((n, n)))
    for i in range(n):
        d = i*n
        m[:,i] = v[d:d+n]
    return m


def var(x0: NDArray[numpy.floating[Any]], μ: NDArray[numpy.floating[Any]], φ: NDArray[numpy.floating[Any]], ω: NDArray[numpy.floating[Any]], npts: int) -> NDArray[numpy.floating[Any]]:
    """
    Simulate a VAR(n) process using the provided parameters.
    
    Parameters
    ----------
    x0: NDArray[numpy.floating[Any]]
        VAR(n) process initial value matrix.
    μ: NDArray[numpy.floating[Any]]
        VAR(n) process offset matrix.
    φ:  list[NDArray[numpy.floating[Any]]]
        VAR(n) process coefficient matrix.
    ω: NDArray[numpy.floating[Any]]
        VAR(n) process gaussian noise autocovariance function.
    npts: int
        Number of samples.

    Returns
    -------
    numpy.matrix
        Simulation results.
    """

    n, m, _ = φ.shape
    xt = numpy.zeros((npts, m))
    ε = multivariate_normal_samples(numpy.zeros(m), ω, npts)
    for i in range(n):
        xt[i] = x0[i]
    for i in range(n, npts):
        xt[i] = ε[i] + μ
        for j in range(n):
            t1 = φ[j]*numpy.matrix(xt[i-j-1]).T
            xt[i] += numpy.squeeze(numpy.array(t1), axis=1)
    return numpy.transpose(xt)
    
def fit(endog: NDArray[numpy.floating[Any]] | DataFrame, maxlags: int=12, trend: str="c") -> VARResultsWrapper:
    """
    Estimate the parameters for and assumed VAR(n) model.

    Parameters
    ----------
    endog: DataFrame
        VAR(n) process endogenous variable samples.
    maxlags: int
        Maximum number of time lags tried. (default is 12)
    trend: str
        Assumed trend (default 'c'). 
        Values 'n'=no trend, 'c'=constant offset, 'ct'=linear trend, 'ctt'=quadratic and linear trend.

    Returns
    -------
    VARResult
        Analysis results.
    """

    return __var_model(endog).fit(maxlags=maxlags, trend=trend)
    
def lag_order_estimate(samples: NDArray[numpy.floating[Any]], maxlags: int=12, trend: str="c") -> LagOrderResults:
    """
    Estimate order of VAR samples.

    Parameters
    ----------
    samples: NDArray[numpy.floating[Any]]
        Samples analyzed.    
    maxlags: int
        Maximum number of lags.
    trend: str
        Assumed trend (default 'c'). 
        Values 'n'=no trend, 'c'=constant offset, 'ct'=linear trend, 'ctt'=quadratic and linear trend.

    Returns
    -------
    LagOrderResults
        Order results.
    """

    return __var_model(samples).select_order(maxlags=maxlags)
        
def __var_model(endog: NDArray[numpy.floating[Any]] | DataFrame) -> VAR:
    """
    Estimate the parameters for and assumed VAR(n) model.

    Parameters
    ----------
    endog: DataFrame
        VAR(n) process endogenous variable samples.

    Returns
    -------
    VAR
        VAR model.
    """

    return VAR(endog)
