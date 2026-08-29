from tabulate import tabulate
from typing import Any
import numpy
from numpy.typing import NDArray
from statsmodels.tsa.arima.model import ARIMAResults, ARIMAResultsWrapper
from statsmodels.tsa.vector_ar.vecm import JohansenTestResult

from lib.data.hyp_test import HypothesisTestType, HypothesisType

class VarianceRatioTestReport:
    """
    Formatted text report of Lo and Mackinlay variance ratio test results.

    Properties
    ----------
    sig_level: float
        Significance level of test.
    hyp_type: HypothesisTestType
        Test hypothesis type. Possible values (BM, FBM_AUTO_CORR, FBM_NEG_AUTO_CORR)
    s_vals: list[int]
        Lag values used in used in test.
    stats: NDArray[numpy.floating[Any]]
        Test statistic values.
    p_vals: list[float]
        Probability values of test statistics.
    critical_values: list[float | None]
        Critical values used in test. Index 0 is the lower tail critical value and 
        index 1 the upper tail critical value.
    status_vals: list[bool]
        Test status for different values of lag, s.
    """

    def __init__(self, sig_level: float, hyp_type: HypothesisType, hyp_test_type: HypothesisTestType, s_vals: list[int], 
                 stats: NDArray[numpy.floating[Any]], p_vals: list[float], critical_values: list[float | None]):
        self.sig_level = sig_level
        self.hyp_type = hyp_type
        self.hyp_test_type = hyp_test_type
        self.s_vals = s_vals
        self.stats = stats
        self.p_vals = p_vals
        self.critical_values = critical_values
        if self.critical_values[0] is None:
            self.status_vals = [self.stats[i] < self.critical_values[1] for i in range(len(self.stats))]
        elif self.critical_values[1] is None:
            self.status_vals = [self.stats[i] > self.critical_values[0] for i in range(len(self.stats))]
        else:
            self.status_vals = [self.critical_values[1] > self.stats[i] > self.critical_values[0] for i in range(len(self.stats))]

    def __repr__(self):
        return f"VarianceRatioTestReport({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status_vals={self.status_vals}, " \
               f"sig_level={self.sig_level}, " \
               f"s_vals={self.s_vals}, " \
               f"stats={self.stats}, " \
               f"p_vals={self.p_vals}, " \
               f"hyp_type={self.hyp_type.value}, " \
               f"hyp_test_type={self.hyp_test_type.value}, " \
               f"critical_values={self.critical_values}"

    def _header(self, tablefmt):
        header = [["Hypothesis Type", self.hyp_type], 
                  ["Hypothesis Test Type", self.hyp_test_type], 
                  ["Significance", f"{int(100.0*self.sig_level)}%"]]
        if self.critical_values[0] is not None:
            header.append(["Lower Critical Value", format(self.critical_values[0], '1.3f')])
        if self.critical_values[1] is not None:
            header.append(["Upper Critical Value", format(self.critical_values[1], '1.3f')])
        return tabulate(header, tablefmt=tablefmt)

    def _results(self, tablefmt):
        status_result = ["Passed" if status else "Failed" for status in self.status_vals]
        s_result = [int(s_val) for s_val in self.s_vals]
        stat_result = [format(stat, '1.3f') for stat in self.stats]
        pval_result = [format(pval, '1.3f') for pval in self.p_vals]
        results = [s_result, stat_result, pval_result, status_result]
        results = numpy.transpose(numpy.array(results))
        return tabulate(results, headers=["s", "Z(s)", "pvalue", "Result"], tablefmt=tablefmt)

    def _table(self, tablefmt):
        header = self._header(tablefmt)
        result = self._results(tablefmt)
        return [header, result]

    def summary(self, tablefmt="fancy_grid"):
        table = self._table(tablefmt)
        print(table[0])
        print(table[1])


class ADFTestReport:
    """
    Formatted text report of ADF test results.

    Properties
    ----------
    stat: float
        ADF test statistic value.
    pval: float
        Test statistic p-value.
    lags: int
        Number of AR(p) lags assumed in solution.
    nobs: int
        Number analyzed data points in performance of test.
    sig_str: list[str]
        Significance level strings ["1%", "5%", "10%"].
    sig: list[float]
        Significance level values [0.01, 0.05, 0.1].
    critical_vals: list[float]
        Value of lower tail critical value used in test for each significance level.
    status_vals: list[bool]
        Test result status values.
    status_str: list[str]
        Test result status as strings.
    """

    def __init__(self, result):
        self.stat = result[0]
        self.pval = result[1]
        self.lags = result[2]
        self.nobs = result[3]
        self.sig_str = ["1%", "5%", "10%"]
        self.sig = [0.01, 0.05, 0.1]
        self.critical_vals = [result[4][sig] for sig in self.sig_str]
        self.status_vals = [self.stat >= val for val in self.critical_vals]
        self.status_str = ["Passed" if status else "Failed" for status in self.status_vals]

    def summary(self, tablefmt="fancy_grid"):
        headers = ["Significance", "Critical Value", "Result"]
        header = [["Test Statistic", self.stat],
                  ["pvalue", self.pval],
                  ["Lags", self.lags],
                  ["Number Obs", self.nobs]]
        results = [[self.sig_str[i], self.critical_vals[i], self.status_str[i]] for i in range(3)]
        print(tabulate(header, tablefmt=tablefmt))
        print(tabulate(results, tablefmt=tablefmt, headers=headers))


class OUEstReport:
    """
    Formatted text report of Ornstein-Uhlenbeck process parameter estimation.
    """

    def __init__(self, result, Δt, x0):
        self.ar_result = result
        self.delta_t = Δt
        self.x0 = x0
        # statsmodels' ARIMA with trend="c" (what arima.ar_offset_fit returns)
        # reports the process mean itself as the constant, while an OLS AR(1)
        # fit reports the intercept μ(1 - φ). The two parameterisations cannot
        # be told apart from the parameter names, so branch on the result type.
        self.mean_parameterised = isinstance(result, (ARIMAResults, ARIMAResultsWrapper))
        self.offset_est = result.params.iloc[0]
        self.offset_error = result.bse.iloc[0]
        self.coeff_est = result.params.iloc[1]
        self.coeff_error = result.bse.iloc[1]
        # the AR innovation variance, not the OU σ² -- sigma2_est() derives that
        self.ar_var_est = result.params.iloc[2]
        self.ar_var_error = result.bse.iloc[2]

    def mu_est(self):
        if self.mean_parameterised:
            return self.offset_est
        return self.offset_est/(1.0 - self.coeff_est)

    def mu_error(self):
        if self.mean_parameterised:
            return self.offset_error
        # d(μ)/d(a) = 1/(1 - φ) and d(μ)/d(φ) = a/(1 - φ)²
        return self.offset_error/(1.0 - self.coeff_est) + \
               self.offset_est*self.coeff_error/(1.0 - self.coeff_est)**2

    def lambda_est(self):
        return -numpy.log(self.coeff_est)/self.delta_t

    def lambda_error(self):
        # an uncertainty is a magnitude; the derivative dλ/dφ is negative
        return abs(self.coeff_error/(self.coeff_est*self.delta_t))

    def sigma2_est(self):
        return 2.0*self.lambda_est()*self.ar_var_est/(1.0 - self.coeff_est**2)

    def sigma2_error(self):
        return 4.0*self.lambda_est()*self.coeff_est*self.ar_var_est*self.coeff_error/(1.0 - self.coeff_est**2)**2 + \
               2.0*self.ar_var_est*self.lambda_error()/(1.0 - self.coeff_est**2) + \
               2.0*self.lambda_est()*self.ar_var_error/(1.0 - self.coeff_est**2)

    def summary(self, tablefmt="fancy_grid"):
        header = [["Δt", self.delta_t],
                  ["X0", self.x0]]
        headers = ["Parameter", "Estimate", "Error"]
        results = [["μ", self.mu_est(), self.mu_error()],
                   ["λ", self.lambda_est(), self.lambda_error()],
                   ["σ2", self.sigma2_est(), self.sigma2_error()]]
        print(tabulate(header, tablefmt=tablefmt))
        print(tabulate(results, tablefmt=tablefmt, headers=headers))


class JohansenTestReport:
    """
    Formatted text report of Johansen cointegration test results.

    Properties
    ----------
    eigen_values: list[float]
        Eigenvalues.
    eigen_vectors: list[float]
        Eigenvectors.
    trace_critical_vals: list[float]
        Critical values used in test.
    trace_statistic: list[bool]
        Test result status values.
    eigen_value_critical_values: list[str]
        Test result status as strings.
    eigen_value_statistic: list[str]
        Test result status as strings.
    """

    def __init__(self, result: JohansenTestResult):
        # eig/evec come back complex128 from numpy>=2 with an identically zero
        # imaginary part; keeping them complex makes tabulate raise ComplexWarning
        # under floatfmt and prints every eigenvector cell with a '+0.j' on it.
        self.eigen_values = numpy.real(result.eig)
        self.eigen_vectors = numpy.real(numpy.array(result.evec))
        self.trace_critical_vals = numpy.array(result.cvt)
        self.trace_statistic = result.lr1
        self.eigen_value_critical_values = numpy.array(result.cvm)
        self.eigen_value_statistic = result.lr2

    def compute_rank(self):
        # One rank per significance level (the columns of the (nvars, 3)
        # critical value matrix), following Johansen's sequential procedure:
        # walk the nulls r<=0, r<=1, ... and stop at the first one not
        # rejected. Iterating the levels -- not the series -- keeps a two
        # variable system from dropping the 99% column and a four variable
        # system from walking off the three column table.
        rejections = numpy.asarray(self.trace_statistic)[:, None] > numpy.asarray(self.trace_critical_vals)
        ranks = []
        for level in range(rejections.shape[1]):
            rank = 0
            for rejected in rejections[:, level]:
                if not rejected:
                    break
                rank += 1
            ranks.append(rank)
        return ranks


    def summary(self, tablefmt="fancy_grid"):
        n = len(self.trace_statistic)
        test_headers = ["Null Hypothesis", "Test Statistic", "Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
        rank_headers = ["Critical Value 90%", "Critical Value 95%", "Critical Value 99%"]
        eigen_headers = ["Eigen Value", "Eigen Vector"]
        null_hypothesis = [f"r <= {i}" for i in range(n)]
        trace_results = [[null_hypothesis[i], self.trace_statistic[i], self.trace_critical_vals[i][0], self.trace_critical_vals[i][1], self.trace_critical_vals[i][2]] for i in range(n)]
        eigen_value_results = [[null_hypothesis[i], self.eigen_value_statistic[i], self.eigen_value_critical_values[i][0], self.eigen_value_critical_values[i][1], self.eigen_value_critical_values[i][2]] for i in range(n)]
        # array2string keeps the bracketed space-separated form but caps the
        # components at 4 decimals instead of numpy's full 8-digit repr
        eigen_values_vectors = [[self.eigen_values[i], numpy.array2string(self.eigen_vectors[:, i], precision=4, suppress_small=True)] for i in range(n)]

        print("Trace Statistic")
        print(tabulate(trace_results, tablefmt=tablefmt, headers=test_headers, floatfmt=".3f"))
        print("\nRank")
        print(tabulate([self.compute_rank()], tablefmt=tablefmt, headers=rank_headers))
        print("\nEigenvalue Statistic")
        print(tabulate(eigen_value_results, tablefmt=tablefmt, headers=test_headers, floatfmt=".3f"))
        print("\nEigenvalues and Eigenvectors")
        print(tabulate(eigen_values_vectors, tablefmt=tablefmt, headers=eigen_headers, floatfmt=".2e"))
