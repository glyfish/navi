import numpy
from enum import Enum
from json import dumps

from statsmodels.tsa.vector_ar.var_model import LagOrderResults

def _json_default(o):
    """json.dumps hook: unwrap numpy scalars, then fall back to __dict__.

    numpy.float64 subclasses float so json handles it, but int64/int32/float32
    do not and have no __dict__, so the bare `lambda o: o.__dict__` raised
    AttributeError instead of producing a document.
    """
    if isinstance(o, numpy.generic):
        return o.item()
    return o.__dict__




class EstModel(str, Enum):
    """
    Estimate model.

    Values
    ------
    ARMA
        Assume an ARMA(p,q) model when performing estimate.
    OLS
        Assume a single variable OLS model when performing regression.
    VAR
        ASSUME a VAR(p) model when performing estimate.
    VECM
        Assume a VECM(p) model when performing estimate.
    """

    ARMA = "ARMA"
    OLS = "OLS"
    VAR = "VAR"
    VECM = "VECM"


class ParamEst:
    """
    Model used to store a parameter estimate result.

    Properties
    ----------
    est_id: str
        Estimate ID.
    est: float
        Estimate value.
    err: float
        Estimate error.
    est_label: str
        Estimate label used when display results.
    err_label: str
        Estimate error label used when display results.
    order: int
        Parameter order index.
    row: int
        Parameter row index.
    column: int
        Parameter column index. 
    param_type: str
        Parameter type.
    """

    def __init__(self, est_id: str, est: float, err: float, est_label: str | None, err_label: str | None, 
                 order: int, row: int, column: int, param_type: str):
            self.est = est
            self.err = err
            self.est_label = est_label
            self.err_label = err_label
            self.row = row
            self.column = column
            self.order = order
            self.est_id = est_id
            self.param_type = param_type

    def set_labels(self, est_label, err_label):
        self.est_label = est_label
        self.err_label = err_label

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return dumps(self, indent=indent, default=_json_default)

    def __repr__(self):
        return f"ParamEst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est=({self.est}), " \
               f"err=({self.err}), " \
               f"est_label=({self.est_label}), " \
               f"err_label=({self.err_label}), " \
               f"order=({self.order}), " \
               f"row=({self.row}), " \
               f"column=({self.column}), " \
               f"est_id=({self.est_id}), " \
               f"param_type=({self.param_type})"

    @staticmethod
    def from_dict(data):
        est = data["est"]
        est_label = data["est_label"] if "est_label" in data else None
        err = data["err"]
        err_label = data["err_label"] if "err_label" in data else None
        order = data["order"] if "order" in data else 0
        row = data["row"] if "row" in data else 0
        column = data["column"] if "column" in data else 0
        est_id = data["est_id"]
        param_type = data["param_type"]
        return ParamEst(est_id, est, err, est_label, err_label, order, row, column, param_type)
    

class OLSTransform:
    """
    OLS result transformation.

    Properties
    ----------
    model: str
        Transformation model.
    """

    def __init__(self, param: ParamEst):
        self.param = param

    def to_json(self, pretty: bool=False):
        indent = 4 if pretty else None
        return dumps(self, indent=indent, default=_json_default)

    def __repr__(self):
        return f"OLSTransform({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"param=({self.param})"


class OLSParamType(str, Enum):
    """
    OLS single variable estimate parameter type.

    Values
    ------
    OLS_CONST
        Estimate of constant parameter.
    OLS_R2
        Estimate of R2 parameter.
    OLS_PARAM
        Estimate of slope parameter.
    TRANS_CONST
        Estimate of slope parameter.
    TRANS_PARAM
        Estimate of slope parameter.
    """

    OLS_CONST = "OLS_CONST"
    OLS_R2 = "OLS_R2"
    OLS_PARAM = "OLS_PARAM"
    TRANS_CONST = "TRANS_CONST"
    TRANS_PARAM = "TRANS_PARAM"


class OLSResult:
    """
    OLS estimate result.

    Properties
    ----------
    est_id: EstModel
        Estimation identifier.
    const: ParamEst
        Constant estimate.
    params: list[ParamEst]
        Parameter estimate.
    r2: ParamEst
        Estimate r^2.
    transforms: list[OLSTransform]
        Estimated parameter transformation.
    """

    def __init__(self, est_id: str, const: ParamEst, params: list[ParamEst], r2: float):
        self.est_model = EstModel.OLS
        self.const = const
        self.params = params
        self.r2 = r2
        self.param_transforms: list[OLSTransform] | None = None
        self.const_transform: OLSTransform | None = None
        self.est_id = est_id
        self.model = None

    def __repr__(self):
        return f"OLSResult({self.__props()})"

    def __str__(self):
        return self.__props()
    
    def __props(self):
        return f"est_id={self.est_id}, " \
               f"est_model=({self.est_model}), " \
               f"const=({self.const}), " \
               f"params=({self.params}), "\
               f"r2=({self.r2}), " \
               f"model=({self.model}), " \
               f"const_transform=({self.const_transform}), " \
               f"param_transforms=({self.param_transforms})"
    
    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=_json_default)
        else:
            return dumps(self, default=_json_default)
    
    def set_transforms(self, model: str, param_transforms: list[OLSTransform], const_transform: OLSTransform):
        self.param_transforms = param_transforms
        self.const_transform = const_transform
        self.model = model


class ARMAEstType(str, Enum):
    """
    ARMA model type.

    Values
    ------
    AR
        AR(p) model.
    AR_OFFSET
        AR(p) model with constant offset.
    MA
        MA(q) model.
    MA_OFFSET
        MA(q) model with constant offset.
    """

    AR = "AR"
    AR_OFFSET = "AR_OFFSET"
    MA = "MA"
    MA_OFFSET = "MA_OFFSET"

    def formula(self):
        if self.value == ARMAEstType.AR.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_{t}$"
        elif self.value == ARMAEstType.AR_OFFSET.value:
            return r"$X_t = \sum_{i=1}^p \varphi_i X_{t-i} + \mu^* + \varepsilon_{t}$"
        elif self.value == ARMAEstType.MA.value:
            return r"$X_t = \sum_{i=1}^q \theta_i \varepsilon_{t-i} + \varepsilon_{t}$"
        elif self.value == ARMAEstType.MA_OFFSET.value:
            return r"$X_t = \sum_{i=1}^q \theta_i \varepsilon_{t-i} + \mu^* + \varepsilon_{t}$"
        else:
            raise Exception(f"Estimate type is invalid: {self}")

    def set_param_labels(self, param, i):
        if self.value == ARMAEstType.AR.value or self.value == ARMAEstType.AR_OFFSET.value:
            # \varphi, matching formula()'s "\sum_{i=1}^p \varphi_i X_{t-i}"
            param.set_labels(est_label=rf"$\hat{{\varphi_{{{i}}}}}$",
                             err_label=rf"$\sigma_{{\hat{{\varphi_{{{i}}}}}}}$")
        elif self.value == ARMAEstType.MA.value or self.value == ARMAEstType.MA_OFFSET.value:
            param.set_labels(est_label=rf"$\hat{{\theta_{{{i}}}}}$",
                             err_label=rf"$\sigma_{{\hat{{\theta_{{{i}}}}}}}$")
        else:
            raise Exception(f"Estimate type is invalid: {self}")


class ARMAParamType(str, Enum):
    """
    ARAM estimate parameter type.

    Values
    ------
    ARMA_CONST
        Estimate of constant parameter.
    ARMA_PARAM
        Estimate of R2 parameter.
    ARMA_SIG2
        Estimate of slope parameter.
    """

    ARMA_CONST = "ARMA_CONST"
    ARMA_PARAM = "ARMA_PARAM"
    ARMA_SIG2 = "ARMA_SIG2"
    ARMA_OFFSET = "ARMA_OFFSET"


class ARMAEst:
    """
    ARMA parameter estimate result.

    Properties
    ----------
    est_id : str
        Estimate identifier
    const: ParamEst
        Estimate of model constant parameter.
    params: list[ParamEst]
        Estimate of model Parameters.
    sigma2: ParamEst
        Estimate of variance of model random component.
    arma_est_type: ARMAEstType
        ARMA model estimate type.
    trend: str | None
        The deterministic term the fit actually used, as reported by the
        estimator ('c' when a constant was fitted, 'n' when none was).
    offset: ParamEst | None
        Estimate of the model offset μ*, present only for the estimate types
        whose model declares one. statsmodels reports the process MEAN as its
        constant, so μ* is derived from it: see __offset_estimate.
    """

    def __init__(self, est_id: str, const: ParamEst, params: list[ParamEst], sigma2: ParamEst,
                 arma_est_type: ARMAEstType=ARMAEstType.AR, trend: str | None=None):
        self.est_model = EstModel.ARMA
        self.arma_est_type = arma_est_type
        self.trend = trend
        self.const = const
        self.order = len(params)
        self.params = params
        self.sigma2 = sigma2
        self.est_id = est_id
        self.offset = self.__offset_estimate()
        self.__set_const_labels()
        self.__set_params_labels()
        self.__set_sigma2_labels()

    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=_json_default)
        else:
            return dumps(self, default=_json_default)

    def __repr__(self):
        return f"ARMAEst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est_model=({self.est_model}), " \
               f"arma_est_type=({self.arma_est_type}), " \
               f"est_id={self.est_id}, " \
               f"const=({self.const}), " \
               f"order=({self.order}), " \
               f"params=({self.params}), " \
               f"trend=({self.trend}), " \
               f"offset=({self.offset}), " \
               f"sigma2=({self.sigma2})"

    def __offset_estimate(self):
        """
        Derive the model offset μ* from the fitted constant.

        The estimator reports the process MEAN as its constant. For an MA(q) the
        mean is μ* already, since the offset does not pass through the moving
        average. For an AR(p) the mean is μ*/(1 - Σφ_i), so μ* = μ(1 - Σφ_i) and
        the two differ by a factor that grows without bound as Σφ_i -> 1.

        Returns None for the estimate types whose model carries no offset term.
        """

        if self.arma_est_type not in (ARMAEstType.AR_OFFSET, ARMAEstType.MA_OFFSET):
            return None

        if self.arma_est_type is ARMAEstType.AR_OFFSET:
            φ_sum = sum(p.est for p in self.params)
            est = self.const.est*(1.0 - φ_sum)
            # first order propagation, in magnitudes: dμ*/dμ = (1 - Σφ), dμ*/dφ_i = -μ
            err = abs(1.0 - φ_sum)*self.const.err + abs(self.const.est)*sum(abs(p.err) for p in self.params)
        else:
            est = self.const.est
            err = self.const.err

        return ParamEst(est_id=self.est_id,
                        est=est,
                        err=err,
                        est_label=r"$\hat{\mu^*}$",
                        err_label=r"$\sigma_{\hat{\mu^*}}$",
                        order=0,
                        row=0,
                        column=0,
                        param_type=ARMAParamType.ARMA_OFFSET.value)

    def __set_const_labels(self):
        # the fitted constant is the process MEAN, not the offset -- μ* is
        # reported separately by __offset_estimate
        self.const.set_labels(est_label=r"$\hat{\mu}$",
                              err_label=r"$\sigma_{\hat{\mu}}$")

    def __set_params_labels(self):
        # formula() sums i = 1..p, so the lag-1 coefficient is subscript 1, not 0.
        # Numbering from the position rather than from param.order keeps the labels
        # distinct even if a caller leaves order unset; the estimators in
        # lib/data/impl/arima.py store order = i + 1, so the two agree there.
        for i, param in enumerate(self.params):
            self.arma_est_type.set_param_labels(param, i + 1)

    def __set_sigma2_labels(self):
        self.sigma2.set_labels(est_label=r"$\hat{\sigma^2}$",
                               err_label=r"$\sigma_{\hat{\sigma^2}}$")


class VARParamType(str, Enum):
    """
    VAR estimate parameter type.

    Values
    ------
    VAR_CONST
        Estimate of constant parameter.
    VAR_PARAM
        Estimate of R2 parameter.
    VAR_OMEGA
        Estimate of slope parameter.
    """

    VAR_CONST = "VAR_CONST"
    VAR_PARAM = "VAR_PARAM"
    VAR_OMEGA = "VAR_OMEGA"


class VAREst:
    """
    VAR parameter estimate result.

    Properties
    ----------
    est_model: EstModel
        Model identifier.
    order: int
        Model order
    const: list[ParamEst]
        Estimate of model constant parameter.
    params:  list[ParamEst]
        Estimate of model Parameters.
    omega: list[ParamEst]
        Estimate of covariance matrix of model random component.
    """

    def __init__(self, order: int, const: list[ParamEst], params: list[ParamEst], omega: list[ParamEst]):
        self.est_model = EstModel.VAR
        self.const = const
        self.order = order
        self.params = params
        self.omega = omega

    def __repr__(self):
        return f"VAREst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est_model=({self.est_model}), " \
               f"const=({self.const}), " \
               f"order=({self.order}), " \
               f"params=({self.params}), " \
               f"omega=({self.omega})"

    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=_json_default)
        else:
            return dumps(self, default=_json_default)
        

class VECMParamType(str, Enum):
    """
    VECM estimate parameter type.

    Values
    ------
    VECM_CONST
        Estimate of constant parameter.
    VECM_ALPHA
        Estimation of matrices multiplying lagged differences of endogenous variables.
    VECM_LAMBDA
        Estimation of α matrix in VECM model.
    VECM_BETA
        Estimation of β matrix in VECM model.
    VECM_OMEGA
        Estimate of covariance matrix of model random component.
    """

    VECM_CONST = "VECM_CONST"
    VECM_ALPHA = "VECM_ALPHA"
    VECM_LAMBDA = "VECM_LAMBDA"
    VECM_BETA = "VECM_BETA"
    VECM_OMEGA = "VECM_OMEGA"


class VECMEst:
    """
    VECM parameter estimate result.

    Properties
    ----------
    rank: int
        Model rank.
    order: int
        Model order
    lambda_est: list[ParamEst]
        VECM lambda matrix estimate.
    beta_est: list[ParamEst]
        VECM beta matrix estimate.
    a_est: list[ParamEst]
        Lag term coefficient matrices.
    omega: list[ParamEst]
        Estimate of covariance matrix of model random component.
    """

    def __init__(self, rank: int, order: int, const: list[ParamEst], lambda_est: list[ParamEst], beta_est: list[ParamEst], a_est: list[ParamEst], omega: list[ParamEst]):
        self.est_model = EstModel.VECM
        self.rank = rank
        self.const = const
        self.order = order
        self.lambda_est = lambda_est
        self.beta_est = beta_est
        self.a_est = a_est
        self.omega = omega

    def __repr__(self):
        return f"VECMEst({self.__props()})"

    def __str__(self):
        return self.__props()

    def __props(self):
        return f"est_model=({self.est_model}), " \
                f"const=({self.const}), " \
                f"order=({self.order}), " \
                f"lambda=({self.lambda_est}), " \
                f"beta=({self.beta_est}), " \
                f"A=({self.a_est}), " \
                f"omega=({self.omega})"

    def to_json(self, pretty: bool=False):
        if pretty:
            return dumps(self, indent=3, default=_json_default)
        else:
            return dumps(self, default=_json_default)
        