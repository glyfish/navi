import os
import json
import numpy
from enum import Enum
from typing import Callable, Any, Sequence, TypeVar, overload
from numpy.typing import NDArray
import shortuuid

from pandas import read_csv, DataFrame

_T = TypeVar("_T")
_N = TypeVar("_N", int, float, str, bool)
Ensemble = tuple[NDArray[numpy.floating[Any]], list[NDArray[numpy.floating[Any]]]]

# create_parameter_scan returns one time array and one sample array per scan
# point, so apply_to_parameter_scan receives lists rather than single arrays.
# Both forms are accepted since scan data is sometimes stacked into an ndarray.
ScanTimes = Sequence[NDArray] | NDArray
ScanData = Sequence[NDArray] | NDArray

@overload
def get_param_throw_if_missing(param: str, expected_type: type[_T], **kwargs) -> _T: ...
@overload
def get_param_throw_if_missing(param: str, **kwargs) -> Any: ...

def get_param_throw_if_missing(param: str, expected_type: type | None = None, **kwargs) -> Any:  # noqa: ARG001
    """
    Raise exception if parameter is missing from kwargs.

    Parameters
    ----------
    param: str
        Parameter to type check.
    expected_type: type | None
        Optional type for typed return inference.
    kwargs
        key word arguments

    Raises
    ------
        Exception(param does not match expected type)

    Returns
    -------
        Specified kwargs parameter.
    """
    if param in kwargs:
        return kwargs[param]
    else:
        raise Exception(f"{param} parameter is required")


@overload
def get_param_default_if_missing(param: str, default: None, **kwargs) -> Any: ...
@overload
def get_param_default_if_missing(param: str, default: _T, **kwargs) -> _T: ...

def get_param_default_if_missing(param: str, default: Any = None, **kwargs) -> Any:
    """
    Get parameter from kwargs and return specified default value if it is missing.

    A default of None carries no type information — the real value, when supplied,
    comes from untyped kwargs — so that case returns Any rather than None. Without
    the overload the return type is inferred as exactly None, which both rejects
    the value at typed call sites and makes `if x is not None:` guards unreachable
    to the type checker.

    Parameters
    ----------
    param: str
        Parameter to type check.
    default
        value returned if specified parameter is not in kwargs.
    kwargs
        key word arguments

    Returns
    -------
        Specified kwargs parameter.
    """
    return kwargs[param] if param in kwargs else default


def verify_condition(param, condition: bool, condition_string: str):
    """
    Raise exception if parameter does not satisfy specified condition.

    Parameters
    ----------
    param: str
        Parameter to type check.
    default
        value returned if specified parameter is not in kwargs.
    kwargs
        key word arguments

    Raises
    ------
        Exception(param does satisfy condition)
    """
    if not condition:
        raise Exception(f"{param} should satisfy {condition_string}")


def verify_type(param, expected_type):
    """
    Raise exception if parameter is not specified type.

    Parameters
    ----------
    param
        Parameter to type check.
    expected_type
        Expected tpe

    Raises
    ------
        Exception(param does not match expected type)
    """

    if not isinstance(param, expected_type):
        raise Exception(f"{param} is type {type(param)}. Expected {expected_type}")


def create_space(**kwargs) -> NDArray[numpy.floating[Any]]:
    """
    Create linear space with specified parameters.

    Parameters
    ----------
    npts: float
        number of steps in simulation.
    xmax: float
        Space maximum value.
    xmin: float
        Space minimum value (default 0.0).
    Δx : float
        Space grid spacing (default 1).

    Raises
    ------
        Exception(xmax or npts is required)

    Returns
    -------
    numpy.ndarray[float]
        Linear space.
    """

    npts = get_param_default_if_missing("npts", None, **kwargs)
    xmax = get_param_default_if_missing("xmax", None, **kwargs)
    xmin = get_param_default_if_missing("xmin", 0.0, **kwargs)
    Δx = get_param_default_if_missing("Δx", 1.0, **kwargs)

    if xmax is None and npts is None:
        raise Exception(f"xmax or npts is required")
    if xmax is None:
        assert npts is not None
        xmax = (npts - 1)*Δx + xmin
    elif npts is None:
        npts = int((xmax-xmin)/Δx) + 1

    return numpy.linspace(xmin, xmax, npts)


def create_logspace(**kwargs) -> NDArray[numpy.floating[Any]]:
    """
    Create log space with specified parameters.

    Parameters
    ----------
    npts: float
        number of steps in simulation.
    xmax: int
        Space maximum value.
    xmin: float
        Space minimum value (default 0.0).
    Returns
    -------
    numpy.ndarray[float]
        Linear space.
    """

    npts = get_param_throw_if_missing("npts", **kwargs)
    xmax = get_param_throw_if_missing("xmax", **kwargs)
    xmin = get_param_default_if_missing("xmin", 1.0, **kwargs)
    return numpy.logspace(numpy.log10(xmin), numpy.log10(xmax/xmin), npts)


def create_parameter_scan(source: Callable[..., tuple[NDArray, NDArray]], *args) -> tuple[list[NDArray], list[NDArray]]:
    """
    Generate a parameter scan for the specified data source using the 
    specified parameters

    Parameters
    ----------
    source: lambda(**kwargs) -> (numpy.ndarray, numpy.ndarray)
        lambda calling source create.
    args : *args
        Array of parameter scan kwargs

    Returns
    -------
    (numpy.ndarray[float], list[numpy.ndarray[float]])
        time and ensemble simulation results.
    """

    scan = []
    t_scan=[]
    for kwargs in args:
        t, samples = source(**kwargs)
        scan.append(samples)
        t_scan.append(t)
    return t_scan, scan


def create_ensemble(source: Callable[..., tuple[NDArray, NDArray]], nsim: int, **kwargs) -> Ensemble:
    """
    Generate a parameter scan for the specified data source using the 
    specified parameters

    Parameters
    ----------
    source: lambda(**kwargs) -> (numpy.ndarray, numpy.ndarray)
        lambda calling source create.
    nsim : int
        Number of simulations in ensemble
    kwargs : **kwargs
        Simulation parameters.

    Returns
    -------
    (numpy.ndarray[float], list[numpy.ndarray[float]])
        time and ensemble simulation results.
    """

    ensemble = []
    t = numpy.array([])
    for _ in range(nsim):
        t, samples = source(**kwargs)
        ensemble.append(samples)
    return t, ensemble


def apply_to_ensemble(func, t: NDArray, ensemble: list[NDArray], **kwargs) -> tuple[NDArray, NDArray]:
    """
    Apply specified function to an ensemble.
    
    Parameters
    ----------
    func: lambda(**kwargs) -> result
        lambda calling source create.
    t: numpy.ndarray[float]
        Time
    ensemble: list[numpy.ndarray[float]]
        Ensemble data
    kwargs : **kwargs
       Function parameters.

    Returns
    -------
    time, list[results]
        List of function results.
    """

    result = [func(t, data, **kwargs) for data in ensemble]
    return result[0][0], numpy.array([data[1] for data in result])


def apply_to_parameter_scan(func: Callable[..., tuple[NDArray, NDArray]],
                            t: ScanTimes,
                            scan: ScanData,
                            **kwargs) -> tuple[NDArray, NDArray]:
    """
    Apply specified function to results of a parameter scan.

    Parameters
    ----------
    func: lambda(t, data, **kwargs) -> (numpy.ndarray, numpy.ndarray)
        lambda applied to each scan point. Receives t unchanged for every point,
        so a func that varies with time should index it itself.
    t: list[numpy.ndarray[float]] or numpy.ndarray[float]
        Time, as returned by create_parameter_scan (one array per scan point).
    scan : list[numpy.ndarray[float]] or numpy.ndarray[float]
        Parameter scan data, as returned by create_parameter_scan.
    kwargs : **kwargs
       Function parameters.

    Returns
    -------
    time, list[results]
        List of function results.
    """

    result = [func(t, data, **kwargs) for data in scan]
    return result[0][0], numpy.array([data[1] for data in result])


def get_s_vals(**kwargs) -> NDArray:
    """
    Compute lags for variance ratio test using provided parameters.

    Parameters
    ----------
    linear: bool
        If true s values are generated on a linear scale. If false they are 
        generated on a logarithmic scale. (default False)
    smin: int
        Minimum lag used in scan.
    smax: int
        Maximum lag used in scan.
    npts: int
        Number of points in scan
    svals: list[int]
        Specify lags used in scan.

    Returns
    -------
    list[int]
        s values used in scan.
    """

    linear = get_param_default_if_missing("linear", False, **kwargs)
    smin = get_param_default_if_missing("smin", 1.0, **kwargs)
    smax = get_param_default_if_missing("smax", None, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    svals = get_param_default_if_missing("svals", None, **kwargs)
    if npts is not None and smax is not None:
        if linear:
            return create_space(npts=npts, xmax=smax, xmin=smin)
        else:
            return create_logspace(npts=npts, xmax=smax, xmin=smin)
    elif svals is not None:
        return svals
    else:
        raise Exception(f"smax and npts or svals is required")
    

def extract_date_range(date: NDArray, data: NDArray, start_date: str, end_date: str) -> tuple[NDArray, NDArray]:
    """
    Extract data from specified start date to specified end date.

    Parameters
    ----------
    date: numpy.ndarray[numpy.datetime64]
        Date array.
    data: numpy.ndarray[float]
        Data to extract from.
    start_date: str
        Start date.
    end_date: str
        End date.

    Returns
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[numpy.datetime64]]
        Extracted data.
    """

    start_index = numpy.where(date >= numpy.datetime64(start_date))[0][0]
    end_index = numpy.where(date == numpy.datetime64(end_date))[0][-1]
    return date[start_index: end_index], data[start_index: end_index]


def read_backtrader_data(file_path: str) -> DataFrame:
    """
    Read a backtrader back test output file at the specified path

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    Pandas DataFrame
        Backtrader output data.        
    """

    data = read_csv(file_path, index_col=0, parse_dates=["datetime"], date_format='%Y-%m-%d %H:%M:%S.%f')
    data.fillna(0.0, inplace=True)
    return data


def read_yahoo_data(file_path: str) -> DataFrame:
    """
    Read a yahoo quote CSV file at the specified path

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    Pandas DataFrame
        Yahoo quote data.
    """
    
    return read_csv(file_path, index_col=0, parse_dates=['Date']).sort_values(by='Date').dropna()


def generate_plot_file_name(file_name: str, 
                            path="./plots", 
                            extension: str = "png", 
                            uuid: str | None= None) -> str:
    """
    Generate a file name with the specified prefix, suffix and extension.

    Parameters
    ----------
    prefix: str
        File name prefix.
    suffix: str
        File name suffix.
    extension: str
        File name extension (default "csv").

    Returns
    -------
    str
        Generated file name.
    """
    
    full_path = os.path.join(path, file_name)
    uuid = uuid if uuid else shortuuid.uuid()
    return f"{full_path}-{uuid}.{extension}"


def print_json_vertical(obj: Any, indent: int = 2) -> None:
    """
    Pretty-print a JSON-serializable object with indentation.
    Helpful when inspecting raw API payloads in notebooks or logs.
    """

    print(json.dumps(obj, indent=indent, sort_keys=True))