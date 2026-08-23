from typing import Any
import numpy
from numpy.typing import NDArray
from matplotlib import pyplot

from lib.utils import get_param_default_if_missing
from lib.plots import comp

def contour(f: NDArray[numpy.floating[Any]], x: NDArray[numpy.floating[Any]], y: NDArray[numpy.floating[Any]], values: NDArray[numpy.floating[Any]], **kwargs):
    """
    Contour plot for f(x,y)

    Parameters
    ----------
    y : NDArray[numpy.floating[Any]]
        Value plotted on y-axis.
    x : NDArray[numpy.floating[Any]]
        Value plotted in x axis
    f : NDArray[numpy.floating[Any]]
        Function contoured.
    values : NDArray[numpy.floating[Any]]
        Values of contours plotted.
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    figsize : (int, int)
        Figure size.
    """

    figsize = get_param_default_if_missing("figsize", (7, 7), **kwargs)

    _, axis = pyplot.subplots(figsize=figsize)
    comp.contour(axis, f, x, y, values, **kwargs)


def contour_hist(samples: NDArray[numpy.floating[Any]],
                 f: NDArray[numpy.floating[Any]],
                 x: NDArray[numpy.floating[Any]], 
                 y: NDArray[numpy.floating[Any]], 
                 values: NDArray[numpy.floating[Any]],
                 **kwargs):
    """
    Overlay data shown in a contour plot with samples shown in a histogram.

    Parameters
    ----------
    samples : NDArray[numpy.floating[Any]]
        Two dimensional array containing sampled data plotted in histogram,
    y : NDArray[numpy.floating[Any]]
        Value plotted on y-axis.
    x : NDArray[numpy.floating[Any]]
        Value plotted in x axis
    f : NDArray[numpy.floating[Any]]
        Function contoured.
    values : NDArray[numpy.floating[Any]]
        Values of contours plotted.
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is 'x')
    ylabel : string, optional
        Plot y-axis label (default is 'y')
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    nbins : int
        Number of bins used in histogram.
    """

    figsize = get_param_default_if_missing("figsize", (9, 7), **kwargs)

    figure, axis = pyplot.subplots(figsize=figsize)
    comp.contour_hist(axis, figure, samples, f, x, y, values, **kwargs)


def colored_scatter(y, x, color_values, **kwargs):
    """
    Make a scatter plot of the x and y data and color the scatter dots with value 
    specified in colors.

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        Plot figure which is needed to add histogram scale.
    y : NDArray[numpy.floating[Any]]
        Data plotted in scatter plot y axis.
    x : NDArray[numpy.floating[Any]]
        Data plotted in scatter plot x axis.
    color_values : NDArray[numpy.floating[Any]]
        Data used to compute scatter point color.
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    ylabel : string, optional
        Plot y-axis label (default is None)
    color_bar_label : str
        Label shown to right of color bar (default None)
    npts : int, optional
        Number of points plotted (default is length of y)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis are labeled using scientific notation. (default (-3, 3))
    plot_axis_type : PlotAxisType
        Plot axis type.
    color_bar_limit : (float, float)
        Color bar limits. (default None)
    legend_loc : string
        Specify legend location. (default best)
    """

    figsize = get_param_default_if_missing("figsize", (9, 7), **kwargs)

    figure, axis = pyplot.subplots(figsize=figsize)
    comp.colored_scatter(axis, figure, y, x, color_values, **kwargs)


def colored_scatter_contour(ydata, xdata, color_values, cont_ydata, cont_xdata, **kwargs):
    """
    Make a scatter plot of the x and y data and color the scatter dots with value 
    specified in colors.

    Parameters
    ----------
    ydata : list[numpy.ndarray]
        Data plotted in scatter plot y axis.
    cont_ydata : list[numpy.ndarray]
        Contour y data
    cont_xdata : list[numpy.ndarray]
        Contour x data
    xdata : list[numpy.ndarray]
        Data plotted in scatter plot x axis.
    color_values : list[numpy.ndarray]
        Data used to compute scatter point color.
    cont_ydata : list[numpy.ndarray]
        Contour y data values
    cont_xdata : list[numpy.ndarray]
        Contour x data values
    title : string, optional
        Plot title (default is None)
    title_offset : float (default is 0.0)
        Plot title off set from top of plot.
    xlabel : string, optional
        Plot x-axis label (default is None)
    ylabel : string, optional
        Plot y-axis label (default is None)
    color_bar_label : str
        Label shown to right of color bar (default None)
    npts : int, optional
        Number of points plotted (default is length of y)
    ylim : (float, float)
        Specify the limits for the y axis. (default None)
    xlim : (float, float)
        Specify the limits for the x axis. (default None)
    scilimits : (-int, int)
        Specify the order where axis are labeled using scientific notation. (default (-3, 3))
    labels : [string], optional
        Curve labels shown in legend.
    plot_axis_type : PlotAxisType
        Plot axis type.
    lw : int, optional
        Plot line width (default is 2)
    color_bar_limit : (float, float)
        Color bar limits. (default None)
    legend_loc : string
        Specify legend location. (default best)
    """

    figsize = get_param_default_if_missing("figsize", (9, 7), **kwargs)

    figure, axis = pyplot.subplots(figsize=figsize)
    comp.colored_scatter_contour(axis, figure, ydata, xdata, color_values, cont_ydata, cont_xdata, **kwargs)
