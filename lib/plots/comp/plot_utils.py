from typing import Any, cast
"""
Reusable plot components.

Functions
---------
__plot_curve
    Plot x and y data on specified axis type.
__plot_curves
    Plot multiple curves on the same x-scale
"""

import numpy
import pandas
import math
from datetime import datetime, date
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib import rcParams

from lib.config import SharedCycler

from lib.plots.comp.axis import (PlotType, logStyle, logXStyle, logYStyle)
from lib.utils import get_param_default_if_missing


def __plot_curve(axis, x, y, n, shared_cycler, **kwargs):
    lw              = get_param_default_if_missing("lw", 2, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    colors          = get_param_default_if_missing("colors", None, **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-4, 4), **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    xlabel          = get_param_default_if_missing("xlabel", None, **kwargs)
    ylabel          = get_param_default_if_missing("ylabel", None, **kwargs)
    npts            = get_param_default_if_missing("npts", min(len(y), len(x)), **kwargs)

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime) or isinstance(x[0], numpy.datetime64) or isinstance(x[0], date):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter 
    else:
        axis.ticklabel_format(style='sci', axis='x', scilimits=scilimits, useMathText=True)

    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    color = colors[n] if colors is not None else shared_cycler.get_next()['color']

    y_is_log = plot_axis_type.value in (PlotType.YLOG.value, PlotType.LOG.value)
    if not y_is_log:
        axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    x = x[:npts]
    y = y[:npts]

    label = labels[n] if labels is not None and n < len(labels) else None

    if plot_axis_type.value == PlotType.LINEAR.value:
        return axis.plot(x, y, lw=lw, label=label, color=color, zorder=5)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        return axis.semilogy(x, y, lw=lw, label=label, color=color, zorder=5)
    elif plot_axis_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        return axis.semilogx(x, y, lw=lw, label=label, color=color, zorder=5)
    elif plot_axis_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        return axis.loglog(x, y, lw=lw, label=label, color=color, zorder=5)
    else:
        raise Exception("Invalid PlotAxisType")


# Plot curves
def __plot_curves(axis, x, y, **kwargs):
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    legend_loc     = cast(Any, get_param_default_if_missing("legend_loc", "best", **kwargs))
    legend_title   = get_param_default_if_missing("legend_title", None, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)
    color_cycler   = get_param_default_if_missing("color_cycler", None, **kwargs)

    ncurve = len(y)

    for i in range(ncurve):
        xplot = x[i]
        yplot = y[i]

        if npts is None or npts > len(yplot):
            npts = len(yplot)
        
        if not isinstance(xplot, (numpy.ndarray, numpy.generic)):
            raise Exception("x must be a numpy.array")

        if not isinstance(yplot, (numpy.ndarray, numpy.generic)):
            raise Exception("y must be a numpy.array")

        color_cycler = color_cycler if color_cycler else SharedCycler(rcParams['axes.prop_cycle'])
        __plot_curve(axis, xplot, yplot, i, color_cycler, **kwargs)

    if labels is not None:
        ncol = math.ceil(ncurve / 6 )
        axis.legend(loc=legend_loc, ncol=ncol, title=legend_title, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9)).set_zorder(20)


def __plot_symbols(axis, x, y, loc, **kwargs):
    labels         = get_param_default_if_missing("labels", None, **kwargs)
    legend_loc     = cast(Any, get_param_default_if_missing("legend_loc", "best", **kwargs))
    legend_title   = get_param_default_if_missing("legend_title", None, **kwargs)
    markers        = get_param_default_if_missing("markers", None, **kwargs)
    marker_colors  = get_param_default_if_missing("marker_colors", None, **kwargs)
    npts           = get_param_default_if_missing("npts", None, **kwargs)

    ncurve = len(y)

    for i in range(ncurve):
        xplot = x[i]
        yplot = y[i]

        if npts is None or npts > len(yplot):
            npts = len(yplot)
        
        if not isinstance(xplot, (numpy.ndarray, numpy.generic)):
            raise Exception("x must be a numpy.array")

        if not isinstance(yplot, (numpy.ndarray, numpy.generic)):
            raise Exception("y must be a numpy.array")

        kwargs['marker'] = markers[i] if markers is not None else 'o'
        kwargs['color'] = marker_colors[i] if marker_colors is not None else None
        __plot_symbol(axis, xplot, yplot, loc+i, **kwargs)

    if labels is not None:
        ncol = math.ceil(ncurve / 6 )
        axis.legend(loc=legend_loc, ncol=ncol, title=legend_title, bbox_to_anchor=(0.1, 0.1, 0.9, 0.9)).set_zorder(20)


def __plot_symbol(axis, x, y, n, **kwargs):
    lw              = get_param_default_if_missing("lw", 2, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    color           = get_param_default_if_missing("color", None, **kwargs)
    plot_axis_type  = get_param_default_if_missing("plot_axis_type", PlotType.LINEAR, **kwargs)
    scilimits       = get_param_default_if_missing("scilimits", (-4, 4), **kwargs)
    ylim            = get_param_default_if_missing("ylim", None, **kwargs)
    xlim            = get_param_default_if_missing("xlim", None, **kwargs)
    labels          = get_param_default_if_missing("labels", None, **kwargs)
    npts            = get_param_default_if_missing("npts", min(len(y), len(x)), **kwargs)
    marker          = get_param_default_if_missing("marker", 'o', **kwargs)
    marker_size     = get_param_default_if_missing("marker_size", 5.0, **kwargs)
    alpha           = get_param_default_if_missing("alpha", 1.0, **kwargs)  

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime) or isinstance(x[0], numpy.datetime64) or isinstance(x[0], date):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter 
    else:
        axis.ticklabel_format(style='sci', axis='x', scilimits=scilimits, useMathText=True)

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    x = x[:npts]
    y = y[:npts]

    label = labels[n] if labels is not None and n < len(labels) else None

    if plot_axis_type.value == PlotType.LOG.value:
        logStyle(axis, x, y)
        axis.loglog(x, y, marker=marker, markersize=marker_size, linestyle="None", markeredgewidth=1.0, 
                    alpha=alpha, zorder=5, label=label, color=color)
    elif plot_axis_type.value == PlotType.XLOG.value:
        logXStyle(axis, x, y)
        axis.semilogx(x, y, marker=marker, markersize=marker_size, linestyle="None", markeredgewidth=1.0, 
                      alpha=alpha, zorder=5, label=label, color=color)
    elif plot_axis_type.value == PlotType.YLOG.value:
        logYStyle(axis, x, y)
        axis.semilogy(x, y, marker=marker, markersize=marker_size, linestyle="None", markeredgewidth=1.0, 
                      alpha=alpha, zorder=5, label=label, color=color)
    else:
        axis.plot(x, y, marker=marker, markersize=marker_size, linestyle="None", markeredgewidth=1.0, 
                  alpha=alpha, zorder=5, label=label, color=color)


def __plot_bar(axis, x, y, shared_cycler, n, zorder=10, **kwargs):
    alpha            = get_param_default_if_missing("alpha", 1.0, **kwargs)
    border_width     = get_param_default_if_missing("border_width", 1, **kwargs)
    bar_width        = get_param_default_if_missing("bar_width", 1.0, **kwargs)
    labels           = get_param_default_if_missing("labels", None, **kwargs)
    colors           = get_param_default_if_missing("colors", None, **kwargs)
    bar_colors       = get_param_default_if_missing("bar_colors", None, **kwargs)
    scilimits        = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)
    xlabel           = get_param_default_if_missing("xlabel", "x", **kwargs)
    xlabel_rotation  = get_param_default_if_missing("xlabel_rotation", None, **kwargs)
    ylabel           = get_param_default_if_missing("ylabel", "y", **kwargs)
    xlim             = get_param_default_if_missing("xlim", None, **kwargs)
    ylim             = get_param_default_if_missing("ylim", None, **kwargs)

    if x is None:
        x = numpy.linspace(0, len(y) - 1, len(y))

    if isinstance(x[0], pandas.Timestamp) or isinstance(x[0], datetime) or isinstance(x[0], numpy.datetime64) or isinstance(x[0], date):
        converter = mdates.ConciseDateConverter()
        munits.registry[numpy.datetime64] = converter
        munits.registry[date] = converter
        munits.registry[datetime] = converter 
    else:
        axis.ticklabel_format(style='sci', axis='x', scilimits=scilimits, useMathText=True)

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    alpha_value = alpha[n] if isinstance(alpha, list) else alpha        

    if bar_colors is not None:
        color = bar_colors
    else:
        color = colors[n] if colors is not None and len(colors) > n else shared_cycler.get_next()['color']    


    label = labels[n] if labels is not None and len(labels) > n else None
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    if isinstance(x[0], float) or isinstance(x[0], int):
        width = bar_width*(x[1]-x[0])
    else:
        width = bar_width

    if xlabel_rotation is not None:
        axis.tick_params("x", rotation=xlabel_rotation)

    return axis.bar(x, y, align='center', width=width, zorder=zorder, alpha=alpha_value, linewidth=border_width, label=label, color=color)


def __plot_multi_bar(axis, x, y, shared_cycler, zorder=10, **kwargs):
    alpha            = get_param_default_if_missing("alpha", 1.0, **kwargs)
    border_width     = get_param_default_if_missing("border_width", 1, **kwargs)
    bar_width        = get_param_default_if_missing("bar_width", 0.8, **kwargs)
    labels           = get_param_default_if_missing("labels", None, **kwargs)
    colors           = get_param_default_if_missing("colors", None, **kwargs)
    scilimits        = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)
    xlabel           = get_param_default_if_missing("xlabel", "x", **kwargs)
    xlabel_rotation  = get_param_default_if_missing("xlabel_rotation", None, **kwargs)
    ylabel           = get_param_default_if_missing("ylabel", "y", **kwargs)
    xlim             = get_param_default_if_missing("xlim", None, **kwargs)
    ylim             = get_param_default_if_missing("ylim", None, **kwargs)
    legend_loc       = cast(Any, get_param_default_if_missing("legend_loc", "best", **kwargs))

    ndatasets = len(y)
    ncategories = len(x)
    positions = numpy.arange(ncategories)
    slot_width = bar_width / ndatasets

    axis.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    if xlabel_rotation is not None:
        axis.tick_params("x", rotation=xlabel_rotation)

    for i, yi in enumerate(y):
        color = colors[i] if colors is not None and len(colors) > i else shared_cycler.get_next()['color']
        alpha_value = alpha[i] if isinstance(alpha, list) else alpha
        label = labels[i] if labels is not None and len(labels) > i else None
        offsets = positions - bar_width / 2 + slot_width * i + slot_width / 2
        axis.bar(offsets, yi, width=slot_width, align='center', zorder=zorder,
                 alpha=alpha_value, linewidth=border_width, label=label, color=color)

    axis.set_xticks(positions)
    axis.set_xticklabels(x)

    if labels is not None:
        axis.legend(loc=legend_loc)


def __axis_twinx(axis, **kwargs):
    ylabel      = get_param_default_if_missing("ylabel", None, **kwargs)
    scilimits   = get_param_default_if_missing("scilimits", (-3, 3), **kwargs)

    axis2 = axis.twinx()

    if ylabel is not None:
        axis2.set_ylabel(ylabel, rotation=-90, labelpad=20)
    axis2.ticklabel_format(style='sci', axis='y', scilimits=scilimits, useMathText=True)

    axis2.grid(False)

    return axis2


def __fit_ylabels(axes, floor: float = 8.0) -> None:
    """Shrink rotated y labels, but only when they actually collide.

    A stack's y labels are rotated, so their length competes with panel height --
    the one dimension a stack spends as panels multiply. The room a centred label
    has is its panel plus the gap it may bleed into, and a long unit string
    outgrows that: 'Index 1982-1984=100' needs 198px against a 163px budget at
    three panels. It stays invisible while the neighbouring label is short, since
    the overflow lands in empty margin, and collides outright when two long ones
    are adjacent.

    Measured and conditional on purpose. Scaling every label by panel count was
    the obvious move and is wrong: it bypasses the two style sheets that exist to
    set text size per display target, gives one label different sizes in one
    document, and shrinks text hardest exactly where the chart is already busiest.
    This leaves the style's size alone unless labels genuinely overlap.

    Two draws, not a search: text height is near-linear in font size, so one
    measurement gives the scale directly. `floor` stops it shrinking into
    illegibility -- past that the labels still overlap and the caller needs a
    different fix (units belong in the per-panel annotation, which has the full
    panel *width*), but a readable overlap beats unreadable text.
    """
    labels = [a.yaxis.label for a in axes if a.yaxis.label.get_text()]
    if len(labels) < 2:
        return
    figure = axes[0].figure

    # `gap` is the separation aimed for, not merely the absence of overlap:
    # scaling to the exact ratio leaves about a pixel between labels, and three
    # labels a pixel apart read as one unbroken column of text.
    gap = 10.0

    # Up to three corrective passes rather than one calculation. Height is only
    # NEAR-linear in font size -- the font's metrics quantise -- so a single
    # estimate undershoots, measurably: 4.7px where 10 was asked for. Each pass
    # re-measures, so the error cannot accumulate, and most charts exit on the
    # first when nothing collides.
    for _ in range(3):
        figure.canvas.draw()
        boxes = [l.get_window_extent() for l in labels]
        centres = [(b.y0 + b.y1) / 2 for b in boxes]
        # Per PAIR, not against the tallest label. Two labels centred on their
        # panels touch when their MEAN height exceeds the centre-to-centre pitch,
        # so a long unit string next to a short one ('Index 1982-1984=100' beside
        # '%') is harmless however far it overruns its own panel -- the overrun
        # lands in empty margin. Testing the tallest shrinks that case for nothing.
        worst = min(((abs(centres[i] - centres[i + 1]) - gap)
                     / ((boxes[i].height + boxes[i + 1].height) / 2))
                    for i in range(len(boxes) - 1))
        if worst >= 1.0:
            return
        size = labels[0].get_fontsize()
        scaled = max(floor, size * worst)
        if scaled >= size:
            return
        for label in labels:
            label.set_fontsize(scaled)
    figure.canvas.draw()


def __twinx_ticks(axis1, axis2):
    """Put one right-axis tick on every left-axis gridline.

    Converts through each axis's own transform -- left data -> display pixels ->
    right data -- rather than interpolating between the two y-limits. The
    interpolation this replaced was linear, which is only correct when *both*
    axes are linear: position on a log axis goes as the log of the value, while
    ``get_ylim`` and ``get_yticks`` report data space. Measured against the
    gridlines it was meant to meet, the linear form landed 0.00 px out for
    linear/linear and 68-218 px out for every combination involving a log axis.

    The three lines after the transform each remove a visible defect:

    - Filtering to the view range. ``get_yticks`` on a log axis returns decades
      *outside* the limits (``[100, 1000, 10000, 100000]`` for a view of 10**3 to
      10**4), and mapping those puts ticks off the end of the right spine.
    - ``NullLocator`` on the minors. Setting only the major locator left the
      right axis with two competing tick systems: these majors, plus the log
      scale's own labelled sub-decade minors, drawn to different rules.
    - A formatter, but only on a log axis. Aligned positions are arbitrary
      values, so a log formatter spells them ``3.76317 x 10^0`` and significant
      figures read better. A linear axis keeps the formatter it has, because
      that is the one carrying the caller's ``scilimits``. See
      ``__twinx_formatter``.

    The tick count is inherited from ``axis1`` by construction -- one tick per
    gridline is the whole point -- so a left axis showing two decades gives the
    right axis two ticks.
    """
    # Sorted, as axis2's limits are two lines below: an inverted axis reports
    # ``(top, bottom)``, and an unsorted ``lo <= t <= hi`` then matches nothing,
    # so every tick is filtered out and the axes are left unaligned.
    lo, hi = sorted(axis1.get_ylim())
    ticks = [t for t in axis1.get_yticks() if lo <= t <= hi]
    if not ticks:
        return
    display = axis1.transData.transform([(0.0, t) for t in ticks])
    right = axis2.transData.inverted().transform(display)[:, 1]
    # The round trip overshoots the end limits by ~1e-15, and matplotlib drops a
    # tick that sits outside the view by any margin at all.
    right = numpy.clip(right, *sorted(axis2.get_ylim()))
    axis2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(right))
    axis2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    formatter = __twinx_formatter(axis2, right)
    if formatter is not None:
        axis2.yaxis.set_major_formatter(formatter)


def __twinx_formatter(axis2, ticks):
    """Pick a formatter for the aligned ticks, or ``None`` to keep the axis's own.

    A LINEAR axis keeps what it has, which is the ``ScalarFormatter`` that every
    twinx entry point configures from the caller's ``scilimits``. Overriding it
    was a regression, and a quiet one, because it applied to the right axis
    alone: the left kept its shared exponent while the right spelled the same
    magnitude in full, so one chart carried two notations (left ``2.1 x10^4``,
    right ``21000``) from GDP magnitudes upward. Worse, full-digit labels grow
    without bound -- at FRED dollar scale they reach past the figure edge and
    push the right y label off a canvas saved without ``bbox_inches="tight"``,
    which is how ``dash`` saves, so the units vanish from the image.

    The duplicate labels that override was written to prevent do not occur:
    ``ScalarFormatter`` renders a normalised index of 0.999 to 1.001 as nine
    distinct labels, because it factors out an offset rather than printing
    decimals.

    A LOG axis does need one. Its own formatter assumes ticks sit on decades and
    spells an aligned position ``3.76317 x 10^0``. Significant figures read far
    better, but a fixed three of them is not always enough: ``%g`` switches to
    exponential once the exponent reaches the precision, which coarsens the
    granularity to 10 across [1000, 10000) and renders neighbouring ticks of a
    sub-percent view identically. So the precision is raised until the labels
    this call will actually draw are distinct, rather than assumed.
    """
    if axis2.get_yscale() == "linear":
        return None
    precision = 3
    while precision < 8 and len({f"{t:.{precision}g}" for t in ticks}) < len(ticks):
        precision += 1
    return matplotlib.ticker.FuncFormatter(
        lambda v, _pos, p=precision: f"{v:.{p}g}")

