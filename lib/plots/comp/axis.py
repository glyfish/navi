from typing import Any

import numpy
from enum import Enum

class PlotType(Enum):
    """
    Supported plot types supported

    Values
    ------
    LINEAR : 1
        Linear x, y plot axis.
    LOG : 2
        Log scale on both x and y axis.
    XLOG : 3
        Log scale on x axis and linear scale on y axis.
    YLOG : 4
        Log scale on y axis and linear scale on x axis.
    """
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

def _style_spine(axis, which: str) -> None:
    """Colour the spine this axis draws on, without erasing the one beneath it.

    Two things go wrong if the spine is named literally, as this module used to
    name ``left``/``bottom``:

    1. A ``twinx`` axes puts its y axis on the RIGHT (``twiny`` its x axis on the
       TOP), so the grey went to a spine hidden behind the original axes and the
       visible one kept the style's ``axes.edgecolor: white`` -- grey tick marks
       with no line under them.
    2. The twin is drawn *after* the axes it overlays, at the same zorder, so its
       unused spine -- white -- paints over the original's grey one and that axis
       disappears. Colouring both sides is not the answer either: that draws a
       spine on the empty side of every ordinary plot, which the style leaves
       blank on purpose. Hiding the unused side is, since it lets whatever is
       underneath show through.
    """
    sides = ("left", "right") if which == "y" else ("bottom", "top")
    position = (axis.yaxis if which == "y" else axis.xaxis).get_ticks_position()
    drawn = sides[1] if position == sides[1] else sides[0]
    axis.spines[drawn].set_color("#b0b0b0")
    if drawn == sides[1]:
        # A twin. Leave the far side transparent rather than white.
        axis.spines[sides[0]].set_visible(False)


def logStyle(axis, x: Any, y: Any):
    """
    Configure axis for log x and y axis.

    Parameters
    ----------
    x : numpy.ndarray
        x axis values.
    y : numpy.ndarray
        y axis values
    """
    minx = min(x) if min(x) > 0.0 else 1.0
    if numpy.log10(max(x)/minx) < 4:
        axis.tick_params(axis='both', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='both', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        # `tick_params` above colours ticks on both sides of each axis already;
        # only the spine needs to know which side this axes actually draws on.
        _style_spine(axis, "x")
        _style_spine(axis, "y")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logXStyle(axis, x: Any, y: Any):
    """
    Configure axis for log x and linear y axis.

    Parameters
    ----------
    x : numpy.ndarray
        x axis values.
    y : numpy.ndarray
        y axis values
    """
    minx = min(x) if min(x) > 0.0 else 1.0
    if numpy.log10(max(x)/minx) < 4:
        axis.tick_params(axis='x', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='x', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        _style_spine(axis, "x")
        axis.set_xlim([min(x)/1.5, 1.5*max(x)])

def logYStyle(axis, x: Any, y: Any):
    """
    Configure axis for linear x and log y axis.

    Parameters
    ----------
    x : numpy.ndarray
        x axis values.
    y : numpy.ndarray
        y axis values
    """
    if numpy.log10(max(y)/min(y)) < 4:
        axis.tick_params(axis='y', which='minor', length=8, color="#b0b0b0", direction="in")
        axis.tick_params(axis='y', which='major', length=15, color="#b0b0b0", direction="in", pad=10)
        _style_spine(axis, "y")
