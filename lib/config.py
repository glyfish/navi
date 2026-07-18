import os
import pathlib
import matplotlib

from importlib import resources
from typing import Literal, Sequence
from matplotlib import colors as mcolors

from IPython.core.display import Image
from cycler import cycler

# Style sheets ship inside the package, so lookup is anchored to the package
# rather than to os.getcwd() and resolves identically from any working directory.
#
# Two display targets, differing only in font sizes:
#   glyfish_style      notebooks and interactive display
#   glyfish_web_style  plots rendered to PNG for embedding in web pages, where
#                      text must stay legible at a fixed image size
def _style(name: str) -> str:
    return str(resources.files("lib") / name)

glyfish_style = _style("gly.fish.mplstyle")
glyfish_web_style = _style("gly.fish-web.mplstyle")

# Output root for figures published as post assets. Relative to the working
# directory, so notebooks write alongside the post they belong to.
plot_asset_path = os.path.join(os.getcwd(), 'plots')

def save_post_asset(figure, post, plot):
    path = os.path.join(plot_asset_path, post, plot) + ".png"
    figure.savefig(path, bbox_inches="tight")

color = mcolors.ColorConverter().to_rgb

ColorKey = Literal["red", "green", "blue", "alpha"]
ColorStops = Sequence[tuple[float, float, float]]

histogram_color_map_cdict: dict[ColorKey, ColorStops] = {
    "red": (
        (0.0, 1.0, 1.0),
        (0.5, 0.19, 0.19),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
    "green": (
        (0.0, 1.0, 1.0),
        (0.5, 0.62, 0.62),
        (0.75, 0.58, 0.58),
        (1.0, 0.91, 0.91),
    ),
    "blue": (
        (0.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}
histogram_color_map = mcolors.LinearSegmentedColormap('HistogramMap', histogram_color_map_cdict)

alternate_color_map_colors = [color('white'), color("#8C35FF"), color("#0067C4"), color("#329EFF"), color("#FF9500"), color("#FFE800")]
alternate_color_map =  mcolors.LinearSegmentedColormap.from_list('AlternateMap', alternate_color_map_colors, N=100 )

contour_color_map = mcolors.ListedColormap(["#0067C4", "#FFE800", "#320075", "#FF9500",
                                                      "#329EFF", "#AC9C00", "#5600C9", "#FFC574",
                                                      "#003B6F", "#FFEB22", "#8C35FF", "#AC6500"])
alternate_contour_color_map = mcolors.ListedColormap(["#003B6F", "#FFEB22", "#FFC574", "#320075"])

distribution_sample_cycler = cycler("color", ["#329EFF", "#320075"])
alternate_cycler = cycler("color", ["#0067C4", "#8C35FF", "#FF9500", "#FFE800", "#329EFF", "#FFC574", "#320075"])
bar_plot_colors = ["#0067C4", "#FF9500", "#320075", "#FFE800", "#329EFF", "#FFC574", "#8C35FF"]
bar_plot_cycler = cycler("color", bar_plot_colors)

class SharedCycler:
    """
    A class to manage a shared cycler instance for consistent styling across plots.
    """

    def __init__(self, cycler_obj):
        """
        Initialize with a cycler object.
        :param cycler_obj: A cycler instance to manage.
        """
        self.cycler_obj = cycler_obj
        self.iterator = iter(self.cycler_obj)

    def get_next(self):
        """
        Retrieve the next style from the cycler.
        :return: A dictionary of properties for the next cycle.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset the iterator if exhausted
            self.iterator = iter(self.cycler_obj)
            return next(self.iterator)

    def reset(self):
        """
        Reset the cycler to the beginning.
        """
        self.iterator = iter(self.cycler_obj)
