from typing import Union

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_series(
    series: pd.Series,
    figsize: tuple[int, int] = (20, 8),
    title: str = None,
    grid: bool = True,
    return_plot: bool = False,
) -> Union[None, tuple[Figure, Axes]]:
    """Draw a simple scatter representation of a pandas series.

    Parameters
    ----------
    series: pd.Series
        The pandas series to be plotted.
    figsize: tuple(int, int)
        A tuple containing the dimensions of the plot.
    title: str
        The plot's title.
    grid: bool
        If True, a grid is added to the plot. By default is True.
    return_plot: bool
        If True, a tuple containing the figure and axes is returned and the plot
        is not shown. By default is False.
    """
    fig, ax = plt.subplots(1, 1, figsize=(figsize))

    ax.scatter(series.index, series)

    add_titles_to_subplot(ax, title, "Observation Index", "Observation Value")
    if grid:
        add_grid_to_axes(ax)

    if return_plot:
        return fig, ax
    plt.show()


def add_titles_to_subplot(
    axis: Axes,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    sizes: tuple[int, int, int] = (16, 14, 14),
) -> None:
    """Set title and axes labels for a given Axes matplotlib object.

    Parameters
    ----------
    axis : Axes
        Object the labels are to be applied to.
    title : str, optional
        Title of the graph. If None, title is not modified.
    x_label : str, optional
        Label of the x axis. If None, x axis label is not modified.
    y_label : str, optional
        Label of the y axis. If None, y axis label is not modified.
    sizes : tuple[int, int, int], optional
        Dimensions of the three labels specified with the previous arguments.
        By default (16, 14, 14)
    """
    if title is not None:
        axis.set_title(title, fontsize=sizes[0])
    if x_label is not None:
        axis.set_xlabel(x_label, fontsize=sizes[1])
    if y_label is not None:
        axis.set_ylabel(y_label, fontsize=sizes[2])


def add_grid_to_axes(axes: Union[list[Axes], Axes], line_style: str = "--") -> None:
    """Add grid to to one or more axes of a matplotlib Axes object.

    Parameters
    ----------
    axes : Union[list[Axes], Axes]
        The axes the grid is to be added to. A list of axes object can be passed
        to apply the grid to multiple items.
    line_style : str, optional
        Code for the line style, by default "--".
    """
    if not isinstance(axes, list):
        axes = [axes]
    for ax in axes:
        ax.grid(True, ls=line_style)
