from typing import Union

from matplotlib.axes import Axes


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
