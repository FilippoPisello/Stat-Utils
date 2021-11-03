from matplotlib.axes import Axes


def add_symmetric_interval(
    axes: Axes,
    central_value: float,
    variation: float,
    double_variation: bool = True,
    color: str = "red",
) -> None:
    """Add horizontal lines to subplot denoting the interval central_value ± variation.

    Parameters
    ----------
    axes : Axes
        The matplotlib axes object to be modified.
    central_value : float
        The value corresponding to the center of the interval.
    variation : float
        The difference between the central value and the limits.
    double_variation : bool, optional
        If True, the variation is multiplied by two. By default is True.
    color : str, optional
        Information over the lines' color. By default is "red".
    """
    if double_variation:
        variation = variation * 2

    axes.axhline(y=(central_value), color=color, ls="-")
    axes.axhline(y=(central_value + variation), color=color, ls="--")
    axes.axhline(y=(central_value - variation), color=color, ls="--")
