import numpy as np
import matplotlib.pyplot as plt
from .math import to_db


def add_detections(
    fig, detections, matrix_shape, extent, ax=None, marker="o", color="r", markersize=8
):
    """
    Create a brand new figure from an existing imshow plot and add detections.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Original figure.
    detections : tuple
        (rows, cols), both NumPy arrays.
    matrix_shape : tuple
        Shape of the plotted matrix, e.g. (ny, nx).
    extent : list or tuple
        [xmin, xmax, ybottom, ytop]
    ax : matplotlib.axes.Axes, optional
        Original axes. If None, uses fig.axes[0].

    Returns
    -------
    new_fig, new_ax
        A completely new figure and axes.
    """
    if ax is None:
        ax = fig.axes[0]

    if len(ax.images) == 0:
        raise ValueError("The provided axes does not contain an imshow image.")

    rows, cols = detections
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have the same shape.")

    old_im = ax.images[0]
    data = np.asarray(old_im.get_array())

    new_fig, new_ax = plt.subplots(figsize=fig.get_size_inches(), dpi=fig.dpi)

    new_im = new_ax.imshow(
        data,
        cmap=old_im.get_cmap(),
        aspect=ax.get_aspect(),
        extent=extent,
        origin=old_im.origin,
        interpolation=old_im.get_interpolation(),
    )
    new_im.set_clim(*old_im.get_clim())

    # copy axes formatting
    new_ax.set_title(ax.get_title())
    new_ax.set_xlabel(ax.get_xlabel())
    new_ax.set_ylabel(ax.get_ylabel())
    new_ax.set_xlim(ax.get_xlim())
    new_ax.set_ylim(ax.get_ylim())

    # if no detections, just return the copied figure
    if rows.size == 0 and cols.size == 0:
        return new_fig, new_ax

    ny, nx = matrix_shape
    xmin, xmax, ybottom, ytop = extent

    dx = (xmax - xmin) / nx
    dy = (ybottom - ytop) / ny

    x = xmin + (cols + 0.5) * dx
    y = ytop + (rows + 0.5) * dy

    new_ax.plot(
        x, y, linestyle="None", marker=marker, color=color, markersize=markersize
    )

    return new_fig, new_ax


def plot_caf(caf, extent, db=False, figsize=(9, 6), **imshow_kwargs):
    """
    Plot the CAF using imshow and return the figure.

    Parameters
    ----------
    caf : 2D array
        The cross-ambiguity function to plot.
    extent : list or tuple
        Same extent passed to imshow: [xmin, xmax, ybottom, ytop].
    db : bool, optional
        If True, plot the CAF in dB scale.
    figsize : tuple, optional
        Size of the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    **imshow_kwargs : additional keyword arguments for imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the CAF plot.
    """

    fig, ax = plt.subplots(figsize=figsize)
    if "aspect" not in imshow_kwargs:
        imshow_kwargs["aspect"] = "auto"
    if db:
        caf_plot = to_db(np.abs(caf))
        label = "CAF (dB)"
    else:
        caf_plot = np.abs(caf)
        label = "CAF"
    im = ax.imshow(caf_plot, extent=extent, **imshow_kwargs)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Cross-Ambiguity Function")

    return fig, ax
