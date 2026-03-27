from __future__ import annotations
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


def plot_caf_cuts(
    caf_state: Any,
    *,
    corte_freq: bool = False,
    corte_range: bool = False,
    f_idx: int,
    r_idx: int,
    sigma_est: np.ndarray | None = None,
    alpha_est: float | None = None,
    en_db: bool = True,
    figsize: tuple[float, float] = (12, 6),
    ax: np.ndarray | None = None,
    plot_sigma: bool = True,
    plot_umbral: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grafica cortes de una CAF en frecuencia y/o en rango.

    Parameters
    ----------
    caf_state : object
        Objeto que debe contener los atributos:
        - caf : np.ndarray de forma (N_range, N_freq)
        - freq_axis : np.ndarray
        - range_axis : np.ndarray
    corte_freq : bool, optional
        Si True, grafica el corte en frecuencia para el índice de rango `r_idx`.
    corte_range : bool, optional
        Si True, grafica el corte en rango para el índice de frecuencia `f_idx`.
    f_idx : int
        Índice de frecuencia (columna) a usar para el corte en rango.
    r_idx : int
        Índice de rango (fila) a usar para el corte en frecuencia.
    sigma_est : np.ndarray | None, optional
        Matriz de estimación de ruido/estadístico base, del mismo tamaño que la CAF.
        Si se pasa, se grafica su corte correspondiente.
    alpha_est : float | None, optional
        Escalar multiplicativo para construir el umbral `sigma_est * alpha_est`.
        Solo se usa si también se pasa `sigma_est`.
    en_db : bool, optional
        Si True, grafica en dB. Si False, grafica en escala lineal.
    figsize : tuple[float, float], optional
        Tamaño de la figura.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura creada.
    ax : np.ndarray
        Arreglo de ejes creados.

    Raises
    ------
    ValueError
        Si ambos flags `corte_freq` y `corte_range` son False.
        Si `sigma_est` no tiene la misma forma que `caf_state.caf`.
        Si `alpha_est` se pasa sin `sigma_est`.
        Si `r_idx` o `f_idx` están fuera de rango.
    """

    if not corte_freq and not corte_range:
        raise ValueError(
            "Debe activarse al menos uno de los flags: 'corte_freq' o 'corte_range'."
        )

    if plot_umbral and (sigma_est is None or alpha_est is None):
        raise ValueError(
            "Para plotear el umbral se deben pasar 'sigma_est' y 'alpha_est'."
        )
    caf = np.asarray(caf_state.caf)
    freq_axis = np.asarray(caf_state.freq_axis)
    range_axis = np.asarray(caf_state.range_axis)

    if caf.ndim != 2:
        raise ValueError("'caf_state.caf' debe ser una matriz 2D.")

    n_range, n_freq = caf.shape

    if not (0 <= r_idx < n_range):
        raise ValueError(
            f"'r_idx'={r_idx} está fuera de rango para caf.shape[0]={n_range}."
        )
    if not (0 <= f_idx < n_freq):
        raise ValueError(
            f"'f_idx'={f_idx} está fuera de rango para caf.shape[1]={n_freq}."
        )

    if sigma_est is not None:
        sigma_est = np.asarray(sigma_est)
        if sigma_est.shape != caf.shape:
            raise ValueError(
                "'sigma_est' debe tener la misma forma que la CAF. "
                f"Se recibió {sigma_est.shape}, esperado {caf.shape}."
            )

    if alpha_est is not None and sigma_est is None:
        raise ValueError(
            "No tiene sentido pasar 'alpha_est' sin pasar también 'sigma_est'."
        )

    def _format_y(x: np.ndarray) -> np.ndarray:
        x = np.abs(np.asarray(x))
        if en_db:
            eps = np.finfo(float).eps
            return 10.0 * np.log10(np.maximum(x, eps))
        return x

    n_plots = int(corte_freq) + int(corte_range)

    if ax is None:
        fig, ax = plt.subplots(1, n_plots, figsize=figsize)
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
    else:
        ax = np.asarray(ax)
        fig = ax.ravel()[0].figure

    i = 0

    if corte_freq:
        freq_cut = _format_y(caf[r_idx, :])

        ax[i].plot(
            freq_axis,
            freq_cut,
            label=f"Corte CAF (pico: {freq_cut[f_idx]:.2f}{' dB' if en_db else ''})",
        )

        if sigma_est is not None:
            if plot_sigma:
                sigma_cut = _format_y(sigma_est[r_idx, :])
                ax[i].plot(freq_axis, sigma_cut, label="Corte de sigma_est")

            if plot_umbral:
                thr_cut = _format_y(sigma_est[r_idx, :] * alpha_est)
                ax[i].plot(freq_axis, thr_cut, label="Umbral sigma_est · alpha_est")

        ax[i].set_title(f"Corte en frecuencia @ bin de rango {r_idx}")
        ax[i].set_xlabel("Frecuencia")
        ax[i].set_ylabel("|CAF| [dB]" if en_db else "|CAF|")
        ax[i].legend()
        i += 1

    if corte_range:
        range_cut = _format_y(caf[:, f_idx])

        ax[i].plot(
            range_axis,
            range_cut,
            label=f"Corte CAF (pico: {range_cut[r_idx]:.2f}{' dB' if en_db else ''})",
        )

        if sigma_est is not None:
            if plot_sigma:
                sigma_cut = _format_y(sigma_est[:, f_idx])
                ax[i].plot(range_axis, sigma_cut, label="Corte de sigma_est")

            if plot_umbral:
                thr_cut = _format_y(sigma_est[:, f_idx] * alpha_est)
                ax[i].plot(range_axis, thr_cut, label="Umbral sigma_est · alpha_est")

        ax[i].set_title(f"Corte en rango @ bin Doppler {f_idx}")
        ax[i].set_xlabel("Rango")
        ax[i].set_ylabel("|CAF| [dB]" if en_db else "|CAF|")
        ax[i].legend()

    fig.tight_layout()
    return fig, ax


def plot_psd(
    x: np.ndarray,
    fs: float,
    *,
    ax: plt.Axes | None = None,
    n_samples: int | None = None,
    NFFT: int = 2**13,
    noverlap: int = 0,
    window: str = "hann",
    label: str | None = None,
    freq_in_khz: bool = True,
    **psd_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grafica la PSD de una señal 1D compleja.

    Parameters
    ----------
    x : np.ndarray
        Señal 1D compleja.
    fs : float
        Frecuencia de muestreo en Hz.
    ax : matplotlib.axes.Axes | None, optional
        Eje sobre el cual graficar. Si es None, se crea una figura nueva.
    n_samples : int | None, optional
        Si se especifica, limita la señal a las primeras `n_samples` muestras.
    NFFT : int, optional
        Cantidad de puntos de FFT usada por `ax.psd`.
    noverlap : int, optional
        Cantidad de muestras de solapamiento entre segmentos.
    window : str, optional
        Tipo de ventana. Soporta: "hann", "hanning", "hamming".
    label : str | None, optional
        Etiqueta de la curva.
    freq_in_khz : bool, optional
        Si True, muestra el eje de frecuencia en kHz. Si False, en Hz.
    **psd_kwargs
        Argumentos extra pasados directamente a `ax.psd`, por ejemplo:
        `color`, `linestyle`, etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura utilizada.
    ax : matplotlib.axes.Axes
        Eje utilizado.

    Raises
    ------
    ValueError
        Si la señal no es 1D compleja, o si la ventana no está soportada.
    """

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"'x' debe ser una señal 1D. Se recibió shape={x.shape}.")

    if not np.iscomplexobj(x):
        raise ValueError("'x' debe ser una señal compleja.")

    if n_samples is not None:
        x = x[: int(n_samples)]

    wname = window.lower()
    if wname in ("hann", "hanning"):
        win = np.hanning(NFFT)
    elif wname == "hamming":
        win = np.hamming(NFFT)
    else:
        raise ValueError("window soportadas: 'hann'/'hanning', 'hamming'")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.psd(
        x,
        NFFT=NFFT,
        Fs=fs,
        window=win,
        noverlap=noverlap,
        label=label,
        **psd_kwargs,
    )

    if freq_in_khz:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / 1e3:.1f}"))
        ax.set_xlabel("Frecuencia [kHz]")
    else:
        ax.set_xlabel("Frecuencia [Hz]")

    ax.set_ylabel("Magnitud [dB]")

    if label is not None:
        ax.legend()

    return fig, ax
