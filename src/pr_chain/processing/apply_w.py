from __future__ import annotations
import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.signal.windows import kaiser


def apply_w(
    reference: np.ndarray,
    beta: float | tuple[float, float] = 14.0,
    freq: bool = False,
    range: bool = False,
) -> np.ndarray:
    """Aplica ventana de Kaiser al vector de referencia para mejorar la resolución en el CAF.

    Parameters
    ----------
    reference : np.ndarray
        Vector de referencia (1D, complejo).
    beta : float
        Parámetro de forma para la ventana de Kaiser.

    Returns
    -------
    np.ndarray
        Vector de referencia con ventana aplicada.
    """
    if isinstance(beta, float):
        beta_freq = beta_range = beta
    elif isinstance(beta, tuple) and len(beta) == 2:
        beta_freq, beta_range = beta
    else:
        raise TypeError("arg must be a float or a tuple of length 2")
    N = len(reference)

    ref_w = reference.copy()

    if freq:
        window_freq = kaiser(N, beta_freq)
        ref_w = ref_w * window_freq
    if range:
        window_range = kaiser(N, beta_range)
        # window_range = fftshift(window_range)
        ref_w = ifft(fft(ref_w) * window_range)

    return ref_w
