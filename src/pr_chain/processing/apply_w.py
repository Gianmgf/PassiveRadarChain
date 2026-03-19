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
    """Aplica una ventana de Kaiser sobre la señal de referencia en frecuencia,
    en rango, o en ambos dominios.

    Parameters
    ----------
    reference : np.ndarray
        Vector de referencia unidimensional.
    beta : float | tuple[float, float], optional
        Parámetro de forma de la ventana de Kaiser. Si es un escalar, se usa
        el mismo valor para las ventanas en frecuencia y en rango. Si es una
        tupla de dos elementos, el primer valor se usa para frecuencia y el
        segundo para rango. Por defecto es ``14.0``.
    freq : bool, optional
        Si es ``True``, aplica la ventana de Kaiser directamente en el dominio
        temporal para reducir lóbulos laterales en frecuencia. Por defecto es
        ``False``.
    range : bool, optional
        Si es ``True``, aplica la ventana de Kaiser en el dominio frecuencial
        para modificar la respuesta en rango. Por defecto es ``False``.

    Returns
    -------
    np.ndarray
        Señal de referencia con el ventaneo aplicado.

    Raises
    ------
    TypeError
        Si ``beta`` no es un escalar ni una tupla de longitud 2.
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
        window_range = fftshift(window_range)
        ref_w = ifft(fft(ref_w) * window_range)

    return ref_w
