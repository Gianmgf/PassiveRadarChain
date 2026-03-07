from __future__ import annotations
import numpy as np
from scipy.signal.windows import kaiser


def apply_w(reference: np.ndarray, beta: float = 14) -> np.ndarray:
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
    N = len(reference)
    window = kaiser(N, beta)
    return reference * window
