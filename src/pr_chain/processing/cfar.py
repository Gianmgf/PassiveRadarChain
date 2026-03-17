"""cfar (simplificado)

Contiene únicamente la función usada por la clase pipeline:
- ca_cfar_1d(caf, Nw, Ng, pfa, return_intermediate=False)

Dependencias: numpy, scipy (scipy.ndimage.convolve)
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve, uniform_filter


def ca_cfar_1d(
    caf: np.ndarray, Nw: int, Ng: int, pfa: float, return_intermediate: bool = False
):
    """CA-CFAR 1D aplicado sobre axis=1 si caf es 2D (por fila).

    Parameters
    ----------
    caf : np.ndarray
        Magnitud (no compleja) 1D o 2D.
    Nw : int
        Ventana de referencia a cada lado.
    Ng : int
        Guard cells a cada lado.
    pfa : float
        Probabilidad de falsa alarma.
    return_intermediate : bool
        Si True, devuelve (idx, sigma_est, alpha_det).

    Returns
    -------
    detections : tuple(np.ndarray, np.ndarray) o np.ndarray
        Índices de detección (np.where-like). Para 2D devuelve (rows, cols).
    sigma_est : np.ndarray (opcional)
        Estimación local de ruido.
    alpha_det : float (opcional)
        Factor de escala del umbral.
    """
    x = np.asarray(caf)
    if x.ndim not in (1, 2):
        raise ValueError(f"caf debe ser 1D o 2D. Got ndim={x.ndim}")
    if Nw <= 0 or Ng < 0:
        raise ValueError(f"Nw debe ser >0 y Ng >=0. Got Nw={Nw}, Ng={Ng}")
    if pfa <= 0 or pfa >= 1:
        raise ValueError(f"pfa debe estar en (0,1). Got {pfa}")

    # Construir kernel: referencias (1), guard (0), celda bajo prueba (0)
    # Longitud total = 2*(Nw+Ng)+1
    kernel = np.ones(2 * (Nw + Ng) + 1, dtype=float)
    center = Nw + Ng
    kernel[center - Ng : center + Ng + 1] = 0.0  # guard + CUT
    n_ref = kernel.sum()
    if n_ref == 0:
        raise ValueError("No hay celdas de referencia (revisá Nw/Ng).")

    noise_sum = convolve(x.astype(float), kernel[None, :], mode="constant", cval=0.0)

    sigma_est = noise_sum / n_ref

    # alpha para CA-CFAR (Richards): Pfa = (1 + alpha/N)^(-N) => alpha = N*(Pfa^(-1/N) - 1)
    alpha_det = float(n_ref * (pfa ** (-1.0 / n_ref) - 1.0))
    threshold = sigma_est * alpha_det

    detections = np.where(x > threshold)

    if return_intermediate:
        return detections, sigma_est, alpha_det
    return detections


def ca_cfar_2d(
    caf: np.ndarray, Nw: int, Ng: int, pfa: float, return_intermediate: bool = False
):
    x = np.asarray(caf, dtype=np.float64)  # ideally power already

    total_ng = (2 * Ng + 1) ** 2
    total = (2 * (Nw + Ng) + 1) ** 2
    total_nw = total - total_ng
    result_ng = uniform_filter(x, 2 * Ng + 1, mode="constant", cval=0.0)
    result_nw = uniform_filter(x, 2 * (Nw + Ng) + 1, mode="constant", cval=0.0)
    sigma_est = result_nw * total / total_nw - result_ng * total_ng / total_nw

    alpha_det = total_nw * (pfa ** (-1.0 / total_nw) - 1.0)
    threshold = sigma_est * alpha_det

    detections = np.where(x > threshold)

    if return_intermediate:
        return detections, sigma_est, alpha_det
    return detections
