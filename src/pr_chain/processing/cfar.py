from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter


def ca_cfar(
    caf: np.ndarray,
    Nw: int,
    Ng: int,
    pfa: float,
    detection_2d: bool = False,
    freq_wrap: bool = False,
):
    """Aplica un detector CA-CFAR sobre una matriz CAF para estimar el umbral
    de detección y localizar las celdas que lo superan.

    Parameters
    ----------
    caf : np.ndarray
        Matriz de entrada sobre la que se realiza la detección.
    Nw : int
        Cantidad de celdas de entrenamiento a cada lado de la celda bajo prueba,
        excluyendo la región de guarda.
    Ng : int
        Cantidad de celdas de guarda a cada lado de la celda bajo prueba.
    pfa : float
        Probabilidad de falsa alarma deseada.
    detection_2d : bool, optional
        Si es ``True``, la estimación del nivel de ruido se utilizando un filtro
        bidimensional. Si es ``False``, se realiza solo a lo largo del
        eje de frecuencia con un filtro unidimensional. Por defecto es ``False``.
    freq_wrap : bool, optional
        Si es ``True``, el eje de frecuencia se trata como periódico durante el
        filtrado de promediado. Si es ``False``, los bordes se completan con
        ceros. Por defecto es ``False``.

    Returns
    -------
    detections : tuple[np.ndarray, np.ndarray]
        Índices de las celdas detectadas, en el formato devuelto por
        ``np.where``.
    sigma_est : np.ndarray
        Estimación local de la potencia de ruido o clutter obtenida a partir de
        las celdas de entrenamiento.
    alpha_det : float
        Factor de escala aplicado sobre ``sigma_est`` para construir el umbral
        de detección a partir de ``pfa``.

    """
    x = np.asarray(caf, dtype=np.float64)  # ideally power already
    if freq_wrap:
        modes = ("constant", "wrap")
    else:
        modes = "constant"
    total_ng = (2 * Ng + 1) ** 2
    total = (2 * (Nw + Ng) + 1) ** 2
    total_nw = total - total_ng
    if detection_2d:
        result_ng = uniform_filter(x, 2 * Ng + 1, mode=modes, cval=0.0)
        result_nw = uniform_filter(x, 2 * (Nw + Ng) + 1, mode=modes, cval=0.0)
    else:
        result_ng = uniform_filter(x, (1, 2 * Ng + 1), mode=modes, cval=0.0)
        result_nw = uniform_filter(x, (1, 2 * (Nw + Ng) + 1), mode=modes, cval=0.0)
    sigma_est = result_nw * total / total_nw - result_ng * total_ng / total_nw

    alpha_det = total_nw * (pfa ** (-1.0 / total_nw) - 1.0)
    threshold = sigma_est * alpha_det

    detections = np.where(x > threshold)

    return detections, sigma_est, alpha_det
