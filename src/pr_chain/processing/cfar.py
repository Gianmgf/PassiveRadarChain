from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter


def _to_2d_tuple(value: int | tuple) -> tuple[int, int]:
    """Convierte un entero o una tupla en una tupla (rango, frecuencia)."""
    if isinstance(value, int):
        return value, value

    if len(value) != 2:
        raise ValueError("El parámetro debe ser un int o una tupla de 2 elementos.")

    return int(value[0]), int(value[1])


def ca_cfar(
    caf: np.ndarray,
    Nw: int | tuple[int, int],
    Ng: int | tuple[int, int],
    pfa: float,
    detection_2d: bool = False,
    freq_wrap: bool = False,
):
    """Aplica un detector CA-CFAR sobre una matriz CAF."""

    """Aplica un detector CA-CFAR sobre una matriz CAF para estimar el umbral
    de detección y localizar las celdas que lo superan.

    Parameters
    ----------
    caf : np.ndarray
        Matriz de entrada sobre la que se realiza la detección.
    Nw : int o tuple
        Cantidad de celdas de entrenamiento a cada lado de la celda bajo prueba,
        excluyendo la región de guarda, , pudiendo las dimensiones en caso de tuple.
    Ng : int o tuple
        Cantidad de celdas de guarda a cada lado de la celda bajo prueba, pudiendo las dimensiones en caso de tuple.
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

    x = np.asarray(caf, dtype=np.float64)

    # Normalización de tamaños
    Nw_r, Nw_f = _to_2d_tuple(Nw)
    Ng_r, Ng_f = _to_2d_tuple(Ng)

    if freq_wrap:
        modes = ("nearest", "wrap")
    else:
        modes = ("nearest", "nearest")

    if detection_2d:
        # Ventana de guarda + CUT
        guard_size = (2 * Ng_r + 1, 2 * Ng_f + 1)

        # Ventana total = entrenamiento + guarda + CUT
        full_size = (2 * (Nw_r + Ng_r) + 1, 2 * (Nw_f + Ng_f) + 1)

        total_ng = guard_size[0] * guard_size[1]
        total = full_size[0] * full_size[1]
        total_nw = total - total_ng

        result_ng = uniform_filter(x, size=guard_size, mode=modes)
        result_nw = uniform_filter(x, size=full_size, mode=modes)

    else:
        # En 1D solo filtramos sobre frecuencia
        guard_size = (1, 2 * Ng_f + 1)
        full_size = (1, 2 * (Nw_f + Ng_f) + 1)

        total_ng = guard_size[1]
        total = full_size[1]
        total_nw = total - total_ng

        result_ng = uniform_filter(x, size=guard_size, mode=modes)
        result_nw = uniform_filter(x, size=full_size, mode=modes)

    sigma_est = result_nw * total / total_nw - result_ng * total_ng / total_nw

    alpha_det = total_nw * (pfa ** (-1.0 / total_nw) - 1.0)
    threshold = sigma_est * alpha_det

    detections = np.where(x > threshold)

    return detections, sigma_est, alpha_det
