from __future__ import annotations
import numpy as np


def block_lattice_filter(
    surveillance: np.ndarray,
    reference: np.ndarray,
    order: int = 100,
) -> np.ndarray:
    """Block-based lattice filter para suprimir clutter/direct path.

    Parámetros
    ----------
    surveillance : np.ndarray (1D, complejo)
    reference : np.ndarray (1D, complejo)
    order : int
        Orden del filtro (cantidad de etapas).

    Returns
    -------
    np.ndarray
        Señal surveillance filtrada (misma forma que entrada).
    """
    surveillance = np.asarray(surveillance)
    reference = np.asarray(reference)

    if surveillance.shape != reference.shape:
        raise ValueError(
            f"Surveillance y reference deben tener la misma forma. "
            f"Got {surveillance.shape} vs {reference.shape}"
        )
    if surveillance.ndim != 1:
        raise ValueError(f"Se esperan vectores 1D. Got ndim={surveillance.ndim}")
    if order <= 0:
        raise ValueError(f"order debe ser positivo. Got {order}")

    # Errores de predicción forward/backward y error de estimación
    b = reference.astype(np.complex128, copy=True)
    f = reference.astype(np.complex128, copy=True)
    e = surveillance.astype(np.complex128, copy=True)

    for _ in range(order):
        # delay de b una muestra
        b_delayed = np.zeros_like(b)
        b_delayed[1:] = b[:-1]

        # coeficiente de reflexión kappa
        numerator = 2.0 * np.sum(b_delayed * np.conj(f))
        denominator = np.sum(np.abs(f) ** 2 + np.abs(b_delayed) ** 2)
        if denominator == 0:
            break
        kappa = numerator / denominator

        # actualizar errores forward/backward
        b_next = b_delayed - kappa * f
        f_next = f - np.conj(kappa) * b_delayed

        # actualizar error de estimación (cancelación)
        denom_h = np.sum(np.abs(b) ** 2)
        if denom_h == 0:
            break
        h = np.sum(e * np.conj(b)) / denom_h
        e_next = e - h * b

        b, f, e = b_next, f_next, e_next

    return e
