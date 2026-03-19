from __future__ import annotations
import numpy as np
from .. import utils
from scipy import fft

C = utils.constants.C


def compute_caf(batch, fs, surveillance, reference):
    """Calcula la función de ambigüedad cruzada (CAF) entre las señales de
    vigilancia y referencia usando un algoritmo de procesamiento por bloques.

    Parameters
    ----------
    batch : int
        Tamaño de bloque utilizado para segmentar las señales. Determina la
        cantidad de bins en rango de la CAF.
    fs : float
        Frecuencia de muestreo de las señales.
    surveillance : np.ndarray
        Señal del canal de vigilancia.
    reference : np.ndarray
        Señal del canal de referencia.

    Returns
    -------
    caf : np.ndarray
        Matriz de la función de ambigüedad cruzada. Solo se devuelven los bins
        de rango no negativos.
    freq_axis : np.ndarray
        Eje de frecuencia Doppler asociado a las columnas de la CAF.
    range_axis : np.ndarray
        Eje de rango asociado a las filas de la CAF.

    Raises
    ------
    ValueError
        Si las señales no tienen la misma longitud, si ``batch`` no es un entero
        positivo, si ``fs`` no es positivo, o si la longitud de señal es menor
        que ``batch``.
    """
    if len(surveillance) != len(reference):
        raise ValueError(
            f"Surveillance and reference signals must have the same length. "
            f"Got {len(surveillance)} and {len(reference)}."
        )

    if not isinstance(batch, int) or batch <= 0:
        raise ValueError(f"Batch size must be a positive integer. Got {batch}.")

    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got {fs}.")

    N = len(surveillance)

    M = N // batch

    if M == 0:
        raise ValueError(
            f"Signal length ({N}) is shorter than batch size ({batch}). "
            f"This would result in zero Doppler bins. Use a smaller batch size."
        )

    N_truncated = M * batch
    surveillance = surveillance[:N_truncated]
    reference = reference[:N_truncated]

    surv_blocks = np.reshape(surveillance, (batch, M), order="F")
    ref_blocks = np.reshape(reference, (batch, M), order="F")

    D = fft.ifft(
        fft.fft(surv_blocks, 2 * batch - 1, axis=0)
        * np.conj(fft.fft(ref_blocks, 2 * batch - 1, axis=0)),
        axis=0,
    )
    D = fft.fftshift(D, axes=0)

    caf = fft.fft(D, axis=1)
    caf = fft.fftshift(caf, axes=1)

    freq_axis = fft.fftshift(fft.fftfreq(M, d=batch / fs))

    range_resolution = C / fs
    range_axis = np.arange(batch) * range_resolution

    # Return only positive range bins (batch-1 to end corresponds to delays >= 0)
    return caf[batch - 1 :, :], freq_axis, range_axis
