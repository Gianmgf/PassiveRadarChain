from __future__ import annotations

import numpy as np

from ..utils.math import awgn


def apply_noise_and_channel(
    surv: np.ndarray,
    ref: np.ndarray,
    add_noise: bool = False,
    noise_on_both_channels: bool = True,
    noise_power_db: float = 0.0,
    channel_response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica una respuesta de canal común y, opcionalmente, ruido AWGN a las
    señales de vigilancia y referencia.

    Parameters
    ----------
    surv : np.ndarray
        Señal del canal de vigilancia.
    ref : np.ndarray
        Señal del canal de referencia.
    add_noise : bool, optional
        Si es ``True``, agrega ruido blanco gaussiano aditivo. Por defecto es ``False``.
    noise_on_both_channels : bool, optional
        Si es ``True``, agrega ruido a ambos canales. Si es ``False``, solo al
        canal de vigilancia. Por defecto es ``True``.
    noise_power_db : float, optional
        Potencia total de ruido en dB. Si el ruido se agrega a ambos canales,
        este valor se reparte entre los dos. Por defecto es ``0.0``.
    channel_response : np.ndarray | None, optional
        Respuesta de canal a aplicar sobre ambas señales. Puede ser ``None``,
        un escalar o un arreglo unidimensional. Si tiene un solo valor, se
        interpreta como una ganancia compleja. Si tiene más de un valor, se
        aplica mediante convolución. Por defecto es ``None``.

    Returns
    -------
    surv_out : np.ndarray
        Señal de vigilancia luego de aplicar canal y/o ruido.
    ref_out : np.ndarray
        Señal de referencia luego de aplicar canal y/o ruido.
    noise_added : None | np.ndarray | tuple[np.ndarray, np.ndarray]
        Ruido agregado durante el procesamiento. Devuelve ``None`` si no se
        agregó ruido, un arreglo si solo se agregó al canal de vigilancia, o
        una tupla ``(noise_ref, noise_surv)`` si se agregó a ambos canales.

    Raises
    ------
    ValueError
        Si ``channel_response`` no es válido, por ejemplo si tiene más de una
        dimensión o está vacío.
    """

    surv = np.asarray(surv)
    ref = np.asarray(ref)

    noise_added = None
    surv_out = surv.astype(np.complex128, copy=True)
    ref_out = ref.astype(np.complex128, copy=True)

    if channel_response is not None:
        h = np.asarray(channel_response, dtype=np.complex128)

        if h.ndim != 1:
            raise ValueError(
                f"channel_response must be None, a scalar, or a 1D array. Got shape {h.shape}."
            )
        if h.size == 0:
            raise ValueError("channel_response must not be empty.")

        if h.size == 1:
            surv_out = surv_out * h[0]
            ref_out = ref_out * h[0]
        else:
            surv_out = np.convolve(surv_out, h, mode="same")
            ref_out = np.convolve(ref_out, h, mode="same")

    if add_noise:
        surv_out, noise_surv = awgn(surv_out, noise_power_db / 2, True)
        noise_added = noise_surv
        if noise_on_both_channels:
            ref_out, noise_ref = awgn(ref_out, noise_power_db / 2, True)
            noise_added = (noise_ref, noise_added)
    return surv_out, ref_out, noise_added
