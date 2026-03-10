from __future__ import annotations

import numpy as np

from ..utils.math import awgn


def apply_noise_and_channel(
    surv: np.ndarray,
    ref: np.ndarray,
    add_noise: bool = False,
    noise_power_db: float = 0.0,
    channel_response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    surv = np.asarray(surv)
    ref = np.asarray(ref)

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
        surv_out = awgn(surv_out, noise_power_db)
        ref

    return surv_out, ref_out
