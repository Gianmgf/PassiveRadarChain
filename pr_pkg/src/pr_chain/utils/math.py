import numpy as np


def to_db(x):
    return 10 * np.log10(x)


def from_db(x):
    return 10 ** (x / 10)


def awgn(
    signal: np.ndarray, noise_power_db: float, return_noise: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Agrega ruido blanco gaussiano a una señal dada un SNR en dB.

    Parameters
    ----------
    signal : np.ndarray
        Señal original (compleja o real).
    snr_db : float
        Relación señal a ruido deseada en decibelios.

    Returns
    -------
    np.ndarray
        Señal con ruido agregado.
    """

    noise_power = from_db(noise_power_db)

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    if return_noise:
        return signal + noise, noise
    else:
        return signal + noise
