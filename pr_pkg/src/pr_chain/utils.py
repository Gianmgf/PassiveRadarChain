import numpy as np

C = 3e8


def to_dB(x):
    return 10 * np.log10(x)


def from_dB(x):
    return 10 ** (x / 10)


def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
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
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = from_dB(snr_db)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise
