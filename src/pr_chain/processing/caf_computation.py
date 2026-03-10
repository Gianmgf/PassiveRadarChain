"""caf_computation (simplificado)

Contiene únicamente la función usada por la clase pipeline:
- compute_caf(batch, fs, surveillance, reference)

Dependencias: numpy, scipy (scipy.fft)
"""

from __future__ import annotations
import numpy as np
from .. import utils
from scipy import fft

# Velocidad de la luz [m/s]
C = utils.constants.C


def compute_caf(batch, fs, surveillance, reference):
    """
    Compute Cross Ambiguity Function using batch algorithm.

    This function calculates the cross ambiguity function between two signals by dividing
    them into batches of length 'batch', with approximately zero Doppler shift within each batch,
    and performing frequency-domain processing.

    The batch algorithm provides efficient computation of the range-Doppler map by:
    1. Dividing signals into batches (range bins)
    2. Computing cross-correlation via FFT
    3. Performing Doppler FFT across batches

    If the signal length N is not evenly divisible by batch, the signals are automatically
    truncated to N_truncated = (N // batch) * batch samples.

    Parameters
    ----------
    batch : int
        Number of range bins (batch size). Must be a positive integer.
        This determines the range resolution: Δr = c / fs meters.
    fs : float
        Sampling frequency [Hz]. Must be a positive float.
    surveillance : numpy.ndarray
        Surveillance channel signal (1D array).
    reference : numpy.ndarray
        Reference signal (1D array). Must have the same length as surveillance.

    Returns
    -------
    caf : numpy.ndarray
        2D complex array representing the cross ambiguity function values.
        Shape: (batch × doppler_bins)
        Only returns positive range bins (delays >= 0).
    freq_axis : numpy.ndarray
        Array of Doppler frequency values [Hz] corresponding to the Doppler bins.
        Length: doppler_bins = N_truncated // batch
    range_axis : numpy.ndarray
        Array of bistatic range values [m] corresponding to the range bins.
        Length: batch

    Raises
    ------
    ValueError
        If surveillance and reference signals have different lengths.
        If batch is not a positive integer.
        If fs is not positive.

    Notes
    -----
    - The Doppler resolution is: Δf = fs * batch / N_truncated [Hz]
    - The maximum unambiguous range is: R_max = batch * c / fs [m]
    - The function automatically handles signal truncation when N % batch != 0

    Examples
    --------
    >>> import numpy as np
    >>> from moving_passive_radar.signal_processing.caf import compute_caf
    >>> # Create test signals
    >>> N = 10000
    >>> fs = 2e6  # 2 MHz
    >>> t = np.arange(N) / fs
    >>> surveillance = np.exp(1j * 2 * np.pi * 100 * t)  # 100 Hz Doppler
    >>> reference = np.ones(N, dtype=complex)
    >>> # Compute CAF
    >>> caf, freq_axis, range_axis = compute_caf(100, fs, surveillance, reference)
    >>> print(f"CAF shape: {caf.shape}")
    CAF shape: (100, 100)
    """
    # Input validation
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

    # Truncate to make N divisible by batch
    M = N // batch  # Number of Doppler bins

    if M == 0:
        raise ValueError(
            f"Signal length ({N}) is shorter than batch size ({batch}). "
            f"This would result in zero Doppler bins. Use a smaller batch size."
        )

    N_truncated = M * batch
    surveillance = surveillance[:N_truncated]
    reference = reference[:N_truncated]

    # Reshape into blocks: [batch x M]
    # Using Fortran order to match the batch algorithm convention
    surv_blocks = np.reshape(surveillance, (batch, M), order="F")
    ref_blocks = np.reshape(reference, (batch, M), order="F")

    # Cross-correlation via FFT
    # D[τ, m] = IFFT(FFT(surv) * conj(FFT(ref)))
    # Zero-pad to 2*batch-1 for full cross-correlation
    D = fft.ifft(
        fft.fft(surv_blocks, 2 * batch - 1, axis=0)
        * np.conj(fft.fft(ref_blocks, 2 * batch - 1, axis=0)),
        axis=0,
    )
    D = fft.fftshift(D, axes=0)  # Center zero-delay

    # Doppler FFT (coherent processing across batches)
    caf = fft.fft(D, axis=1)
    caf = fft.fftshift(caf, axes=1)  # Center zero-Doppler

    # Generate frequency axis (Doppler)
    freq_axis = fft.fftshift(fft.fftfreq(M, d=batch / fs))

    # Generate range axis
    range_resolution = C / fs  # [m]
    range_axis = np.arange(batch) * range_resolution  # [m]

    # Return only positive range bins (batch-1 to end corresponds to delays >= 0)
    return caf[batch - 1 :, :], freq_axis, range_axis
