from __future__ import annotations
import numpy as np
from ..utils import from_dB, C


class ClutterGenerator:
    """
    Generates a simulated clutter signal for a moving passive bistatic radar.
    """

    def __init__(
        self,
        fs: float = 100e6,
        N_CLUTT: int = 15,
        clutter_rcs_min_db: float = -20,
        clutter_rcs_max_db: float = 1,
        rand_clutter: bool = True,
        clutter_positions: np.ndarray = None,
        clutter_limits: np.ndarray = np.array([0, 500, 40, 220]),
        Tx_position: np.ndarray = None,
        Rx_position: np.ndarray = None,
    ):
        self.fs = fs

        self.N_CLUTT = N_CLUTT
        self.clutter_rcs_min_db = clutter_rcs_min_db
        self.clutter_rcs_max_db = clutter_rcs_max_db

        if rand_clutter or clutter_positions is None:
            self.clutter_positions = np.array(
                [
                    np.random.randint(clutter_limits[0], clutter_limits[1], N_CLUTT),
                    np.random.randint(clutter_limits[2], clutter_limits[3], N_CLUTT),
                ]
            ).T

        else:
            if clutter_positions.shape[0] != self.N_CLUTT:
                raise ValueError(
                    "clutter_positions must match the number clutter objects N_CLUTT."
                )
            self.clutter_positions = clutter_positions

        if Tx_position is None:
            self.Tx_position = np.array([0.0, 0.0])
        else:
            self.Tx_position = np.array(Tx_position)

        if Rx_position is None:
            self.Rx_position = np.array([0.0, 0.0])

        else:
            self.Rx_position = np.array(Rx_position)

        clutter_range = (
            np.linalg.norm(self.clutter_positions - self.Tx_position, axis=1)
            + np.linalg.norm(self.clutter_positions - self.Rx_position, axis=1)
            - np.linalg.norm(self.Tx_position - self.Rx_position)
        )

        self.clutter_sample_delays = clutter_range * fs / C

        self.clutter_sample_delays = self.clutter_sample_delays.astype(int)

    def generate(self, reference_signal: np.ndarray) -> np.ndarray:
        """
        Generate simulated clutter signal.

        Parameters
        ----------
        reference_signal : np.ndarray
            Complex baseband reference signal.
        fs : float
            Sampling frequency in Hz.
        clutter_sample_delays : np.ndarray
            Integer delay samples for each clutter patch.
        cos_clutter : np.ndarray
            Cosine of the angle for each clutter patch.

        Returns
        -------
        clutt : np.ndarray
            Complex baseband clutter signal.
        """
        clutter_rcs = np.sqrt(
            from_dB(
                np.random.uniform(
                    self.clutter_rcs_min_db, self.clutter_rcs_max_db, self.N_CLUTT
                )
            )
        )

        ref = reference_signal.copy()
        clutter = np.zeros_like(ref)
        for i in range(self.N_CLUTT):
            clutter += (
                np.concatenate(
                    (
                        np.zeros(self.clutter_sample_delays[i]),
                        np.roll(ref, self.clutter_sample_delays[i])[
                            self.clutter_sample_delays[i] :
                        ],
                    )
                )
                * clutter_rcs[i]
            )

        return clutter
