from __future__ import annotations
import numpy as np
from ..utils.math import from_db
from ..utils.constants import C


class EchoGenerator:
    """
    Generates a simulated clutter signal for a moving passive bistatic radar.
    """

    def __init__(
        self,
        fs: float = 100e6,
        f_c: float = 20e9,
        V_b: np.ndarray = None,
        target_rcs_db: float = 0,
        rand_target: bool = True,
        target_position: np.ndarray = None,
        target_limits: np.ndarray = np.array([0, 500, 40, 220]),
        Tx_position: np.ndarray = None,
        Rx_position: np.ndarray = None,
    ):

        self.fs = fs
        self.f_c = f_c
        self.target_rcs_db = target_rcs_db

        if V_b is None:
            self.V_b = np.array([20.0, 10.0])
        else:
            self.V_b = V_b
        if rand_target or target_position is None:
            self.target_position = np.array(
                [
                    np.random.randint(target_limits[0], target_limits[1]),
                    np.random.randint(target_limits[2], target_limits[3]),
                ]
            ).T

        else:
            self.target_position = target_position

        if Tx_position is None:
            self.Tx_position = np.array([0.0, 0.0])
        else:
            self.Tx_position = np.array(Tx_position)

        if Rx_position is None:
            self.Rx_position = np.array([0.0, 0.0])

        else:
            self.Rx_position = np.array(Rx_position)

        target_range = (
            np.linalg.norm(self.target_position - self.Tx_position)
            + np.linalg.norm(self.target_position - self.Rx_position)
            - np.linalg.norm(self.Tx_position - self.Rx_position)
        )

        self.target_sample_delay = target_range * fs / C
        self.target_sample_delay = self.target_sample_delay.astype(int)

    def generate(self, reference_signal: np.ndarray) -> np.ndarray:
        wavelength = C / self.f_c
        N = len(reference_signal)
        # Doppler shift and biestatic velocity calculation
        u_target_to_transmitter = self.Tx_position - self.target_position
        u_target_to_transmitter = u_target_to_transmitter / np.linalg.norm(
            u_target_to_transmitter
        )

        u_target_to_radar = self.Rx_position - self.target_position
        u_target_to_radar = u_target_to_radar / np.linalg.norm(u_target_to_radar)
        v_b = np.dot(u_target_to_transmitter, self.V_b) + np.dot(
            u_target_to_radar, self.V_b
        )
        f_doppler = -v_b / wavelength

        echo = np.zeros_like(reference_signal)
        echo_power = np.sqrt(from_db(self.target_rcs_db))

        echo = (
            np.concatenate(
                (
                    np.zeros(self.target_sample_delay),
                    np.roll(reference_signal, self.target_sample_delay)[
                        self.target_sample_delay :
                    ]
                    * np.exp(
                        1j
                        * 2
                        * np.pi
                        * f_doppler
                        * np.arange(0, N - self.target_sample_delay)
                        / self.fs
                    ),
                )
            )
            * echo_power
        )

        return echo, f_doppler

    def target_position(self) -> np.ndarray:
        return self.target_position, self.target_sample_delay * C / self.fs
