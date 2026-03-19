from __future__ import annotations
import numpy as np
from ..utils.math import from_db
from ..utils.constants import C


class ClutterGenerator:
    """Genera una señal de clutter simulada para un escenario de radar pasivo bistático.

    La clase modela múltiples reflectores estacionarios distribuidos en posiciones
    del plano. A partir de la geometría transmisor–clutter–receptor, calcula el
    retardo bistático de cada reflector y construye la señal de clutter como la
    suma de copias retardadas y escaladas de la señal de referencia.

    Parameters
    ----------
    fs : float, optional
        Frecuencia de muestreo en Hz. Por defecto es ``100e6``.
    N_CLUTT : int, optional
        Cantidad de reflectores de clutter a simular. Por defecto es ``15``.
    clutter_rcs_min_db : float, optional
        Valor mínimo de RCS del clutter en dB. Por defecto es ``-20``.
    clutter_rcs_max_db : float, optional
        Valor máximo de RCS del clutter en dB. Por defecto es ``1``.
    rand_clutter : bool, optional
        Si es ``True``, las posiciones de clutter se generan aleatoriamente
        dentro de ``clutter_limits``. Si es ``False``, se usan las posiciones
        dadas en ``clutter_positions``. Por defecto es ``True``.
    clutter_positions : np.ndarray, optional
        Posiciones de los reflectores de clutter con forma ``(N_CLUTT, 2)``.
        Cada fila representa una posición ``[x, y]``. Solo se usa si
        ``rand_clutter=False``.
    clutter_limits : np.ndarray, optional
        Límites espaciales para la generación aleatoria del clutter en el
        formato ``[xmin, xmax, ymin, ymax]``. Por defecto es
        ``np.array([0, 500, 40, 220])``.
    Tx_position : np.ndarray, optional
        Posición del transmisor en el plano, con formato ``[x, y]``.
        Por defecto es ``[0.0, 0.0]``.
    Rx_position : np.ndarray, optional
        Posición del receptor en el plano, con formato ``[x, y]``.
        Por defecto es ``[0.0, 0.0]``.

    Raises
    ------
    ValueError
        Si ``clutter_positions`` es provisto y su cantidad de filas no coincide
        con ``N_CLUTT``.

    Attributes
    ----------
    clutter_positions : np.ndarray
        Posiciones finales de los reflectores de clutter.
    clutter_sample_delays : np.ndarray
        Retardos enteros, en muestras, asociados a cada reflector de clutter.
    Tx_position : np.ndarray
        Posición del transmisor.
    Rx_position : np.ndarray
        Posición del receptor.
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
        Genera la señal de clutter a partir de una señal de referencia.

        Cada reflector de clutter se modela como una copia retardada y escalada de
        la señal de referencia. La amplitud de cada contribución se obtiene a partir
        de un valor de RCS aleatorio uniforme en dB entre ``clutter_rcs_min_db`` y
        ``clutter_rcs_max_db``.

        Parameters
        ----------
        reference_signal : np.ndarray
            Señal de referencia compleja en banda base.

        Returns
        -------
        np.ndarray
            Señal compleja de clutter con la misma forma que ``reference_signal``.

        Notes
        -----
        Los retardos usados en la generación son enteros, ya que se obtienen
        discretizando el retardo bistático en muestras.
        """
        clutter_rcs = np.sqrt(
            from_db(
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
