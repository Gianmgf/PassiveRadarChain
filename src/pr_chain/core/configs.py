from typing import Any
from dataclasses import dataclass, field
import numpy as np


def _normalize_optional_array(
    value: Any, *, ndim: int | None = None
) -> np.ndarray | None:
    """Convert optional sequence-like config values into NumPy arrays."""
    if value is None:
        return None
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected an array with ndim={ndim}. Got ndim={arr.ndim}.")
    return arr


@dataclass
class ChannelConfig:
    enable: bool = False
    add_noise: bool = False
    noise_power_db: float = 0.0
    channel_respone: np.ndarray | None = None


@dataclass
class InputConfig:
    """Configuration for the input signals and top-level acquisition parameters."""

    fs: float = 8e6
    f_c: float = 700e6
    N: int = 500_000
    seed: int | None = None
    use_simulated_data: bool = True

    def __post_init__(self) -> None:
        """Validate input configuration values."""
        if self.fs <= 0:
            raise ValueError(f"fs must be positive. Got {self.fs}.")
        if self.f_c <= 0:
            raise ValueError(f"f_c must be positive. Got {self.f_c}.")
        if self.N <= 0:
            raise ValueError(f"N must be positive. Got {self.N}.")


@dataclass
class ClutterConfig:
    """Configuration mirroring ``ClutterGenerator`` arguments."""

    N_CLUTT: int = 20
    clutter_rcs_min_db: float = 0.0
    clutter_rcs_max_db: float = 0.0
    rand_clutter: bool = True
    clutter_positions: np.ndarray | None = None
    clutter_limits: np.ndarray = field(
        default_factory=lambda: np.array([-10, 500, 5, 150])
    )

    def __post_init__(self) -> None:
        """Normalize clutter geometry arrays."""
        self.clutter_positions = _normalize_optional_array(
            self.clutter_positions, ndim=2
        )
        self.clutter_limits = np.asarray(self.clutter_limits)
        if self.N_CLUTT <= 0:
            raise ValueError(f"N_CLUTT must be positive. Got {self.N_CLUTT}.")


@dataclass
class EchoConfig:
    """Configuration mirroring ``EchoGenerator`` arguments."""

    V_b: np.ndarray = field(default_factory=lambda: np.array([10.0, 100.0]))
    target_rcs_db: float = -3.0
    rand_target: bool = False
    target_position: np.ndarray = field(default_factory=lambda: np.array([20.0, 220.0]))
    target_limits: np.ndarray = field(
        default_factory=lambda: np.array([0, 500, 40, 220])
    )

    def __post_init__(self) -> None:
        """Normalize target and velocity arrays."""
        self.V_b = np.asarray(self.V_b, dtype=float)
        self.target_position = np.asarray(self.target_position, dtype=float)
        self.target_limits = np.asarray(self.target_limits)
        if self.V_b.shape != (2,):
            raise ValueError(f"V_b must have shape (2,). Got {self.V_b.shape}.")
        if self.target_position.shape != (2,):
            raise ValueError(
                f"target_position must have shape (2,). Got {self.target_position.shape}."
            )


@dataclass
class SimulationConfig:
    """Configuration for simulated-data generation and geometry."""

    direct_signal: bool = True
    reference_scale: float = 1.0
    transmitter_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    radar_position: np.ndarray = field(default_factory=lambda: np.array([70.0, 150.0]))
    clutter: ClutterConfig = field(default_factory=ClutterConfig)
    echo: EchoConfig = field(default_factory=EchoConfig)

    def __post_init__(self) -> None:
        """Normalize geometry arrays."""
        self.transmitter_position = np.asarray(self.transmitter_position, dtype=float)
        self.radar_position = np.asarray(self.radar_position, dtype=float)
        if self.transmitter_position.shape != (2,):
            raise ValueError(
                "transmitter_position must have shape (2,). "
                f"Got {self.transmitter_position.shape}."
            )
        if self.radar_position.shape != (2,):
            raise ValueError(
                f"radar_position must have shape (2,). Got {self.radar_position.shape}."
            )


@dataclass
class WindowConfig:
    """Configuration for optional reference windowing before CAF computation."""

    enabled: bool = True
    beta: float | tuple[float, float] = (14.0, 14.0)
    freq: bool = True
    range: bool = False

    def __post_init__(self) -> None:
        """Normalize the window beta parameter after JSON deserialization."""
        if isinstance(self.beta, list):
            if len(self.beta) != 2:
                raise ValueError(
                    "beta list must have length 2 when provided as a sequence."
                )
            self.beta = (float(self.beta[0]), float(self.beta[1]))


@dataclass
class FilterConfig:
    """Configuration for optional clutter filtering on the surveillance channel."""

    enabled: bool = True
    order: int = 30

    def __post_init__(self) -> None:
        """Validate filter settings."""
        if self.order <= 0:
            raise ValueError(f"order must be positive. Got {self.order}.")


@dataclass
class CAFConfig:
    """Configuration for the cross-ambiguity function computation."""

    batch: int = 200

    def __post_init__(self) -> None:
        """Validate CAF settings."""
        if self.batch <= 0:
            raise ValueError(f"batch must be positive. Got {self.batch}.")


@dataclass
class CFARConfig:
    """Configuration for CA-CFAR detection on the CAF magnitude."""

    enabled: bool = True
    bidimensional: bool = False
    Nw: int = 512
    Ng: int = 8
    P_fa: float = 1e-6
    return_intermediate: bool = True

    def __post_init__(self) -> None:
        """Validate CFAR settings."""
        if self.Nw <= 0:
            raise ValueError(f"Nw must be positive. Got {self.Nw}.")
        if self.Ng < 0:
            raise ValueError(f"Ng must be non-negative. Got {self.Ng}.")
        if not (0.0 < self.P_fa < 1.0):
            raise ValueError(f"P_fa must be in (0, 1). Got {self.P_fa}.")


@dataclass
class PlotConfig:
    """Configuration for visualization helpers."""

    show: bool = False
    save: bool = False
    db: bool = True
    figsize: tuple[float, float] = (9, 6)
    cmap: str = "viridis"
    aspect: str = "auto"
    xlim: tuple[float, float] | None = (-10.0, 10.0)
    ylim: tuple[float, float] | None = (1000.0, 0.0)
    marker: str = "o"
    color: str = "r"
    markersize: int = 8

    def __post_init__(self) -> None:
        """Normalize tuple-like plot configuration values after deserialization."""
        if isinstance(self.figsize, list):
            self.figsize = tuple(self.figsize)
        if isinstance(self.xlim, list):
            self.xlim = tuple(self.xlim)
        if isinstance(self.ylim, list):
            self.ylim = tuple(self.ylim)


@dataclass
class IOConfig:
    """Configuration for saving configs, states, and figures."""

    output_root: str | None = None
    figure_format: str = "png"


@dataclass
class PassiveRadarChainConfig:
    """Top-level configuration for ``PassiveRadarChain``."""

    input: InputConfig = field(default_factory=InputConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    caf: CAFConfig = field(default_factory=CAFConfig)
    cfar: CFARConfig = field(default_factory=CFARConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    io: IOConfig = field(default_factory=IOConfig)
