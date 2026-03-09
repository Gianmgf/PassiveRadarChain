from __future__ import annotations

import copy
import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from ..generators import ClutterGenerator, EchoGenerator
from ..processing import apply_w, block_lattice_filter, ca_cfar_1d, compute_caf
from ..utils import add_detections, plot_caf

StageName = Literal["inputs", "window", "filter", "caf", "detect"]


def _jsonify(value: Any) -> Any:
    """Convert nested Python/NumPy objects into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@contextmanager
def _temporary_seed(seed: int | None):
    """Temporarily seed NumPy's global RNG and restore its previous state on exit."""
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _as_complex_1d(signal: np.ndarray | list[complex] | tuple[complex, ...], name: str) -> np.ndarray:
    """Validate and convert an input signal into a 1D complex NumPy array."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array. Got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return np.asarray(arr, dtype=np.complex128)


def _normalize_optional_array(value: Any, *, ndim: int | None = None) -> np.ndarray | None:
    """Convert optional sequence-like config values into NumPy arrays."""
    if value is None:
        return None
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected an array with ndim={ndim}. Got ndim={arr.ndim}.")
    return arr


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
        self.clutter_positions = _normalize_optional_array(self.clutter_positions, ndim=2)
        self.clutter_limits = np.asarray(self.clutter_limits)
        if self.N_CLUTT <= 0:
            raise ValueError(f"N_CLUTT must be positive. Got {self.N_CLUTT}.")


@dataclass
class EchoConfig:
    """Configuration mirroring ``EchoGenerator`` arguments."""

    V_b: np.ndarray = field(default_factory=lambda: np.array([10.0, 100.0]))
    target_rcs_db: float = -3.0
    add_noise: bool = False
    noise_power_db: float = 10.0
    rand_target: bool = False
    target_position: np.ndarray = field(default_factory=lambda: np.array([20.0, 220.0]))
    target_limits: np.ndarray = field(default_factory=lambda: np.array([0, 500, 40, 220]))

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

    reference_scale: float = 1.0
    transmitter_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
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
                raise ValueError("beta list must have length 2 when provided as a sequence.")
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
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    caf: CAFConfig = field(default_factory=CAFConfig)
    cfar: CFARConfig = field(default_factory=CFARConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    io: IOConfig = field(default_factory=IOConfig)


@dataclass
class InputState:
    """Runtime state for the input channels."""

    reference: np.ndarray
    surveillance: np.ndarray
    source_mode: Literal["real", "simulated"]
    original_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    """Runtime state for simulated intermediate signals."""

    clutter: np.ndarray | None = None
    echo: np.ndarray | None = None
    doppler_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowState:
    """Runtime state for the windowed reference signal."""

    reference: np.ndarray
    applied: bool
    beta: float | tuple[float, float]
    freq: bool
    range: bool


@dataclass
class FilterState:
    """Runtime state for the surveillance channel after clutter filtering."""

    surveillance: np.ndarray
    applied: bool
    order: int


@dataclass
class CAFState:
    """Runtime state for the most recent CAF computation."""

    caf: np.ndarray
    freq_axis: np.ndarray
    range_axis: np.ndarray
    extent: list[float]
    input_length: int
    truncated_length: int


@dataclass
class DetectionState:
    """Runtime state for the most recent detection output."""

    detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None
    sigma_est: np.ndarray | None = None
    alpha_det: float | None = None


@dataclass
class PipelineState:
    """Aggregate runtime state for the entire processing chain."""

    inputs: InputState | None = None
    simulation: SimulationState | None = None
    window: WindowState | None = None
    filter: FilterState | None = None
    caf: CAFState | None = None
    detection: DetectionState | None = None
    completed_stages: dict[str, bool] = field(
        default_factory=lambda: {
            "inputs": False,
            "window": False,
            "filter": False,
            "caf": False,
            "detect": False,
        }
    )
    stage_snapshots: dict[str, Any] = field(default_factory=dict)
    last_completed_stage: str | None = None


class PassiveRadarChain:
    """Stateful single-channel passive-radar processing chain.

    The chain can either simulate reference/surveillance inputs or accept externally
    provided real data. It caches intermediate results, supports reruns from a chosen
    stage, and wraps the existing ``pr_chain`` modules without modifying them.
    """

    _STAGE_ORDER: tuple[StageName, ...] = ("inputs", "window", "filter", "caf", "detect")

    def __init__(
        self,
        config: PassiveRadarChainConfig | None = None,
        *,
        verbose: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the chain with configuration, state, and logging."""
        self.config = config or PassiveRadarChainConfig()
        self.state = PipelineState()
        self._external_inputs_were_set = False
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._configure_logger(verbose=verbose)
        self.output_root = self._resolve_output_root()
        self._ensure_output_directories()

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        verbose: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> "PassiveRadarChain":
        """Construct a chain instance from a previously saved JSON configuration."""
        config = cls._config_from_json(path)
        return cls(config=config, verbose=verbose, logger=logger)

    def _configure_logger(self, verbose: bool | None = None) -> None:
        """Configure the internal logger if the caller did not configure one already."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        if verbose is None:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _resolve_output_root(self) -> Path:
        """Resolve the package-level ``simulated_data`` directory requested by the user."""
        if self.config.io.output_root is not None:
            return Path(self.config.io.output_root).expanduser().resolve()
        return Path(__file__).resolve().parents[3] / "simulated_data"

    def _ensure_output_directories(self) -> None:
        """Create output directories for configs, states, and figures if needed."""
        for directory in (self.output_root, self.output_root / "configs", self.output_root / "states", self.output_root / "figures"):
            directory.mkdir(parents=True, exist_ok=True)

    def _default_stem(self, prefix: str) -> str:
        """Generate a timestamped default filename stem."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _validate_stage_name(self, stage: str) -> StageName:
        """Validate that a stage name is one of the supported pipeline stages."""
        if stage not in self._STAGE_ORDER:
            raise ValueError(
                f"Invalid stage '{stage}'. Expected one of {self._STAGE_ORDER}."
            )
        return stage  # type: ignore[return-value]

    def _stage_index(self, stage: StageName) -> int:
        """Return the ordinal index of a pipeline stage."""
        return self._STAGE_ORDER.index(stage)

    def _mark_completed(self, stage: StageName) -> None:
        """Mark a stage as completed and all earlier stages as available."""
        for known_stage in self._STAGE_ORDER:
            if self._stage_index(known_stage) <= self._stage_index(stage):
                self.state.completed_stages[known_stage] = True
        self.state.last_completed_stage = stage

    def _store_stage_snapshot(self, stage: StageName) -> None:
        """Store a serializable snapshot of the current config relevant to a stage."""
        self.state.stage_snapshots[stage] = self._stage_snapshot(stage)

    def _stage_snapshot(self, stage: StageName) -> dict[str, Any]:
        """Build a serializable snapshot of the configuration relevant to one stage."""
        if stage == "inputs":
            desired_mode = "simulated" if self.config.input.use_simulated_data else "real"
            return _jsonify(
                {
                    "input": asdict(self.config.input),
                    "simulation": asdict(self.config.simulation),
                    "desired_mode": desired_mode,
                }
            )
        if stage == "window":
            return _jsonify(asdict(self.config.window))
        if stage == "filter":
            return _jsonify(asdict(self.config.filter))
        if stage == "caf":
            return _jsonify(asdict(self.config.caf))
        if stage == "detect":
            return _jsonify(asdict(self.config.cfar))
        raise ValueError(f"Unknown stage '{stage}'.")

    def _auto_invalidate_if_needed(self, stage: StageName) -> None:
        """Invalidate a stage and all downstream stages when its config changed."""
        if stage not in self.state.stage_snapshots:
            return
        current = self._stage_snapshot(stage)
        previous = self.state.stage_snapshots.get(stage)
        if current != previous:
            self.logger.info("Configuration changed for stage '%s'; invalidating downstream cache.", stage)
            self.invalidate_from(stage)

    def invalidate_from(self, stage: str, *, include_stage: bool = True) -> None:
        """Invalidate cached results from a given stage onward."""
        valid_stage = self._validate_stage_name(stage)
        start_index = self._stage_index(valid_stage)
        if not include_stage:
            start_index += 1

        for known_stage in self._STAGE_ORDER[start_index:]:
            if known_stage == "inputs":
                self.state.inputs = None
                self.state.simulation = None
                self._external_inputs_were_set = False
            elif known_stage == "window":
                self.state.window = None
            elif known_stage == "filter":
                self.state.filter = None
            elif known_stage == "caf":
                self.state.caf = None
            elif known_stage == "detect":
                self.state.detection = None
            self.state.completed_stages[known_stage] = False
            self.state.stage_snapshots.pop(known_stage, None)

        remaining_completed = [
            s for s in self._STAGE_ORDER if self.state.completed_stages.get(s, False)
        ]
        self.state.last_completed_stage = remaining_completed[-1] if remaining_completed else None

    def reset(self) -> None:
        """Reset the entire chain state while preserving the current configuration."""
        self.logger.info("Resetting chain state.")
        self.state = PipelineState()
        self._external_inputs_were_set = False

    def update_input_config(self, **kwargs: Any) -> None:
        """Update input configuration values and invalidate the input stage if needed."""
        self._update_dataclass(self.config.input, **kwargs)
        self._auto_invalidate_if_needed("inputs")

    def update_simulation_config(self, **kwargs: Any) -> None:
        """Update simulation configuration values and invalidate the input stage if needed."""
        self._update_dataclass(self.config.simulation, **kwargs)
        self._auto_invalidate_if_needed("inputs")

    def update_clutter_config(self, **kwargs: Any) -> None:
        """Update clutter generator configuration and invalidate the input stage if needed."""
        self._update_dataclass(self.config.simulation.clutter, **kwargs)
        self._auto_invalidate_if_needed("inputs")

    def update_echo_config(self, **kwargs: Any) -> None:
        """Update echo generator configuration and invalidate the input stage if needed."""
        self._update_dataclass(self.config.simulation.echo, **kwargs)
        self._auto_invalidate_if_needed("inputs")

    def update_window_config(self, **kwargs: Any) -> None:
        """Update window configuration and invalidate the window stage if needed."""
        self._update_dataclass(self.config.window, **kwargs)
        self._auto_invalidate_if_needed("window")

    def update_filter_config(self, **kwargs: Any) -> None:
        """Update filter configuration and invalidate the filter stage if needed."""
        self._update_dataclass(self.config.filter, **kwargs)
        self._auto_invalidate_if_needed("filter")

    def update_caf_config(self, **kwargs: Any) -> None:
        """Update CAF configuration and invalidate the CAF stage if needed."""
        self._update_dataclass(self.config.caf, **kwargs)
        self._auto_invalidate_if_needed("caf")

    def update_cfar_config(self, **kwargs: Any) -> None:
        """Update CFAR configuration and invalidate the detection stage if needed."""
        self._update_dataclass(self.config.cfar, **kwargs)
        self._auto_invalidate_if_needed("detect")

    def update_plot_config(self, **kwargs: Any) -> None:
        """Update plotting configuration values."""
        self._update_dataclass(self.config.plot, **kwargs)

    def update_io_config(self, **kwargs: Any) -> None:
        """Update I/O configuration values and refresh the output directories if needed."""
        self._update_dataclass(self.config.io, **kwargs)
        self.output_root = self._resolve_output_root()
        self._ensure_output_directories()

    def _update_dataclass(self, target: Any, **kwargs: Any) -> None:
        """Assign keyword values into a dataclass and rerun its validation hook."""
        for key, value in kwargs.items():
            if not hasattr(target, key):
                raise AttributeError(f"Unknown configuration field '{key}' for {type(target).__name__}.")
            setattr(target, key, value)
        post_init = getattr(target, "__post_init__", None)
        if callable(post_init):
            post_init()

    def set_inputs(
        self,
        reference: np.ndarray | list[complex] | tuple[complex, ...],
        surveillance: np.ndarray | list[complex] | tuple[complex, ...],
        *,
        fs: float | None = None,
        f_c: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set external real-data inputs and invalidate downstream cached results."""
        ref = _as_complex_1d(reference, "reference")
        surv = _as_complex_1d(surveillance, "surveillance")
        if ref.shape != surv.shape:
            raise ValueError(
                f"reference and surveillance must have the same shape. Got {ref.shape} and {surv.shape}."
            )

        if fs is not None:
            self.config.input.fs = fs
        if f_c is not None:
            self.config.input.f_c = f_c
        self.config.input.use_simulated_data = False

        self.state.inputs = InputState(
            reference=ref,
            surveillance=surv,
            source_mode="real",
            original_length=len(ref),
            metadata=metadata or {},
        )
        self.state.simulation = None
        self._external_inputs_were_set = True
        self.invalidate_from("window")
        self.state.completed_stages["inputs"] = True
        self._store_stage_snapshot("inputs")
        self.state.last_completed_stage = "inputs"
        self.logger.info("External inputs set successfully (length=%d).", len(ref))

    def load_inputs(self, path: str | Path) -> None:
        """Load external real-data inputs from an ``.npz`` file.

        The input file must contain ``reference`` and ``surveillance`` arrays. It may
        also contain ``fs`` and ``f_c`` scalar values.
        """
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as npz:
            if "reference" not in npz or "surveillance" not in npz:
                raise KeyError("The input file must contain 'reference' and 'surveillance' arrays.")
            ref = npz["reference"]
            surv = npz["surveillance"]
            fs = float(npz["fs"]) if "fs" in npz else None
            f_c = float(npz["f_c"]) if "f_c" in npz else None
        self.set_inputs(ref, surv, fs=fs, f_c=f_c, metadata={"loaded_from": str(path)})

    def simulate_inputs(self) -> InputState:
        """Simulate reference and surveillance signals using the existing generator classes."""
        self.config.input.use_simulated_data = True
        self._auto_invalidate_if_needed("inputs")
        self.logger.info("Simulating input signals.")

        with _temporary_seed(self.config.input.seed):
            reference = self.config.simulation.reference_scale * (
                np.random.randn(self.config.input.N) + 1j * np.random.randn(self.config.input.N)
            )
            reference = np.asarray(reference, dtype=np.complex128)

            clutter_generator = ClutterGenerator(
                fs=self.config.input.fs,
                N_CLUTT=self.config.simulation.clutter.N_CLUTT,
                clutter_rcs_min_db=self.config.simulation.clutter.clutter_rcs_min_db,
                clutter_rcs_max_db=self.config.simulation.clutter.clutter_rcs_max_db,
                rand_clutter=self.config.simulation.clutter.rand_clutter,
                clutter_positions=self.config.simulation.clutter.clutter_positions,
                clutter_limits=self.config.simulation.clutter.clutter_limits,
                Tx_position=self.config.simulation.transmitter_position,
                Rx_position=self.config.simulation.radar_position,
            )
            echo_generator = EchoGenerator(
                fs=self.config.input.fs,
                f_c=self.config.input.f_c,
                V_b=self.config.simulation.echo.V_b,
                target_rcs_db=self.config.simulation.echo.target_rcs_db,
                add_noise=self.config.simulation.echo.add_noise,
                noise_power_db=self.config.simulation.echo.noise_power_db,
                rand_target=self.config.simulation.echo.rand_target,
                target_position=self.config.simulation.echo.target_position,
                target_limits=self.config.simulation.echo.target_limits,
                Tx_position=self.config.simulation.transmitter_position,
                Rx_position=self.config.simulation.radar_position,
            )

            clutter = clutter_generator.generate(reference)
            echo, doppler_hz = echo_generator.generate(reference)
            surveillance = np.asarray(clutter + echo, dtype=np.complex128)

        self.state.inputs = InputState(
            reference=reference,
            surveillance=surveillance,
            source_mode="simulated",
            original_length=len(reference),
            metadata={"seed": self.config.input.seed},
        )
        self.state.simulation = SimulationState(
            clutter=np.asarray(clutter, dtype=np.complex128),
            echo=np.asarray(echo, dtype=np.complex128),
            doppler_hz=float(doppler_hz),
            metadata={
                "transmitter_position": np.asarray(self.config.simulation.transmitter_position).tolist(),
                "radar_position": np.asarray(self.config.simulation.radar_position).tolist(),
                "target_position": np.asarray(self.config.simulation.echo.target_position).tolist(),
            },
        )
        self._external_inputs_were_set = False
        self.invalidate_from("window")
        self.state.completed_stages["inputs"] = True
        self._store_stage_snapshot("inputs")
        self.state.last_completed_stage = "inputs"
        return self.state.inputs

    def _ensure_inputs_available(self) -> InputState:
        """Ensure that the chain has input signals available for downstream stages."""
        self._auto_invalidate_if_needed("inputs")
        if self.state.inputs is not None:
            return self.state.inputs
        if self.config.input.use_simulated_data:
            return self.simulate_inputs()
        if self._external_inputs_were_set:
            if self.state.inputs is None:
                raise RuntimeError("External inputs were indicated but are not available in state.")
            return self.state.inputs
        raise RuntimeError(
            "No inputs are available. Use set_inputs(), load_inputs(), or enable simulated inputs."
        )

    def apply_window(self) -> WindowState:
        """Apply the optional reference windowing stage and cache its output."""
        self._auto_invalidate_if_needed("window")
        if self.state.window is not None:
            return self.state.window

        inputs = self._ensure_inputs_available()
        self.logger.info("Running window stage (enabled=%s).", self.config.window.enabled)
        if self.config.window.enabled:
            ref_w = apply_w(
                inputs.reference,
                beta=self.config.window.beta,
                freq=self.config.window.freq,
                range=self.config.window.range,
            )
        else:
            ref_w = inputs.reference.copy()

        self.state.window = WindowState(
            reference=np.asarray(ref_w, dtype=np.complex128),
            applied=bool(self.config.window.enabled),
            beta=self.config.window.beta,
            freq=bool(self.config.window.freq),
            range=bool(self.config.window.range),
        )
        self.state.completed_stages["window"] = True
        self._store_stage_snapshot("window")
        self.state.last_completed_stage = "window"
        return self.state.window

    def apply_filter(self) -> FilterState:
        """Apply the optional clutter filter to the surveillance signal and cache its output."""
        self._auto_invalidate_if_needed("filter")
        if self.state.filter is not None:
            return self.state.filter

        inputs = self._ensure_inputs_available()
        self.logger.info("Running filter stage (enabled=%s).", self.config.filter.enabled)
        if self.config.filter.enabled:
            filtered = block_lattice_filter(
                surveillance=inputs.surveillance,
                reference=inputs.reference,
                order=self.config.filter.order,
            )
        else:
            filtered = inputs.surveillance.copy()

        self.state.filter = FilterState(
            surveillance=np.asarray(filtered, dtype=np.complex128),
            applied=bool(self.config.filter.enabled),
            order=int(self.config.filter.order),
        )
        self.state.completed_stages["filter"] = True
        self._store_stage_snapshot("filter")
        self.state.last_completed_stage = "filter"
        return self.state.filter

    def compute_caf(self) -> CAFState:
        """Compute the cross-ambiguity function using the current upstream cached signals."""
        self._auto_invalidate_if_needed("caf")
        if self.state.caf is not None:
            return self.state.caf

        inputs = self._ensure_inputs_available()
        window_state = self.apply_window()
        filter_state = self.apply_filter()

        self.logger.info("Running CAF stage with batch=%d.", self.config.caf.batch)
        caf, freq_axis, range_axis = compute_caf(
            batch=self.config.caf.batch,
            fs=self.config.input.fs,
            surveillance=filter_state.surveillance,
            reference=window_state.reference,
        )
        truncated_length = (len(inputs.reference) // self.config.caf.batch) * self.config.caf.batch
        extent = [float(freq_axis[0] / 1e3), float(freq_axis[-1] / 1e3), float(range_axis[-1]), float(range_axis[0])]

        self.state.caf = CAFState(
            caf=np.asarray(caf),
            freq_axis=np.asarray(freq_axis),
            range_axis=np.asarray(range_axis),
            extent=extent,
            input_length=int(len(inputs.reference)),
            truncated_length=int(truncated_length),
        )
        self.state.completed_stages["caf"] = True
        self._store_stage_snapshot("caf")
        self.state.last_completed_stage = "caf"
        return self.state.caf

    def run_detection(self) -> DetectionState:
        """Run CA-CFAR detection on the magnitude of the current CAF."""
        self._auto_invalidate_if_needed("detect")
        if self.state.detection is not None:
            return self.state.detection

        caf_state = self.compute_caf()
        self.logger.info("Running detection stage (enabled=%s).", self.config.cfar.enabled)
        if not self.config.cfar.enabled:
            self.state.detection = DetectionState(detections=None, sigma_est=None, alpha_det=None)
        else:
            result = ca_cfar_1d(
                np.abs(caf_state.caf),
                Nw=self.config.cfar.Nw,
                Ng=self.config.cfar.Ng,
                pfa=self.config.cfar.P_fa,
                return_intermediate=self.config.cfar.return_intermediate,
            )
            if self.config.cfar.return_intermediate:
                detections, sigma_est, alpha_det = result
                self.state.detection = DetectionState(
                    detections=detections,
                    sigma_est=np.asarray(sigma_est),
                    alpha_det=float(alpha_det),
                )
            else:
                self.state.detection = DetectionState(detections=result)

        self.state.completed_stages["detect"] = True
        self._store_stage_snapshot("detect")
        self.state.last_completed_stage = "detect"
        return self.state.detection

    def run(self, *, start_from: str = "inputs", stop_at: str = "detect") -> PipelineState:
        """Run the chain from one stage to another, inclusive."""
        start = self._validate_stage_name(start_from)
        stop = self._validate_stage_name(stop_at)
        if self._stage_index(start) > self._stage_index(stop):
            raise ValueError(
                f"start_from ('{start}') must come before or equal to stop_at ('{stop}')."
            )

        self.logger.info("Running chain from '%s' to '%s'.", start, stop)
        for stage in self._STAGE_ORDER[self._stage_index(start) : self._stage_index(stop) + 1]:
            if stage == "inputs":
                self._ensure_inputs_available()
            elif stage == "window":
                self.apply_window()
            elif stage == "filter":
                self.apply_filter()
            elif stage == "caf":
                self.compute_caf()
            elif stage == "detect":
                self.run_detection()
        return self.get_state()

    def run_from(self, stage: str) -> PipelineState:
        """Run the chain starting from a chosen stage up to detection."""
        return self.run(start_from=stage, stop_at="detect")

    def run_until(self, stage: str) -> PipelineState:
        """Run the chain from the beginning up to a chosen stage."""
        return self.run(start_from="inputs", stop_at=stage)

    def get_state(self) -> PipelineState:
        """Return a deep copy of the current runtime state."""
        return copy.deepcopy(self.state)

    def save_config(self, path: str | Path | None = None) -> Path:
        """Serialize the current configuration to a JSON file."""
        path = Path(path) if path is not None else self.output_root / "configs" / f"{self._default_stem('config')}.json"
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonify(asdict(self.config)), f, indent=2)
        self.logger.info("Configuration saved to %s", path)
        return path

    def load_config(self, path: str | Path, *, reset_state: bool = True) -> None:
        """Load a JSON configuration into the current chain instance."""
        self.config = self._config_from_json(path)
        self.output_root = self._resolve_output_root()
        self._ensure_output_directories()
        if reset_state:
            self.reset()
        self.logger.info("Configuration loaded from %s", Path(path).expanduser().resolve())

    @staticmethod
    def _config_from_json(path: str | Path) -> PassiveRadarChainConfig:
        """Deserialize a JSON configuration file into config dataclasses."""
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return PassiveRadarChainConfig(
            input=InputConfig(**data.get("input", {})),
            simulation=SimulationConfig(
                transmitter_position=data.get("simulation", {}).get("transmitter_position", [0.0, 0.0]),
                radar_position=data.get("simulation", {}).get("radar_position", [70.0, 150.0]),
                reference_scale=data.get("simulation", {}).get("reference_scale", 1.0),
                clutter=ClutterConfig(**data.get("simulation", {}).get("clutter", {})),
                echo=EchoConfig(**data.get("simulation", {}).get("echo", {})),
            ),
            window=WindowConfig(**data.get("window", {})),
            filter=FilterConfig(**data.get("filter", {})),
            caf=CAFConfig(**data.get("caf", {})),
            cfar=CFARConfig(**data.get("cfar", {})),
            plot=PlotConfig(**data.get("plot", {})),
            io=IOConfig(**data.get("io", {})),
        )

    def save_state(self, path: str | Path | None = None) -> tuple[Path, Path]:
        """Save the current numerical pipeline state to ``.npz`` plus JSON metadata."""
        stem_path = (
            Path(path)
            if path is not None
            else self.output_root / "states" / self._default_stem("state")
        )
        stem_path = stem_path.expanduser().resolve()
        stem_path.parent.mkdir(parents=True, exist_ok=True)

        npz_path = stem_path.with_suffix(".npz")
        meta_path = stem_path.with_suffix(".json")

        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {
            "completed_stages": self.state.completed_stages,
            "stage_snapshots": self.state.stage_snapshots,
            "last_completed_stage": self.state.last_completed_stage,
            "external_inputs_were_set": self._external_inputs_were_set,
        }

        if self.state.inputs is not None:
            arrays["inputs_reference"] = self.state.inputs.reference
            arrays["inputs_surveillance"] = self.state.inputs.surveillance
            meta["inputs"] = {
                "source_mode": self.state.inputs.source_mode,
                "original_length": self.state.inputs.original_length,
                "metadata": _jsonify(self.state.inputs.metadata),
            }
        if self.state.simulation is not None:
            if self.state.simulation.clutter is not None:
                arrays["simulation_clutter"] = self.state.simulation.clutter
            if self.state.simulation.echo is not None:
                arrays["simulation_echo"] = self.state.simulation.echo
            meta["simulation"] = {
                "doppler_hz": self.state.simulation.doppler_hz,
                "metadata": _jsonify(self.state.simulation.metadata),
            }
        if self.state.window is not None:
            arrays["window_reference"] = self.state.window.reference
            meta["window"] = {
                "applied": self.state.window.applied,
                "beta": _jsonify(self.state.window.beta),
                "freq": self.state.window.freq,
                "range": self.state.window.range,
            }
        if self.state.filter is not None:
            arrays["filter_surveillance"] = self.state.filter.surveillance
            meta["filter"] = {
                "applied": self.state.filter.applied,
                "order": self.state.filter.order,
            }
        if self.state.caf is not None:
            arrays["caf_matrix"] = self.state.caf.caf
            arrays["caf_freq_axis"] = self.state.caf.freq_axis
            arrays["caf_range_axis"] = self.state.caf.range_axis
            meta["caf"] = {
                "extent": self.state.caf.extent,
                "input_length": self.state.caf.input_length,
                "truncated_length": self.state.caf.truncated_length,
            }
        if self.state.detection is not None:
            if isinstance(self.state.detection.detections, tuple):
                arrays["detection_rows"] = np.asarray(self.state.detection.detections[0])
                arrays["detection_cols"] = np.asarray(self.state.detection.detections[1])
                meta["detection"] = {"kind": "tuple"}
            elif self.state.detection.detections is not None:
                arrays["detection_indices"] = np.asarray(self.state.detection.detections)
                meta["detection"] = {"kind": "array"}
            else:
                meta["detection"] = {"kind": "none"}
            if self.state.detection.sigma_est is not None:
                arrays["detection_sigma_est"] = self.state.detection.sigma_est
            if self.state.detection.alpha_det is not None:
                meta.setdefault("detection", {})["alpha_det"] = self.state.detection.alpha_det

        np.savez_compressed(npz_path, **arrays)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonify(meta), f, indent=2)
        self.logger.info("State saved to %s and %s", npz_path, meta_path)
        return npz_path, meta_path

    def load_state(self, path: str | Path) -> None:
        """Load a previously saved numerical pipeline state from ``.npz`` and JSON metadata."""
        path = Path(path).expanduser().resolve()
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = path.with_suffix(".json")

        with np.load(npz_path, allow_pickle=False) as npz:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            self.state = PipelineState(
                completed_stages=meta.get("completed_stages", {}),
                stage_snapshots=meta.get("stage_snapshots", {}),
                last_completed_stage=meta.get("last_completed_stage"),
            )
            self._external_inputs_were_set = bool(meta.get("external_inputs_were_set", False))

            if "inputs" in meta:
                self.state.inputs = InputState(
                    reference=np.asarray(npz["inputs_reference"]),
                    surveillance=np.asarray(npz["inputs_surveillance"]),
                    source_mode=meta["inputs"]["source_mode"],
                    original_length=int(meta["inputs"]["original_length"]),
                    metadata=meta["inputs"].get("metadata", {}),
                )
            if "simulation" in meta:
                self.state.simulation = SimulationState(
                    clutter=np.asarray(npz["simulation_clutter"]) if "simulation_clutter" in npz else None,
                    echo=np.asarray(npz["simulation_echo"]) if "simulation_echo" in npz else None,
                    doppler_hz=meta["simulation"].get("doppler_hz"),
                    metadata=meta["simulation"].get("metadata", {}),
                )
            if "window" in meta:
                self.state.window = WindowState(
                    reference=np.asarray(npz["window_reference"]),
                    applied=bool(meta["window"]["applied"]),
                    beta=meta["window"]["beta"],
                    freq=bool(meta["window"]["freq"]),
                    range=bool(meta["window"]["range"]),
                )
            if "filter" in meta:
                self.state.filter = FilterState(
                    surveillance=np.asarray(npz["filter_surveillance"]),
                    applied=bool(meta["filter"]["applied"]),
                    order=int(meta["filter"]["order"]),
                )
            if "caf" in meta:
                self.state.caf = CAFState(
                    caf=np.asarray(npz["caf_matrix"]),
                    freq_axis=np.asarray(npz["caf_freq_axis"]),
                    range_axis=np.asarray(npz["caf_range_axis"]),
                    extent=list(meta["caf"]["extent"]),
                    input_length=int(meta["caf"]["input_length"]),
                    truncated_length=int(meta["caf"]["truncated_length"]),
                )
            if "detection" in meta:
                kind = meta["detection"].get("kind", "none")
                detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None
                if kind == "tuple":
                    detections = (np.asarray(npz["detection_rows"]), np.asarray(npz["detection_cols"]))
                elif kind == "array":
                    detections = np.asarray(npz["detection_indices"])
                else:
                    detections = None
                self.state.detection = DetectionState(
                    detections=detections,
                    sigma_est=np.asarray(npz["detection_sigma_est"]) if "detection_sigma_est" in npz else None,
                    alpha_det=meta["detection"].get("alpha_det"),
                )
        self.logger.info("State loaded from %s and %s", npz_path, meta_path)

    def _prepare_figure_path(self, filename: str | None, stem_prefix: str) -> Path:
        """Build a figure output path inside the package-level ``simulated_data/figures`` folder."""
        if filename is None:
            filename = f"{self._default_stem(stem_prefix)}.{self.config.io.figure_format}"
        output_path = self.output_root / "figures" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _apply_axes_limits(self, ax: plt.Axes, *, xlim: tuple[float, float] | None, ylim: tuple[float, float] | None) -> None:
        """Apply configured x/y limits to a Matplotlib axes when provided."""
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def plot_caf(
        self,
        *,
        show: bool | None = None,
        save: bool | None = None,
        filename: str | None = None,
        title: str | None = None,
        db: bool | None = None,
        **imshow_kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the most recent CAF and optionally save/show the figure."""
        caf_state = self.compute_caf()
        show = self.config.plot.show if show is None else show
        save = self.config.plot.save if save is None else save
        db = self.config.plot.db if db is None else db

        fig, ax = plot_caf(
            caf_state.caf,
            caf_state.extent,
            db=db,
            figsize=self.config.plot.figsize,
            aspect=self.config.plot.aspect,
            cmap=self.config.plot.cmap,
            **imshow_kwargs,
        )
        if title is not None:
            ax.set_title(title)
        self._apply_axes_limits(ax, xlim=self.config.plot.xlim, ylim=self.config.plot.ylim)

        if save:
            output_path = self._prepare_figure_path(filename, stem_prefix="caf")
            fig.savefig(output_path, bbox_inches="tight")
            self.logger.info("CAF figure saved to %s", output_path)
        if show:
            plt.show()
        return fig, ax

    def plot_detections(
        self,
        *,
        show: bool | None = None,
        save: bool | None = None,
        filename: str | None = None,
        title: str | None = None,
        db: bool | None = None,
        **imshow_kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the most recent CAF with the current detections overlaid on a copied figure."""
        detection_state = self.run_detection()
        caf_state = self.compute_caf()
        show = self.config.plot.show if show is None else show
        save = self.config.plot.save if save is None else save

        base_title = title or "Cross-Ambiguity Function with Detections"
        fig, ax = self.plot_caf(show=False, save=False, title=base_title, db=db, **imshow_kwargs)
        detections = detection_state.detections
        if isinstance(detections, tuple):
            fig_det, ax_det = add_detections(
                fig,
                detections,
                caf_state.caf.shape,
                caf_state.extent,
                ax=ax,
                marker=self.config.plot.marker,
                color=self.config.plot.color,
                markersize=self.config.plot.markersize,
            )
        else:
            fig_det, ax_det = fig, ax

        self._apply_axes_limits(ax_det, xlim=self.config.plot.xlim, ylim=self.config.plot.ylim)
        if title is not None:
            ax_det.set_title(title)

        if save:
            output_path = self._prepare_figure_path(filename, stem_prefix="detections")
            fig_det.savefig(output_path, bbox_inches="tight")
            self.logger.info("Detection figure saved to %s", output_path)
        if show:
            plt.show()
        return fig_det, ax_det
