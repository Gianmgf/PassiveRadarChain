from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from ..generators import ClutterGenerator, EchoGenerator
from ..processing import (
    apply_w,
    block_lattice_filter,
    ca_cfar,
    compute_caf,
    apply_noise_and_channel,
)
from ..utils import plot_caf
from .configs import (
    CAFConfig,
    CFARConfig,
    ClutterConfig,
    EchoConfig,
    FilterConfig,
    IOConfig,
    InputConfig,
    PassiveRadarChainConfig,
    PlotConfig,
    SimulationConfig,
    WindowConfig,
    ChannelConfig,
)


StageName = Literal["inputs", "channel", "filter", "window", "caf", "detect"]


def _jsonify(value: Any) -> Any:
    """Convierte objetos anidados de Python/NumPy en estructuras serializables a JSON."""
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


def _as_complex_1d(
    signal: np.ndarray | list[complex] | tuple[complex, ...], name: str
) -> np.ndarray:
    """Valida que la señal sea un arreglo unidimensional no vacío y la convierte a un arreglo complejo de NumPy."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array. Got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return np.asarray(arr, dtype=np.complex128)


@dataclass
class InputState:
    """Estado de ejecución de las señales de entrada de la cadena, incluyendo
    referencia, vigilancia y origen de los datos."""

    reference: np.ndarray
    surveillance: np.ndarray
    source_mode: Literal["real", "simulated"]
    original_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationState:
    """Estado de ejecución de las señales intermedias simuladas, incluyendo
    clutter, eco, Doppler y geometría efectiva del escenario simulado."""

    clutter: np.ndarray | None = None
    echo: np.ndarray | None = None
    doppler_hz: float | None = None
    clutter_positions: np.ndarray | None = None
    target_position: np.ndarray | None = None
    radar_position: np.ndarray | None = None
    transmitter_position: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowState:
    """Estado de ejecución de la señal de referencia luego de aplicar el
    ventaneo configurado."""

    reference: np.ndarray
    applied: bool
    beta: float | tuple[float, float]
    freq: bool
    range: bool


@dataclass
class FilterState:
    """Estado de ejecución de la señal de vigilancia luego del filtrado de
    clutter."""

    surveillance: np.ndarray
    applied: bool
    order: int


@dataclass
class CAFState:
    """Estado de ejecución del cálculo más reciente de la función de
    ambigüedad cruzada."""

    caf: np.ndarray
    freq_axis: np.ndarray
    range_axis: np.ndarray
    extent: list[float]
    input_length: int
    truncated_length: int


@dataclass
class DetectionState:
    """Estado de ejecución de la etapa de detección, incluyendo detecciones,
    estimación de ruido y umbral aplicado."""

    detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None
    sigma_est: np.ndarray | None = None
    alpha_det: float | None = None


@dataclass
class ChannelState:
    """Estado de ejecución de la etapa opcional de canal y ruido aplicada
    sobre las señales de entrada."""

    reference_ch: np.ndarray | None = None
    surveillance_ch: np.ndarray | None = None
    applied: bool = False
    add_noise: bool = False
    noise_power_db: float | None = None
    noise_added: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None


@dataclass
class PipelineState:
    """Estado agregado de ejecución de toda la cadena de procesamiento,
    incluyendo etapas, resultados intermedios y metadatos de ejecución."""

    inputs: InputState | None = None
    channel: ChannelState | None = None
    simulation: SimulationState | None = None
    window: WindowState | None = None
    filter: FilterState | None = None
    caf: CAFState | None = None
    detection: DetectionState | None = None
    completed_stages: dict[str, bool] = field(
        default_factory=lambda: {
            "inputs": False,
            "channel": False,
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

    _STAGE_ORDER: tuple[StageName, ...] = (
        "inputs",
        "channel",
        "filter",
        "window",
        "caf",
        "detect",
    )

    def __init__(
        self,
        config: PassiveRadarChainConfig | None = None,
        *,
        verbose: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Inicializa la cadena de procesamiento con su configuración, estado
        interno, sistema de logging y posibles overrides para la CAF."""
        self.config = config or PassiveRadarChainConfig()
        self.state = PipelineState()
        self._external_inputs_were_set = False

        # Override opcional de la referencia que entra a la CAF.
        self._caf_reference_override: np.ndarray | None = None
        self._caf_reference_override_metadata: dict[str, Any] = {}

        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
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
        """Crea una instancia de la cadena a partir de un archivo JSON de configuración previamente guardado."""
        config = cls._config_from_json(path)
        return cls(config=config, verbose=verbose, logger=logger)

    def _configure_logger(self, verbose: bool | None = None) -> None:
        """Configura el logger interno si no fue configurado externamente."""
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
        """Determina el directorio raíz donde se almacenarán configuraciones, estados y figuras generadas."""
        if self.config.io.output_root is not None:
            return Path(self.config.io.output_root).expanduser().resolve()
        return Path(__file__).resolve().parents[3] / "simulated_data"

    def _ensure_output_directories(self) -> None:
        """Crea los directorios de salida necesarios si todavía no existen."""
        for directory in (
            self.output_root,
            self.output_root / "configs",
            self.output_root / "states",
            self.output_root / "figures",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _default_stem(self, prefix: str) -> str:
        """Genera un nombre base con prefijo y marca temporal para archivos de salida."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _validate_stage_name(self, stage: str) -> StageName:
        """Verifica que el nombre de etapa sea válido dentro de la cadena de procesamiento."""
        if stage not in self._STAGE_ORDER:
            raise ValueError(
                f"Invalid stage '{stage}'. Expected one of {self._STAGE_ORDER}."
            )
        return stage  # type: ignore[return-value]

    def _stage_index(self, stage: StageName) -> int:
        """Devuelve el índice ordinal asociado a una etapa de la cadena."""
        return self._STAGE_ORDER.index(stage)

    def _stage_state_map(self) -> dict[str, Any]:
        """Devuelve un mapeo entre nombres públicos de etapa y su estado actual."""
        return {
            "inputs": self.state.inputs,
            "simulation": self.state.simulation,
            "channel": self.state.channel,
            "filter": self.state.filter,
            "window": self.state.window,
            "caf": self.state.caf,
            "detection": self.state.detection,
        }

    def _finalize_stage(self, stage: StageName) -> None:
        """Marca la etapa como completada, actualiza las previas, guarda su snapshot y actualiza el último estado válido."""
        stage_idx = self._stage_index(stage)
        for known_stage in self._STAGE_ORDER:
            if self._stage_index(known_stage) <= stage_idx:
                self.state.completed_stages[known_stage] = True

        self._store_stage_snapshot(stage)
        self.state.last_completed_stage = stage

    def _store_stage_snapshot(self, stage: StageName) -> None:
        """Guarda una instantánea serializable de la configuración relevante para una etapa."""
        self.state.stage_snapshots[stage] = self._stage_snapshot(stage)

    def _stage_snapshot(self, stage: StageName) -> dict[str, Any]:
        """Construye una instantánea serializable de la configuración asociada a una etapa específica."""
        if stage == "inputs":
            desired_mode = (
                "simulated" if self.config.input.use_simulated_data else "real"
            )
            return _jsonify(
                {
                    "input": asdict(self.config.input),
                    "simulation": asdict(self.config.simulation),
                    "desired_mode": desired_mode,
                }
            )

        if stage == "channel":
            return _jsonify(asdict(self.config.channel))

        if stage == "window":
            return _jsonify(asdict(self.config.window))

        if stage == "filter":
            return _jsonify(asdict(self.config.filter))

        if stage == "caf":
            return _jsonify(
                {
                    "caf": asdict(self.config.caf),
                    "reference_override": {
                        "active": self._caf_reference_override is not None,
                        "length": (
                            None
                            if self._caf_reference_override is None
                            else int(len(self._caf_reference_override))
                        ),
                        "metadata": self._caf_reference_override_metadata,
                    },
                }
            )

        if stage == "detect":
            return _jsonify(asdict(self.config.cfar))

        raise ValueError(f"Unknown stage '{stage}'.")

    def _auto_invalidate_if_needed(self, stage: StageName) -> None:
        """Invalida una etapa y las posteriores si detecta cambios en su configuración respecto de la última ejecución."""
        previous = self.state.stage_snapshots.get(stage)
        if previous is None:
            return

        current = self._stage_snapshot(stage)
        if current != previous:
            self.logger.info(
                "Configuration changed for stage '%s'; invalidating downstream cache.",
                stage,
            )
            self.invalidate_from(stage)

    def invalidate_from(self, stage: str, *, include_stage: bool = True) -> None:
        """Invalida los resultados almacenados desde una etapa dada en adelante."""
        valid_stage = self._validate_stage_name(stage)
        start_index = self._stage_index(valid_stage)
        if not include_stage:
            start_index += 1

        for known_stage in self._STAGE_ORDER[start_index:]:
            if known_stage == "inputs":
                self.state.inputs = None
                self.state.simulation = None
                self._external_inputs_were_set = False
            elif known_stage == "channel":
                self.state.channel = None
            elif known_stage == "filter":
                self.state.filter = None
            elif known_stage == "window":
                self.state.window = None
            elif known_stage == "caf":
                self.state.caf = None
            elif known_stage == "detect":
                self.state.detection = None

            self.state.completed_stages[known_stage] = False
            self.state.stage_snapshots.pop(known_stage, None)

        remaining_completed = [
            s for s in self._STAGE_ORDER if self.state.completed_stages.get(s, False)
        ]
        self.state.last_completed_stage = (
            remaining_completed[-1] if remaining_completed else None
        )

    def reset(self, *, clear_reconstructed_reference: bool = True) -> None:
        """Reinicia completamente el estado de la cadena manteniendo intacta la configuración actual."""
        self.logger.info("Resetting chain state.")
        self.state = PipelineState()
        self._external_inputs_were_set = False

        if clear_reconstructed_reference:
            self._caf_reference_override = None
            self._caf_reference_override_metadata = {}

    def _update_dataclass(self, target: Any, **kwargs: Any) -> None:
        """Asigna valores a un dataclass de configuración y ejecuta su validación posterior si existe."""
        for key, value in kwargs.items():
            if not hasattr(target, key):
                raise AttributeError(
                    f"Unknown configuration field '{key}' for {type(target).__name__}."
                )
            setattr(target, key, value)

        post_init = getattr(target, "__post_init__", None)
        if callable(post_init):
            post_init()

    def update_config(self, section: str, **kwargs: Any) -> None:
        """Actualiza una sección de configuración y realiza la invalidación necesaria."""
        section_map: dict[str, tuple[Any, StageName | None]] = {
            "input": (self.config.input, "inputs"),
            "channel": (self.config.channel, "channel"),
            "simulation": (self.config.simulation, "inputs"),
            "clutter": (self.config.simulation.clutter, "inputs"),
            "echo": (self.config.simulation.echo, "inputs"),
            "window": (self.config.window, "window"),
            "filter": (self.config.filter, "filter"),
            "caf": (self.config.caf, "caf"),
            "cfar": (self.config.cfar, "detect"),
            "plot": (self.config.plot, None),
            "io": (self.config.io, None),
        }

        if section not in section_map:
            raise ValueError(
                f"Unknown config section '{section}'. Expected one of {tuple(section_map)}."
            )

        target, stage_to_invalidate = section_map[section]
        self._update_dataclass(target, **kwargs)

        if section == "io":
            self.output_root = self._resolve_output_root()
            self._ensure_output_directories()
            return

        if stage_to_invalidate is not None:
            self._auto_invalidate_if_needed(stage_to_invalidate)

    def update_input_config(self, **kwargs: Any) -> None:
        self.update_config("input", **kwargs)

    def update_channel_config(self, **kwargs: Any) -> None:
        self.update_config("channel", **kwargs)

    def update_simulation_config(self, **kwargs: Any) -> None:
        self.update_config("simulation", **kwargs)

    def update_clutter_config(self, **kwargs: Any) -> None:
        self.update_config("clutter", **kwargs)

    def update_echo_config(self, **kwargs: Any) -> None:
        self.update_config("echo", **kwargs)

    def update_window_config(self, **kwargs: Any) -> None:
        self.update_config("window", **kwargs)

    def update_filter_config(self, **kwargs: Any) -> None:
        self.update_config("filter", **kwargs)

    def update_caf_config(self, **kwargs: Any) -> None:
        self.update_config("caf", **kwargs)

    def update_cfar_config(self, **kwargs: Any) -> None:
        self.update_config("cfar", **kwargs)

    def update_plot_config(self, **kwargs: Any) -> None:
        self.update_config("plot", **kwargs)

    def update_io_config(self, **kwargs: Any) -> None:
        self.update_config("io", **kwargs)

    def _drop_reconstructed_reference_if_incompatible(
        self, expected_length: int
    ) -> None:
        """Borra el override de referencia si su longitud no es compatible con las señales actuales."""
        if self._caf_reference_override is None:
            return

        if len(self._caf_reference_override) != expected_length:
            self.logger.warning(
                "Clearing reconstructed reference override because its length (%d) "
                "does not match the current input length (%d).",
                len(self._caf_reference_override),
                expected_length,
            )
            self._caf_reference_override = None
            self._caf_reference_override_metadata = {}
            self.invalidate_from("caf")

    def set_reconstructed_reference(
        self,
        reference: np.ndarray | list[complex] | tuple[complex, ...],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Define una señal de referencia reconstruida para reemplazar la entrada
        de referencia del bloque CAF.

        La señal se usa directamente en `compute_caf()` en lugar de la salida del
        bloque `window`.
        """
        ref = _as_complex_1d(reference, "reconstructed_reference")

        if self.state.inputs is not None and len(ref) != len(
            self.state.inputs.reference
        ):
            raise ValueError(
                "reconstructed_reference must have the same length as the current input signals. "
                f"Got {len(ref)} and {len(self.state.inputs.reference)}."
            )

        self._caf_reference_override = ref
        self._caf_reference_override_metadata = metadata or {}
        self.invalidate_from("caf")
        self.logger.info(
            "Reconstructed reference set successfully (length=%d).", len(ref)
        )

    def clear_reconstructed_reference(self) -> None:
        """Elimina el override de referencia reconstruida y vuelve a usar la rama normal hacia la CAF."""
        if self._caf_reference_override is None:
            return

        self._caf_reference_override = None
        self._caf_reference_override_metadata = {}
        self.invalidate_from("caf")
        self.logger.info("Reconstructed reference cleared.")

    # Alias más genérico, por comodidad.
    set_caf_reference = set_reconstructed_reference
    clear_caf_reference = clear_reconstructed_reference

    def load_reconstructed_reference(
        self,
        path: str | Path,
        *,
        key: str = "reference",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Carga una referencia reconstruida desde un archivo .npz y la utiliza como override para la CAF."""
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as npz:
            if key not in npz:
                raise KeyError(
                    f"The reconstructed reference file must contain an array named '{key}'."
                )
            ref = npz[key]

        merged_metadata = {"loaded_from": str(path), "key": key}
        if metadata:
            merged_metadata.update(metadata)

        self.set_reconstructed_reference(ref, metadata=merged_metadata)

    def _get_reference_for_caf(self, fallback_reference: np.ndarray) -> np.ndarray:
        """Devuelve la referencia que se utilizará en la CAF."""
        if self._caf_reference_override is not None:
            return self._caf_reference_override
        return fallback_reference

    def set_inputs(
        self,
        reference: np.ndarray | list[complex] | tuple[complex, ...],
        surveillance: np.ndarray | list[complex] | tuple[complex, ...],
        *,
        fs: float | None = None,
        f_c: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Define señales de entrada externas, actualiza la configuración asociada e invalida las etapas posteriores."""
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

        self._drop_reconstructed_reference_if_incompatible(len(ref))

        self.state.inputs = InputState(
            reference=ref,
            surveillance=surv,
            source_mode="real",
            original_length=len(ref),
            metadata=metadata or {},
        )
        self.state.simulation = None
        self._external_inputs_were_set = True

        self.invalidate_from("channel")
        self._finalize_stage("inputs")
        self.logger.info("External inputs set successfully (length=%d).", len(ref))

    def load_inputs(self, path: str | Path) -> None:
        """Carga señales de referencia y vigilancia desde un archivo .npz y las establece como entradas externas."""
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as npz:
            if "reference" not in npz or "surveillance" not in npz:
                raise KeyError(
                    "The input file must contain 'reference' and 'surveillance' arrays."
                )
            ref = npz["reference"]
            surv = npz["surveillance"]
            fs = float(npz["fs"]) if "fs" in npz else None
            f_c = float(npz["f_c"]) if "f_c" in npz else None

        self.set_inputs(ref, surv, fs=fs, f_c=f_c, metadata={"loaded_from": str(path)})

    def simulate_inputs(self, reference: np.ndarray | None = None) -> InputState:
        """Genera señales simuladas de referencia y vigilancia a partir de la configuración actual del escenario."""
        self.config.input.use_simulated_data = True
        self._auto_invalidate_if_needed("inputs")
        self.logger.info("Simulating input signals.")

        if reference is None:
            reference = self.config.simulation.reference_scale * (
                np.random.randn(self.config.input.N)
                + 1j * np.random.randn(self.config.input.N)
            )
        else:
            reference = np.asarray(reference, dtype=np.complex128)

        self._drop_reconstructed_reference_if_incompatible(len(reference))

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
            rand_target=self.config.simulation.echo.rand_target,
            target_position=self.config.simulation.echo.target_position,
            target_limits=self.config.simulation.echo.target_limits,
            Tx_position=self.config.simulation.transmitter_position,
            Rx_position=self.config.simulation.radar_position,
        )

        clutter = clutter_generator.generate(reference)
        echo, doppler_hz = echo_generator.generate(reference)

        if self.config.simulation.direct_signal:
            surveillance = np.asarray(clutter + echo + reference, dtype=np.complex128)
        else:
            surveillance = np.asarray(clutter + echo, dtype=np.complex128)

        clutter_positions = getattr(clutter_generator, "clutter_positions", None)
        if clutter_positions is not None:
            clutter_positions = np.asarray(clutter_positions, dtype=float)
        elif self.config.simulation.clutter.clutter_positions is not None:
            clutter_positions = np.asarray(
                self.config.simulation.clutter.clutter_positions, dtype=float
            )

        target_position = getattr(echo_generator, "target_position", None)
        if target_position is None:
            target_position = getattr(echo_generator, "target_positions", None)

        if target_position is not None:
            target_position = np.asarray(target_position, dtype=float)
            if target_position.ndim == 2 and target_position.shape[0] == 1:
                target_position = target_position[0]
        else:
            target_position = np.asarray(
                self.config.simulation.echo.target_position, dtype=float
            )

        self.state.inputs = InputState(
            reference=np.asarray(reference, dtype=np.complex128),
            surveillance=np.asarray(surveillance, dtype=np.complex128),
            source_mode="simulated",
            original_length=len(reference),
        )
        self.state.simulation = SimulationState(
            clutter=np.asarray(clutter, dtype=np.complex128),
            echo=np.asarray(echo, dtype=np.complex128),
            doppler_hz=float(doppler_hz) if doppler_hz is not None else None,
            clutter_positions=clutter_positions,
            target_position=target_position,
            radar_position=np.asarray(
                self.config.simulation.radar_position, dtype=float
            ),
            transmitter_position=np.asarray(
                self.config.simulation.transmitter_position, dtype=float
            ),
            metadata={
                "transmitter_position": np.asarray(
                    self.config.simulation.transmitter_position
                ).tolist(),
                "radar_position": np.asarray(
                    self.config.simulation.radar_position
                ).tolist(),
                "target_position": np.asarray(target_position).tolist(),
            },
        )

        self._external_inputs_were_set = False
        self.invalidate_from("channel")
        self._finalize_stage("inputs")
        return self.state.inputs

    def _ensure_inputs_available(self) -> InputState:
        """Garantiza que existan señales de entrada disponibles, ya sea cargadas externamente o simuladas."""
        self._auto_invalidate_if_needed("inputs")

        if self.state.inputs is not None:
            return self.state.inputs

        if self.config.input.use_simulated_data:
            return self.simulate_inputs()

        raise RuntimeError(
            "No inputs are available. Use set_inputs(), load_inputs(), or enable simulated inputs."
        )

    def apply_channel(self) -> ChannelState:
        """Aplica la etapa opcional de canal y ruido a las señales de entrada y almacena sus resultados."""
        self._auto_invalidate_if_needed("channel")
        if self.state.channel is not None:
            return self.state.channel

        inputs = self._ensure_inputs_available()

        self.logger.info(
            "Running channel stage (enabled=%s, add_noise=%s).",
            self.config.channel.enable,
            self.config.channel.add_noise,
        )

        noise: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None
        if self.config.channel.enable:
            surv_ch, ref_ch, noise = apply_noise_and_channel(
                surv=inputs.surveillance,
                ref=inputs.reference,
                add_noise=self.config.channel.add_noise,
                noise_on_both_channels=self.config.channel.noise_on_both_channels,
                noise_power_db=self.config.channel.noise_power_db,
                channel_response=self.config.channel.channel_response,
            )
        else:
            surv_ch = inputs.surveillance
            ref_ch = inputs.reference

        self.state.channel = ChannelState(
            reference_ch=np.asarray(ref_ch, dtype=np.complex128),
            surveillance_ch=np.asarray(surv_ch, dtype=np.complex128),
            applied=bool(self.config.channel.enable),
            add_noise=bool(self.config.channel.add_noise),
            noise_power_db=(
                float(self.config.channel.noise_power_db)
                if self.config.channel.add_noise
                else None
            ),
            noise_added=noise if self.config.channel.add_noise else None,
        )
        self._finalize_stage("channel")
        return self.state.channel

    def apply_filter(self) -> FilterState:
        """Aplica el filtro de clutter sobre la señal de vigilancia y almacena la salida resultante."""
        self._auto_invalidate_if_needed("filter")
        if self.state.filter is not None:
            return self.state.filter

        channel_state = self.apply_channel()

        self.logger.info(
            "Running filter stage (enabled=%s).", self.config.filter.enabled
        )
        if self.config.filter.enabled:
            filtered = block_lattice_filter(
                surveillance=channel_state.surveillance_ch,
                reference=channel_state.reference_ch,
                order=self.config.filter.order,
            )
        else:
            filtered = channel_state.surveillance_ch

        self.state.filter = FilterState(
            surveillance=np.asarray(filtered, dtype=np.complex128),
            applied=bool(self.config.filter.enabled),
            order=int(self.config.filter.order),
        )
        self._finalize_stage("filter")
        return self.state.filter

    def apply_window(self) -> WindowState:
        """Aplica el ventaneo configurado sobre la señal de referencia y almacena la salida resultante."""
        self._auto_invalidate_if_needed("window")
        if self.state.window is not None:
            return self.state.window

        channel_state = self.apply_channel()

        self.logger.info(
            "Running window stage (enabled=%s).", self.config.window.enabled
        )
        if self.config.window.enabled:
            ref_w = apply_w(
                channel_state.reference_ch,
                beta=self.config.window.beta,
                freq=self.config.window.freq,
                range=self.config.window.range,
            )
        else:
            ref_w = channel_state.reference_ch

        self.state.window = WindowState(
            reference=np.asarray(ref_w, dtype=np.complex128),
            applied=bool(self.config.window.enabled),
            beta=self.config.window.beta,
            freq=bool(self.config.window.freq),
            range=bool(self.config.window.range),
        )
        self._finalize_stage("window")
        return self.state.window

    def compute_caf(self) -> CAFState:
        """Calcula la función de ambigüedad cruzada a partir de las señales procesadas en las etapas previas."""
        self._auto_invalidate_if_needed("caf")
        if self.state.caf is not None:
            return self.state.caf

        filter_state = self.apply_filter()
        window_state = self.apply_window()

        reference_for_caf = self._get_reference_for_caf(window_state.reference)
        if len(reference_for_caf) != len(filter_state.surveillance):
            raise ValueError(
                "The reference signal used for CAF and the surveillance signal must have the same length. "
                f"Got {len(reference_for_caf)} and {len(filter_state.surveillance)}."
            )

        self.logger.info(
            "Running CAF stage with batch=%d (reference_override=%s).",
            self.config.caf.batch,
            self._caf_reference_override is not None,
        )

        caf, freq_axis, range_axis = compute_caf(
            batch=self.config.caf.batch,
            fs=self.config.input.fs,
            surveillance=filter_state.surveillance,
            reference=reference_for_caf,
        )

        input_length = int(len(filter_state.surveillance))
        truncated_length = (
            input_length // self.config.caf.batch
        ) * self.config.caf.batch

        extent = [
            float(freq_axis[0] / 1e3),
            float(freq_axis[-1] / 1e3),
            float(range_axis[-1]),
            float(range_axis[0]),
        ]

        self.state.caf = CAFState(
            caf=np.asarray(caf),
            freq_axis=np.asarray(freq_axis),
            range_axis=np.asarray(range_axis),
            extent=extent,
            input_length=input_length,
            truncated_length=int(truncated_length),
        )
        self._finalize_stage("caf")
        return self.state.caf

    def run_detection(self) -> DetectionState:
        """Ejecuta la detección CFAR sobre la magnitud de la CAF y almacena el resultado."""
        self._auto_invalidate_if_needed("detect")
        if self.state.detection is not None:
            return self.state.detection

        caf_state = self.compute_caf()
        self.logger.info(
            "Running detection stage (enabled=%s).", self.config.cfar.enabled
        )

        if not self.config.cfar.enabled:
            self.state.detection = DetectionState(
                detections=None, sigma_est=None, alpha_det=None
            )
        else:
            detections, sigma_est, alpha_det = ca_cfar(
                np.abs(caf_state.caf),
                Nw=self.config.cfar.Nw,
                Ng=self.config.cfar.Ng,
                pfa=self.config.cfar.P_fa,
                detection_2d=self.config.cfar.bidimensional,
                freq_wrap=self.config.cfar.freq_wrap,
            )
            self.state.detection = DetectionState(
                detections=detections,
                sigma_est=np.asarray(sigma_est),
                alpha_det=float(alpha_det),
            )

        self._finalize_stage("detect")
        return self.state.detection

    def run(
        self,
        *,
        start_from: str = "inputs",
        stop_at: str = "detect",
        copy_state: bool = True,
    ) -> PipelineState:
        """Ejecuta la cadena de procesamiento entre dos etapas especificadas, inclusive.

        Parameters
        ----------
        start_from : str
            Etapa inicial.
        stop_at : str
            Etapa final.
        copy_state : bool, optional
            Si es True devuelve una copia profunda del estado.
            Si es False devuelve el estado interno directo, evitando el costo
            de `deepcopy` en loops o benchmarking.
        """
        start = self._validate_stage_name(start_from)
        stop = self._validate_stage_name(stop_at)

        if self._stage_index(start) > self._stage_index(stop):
            raise ValueError(
                f"start_from ('{start}') must come before or equal to stop_at ('{stop}')."
            )

        handlers: dict[StageName, Any] = {
            "inputs": self._ensure_inputs_available,
            "channel": self.apply_channel,
            "filter": self.apply_filter,
            "window": self.apply_window,
            "caf": self.compute_caf,
            "detect": self.run_detection,
        }

        self.logger.info("Running chain from '%s' to '%s'.", start, stop)
        for stage in self._STAGE_ORDER[
            self._stage_index(start) : self._stage_index(stop) + 1
        ]:
            handlers[stage]()

        return self.get_state(copy_state=copy_state)

    def run_from(self, stage: str, *, copy_state: bool = True) -> PipelineState:
        """Ejecuta la cadena desde una etapa dada hasta la etapa final de detección."""
        return self.run(start_from=stage, stop_at="detect", copy_state=copy_state)

    def run_until(self, stage: str, *, copy_state: bool = True) -> PipelineState:
        """Ejecuta la cadena desde el inicio hasta la etapa indicada."""
        return self.run(start_from="inputs", stop_at=stage, copy_state=copy_state)

    def get_state(
        self,
        stage: str | None = None,
        *,
        copy_state: bool = True,
    ) -> (
        PipelineState
        | SimulationState
        | InputState
        | ChannelState
        | FilterState
        | WindowState
        | CAFState
        | DetectionState
        | None
    ):
        """Devuelve el estado completo de la cadena o del bloque indicado.

        Parameters
        ----------
        stage : str | None
            Nombre de la etapa. Si es None, devuelve el estado completo.
            Stages:
            inputs, simulation, channel, filter, window, caf, detection
        copy_state : bool, optional
            Si es True devuelve una copia profunda del estado solicitado.
            Si es False devuelve una referencia directa al estado interno.
        """
        if stage is None:
            return copy.deepcopy(self.state) if copy_state else self.state

        stage_map = self._stage_state_map()

        if stage not in stage_map:
            self.logger.warning(
                "Invalid stage '%s'. Must be one of %s. Returning the full pipeline state.",
                stage,
                tuple(stage_map.keys()),
            )
            return copy.deepcopy(self.state) if copy_state else self.state

        target = stage_map[stage]
        return copy.deepcopy(target) if copy_state else target

    def peek_state(
        self, stage: str | None = None
    ) -> (
        PipelineState
        | SimulationState
        | InputState
        | ChannelState
        | FilterState
        | WindowState
        | CAFState
        | DetectionState
        | None
    ):
        """Devuelve una referencia directa al estado interno, sin copiar."""
        return self.get_state(stage=stage, copy_state=False)

    def save_config(
        self,
        path: str | Path | None = None,
        filename: str | None = None,
    ) -> Path:
        """Guarda la configuración actual de la cadena en un archivo JSON."""
        if path is not None:
            path = Path(path)
        else:
            default_name = (
                f"{filename}.json"
                if filename is not None
                else f"{self._default_stem('config')}.json"
            )
            path = self.output_root / "configs" / default_name

        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonify(asdict(self.config)), f, indent=2)

        self.logger.info("Configuration saved to %s", path)
        return path

    def load_config(self, path: str | Path, *, reset_state: bool = True) -> None:
        """Carga una configuración desde un archivo JSON y opcionalmente reinicia el estado actual."""
        self.config = self._config_from_json(path)
        self.output_root = self._resolve_output_root()
        self._ensure_output_directories()

        if reset_state:
            self.reset(clear_reconstructed_reference=False)

        self.logger.info(
            "Configuration loaded from %s", Path(path).expanduser().resolve()
        )

    @staticmethod
    def _config_from_json(path: str | Path) -> PassiveRadarChainConfig:
        """Reconstruye la configuración completa de la cadena a partir de un archivo JSON."""
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return PassiveRadarChainConfig(
            input=InputConfig(**data.get("input", {})),
            channel=ChannelConfig(**data.get("channel", {})),
            simulation=SimulationConfig(
                transmitter_position=data.get("simulation", {}).get(
                    "transmitter_position", [0.0, 0.0]
                ),
                radar_position=data.get("simulation", {}).get(
                    "radar_position", [70.0, 150.0]
                ),
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

    def save_state(
        self,
        path: str | Path | None = None,
        filename: str | None = None,
    ) -> tuple[Path, Path]:
        """Guarda el estado numérico actual de la cadena en archivos .npz y .json."""
        if path is not None:
            stem_path = Path(path)
        else:
            default_name = filename or self._default_stem("state")
            stem_path = self.output_root / "states" / default_name

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

        if self._caf_reference_override is not None:
            arrays["caf_reference_override"] = self._caf_reference_override
            meta["caf_reference_override"] = {
                "metadata": _jsonify(self._caf_reference_override_metadata)
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
            if self.state.simulation.clutter_positions is not None:
                arrays["simulation_clutter_positions"] = (
                    self.state.simulation.clutter_positions
                )
            if self.state.simulation.target_position is not None:
                arrays["simulation_target_position"] = (
                    self.state.simulation.target_position
                )

            meta["simulation"] = {
                "doppler_hz": self.state.simulation.doppler_hz,
                "metadata": _jsonify(self.state.simulation.metadata),
            }

        if self.state.channel is not None:
            if self.state.channel.reference_ch is not None:
                arrays["channel_reference"] = self.state.channel.reference_ch
            if self.state.channel.surveillance_ch is not None:
                arrays["channel_surveillance"] = self.state.channel.surveillance_ch

            meta["channel"] = {
                "applied": self.state.channel.applied,
                "add_noise": self.state.channel.add_noise,
                "noise_power_db": self.state.channel.noise_power_db,
            }

            if self.state.channel.noise_added is not None:
                noise = self.state.channel.noise_added
                if isinstance(noise, tuple):
                    arrays["channel_noise_0"] = np.asarray(noise[0])
                    arrays["channel_noise_1"] = np.asarray(noise[1])
                    meta["channel"]["noise_kind"] = "tuple"
                else:
                    arrays["channel_noise"] = np.asarray(noise)
                    meta["channel"]["noise_kind"] = "array"

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
                arrays["detection_rows"] = np.asarray(
                    self.state.detection.detections[0]
                )
                arrays["detection_cols"] = np.asarray(
                    self.state.detection.detections[1]
                )
                meta["detection"] = {"kind": "tuple"}
            elif self.state.detection.detections is not None:
                arrays["detection_indices"] = np.asarray(
                    self.state.detection.detections
                )
                meta["detection"] = {"kind": "array"}
            else:
                meta["detection"] = {"kind": "none"}

            if self.state.detection.sigma_est is not None:
                arrays["detection_sigma_est"] = self.state.detection.sigma_est
            if self.state.detection.alpha_det is not None:
                meta.setdefault("detection", {})["alpha_det"] = (
                    self.state.detection.alpha_det
                )

        np.savez_compressed(npz_path, **arrays)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonify(meta), f, indent=2)

        self.logger.info("State saved to %s and %s", npz_path, meta_path)
        return npz_path, meta_path

    def load_state(self, path: str | Path) -> None:
        """Carga un estado previamente guardado desde archivos .npz y .json."""
        path = Path(path).expanduser().resolve()
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = path.with_suffix(".json")

        with np.load(npz_path, allow_pickle=False) as npz:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            self.state = PipelineState()
            self.state.completed_stages.update(meta.get("completed_stages", {}))
            self.state.stage_snapshots = meta.get("stage_snapshots", {})
            self.state.last_completed_stage = meta.get("last_completed_stage")
            self._external_inputs_were_set = bool(
                meta.get("external_inputs_were_set", False)
            )

            if "caf_reference_override" in npz:
                self._caf_reference_override = np.asarray(
                    npz["caf_reference_override"], dtype=np.complex128
                )
                self._caf_reference_override_metadata = meta.get(
                    "caf_reference_override", {}
                ).get("metadata", {})
            else:
                self._caf_reference_override = None
                self._caf_reference_override_metadata = {}

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
                    clutter=np.asarray(npz["simulation_clutter"])
                    if "simulation_clutter" in npz
                    else None,
                    echo=np.asarray(npz["simulation_echo"])
                    if "simulation_echo" in npz
                    else None,
                    doppler_hz=meta["simulation"].get("doppler_hz"),
                    clutter_positions=np.asarray(npz["simulation_clutter_positions"])
                    if "simulation_clutter_positions" in npz
                    else None,
                    target_position=np.asarray(npz["simulation_target_position"])
                    if "simulation_target_position" in npz
                    else None,
                    metadata=meta["simulation"].get("metadata", {}),
                )

            if "channel" in meta:
                noise_added: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None
                noise_kind = meta["channel"].get("noise_kind")

                if noise_kind == "tuple":
                    noise_added = (
                        np.asarray(npz["channel_noise_0"]),
                        np.asarray(npz["channel_noise_1"]),
                    )
                elif noise_kind == "array":
                    noise_added = np.asarray(npz["channel_noise"])

                self.state.channel = ChannelState(
                    reference_ch=np.asarray(npz["channel_reference"])
                    if "channel_reference" in npz
                    else None,
                    surveillance_ch=np.asarray(npz["channel_surveillance"])
                    if "channel_surveillance" in npz
                    else None,
                    applied=bool(meta["channel"].get("applied", False)),
                    add_noise=bool(meta["channel"].get("add_noise", False)),
                    noise_power_db=meta["channel"].get("noise_power_db"),
                    noise_added=noise_added,
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
                    detections = (
                        np.asarray(npz["detection_rows"]),
                        np.asarray(npz["detection_cols"]),
                    )
                elif kind == "array":
                    detections = np.asarray(npz["detection_indices"])
                else:
                    detections = None

                self.state.detection = DetectionState(
                    detections=detections,
                    sigma_est=np.asarray(npz["detection_sigma_est"])
                    if "detection_sigma_est" in npz
                    else None,
                    alpha_det=meta["detection"].get("alpha_det"),
                )

        self.logger.info("State loaded from %s and %s", npz_path, meta_path)

    def _prepare_figure_path(self, filename: str | None, stem_prefix: str) -> Path:
        """Construye la ruta de salida para una figura dentro del directorio configurado."""
        if filename is None:
            filename = (
                f"{self._default_stem(stem_prefix)}.{self.config.io.figure_format}"
            )
        output_path = self.output_root / "figures" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _apply_axes_limits(
        self,
        ax: plt.Axes,
        *,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
    ) -> None:
        """Aplica límites en los ejes x e y de una figura si fueron definidos en la configuración."""
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def _normalize_detection_indices(
        self, detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Normaliza el formato de detecciones a dos vectores: filas y columnas."""
        if detections is None:
            return None

        if isinstance(detections, tuple):
            rows = np.asarray(detections[0])
            cols = np.asarray(detections[1])

            if rows.shape != cols.shape:
                raise ValueError(
                    "Detection rows and cols must have the same shape. "
                    f"Got {rows.shape} and {cols.shape}."
                )

            if rows.size == 0:
                return None

            return rows, cols

        arr = np.asarray(detections)
        if arr.size == 0:
            return None

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(
                    "detections as a 1D array must have shape (2,), with [row, col]."
                )
            arr = arr[None, :]

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "detections must have shape (N, 2) or be a tuple of (rows, cols)."
            )

        return arr[:, 0], arr[:, 1]

    def _overlay_detections_on_caf(
        self,
        ax: plt.Axes,
        *,
        caf_shape: tuple[int, int],
        extent: list[float],
        detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None,
    ) -> None:
        """Superpone detecciones sobre el gráfico de la CAF."""
        normalized = self._normalize_detection_indices(detections)
        if normalized is None:
            return

        rows, cols = normalized
        ny, nx = caf_shape
        xmin, xmax, ybottom, ytop = extent

        dx = (xmax - xmin) / nx
        dy = (ybottom - ytop) / ny

        x = xmin + (cols + 0.5) * dx
        y = ytop + (rows + 0.5) * dy

        ax.plot(x, y, linestyle="None", marker="o", color="r", markersize=8)

    def plot_scenario_geometry(
        self,
        scale: float = 1.0,
        *,
        show_velocities: bool = True,
        arrow_len_px: int = 45,
        clutter_marker_size: int = 25,
        show_labels: bool = True,
        show: bool | None = None,
        save: bool | None = None,
        filename: str | None = None,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grafica la geometría del escenario simulado, incluyendo transmisor,
        receptor, blanco, clutter y flecha de velocidad del blanco."""
        if scale == 0:
            raise ValueError("scale must be non-zero.")

        show = self.config.plot.show if show is None else show
        save = self.config.plot.save if save is None else save

        inputs = self._ensure_inputs_available()
        if inputs.source_mode != "simulated" or self.state.simulation is None:
            raise RuntimeError(
                "plot_scenario_geometry() is only available for simulated inputs."
            )

        sim_state = self.state.simulation
        fig, ax = plt.subplots(figsize=self.config.plot.figsize)

        P_tx = (
            np.asarray(
                self.config.simulation.transmitter_position, dtype=float
            ).reshape(2)
            / scale
        )
        P_rx = (
            np.asarray(self.config.simulation.radar_position, dtype=float).reshape(2)
            / scale
        )

        if sim_state.target_position is not None:
            P_tgt = np.asarray(sim_state.target_position, dtype=float)
        else:
            P_tgt = np.asarray(self.config.simulation.echo.target_position, dtype=float)

        if P_tgt.ndim == 2 and P_tgt.shape[0] == 1:
            P_tgt = P_tgt[0]
        if P_tgt.shape != (2,):
            raise ValueError(f"target_position must be shape (2,). Got {P_tgt.shape}")
        P_tgt = P_tgt / scale

        if sim_state.clutter_positions is None:
            P_cl = np.empty((0, 2), dtype=float)
        else:
            P_cl = np.asarray(sim_state.clutter_positions, dtype=float)

        if P_cl.size == 0:
            P_cl = np.empty((0, 2), dtype=float)
        elif P_cl.ndim != 2 or P_cl.shape[1] != 2:
            raise ValueError(
                f"clutter_positions must be shape (N, 2). Got {P_cl.shape}"
            )
        else:
            P_cl = P_cl / scale

        if P_cl.size > 0:
            ax.scatter(
                P_cl[:, 0],
                P_cl[:, 1],
                s=clutter_marker_size,
                marker=".",
                label=f"Clutter (N={P_cl.shape[0]})",
                zorder=2,
            )

        ax.scatter(
            P_tx[0],
            P_tx[1],
            marker="^",
            s=120,
            label=f"Tx ({P_tx[0]:.1f}, {P_tx[1]:.1f})" if show_labels else "Tx",
            zorder=5,
        )
        ax.scatter(
            P_rx[0],
            P_rx[1],
            marker="s",
            s=120,
            label=f"Rx ({P_rx[0]:.1f}, {P_rx[1]:.1f})" if show_labels else "Rx",
            zorder=5,
        )
        ax.scatter(
            P_tgt[0],
            P_tgt[1],
            color="red",
            marker="x",
            s=140,
            label=f"Target ({P_tgt[0]:.1f}, {P_tgt[1]:.1f})"
            if show_labels
            else "Target",
            zorder=6,
        )

        fig.canvas.draw()

        def _draw_velocity_arrow(
            P0_data: np.ndarray,
            V_vec: np.ndarray | list[float] | tuple[float, float],
            *,
            color: str,
            name: str,
        ) -> None:
            V = np.asarray(V_vec, dtype=float).reshape(-1)
            if V.shape != (2,):
                return

            vmag = float(np.linalg.norm(V))
            if np.isclose(vmag, 0.0):
                return

            U = V / vmag
            p0 = ax.transData.transform(P0_data)
            p1 = p0 + arrow_len_px * U
            p1_data = ax.transData.inverted().transform(p1)

            ax.annotate(
                "",
                xy=p1_data,
                xytext=P0_data,
                arrowprops=dict(arrowstyle="->", linewidth=2, color=color),
                zorder=7,
            )

            if show_labels:
                label_offset_px = 12
                p_label = p1 + label_offset_px * U
                p_label_data = ax.transData.inverted().transform(p_label)

                ax.text(
                    p_label_data[0],
                    p_label_data[1],
                    f"{name}: {vmag:.2f} m/s",
                    color=color,
                    ha="left",
                    va="bottom",
                    zorder=8,
                )

        if show_velocities:
            _draw_velocity_arrow(
                P_tgt,
                self.config.simulation.echo.V_b,
                color="red",
                name="Vb",
            )

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        if scale == 1.0:
            unit = "m"
        elif scale == 1000.0:
            unit = "km"
        else:
            unit = f"m/{scale:g}"

        if title is None:
            title = f"Scenario geometry (scale={scale:g})"

        ax.set_title(title)
        ax.set_xlabel(f"x [{unit}]")
        ax.set_ylabel(f"y [{unit}]")

        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=True)

        fig.tight_layout()

        if save:
            output_path = self._prepare_figure_path(
                filename, stem_prefix="scenario_geometry"
            )
            fig.savefig(output_path, bbox_inches="tight")
            self.logger.info("Scenario geometry figure saved to %s", output_path)

        if show:
            plt.show()

        return fig, ax

    def plot_caf(
        self,
        *,
        show: bool | None = None,
        save: bool | None = None,
        filename: str | None = None,
        title: str | None = None,
        db: bool | None = None,
        detections: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
        **imshow_kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grafica la CAF actual, opcionalmente superpone detecciones y permite mostrar o guardar la figura."""
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

        self._apply_axes_limits(
            ax, xlim=self.config.plot.xlim, ylim=self.config.plot.ylim
        )

        self._overlay_detections_on_caf(
            ax,
            caf_shape=caf_state.caf.shape,
            extent=caf_state.extent,
            detections=detections,
        )

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
        """Grafica la CAF actual con las detecciones estimadas superpuestas y permite mostrar o guardar la figura."""
        detection_state = self.run_detection()
        show = self.config.plot.show if show is None else show
        save = self.config.plot.save if save is None else save

        fig, ax = self.plot_caf(
            show=False,
            save=False,
            title=title or "Cross-Ambiguity Function with Detections",
            db=db,
            detections=detection_state.detections,
            **imshow_kwargs,
        )

        self._apply_axes_limits(
            ax, xlim=self.config.plot.xlim, ylim=self.config.plot.ylim
        )

        if title is not None:
            ax.set_title(title)

        if save:
            output_path = self._prepare_figure_path(filename, stem_prefix="detections")
            fig.savefig(output_path, bbox_inches="tight")
            self.logger.info("Detection figure saved to %s", output_path)

        if show:
            plt.show()

        return fig, ax
