"""Shared helpers for the PassiveRadarChain example scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pr_chain.core import (
    CAFConfig,
    CFARConfig,
    ClutterConfig,
    EchoConfig,
    FilterConfig,
    InputConfig,
    PassiveRadarChainConfig,
    PlotConfig,
    SimulationConfig,
    WindowConfig,
)


def build_example_config(
    *, seed: int | None = 42, save_figures: bool = False, show_figures: bool = False
) -> PassiveRadarChainConfig:
    """Build a compact, reproducible example configuration.

    The defaults are intentionally smaller than the original notebook-style demo so the
    example scripts run faster while still exercising the whole API.
    """
    return PassiveRadarChainConfig(
        input=InputConfig(
            N=120_000,
            fs=8e6,
            f_c=700e6,
            seed=seed,
            use_simulated_data=True,
        ),
        simulation=SimulationConfig(
            transmitter_position=[0.0, 0.0],
            radar_position=[70.0, 150.0],
            clutter=ClutterConfig(
                N_CLUTT=12,
                clutter_rcs_min_db=0.0,
                clutter_rcs_max_db=0.0,
                rand_clutter=True,
                clutter_limits=[-10, 500, 5, 150],
            ),
            echo=EchoConfig(
                V_b=[10.0, 100.0],
                add_noise=False,
                rand_target=False,
                target_rcs_db=-3.0,
                target_position=[20.0, 220.0],
            ),
        ),
        filter=FilterConfig(enabled=True, order=30),
        window=WindowConfig(enabled=True, beta=(14.0, 14.0), freq=True, range=False),
        caf=CAFConfig(batch=200),
        cfar=CFARConfig(
            enabled=True, Nw=128, Ng=8, P_fa=1e-5, return_intermediate=True
        ),
        plot=PlotConfig(
            show=show_figures,
            save=save_figures,
            db=True,
            cmap="viridis",
            aspect="auto",
            xlim=(-10.0, 10.0),
            ylim=(1000.0, 0.0),
        ),
    )


def save_example_real_inputs(
    path: str | Path,
    *,
    seed: int = 7,
    N: int = 80_000,
    fs: float = 8e6,
    f_c: float = 700e6,
) -> Path:
    """Create a small synthetic `.npz` file that mimics real loaded inputs.

    This is useful for the real-data demos without requiring external files.
    """
    rng = np.random.default_rng(seed)
    reference = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(
        np.complex128
    )

    # Build a simple delayed + Doppler-shifted surveillance signal so the real-data
    # loading demos are self-contained.
    delay = 25
    doppler_hz = 150.0
    time = np.arange(N) / fs
    echo = np.zeros_like(reference)
    echo[delay:] = reference[:-delay] * np.exp(
        1j * 2 * np.pi * doppler_hz * time[:-delay]
    )
    surveillance = 0.6 * reference + 0.15 * echo

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, reference=reference, surveillance=surveillance, fs=fs, f_c=f_c
    )
    return path
