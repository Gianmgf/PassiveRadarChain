"""Shared helpers for the PassiveRadarChain example scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pr_chain.core.configs import (
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
    ChannelConfig,
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
            N=1_000_000, fs=8126984.0, f_c=700e6, use_simulated_data=True
        ),
        simulation=SimulationConfig(
            transmitter_position=[0.0, 0.0],
            radar_position=[50.0, 500.0],
            direct_signal=True,
            clutter=ClutterConfig(
                N_CLUTT=10,
                clutter_rcs_min_db=-3.0,
                clutter_rcs_max_db=-5.0,
                rand_clutter=True,
                clutter_limits=[-100, 2000, 5, 150],
            ),
            echo=EchoConfig(
                V_b=[10.0, 100.0],
                rand_target=False,
                target_rcs_db=-15.0,
                target_position=[2500.0, 120.0],
            ),
        ),
        channel=ChannelConfig(
            enable=True,
            add_noise=True,
            noise_on_both_channels=True,
            noise_power_db=1,
        ),
        filter=FilterConfig(enabled=False, order=400),
        window=WindowConfig(enabled=False, beta=(150.0, 50.0), freq=True, range=True),
        caf=CAFConfig(batch=500),
        cfar=CFARConfig(
            enabled=True,
            bidimensional=True,
            Nw=1,
            Ng=(64, 256),
            P_fa=1e-5,
            freq_wrap=True,
        ),
        plot=PlotConfig(
            show=False,
            save=False,
            db=True,
            cmap="viridis",
            aspect="auto",
            xlim=(-4.2, 4.2),
            ylim=(18400, 0.0),
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
