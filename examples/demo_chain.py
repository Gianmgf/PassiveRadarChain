"""Notebook-style demo for the PassiveRadarChain class.

This script mirrors the uploaded notebook flow using the new stateful class:
1. simulate reference/surveillance data,
2. compute a raw CAF,
3. apply clutter filtering and reference windowing,
4. rerun CAF and CFAR,
5. overlay detections.
"""

from __future__ import annotations

from pr_chain import PassiveRadarChain
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

N_samples = 5_000_000


def build_notebook_like_config() -> PassiveRadarChainConfig:
    """Build a configuration that closely matches the uploaded notebook."""
    return PassiveRadarChainConfig(
        input=InputConfig(
            N=N_samples, fs=8126984.0, f_c=700e6, seed=None, use_simulated_data=True
        ),
        simulation=SimulationConfig(
            transmitter_position=[0.0, 0.0],
            radar_position=[70.0, 150.0],
            clutter=ClutterConfig(
                N_CLUTT=20,
                clutter_rcs_min_db=0.0,
                clutter_rcs_max_db=0.0,
                rand_clutter=True,
                clutter_limits=[-10, 500, 5, 150],
            ),
            echo=EchoConfig(
                V_b=[10.0, 100.0],
                rand_target=False,
                target_rcs_db=-6.0,
                target_position=[20.0, 220.0],
            ),
        ),
        channel=ChannelConfig(enable=True, add_noise=True, noise_power_db=20.0),
        filter=FilterConfig(enabled=False, order=30),
        window=WindowConfig(enabled=False, beta=(50.0, 50.0), freq=True, range=True),
        caf=CAFConfig(batch=200),
        cfar=CFARConfig(
            enabled=True, Nw=512, Ng=8, P_fa=1e-6, return_intermediate=True
        ),
        plot=PlotConfig(
            show=False,
            save=True,
            db=True,
            cmap="viridis",
            aspect="auto",
            xlim=(-0.2, 0.6),
            ylim=(800.0, 0.0),
        ),
    )


def main() -> None:
    """Run the notebook-style demo using the chain class."""
    chain = PassiveRadarChain(config=build_notebook_like_config(), verbose=True)

    # Raw CAF (equivalent to notebook cells 1-3).
    # reference = np.load("isdbt_signal.npy")
    # reference = reference[0:N_samples]
    # chain.simulate_inputs(reference=reference)

    chain.run_until("caf")
    chain.plot_caf(filename="demo_raw_caf.png", title="Raw CAF")

    # Filtered CAF without windowing (notebook cell 4-5, first branch).
    chain.update_filter_config(enabled=True, order=30)
    chain.run_from("filter")
    chain.plot_caf(
        filename="demo_filtered_caf.png",
        title="CAF after block lattice filtering (Without windowing)",
    )

    # Filtered CAF with windowing and detections (notebook cells 4-7, second branch).
    chain.update_window_config(enabled=True, beta=(50.0, 50.0), freq=True, range=False)
    chain.run_from("window")
    chain.plot_caf(
        filename="demo_filtered_windowed_caf.png",
        title="CAF after block lattice filtering (With windowing)",
    )
    chain.run_detection()
    chain.plot_detections(
        filename="demo_filtered_windowed_detections.png",
        title="Filtered + Windowed CAF with Detections",
    )

    # chain.save_config()
    # chain.save_state()


if __name__ == "__main__":
    main()
