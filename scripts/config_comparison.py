from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pr_chain import PassiveRadarChain
from pr_chain import utils
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
from graphs import graph

N_SAMPELS = 1_000_000
INCLUDE_ISDBT = True

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
np.random.seed(47)


def build_config(
    cfar: tuple[int, tuple[int, int]] = (1, (64, 256)),
    beta: tuple[float, float] = (150.0, 50.0),
    save: bool = False,
) -> PassiveRadarChainConfig:
    """Build a configuration that closely matches the uploaded notebook."""
    return PassiveRadarChainConfig(
        input=InputConfig(
            N=N_SAMPELS, fs=8126984.0, f_c=700e6, use_simulated_data=True
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
        window=WindowConfig(enabled=False, beta=beta, freq=True, range=True),
        caf=CAFConfig(batch=500),
        cfar=CFARConfig(
            enabled=True,
            bidimensional=True,
            Nw=cfar[1],
            Ng=cfar[0],
            P_fa=1e-5,
            freq_wrap=True,
        ),
        plot=PlotConfig(
            show=False,
            save=save,
            db=True,
            cmap="viridis",
            aspect="auto",
            xlim=(-4.2, 4.2),
            ylim=(18400, 0.0),
        ),
    )


def main(remod, pass_parameters, beta, cfar, save=False) -> None:
    REMOD_TITLE = "_remod" if remod else "_no_remod"
    RECONSTRUCTOR_TITLE = "con reconstrucción" if remod else "sin reconstrucción"
    if pass_parameters:
        chain = PassiveRadarChain(
            config=build_config(cfar=cfar, beta=beta, save=save), verbose=True
        )
    else:
        chain = PassiveRadarChain(config=build_config(), verbose=True)

    if INCLUDE_ISDBT:
        isdbt = np.load(DATA_DIR / "isdbt_signal.npy")
        isdbt = isdbt[0:N_SAMPELS]
        E = np.mean(np.abs(isdbt) ** 2)
        N_o_in = utils.math.to_db(E / utils.math.from_db(12))
        chain.update_channel_config(
            noise_on_both_channels=not remod,
            noise_power_db=N_o_in,
        )
        chain.simulate_inputs(isdbt)
    # CAF Sin Filtro CF ni ventanas
    chain.run(start_from="channel", stop_at="caf")

    chain.plot_caf(
        filename=f"caf{REMOD_TITLE}.png",
        title=f"CAF {RECONSTRUCTOR_TITLE}",
    )

    # CAF con Filtro CF y sin ventanas
    chain.update_filter_config(enabled=True)
    chain.run(start_from="filter", stop_at="caf")
    chain.plot_caf(
        filename=f"filtered_caf{REMOD_TITLE}.png",
        title=f"CAF {RECONSTRUCTOR_TITLE} (CF)",
    )

    # CAF con Filtro CF y ventanas
    chain.update_window_config(enabled=True)
    chain.run(start_from="window", stop_at="detect")
    chain.plot_caf(
        filename=f"filtered_w_caf{REMOD_TITLE}.png",
        title=f"CAF {RECONSTRUCTOR_TITLE} (CF + Ventanas)",
    )

    chain.plot_detections(
        filename=f"filtered_w_detections{REMOD_TITLE}.png",
        title=f"Detecciones CAF {RECONSTRUCTOR_TITLE} (CF + Ventanas) ",
    )
    chain.save_config(filename=f"config{REMOD_TITLE}")
    chain.save_state(filename=f"state{REMOD_TITLE}")
    sim = chain.peek_state("simulation")
    target_p = sim.target_position
    target_doppler = sim.doppler_hz
    radar_p = sim.radar_position
    tx_p = sim.transmitter_position
    target_range = (
        np.linalg.norm(target_p - radar_p)
        + np.linalg.norm(target_p - tx_p)
        - np.linalg.norm(radar_p - tx_p)
    )
    caf_state = chain.peek_state("caf")
    r = caf_state.range_axis
    f = caf_state.freq_axis
    print(
        f"True:\n\tTarget Doppler: {target_doppler:.2f} \tTarget Range: {target_range:.2f}"
    )
    print(f"Detected:\n\tTarget Doppler: {f[1003]:.2f} \tTarget Range: {r[121]:.2f}")


if __name__ == "__main__":
    main(True, True, (200.0, 100.0), (1, (254, 600)), True)
    main(False, True, (200.0, 100.0), (1, (254, 600)), True)
    graph(
        "config_remod.json",
        "config_no_remod.json",
        "state_remod.npz",
        "state_no_remod.npz",
        True,
    )

    plt.show()
