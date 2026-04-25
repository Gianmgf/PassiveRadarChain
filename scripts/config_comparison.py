from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pr_chain import PassiveRadarChain
from matplotlib.ticker import MaxNLocator
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
FIG_DIR = BASE_DIR.parent / "simulated_data" / "figures"
np.random.seed(47)


def build_config(
    cfar: tuple[int, tuple[int, int]] = (1, (64, 256)),
    beta: tuple[float, float] = (150.0, 50.0),
    samples: int = N_SAMPELS,
    save: bool = False,
) -> PassiveRadarChainConfig:
    """Build a configuration that closely matches the uploaded notebook."""
    return PassiveRadarChainConfig(
        input=InputConfig(
            N=samples, fs=8126984.0, f_c=700e6, use_simulated_data=False
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
                target_rcs_db=-36.0,
                target_position=[2500.0, 120.0],
            ),
        ),
        channel=ChannelConfig(
            enable=False,
            add_noise=False,
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
        ),
    )


def run_config(remod, pass_parameters, beta, cfar, samples=N_SAMPELS, save=False) -> None:
    REMOD_TITLE = "_remod" if remod else "_no_remod"
    RECONSTRUCTOR_TITLE = "con reconstrucción" if remod else "sin reconstrucción"
    if pass_parameters:
        chain = PassiveRadarChain(
            config=build_config(cfar=cfar, beta=beta,samples=samples, save=save), verbose=True
        )
    else:
        chain = PassiveRadarChain(config=build_config(), verbose=True)

    signals = np.load(DATA_DIR / "pr_signals" / "signals.npz")
    
    surv = signals['surv']
    if remod:
        ref = signals['remod']
    else:
        ref = signals['noisy']

    fig1, ax1 = plt.subplots()
    utils.plot_psd(x= ref, fs= 8126984.0, ax=ax1, n_samples=N_SAMPELS, freq_in_khz=True)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))
    ax1.set_title(f"Señal de referencia {RECONSTRUCTOR_TITLE}")
    if save:
        fig1.savefig(FIG_DIR / f"ref{REMOD_TITLE}.png", dpi=300, bbox_inches="tight")
        
    chain.set_inputs(reference= ref, surveillance= surv)
    # CAF Sin Filtro CF ni ventanas
    chain.run_until(stage="caf")
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
    




if __name__ == "__main__":
    run_config(True, True, (200.0, 100.0), (1, (254, 600)), True)
    run_config(False, True, (200.0, 100.0), (1, (254, 600)), True)
    graph(
        "config_remod.json",
        "config_no_remod.json",
        "state_remod.npz",
        "state_no_remod.npz",
        True,
    )

    plt.show()
