from __future__ import annotations
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

N_SAMPELS = 1_000_000
INCLUDE_ISDBT = True


def build_config() -> PassiveRadarChainConfig:
    """Build a configuration that closely matches the uploaded notebook."""
    return PassiveRadarChainConfig(
        input=InputConfig(
            N=N_SAMPELS, fs=8126984.0, f_c=700e6, seed=None, use_simulated_data=True
        ),
        simulation=SimulationConfig(
            transmitter_position=[0.0, 0.0],
            radar_position=[50.0, 500.0],
            direct_signal=True,
            clutter=ClutterConfig(
                N_CLUTT=10,
                clutter_rcs_min_db=0.0,
                clutter_rcs_max_db=0.0,
                rand_clutter=True,
                clutter_limits=[-100, 2000, 5, 150],
            ),
            echo=EchoConfig(
                V_b=[10.0, 100.0],
                rand_target=False,
                target_rcs_db=-10.0,
                target_position=[2500.0, 120.0],
            ),
        ),
        channel=ChannelConfig(enable=True, add_noise=True, noise_power_db=1.0),
        filter=FilterConfig(enabled=False, order=30),
        window=WindowConfig(enabled=False, beta=(15.0, 10.0), freq=True, range=True),
        caf=CAFConfig(batch=500),
        cfar=CFARConfig(
            enabled=True,
            bidimensional=True,
            Nw=256,
            Ng=16,
            P_fa=1e-6,
            return_intermediate=True,
        ),
        plot=PlotConfig(
            show=True,
            save=True,
            db=True,
            cmap="viridis",
            aspect="auto",
            xlim=(-4.2, 4.2),
            ylim=(19000, 0.0),
        ),
    )


def main() -> None:
    chain = PassiveRadarChain(config=build_config(), verbose=True)

    if INCLUDE_ISDBT:
        isdbt = np.load("isdbt_signal.npy")
        isdbt = isdbt[0:N_SAMPELS]
        E = np.mean(np.abs(isdbt) ** 2)
        N_o = E / utils.math.from_db(13)
        chain.update_channel_config(noise_power_db=N_o)
        chain.simulate_inputs(isdbt)

    chain.run_until("caf")
    _, ax1 = chain.plot_caf(filename="caf_isdbt2.png", title="CAF")
    ax1.set_xlabel("kHz")
    ax1.set_ylabel("m")

    chain.update_filter_config(enabled=True, order=100)
    chain.run_from("filter")
    _, ax2 = chain.plot_caf(
        filename="filtered_caf_isdbt2.png",
        title="CAF after block lattice filtering (Without windowing)",
    )
    ax2.set_xlabel("kHz")
    ax2.set_ylabel("m")

    chain.update_window_config(enabled=True, beta=(10.0, 10.0), freq=True, range=True)
    chain.run_from("window")
    _, ax3 = chain.plot_caf(
        filename="filtered_w_caf_isdbt2.png",
        title="CAF con cf (windowing)",
    )
    ax3.set_xlabel("kHz")
    ax3.set_ylabel("m")

    detection_state = chain.run_detection()
    caf_state = chain.get_state().caf
    _, ax4 = chain.plot_detections(
        filename="filtered_w_detections_isdbt2.png",
        title="CF + Windowed CAF Detections",
    )
    ax4.set_xlabel("kHz")
    ax4.set_ylabel("m")

    fig5, ax5 = utils.plot_caf(
        detection_state.sigma_est,
        caf_state.extent,
    )
    ax5.set_xlabel("kHz")
    ax5.set_ylabel("m")
    ax5.set_title(r"CA-CFAR $\hat{\sigma}$ [ m, k]")

    fig6, ax6 = plt.subplots(figsize=(9, 9), ncols=2)
    rows, cols = detection_state.detections

    r_idx = rows[0]
    f_idx = cols[0]

    freq_cut = utils.math.to_db(np.abs(caf_state.caf[r_idx, :]))
    freq_thr = utils.math.to_db(
        detection_state.sigma_est[r_idx, :] * detection_state.alpha_det
    )

    range_cut = utils.math.to_db(np.abs(caf_state.caf[:, f_idx]))
    range_thr = utils.math.to_db(
        detection_state.sigma_est[:, f_idx] * detection_state.alpha_det
    )

    ax6[0].plot(caf_state.freq_axis, freq_cut, label="CAF cut")
    ax6[0].plot(caf_state.freq_axis, freq_thr, label="CFAR threshold")
    ax6[0].set_title(f"Frequency cut at range bin {r_idx}")
    ax6[0].set_xlabel("Frequency [Hz]")
    ax6[0].set_ylabel("|CAF|[dB]")
    ax6[0].legend()

    ax6[1].plot(caf_state.range_axis, range_cut, label="CAF cut")
    ax6[1].plot(caf_state.range_axis, range_thr, label="CFAR threshold")
    ax6[1].set_title(f"Range cut at Doppler bin {f_idx}")
    ax6[1].set_xlabel("Range [m]")
    ax6[1].set_ylabel("|CAF|[dB]")
    ax6[1].legend()
    plt.show()
    # chain.save_config()
    # chain.save_state()  885


if __name__ == "__main__":
    main()
