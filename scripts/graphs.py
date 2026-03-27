from __future__ import annotations
from pathlib import Path
from pr_chain import PassiveRadarChain, utils

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR.parent / "simulated_data" / "configs"
STATE_DIR = BASE_DIR.parent / "simulated_data" / "states"
FIG_DIR = BASE_DIR.parent / "simulated_data" / "figures"
FIG_SIZE = (9, 6)


def main() -> None:

    chain_remod = PassiveRadarChain.from_config_file(
        CONFIG_DIR / "config_remod.json", verbose=True
    )
    chain_no_remod = PassiveRadarChain.from_config_file(
        CONFIG_DIR / "config_no_remod.json", verbose=True
    )

    chain_remod.load_state(STATE_DIR / "state_remod.npz")
    chain_no_remod.load_state(STATE_DIR / "state_no_remod.npz")

    caf_state_remod = chain_remod.get_state("caf")
    detection_state_remod = chain_remod.get_state("detection")

    caf_state_no_remod = chain_no_remod.get_state("caf")
    detection_state_no_remod = chain_no_remod.get_state("detection")

    fig1, ax1 = utils.plot_caf(
        detection_state_no_remod.sigma_est,
        caf_state_no_remod.extent,
    )
    ax1.set_xlabel("kHz")
    ax1.set_ylabel("m")
    ax1.set_title(r"CA-CFAR $\hat{\sigma}$ [ m, k]")
    fig1.savefig(FIG_DIR / "sigma_est_no_remod.png", dpi=300, bbox_inches="tight")

    rows, cols = detection_state_no_remod.detections
    detected_values = np.abs(caf_state_no_remod.caf[rows, cols])
    max_detection_idx = np.argmax(detected_values)

    r_idx = rows[max_detection_idx]
    f_idx = cols[max_detection_idx]

    max_peak_cut_db = utils.math.to_db(np.abs(caf_state_no_remod.caf[r_idx, f_idx]))

    _, ax2 = plt.subplots(figsize=FIG_SIZE, ncols=2)

    fig2, ax2 = utils.plotting.plot_caf_cuts(
        caf_state_no_remod,
        corte_freq=True,
        corte_range=True,
        f_idx=f_idx,
        r_idx=r_idx,
        sigma_est=detection_state_no_remod.sigma_est,
        alpha_est=detection_state_no_remod.alpha_det,
        plot_sigma=False,
        plot_umbral=True,
        en_db=True,
        ax=ax2,
    )

    for axis in ax2:
        for collection in axis.collections:
            collection.set_label("_nolegend_")

    lineas_freq = ax2[0].lines
    lineas_freq[0].set_label(f"CAF (pico: {max_peak_cut_db:.2f} dB)")
    lineas_freq[1].set_label("Umbral CFAR")

    lineas_range = ax2[1].lines
    lineas_range[0].set_label(f"CAF (pico: {max_peak_cut_db:.2f} dB)")
    lineas_range[1].set_label("Umbral CFAR")

    ax2[0].set_title(f"Corte en frecuencia @ range bin {r_idx}")
    ax2[0].set_xlabel("Freq [kHz]")
    ax2[0].set_ylabel("|CAF| [dB]")
    ax2[0].legend()

    ax2[1].set_title(f"Corte en rango @ Doppler bin {f_idx}")
    ax2[1].set_xlabel("Rango [m]")
    ax2[1].set_ylabel("|CAF| [dB]")
    ax2[1].legend()

    ax2[0].yaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2[1].yaxis.set_major_locator(MaxNLocator(nbins=20))

    fig2.savefig(FIG_DIR / "cortes_rf_no_remod.png", dpi=300, bbox_inches="tight")

    fig3, ax3 = utils.plot_caf(
        detection_state_remod.sigma_est,
        caf_state_remod.extent,
    )
    ax3.set_xlabel("kHz")
    ax3.set_ylabel("m")
    ax3.set_title(r"CA-CFAR $\hat{\sigma}$ [ m, k]")
    fig3.savefig(FIG_DIR / "sigma_est_remod.png", dpi=300, bbox_inches="tight")

    rows, cols = detection_state_remod.detections
    detected_values = np.abs(caf_state_remod.caf[rows, cols])
    max_detection_idx = np.argmax(detected_values)

    r_idx = rows[max_detection_idx]
    f_idx = cols[max_detection_idx]

    max_peak_cut_db = utils.math.to_db(np.abs(caf_state_remod.caf[r_idx, f_idx]))

    _, ax4 = plt.subplots(figsize=FIG_SIZE, ncols=2)

    fig4, ax4 = utils.plotting.plot_caf_cuts(
        caf_state_remod,
        corte_freq=True,
        corte_range=True,
        f_idx=f_idx,
        r_idx=r_idx,
        sigma_est=detection_state_no_remod.sigma_est,
        alpha_est=detection_state_no_remod.alpha_det,
        plot_sigma=False,
        plot_umbral=True,
        en_db=True,
        ax=ax4,
    )

    for axis in ax4:
        for collection in axis.collections:
            collection.set_label("_nolegend_")

    lineas_freq = ax4[0].lines
    lineas_freq[0].set_label(f"CAF (pico: {max_peak_cut_db:.2f} dB)")
    lineas_freq[1].set_label("Umbral CFAR")

    lineas_range = ax4[1].lines
    lineas_range[0].set_label(f"CAF (pico: {max_peak_cut_db:.2f} dB)")
    lineas_range[1].set_label("Umbral CFAR")

    ax4[0].set_title(f"Corte en frecuencia @ range bin {r_idx}")
    ax4[0].set_xlabel("Freq [kHz]")
    ax4[0].set_ylabel("|CAF| [dB]")
    ax4[0].legend()

    ax4[1].set_title(f"Corte en rango @ Doppler bin {f_idx}")
    ax4[1].set_xlabel("Rango [m]")
    ax4[1].set_ylabel("|CAF| [dB]")
    ax4[1].legend()

    ax4[0].yaxis.set_major_locator(MaxNLocator(nbins=20))
    ax4[1].yaxis.set_major_locator(MaxNLocator(nbins=20))

    fig4.savefig(FIG_DIR / "cortes_rf_remod.png", dpi=300, bbox_inches="tight")

    rows, cols = detection_state_no_remod.detections
    detected_values = np.abs(caf_state_no_remod.caf[rows, cols])
    max_detection_idx = np.argmax(detected_values)

    r_idx = rows[max_detection_idx]
    f_idx = cols[max_detection_idx]

    max_peak_cut_no_remod_db = utils.math.to_db(
        np.abs(caf_state_no_remod.caf[r_idx, f_idx])
    )
    max_peak_cut_remod_db = utils.math.to_db(np.abs(caf_state_remod.caf[r_idx, f_idx]))

    _, ax5 = plt.subplots(figsize=FIG_SIZE, ncols=2)

    fig5, ax5 = utils.plotting.plot_caf_cuts(
        caf_state_remod,
        corte_freq=True,
        corte_range=True,
        f_idx=f_idx,
        r_idx=r_idx,
        sigma_est=detection_state_remod.sigma_est,
        alpha_est=detection_state_no_remod.alpha_det,
        plot_sigma=True,
        plot_umbral=False,
        en_db=True,
        ax=ax5,
    )

    _, _ = utils.plotting.plot_caf_cuts(
        caf_state_no_remod,
        corte_freq=True,
        corte_range=True,
        f_idx=f_idx,
        r_idx=r_idx,
        sigma_est=detection_state_no_remod.sigma_est,
        alpha_est=detection_state_no_remod.alpha_det,
        plot_sigma=True,
        plot_umbral=False,
        en_db=True,
        ax=ax5,
    )

    for axis in ax5:
        for collection in axis.collections:
            collection.set_label("_nolegend_")

    lineas_freq = ax5[0].lines
    lineas_freq[0].set_label(f"CAF remod (pico: {max_peak_cut_remod_db:.2f} dB)")
    lineas_freq[1].set_label(r"$\hat{\sigma}$ remod")
    lineas_freq[2].set_label(f"CAF no remod (pico: {max_peak_cut_no_remod_db:.2f} dB)")
    lineas_freq[3].set_label(r"$\hat{\sigma}$ no remod")
    lineas_freq[2].set_linestyle("--")
    lineas_freq[3].set_linestyle("--")

    lineas_range = ax5[1].lines
    lineas_range[0].set_label(f"CAF remod (pico: {max_peak_cut_remod_db:.2f} dB)")
    lineas_range[1].set_label(r"$\hat{\sigma}$ remod")
    lineas_range[2].set_label(f"CAF no remod (pico: {max_peak_cut_no_remod_db:.2f} dB)")
    lineas_range[3].set_label(r"$\hat{\sigma}$ no remod")
    lineas_range[2].set_linestyle("--")
    lineas_range[3].set_linestyle("--")

    ax5[0].set_title(f"Corte en frecuencia @ range bin {r_idx}")
    ax5[0].set_xlabel("Freq [kHz]")
    ax5[0].set_ylabel("|CAF| [dB]")
    ax5[0].legend()

    ax5[1].set_title(f"Corte en rango @ Doppler bin {f_idx}")
    ax5[1].set_xlabel("Rango [m]")
    ax5[1].set_ylabel("|CAF| [dB]")
    ax5[1].legend()

    ax5[0].yaxis.set_major_locator(MaxNLocator(nbins=20))
    ax5[1].yaxis.set_major_locator(MaxNLocator(nbins=20))

    plt.tight_layout()
    plt.show()
    fig5.savefig(
        FIG_DIR / "cortes_rf_remod_vs_noremod.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
