from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

from gen_signals import gen_signals
from config_comparison import run_config
from graphs import graph

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
N_SAMPLES = 1_000_000
np.random.seed(47)


def test_full_chain(gen_isdtb_signals:bool = False):
    if gen_isdtb_signals:
        subprocess.run([sys.executable, DATA_DIR/"gr_files"/"tx_tesis.py"])
    reference = np.fromfile(DATA_DIR / "gr_files" / "isdtb_clean.cfile", dtype=np.complex64)
    reference_remod = np.fromfile(DATA_DIR / "gr_files" / "isdbt_remod.cfile", dtype=np.complex64)
    E = np.mean(np.abs(reference) ** 2)

    snr = 16
    noisy_ref, surv, fd, target_p = gen_signals(snr, reference)
    

    np.savez(
    DATA_DIR / "pr_signals"/"signals.npz",
    remod=reference_remod[:N_SAMPLES],
    noisy=noisy_ref,
    surv= surv,
    target_doppler = fd,
    target_p = target_p
    )
    if gen_isdtb_signals:
        subprocess.run([sys.executable, DATA_DIR/"gr_files"/"rx_tesis_A.py"])
        subprocess.run([sys.executable, DATA_DIR/"gr_files"/"rx_tesis_B.py"])
        subprocess.run([sys.executable, DATA_DIR/"gr_files"/"tx_tesis_remod.py"])
    

    run_config(remod= True, pass_parameters= True, beta=(170.0, 20.0), cfar=(1, (128, 512)), save= True)
    run_config(remod= False, pass_parameters= True, beta=(170.0, 20.0), cfar=(1, (128,512)), save= True)
    graph(
        "config_remod.json",
        "config_no_remod.json",
        "state_remod.npz",
        "state_no_remod.npz",
        True,
    )


if __name__ == "__main__":
    test_full_chain(False)
    plt.show()

    



