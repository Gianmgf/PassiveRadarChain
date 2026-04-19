import numpy as np
from pathlib import Path
from pr_chain import utils
from pr_chain import PassiveRadarChain
from pr_chain.core.configs import (
    ClutterConfig,
    EchoConfig,
    InputConfig,
    PassiveRadarChainConfig,
    SimulationConfig,
    ChannelConfig,   
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
N_SAMPLES = 1_000_000
np.random.seed(47)

def gen_signals(snr_db, reference):
    N = len(reference)
    E_ref = np.mean(np.abs(reference) ** 2)
    noise_power = E_ref / utils.from_db(snr_db)

    noise = (
        np.sqrt(noise_power)
        * (np.random.randn(N) + 1j * np.random.randn(N))
    ).astype(np.complex64)

    noisy_ref = (reference + noise).astype(np.complex64)

    noisy_ref.tofile(DATA_DIR / "gr_files" / "isdbt_noisy.cfile")

    pr_config = PassiveRadarChainConfig(
        input=InputConfig(
            N=N_SAMPLES, fs=8126984.0, f_c=700e6, use_simulated_data=True
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
                target_rcs_db=-18.0,
                target_position=[2500.0, 120.0],
            ),
        ),
        channel=ChannelConfig(
            enable=False,
        ))
    
    noise = (
        np.sqrt(noise_power)
        * (np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES))
    )

    
    pr_chain = PassiveRadarChain(pr_config)
    pr_chain.simulate_inputs(reference[:N_SAMPLES])
    
    
    input_state= pr_chain.get_state(stage= "inputs", copy_state=False)
    sim_state = pr_chain.get_state(stage="simulation", copy_state=False)

    surv = input_state.surveillance + noise
    fd, target_p = sim_state.doppler_hz, sim_state.target_position

    return noisy_ref[0:N_SAMPLES], surv, fd, target_p




if __name__ == "__main__":
    reference = np.fromfile(DATA_DIR / "gr_files" / "isdtb_clean.cfile", dtype=np.complex64)
    reference_remod = np.fromfile(DATA_DIR / "gr_files" / "isdbt_remod.cfile", dtype=np.complex64)
    E = np.mean(np.abs(reference) ** 2)

    snr = 16
    noisy_ref, surv, fd, target_p = gen_signals(snr, reference)


    np.savez(
    DATA_DIR / "pr_signals"/"signals.npz",
    remod=reference_remod[:N_SAMPLES],
    clean= reference[:N_SAMPLES],
    noisy=noisy_ref,
    surv= surv,
    target_doppler = fd,
    target_p = target_p
)

