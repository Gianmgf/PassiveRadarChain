"""Demonstrate both real-data entry points: `set_inputs()` and `load_inputs()`.

The example creates a small synthetic `.npz` file so the workflow is self-contained,
then shows:
- how to call `set_inputs(reference, surveillance, ...)`
- how to call `load_inputs(path)`
- how to process only the later stages once inputs are already available
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from pr_chain import PassiveRadarChain

from example_common import build_example_config, save_example_real_inputs


def main() -> None:
    # Build an example file that mimics saved real inputs.
    input_file = save_example_real_inputs(Path(__file__).resolve().parents[1] / "simulated_data" / "states" / "example_real_inputs.npz")
    print("Created example real-input file:", input_file)

    # Method 1: set_inputs() directly from arrays already in memory.
    with np.load(input_file, allow_pickle=False) as npz:
        reference = npz["reference"]
        surveillance = npz["surveillance"]
        fs = float(npz["fs"])
        f_c = float(npz["f_c"])

    direct_chain = PassiveRadarChain(config=build_example_config(seed=None), verbose=True)
    direct_chain.set_inputs(
        reference,
        surveillance,
        fs=fs,
        f_c=f_c,
        metadata={"description": "loaded manually into set_inputs"},
    )
    direct_chain.run_from("window")
    direct_chain.plot_detections(title="Detections from set_inputs()")

    # Method 2: load_inputs() directly from a `.npz` file.
    file_chain = PassiveRadarChain(config=build_example_config(seed=None), verbose=True)
    file_chain.load_inputs(input_file)
    file_chain.run_until("caf")
    file_chain.plot_caf(title="CAF from load_inputs()")


if __name__ == "__main__":
    main()
