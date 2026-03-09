"""Load a previously saved state and display the stored results.

Workflow:
1. Run once and save the numerical state if needed.
2. Create a fresh chain.
3. Load the saved state.
4. Plot the CAF and detections from the loaded cached results.

Important note:
- `plot_caf()` and `plot_detections()` will use the loaded cached state when it is
  already available, so this is the easiest way to recover past results visually.
"""

from __future__ import annotations

from pathlib import Path

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    chain = PassiveRadarChain(config=build_example_config(seed=42), verbose=True)

    # Create one saved state if you do not already have one.
    chain.run()
    npz_path, json_path = chain.save_state()
    print("Saved reference state files:")
    print("  ", npz_path)
    print("  ", json_path)

    # Load into a brand-new chain instance.
    reloaded = PassiveRadarChain(config=build_example_config(seed=999), verbose=True)
    reloaded.load_state(npz_path)

    loaded_state = reloaded.get_state()
    print("Loaded stages:", loaded_state.completed_stages)
    if loaded_state.caf is not None:
        print("Loaded CAF shape:", loaded_state.caf.caf.shape)
    if loaded_state.detection is not None and isinstance(loaded_state.detection.detections, tuple):
        print("Loaded detections:", len(loaded_state.detection.detections[0]))

    reloaded.plot_caf(show=True, save=False, title="CAF loaded from saved state")
    reloaded.plot_detections(show=True, save=False, title="Detections loaded from saved state")

    # Example of loading from the stem instead of the explicit `.npz` path.
    reloaded_again = PassiveRadarChain(config=build_example_config(seed=1), verbose=False)
    reloaded_again.load_state(Path(npz_path).with_suffix(""))
    print("Loaded again from stem path:", Path(npz_path).with_suffix(""))


if __name__ == "__main__":
    main()
