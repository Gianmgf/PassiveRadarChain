"""Run the full chain once, with no mid-run configuration changes.

What it shows:
- creating the chain from a config object
- running every stage with `run()`
- plotting the final CAF and detections
- inspecting the cached state
- saving config and numerical state
"""

from __future__ import annotations

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    config = build_example_config(seed=42, save_figures=False, show_figures=False)
    chain = PassiveRadarChain(config=config, verbose=True)

    state = chain.run()
    print("Completed stages:", state.completed_stages)
    print("Last completed stage:", state.last_completed_stage)

    fig_caf, ax_caf = chain.plot_caf(title="Full chain CAF")
    fig_det, ax_det = chain.plot_detections(title="Full chain detections")
    print("Created figures:", fig_caf, ax_caf, fig_det, ax_det)

    config_path = chain.save_config()
    state_paths = chain.save_state()
    print("Saved config to:", config_path)
    print("Saved state to:", state_paths)


if __name__ == "__main__":
    main()
