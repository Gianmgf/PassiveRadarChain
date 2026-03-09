"""Show the two main ways to load configuration files.

This demo covers:
- saving a config to JSON
- constructing a new chain with `PassiveRadarChain.from_config_file(...)`
- loading a config into an existing chain with `load_config(...)`
- modifying the loaded config and rerunning
"""

from __future__ import annotations

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    original = PassiveRadarChain(config=build_example_config(seed=123), verbose=True)
    config_path = original.save_config()
    print("Saved config file:", config_path)

    # Option 1: create a new chain directly from a config file.
    chain_from_file = PassiveRadarChain.from_config_file(config_path, verbose=True)
    print("Loaded via classmethod. Seed:", chain_from_file.config.input.seed)
    chain_from_file.run_until("caf")
    chain_from_file.plot_caf(title="CAF from classmethod-loaded config")

    # Option 2: load a config into an existing chain object.
    chain_existing = PassiveRadarChain(config=build_example_config(seed=None), verbose=True)
    chain_existing.load_config(config_path)
    print("Loaded into existing chain. Seed:", chain_existing.config.input.seed)

    # Modify the loaded config and rerun from the affected stage onward.
    chain_existing.update_filter_config(order=20)
    chain_existing.update_window_config(beta=(8.0, 8.0))
    chain_existing.run()
    chain_existing.plot_detections(title="Detections after loading and editing config")


if __name__ == "__main__":
    main()
