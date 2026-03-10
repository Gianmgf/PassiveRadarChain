"""End-to-end example of saving, resetting, and resuming a workflow.

This demo shows:
- saving config and numerical state
- resetting the chain state while keeping config
- loading the saved state back
- loading the saved config into a new chain
- continuing to rerun later stages after reload
"""

from __future__ import annotations

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    chain = PassiveRadarChain(config=build_example_config(seed=24), verbose=True)
    chain.run()

    config_path = chain.save_config()
    state_npz_path, state_json_path = chain.save_state()
    print("Saved config:", config_path)
    print("Saved state files:", state_npz_path, state_json_path)

    chain.reset()
    print("After reset:", chain.get_state().completed_stages)

    chain.load_state(state_npz_path)
    print("After load_state:", chain.get_state().completed_stages)
    chain.plot_detections(title="Detections after reloading saved state")

    fresh_chain = PassiveRadarChain.from_config_file(config_path, verbose=True)
    fresh_chain.load_state(state_npz_path)
    fresh_chain.update_cfar_config(P_fa=1e-4)
    fresh_chain.run_from("detect")
    fresh_chain.plot_detections(title="Reloaded state with updated CFAR")


if __name__ == "__main__":
    main()
