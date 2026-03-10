"""Inspect the chain state, stage cache, and configuration objects.

This is useful when learning the class because it shows where values live after each
stage and what information is available to downstream analysis code.
"""

from __future__ import annotations

from pprint import pprint

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    chain = PassiveRadarChain(config=build_example_config(seed=42), verbose=True)

    print("Initial config:")
    pprint(chain.config)
    print()

    chain.run_until("inputs")
    state = chain.get_state()
    print("After inputs stage:")
    print("  completed_stages:", state.completed_stages)
    print("  input source mode:", state.inputs.source_mode if state.inputs else None)
    print("  original input length:", state.inputs.original_length if state.inputs else None)
    print()

    chain.run_until("caf")
    state = chain.get_state()
    print("After CAF stage:")
    print("  completed_stages:", state.completed_stages)
    print("  CAF shape:", state.caf.caf.shape if state.caf else None)
    print("  CAF extent:", state.caf.extent if state.caf else None)
    print("  stage snapshots:")
    pprint(state.stage_snapshots)
    print()

    chain.run_detection()
    state = chain.get_state()
    print("After detection stage:")
    print("  last completed stage:", state.last_completed_stage)
    if state.detection is not None and isinstance(state.detection.detections, tuple):
        print("  number of detections:", len(state.detection.detections[0]))
        print("  alpha_det:", state.detection.alpha_det)


if __name__ == "__main__":
    main()
