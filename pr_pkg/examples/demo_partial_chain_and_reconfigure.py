"""Run partial chains, change later-stage configs, and rerun efficiently.

This demo is the most useful one for day-to-day experimentation because it shows the
cached pipeline workflow:
- run only up to CAF
- modify filter settings and rerun from `filter`
- modify window settings and rerun from `window`
- modify CFAR settings and rerun only detection
- inspect invalidation behavior in the cached state
"""

from __future__ import annotations

from pr_chain import PassiveRadarChain

from example_common import build_example_config


def main() -> None:
    chain = PassiveRadarChain(config=build_example_config(seed=42), verbose=True)

    # Baseline run only up to CAF.
    chain.run_until("caf")
    chain.plot_caf(title="Baseline CAF")

    # Change only the filter settings and rerun from that stage onward.
    chain.update_filter_config(enabled=True, order=10)
    state_after_filter_change = chain.run_from("filter")
    print("After filter change:", state_after_filter_change.completed_stages)
    chain.plot_caf(title="CAF after filter change")

    # Change only the window settings and rerun from window onward.
    chain.update_window_config(enabled=True, beta=(6.0, 12.0), freq=True, range=False)
    state_after_window_change = chain.run_from("window")
    print("After window change:", state_after_window_change.completed_stages)
    chain.plot_caf(title="CAF after window change")

    # Change only CFAR settings and rerun detection.
    chain.update_cfar_config(Nw=96, Ng=6, P_fa=1e-4)
    state_after_cfar_change = chain.run_from("detect")
    print("After CFAR change:", state_after_cfar_change.completed_stages)
    chain.plot_detections(title="Detections after CFAR change")

    # Manual invalidation example.
    chain.invalidate_from("caf")
    print("After manual invalidate_from('caf'):", chain.get_state().completed_stages)
    chain.run_from("caf")
    chain.plot_detections(title="Detections after manual CAF invalidation")


if __name__ == "__main__":
    main()
