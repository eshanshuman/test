"""Headless test of dashboard.py with streamlit.testing.v1.AppTest.

Purpose
    Proves the dashboard runs end to end without a browser: it starts the app,
    checks there is no exception, then selects each preset and checks that the
    sliders move, the status card shows what recommend.py returns for those
    readings, the three headline metrics are present and the "disagree"
    warning appears exactly when rules and model disagree.

Outputs
    A PASS/FAIL line per check; exit code 1 on any failure.
    Run: python tools/test_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import config as cfg
import recommend as rec


def main() -> None:
    cfg.section(f"Dashboard AppTest on the {cfg.DATASET} dataset ({cfg.DATASET_NOTE})")
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=180).run()
    check("app starts without exception", not at.exception, "; ".join(str(e.value) for e in at.exception))
    check("four sliders in the sidebar", len(at.sidebar.slider) == 4)
    check("three headline metrics", len(at.metric) == 3, ", ".join(m.label for m in at.metric))
    print(f"         forecast metric: {at.metric[1].label} = {at.metric[1].value}")
    captions = [c.value for c in at.caption]
    print(f"         forecast warning shown: {any('Low-confidence forecast' in c for c in captions)}")

    for preset, reading in rec.PRESETS.items():
        at.sidebar.selectbox[0].set_value(preset).run()
        expected = rec.recommend(*reading)
        sliders = [float(s.value) for s in at.sidebar.slider]
        card = " ".join(m.value for m in at.markdown)
        warned = any("disagree" in w.value for w in at.warning)
        passed = (not at.exception and sliders == [float(v) for v in reading]
                  and f">{expected['status']}</div>" in card
                  and str(at.metric[2].value) == str(expected["n_immediate"])
                  and warned == (expected["agreement"] is False))
        check(f"preset '{preset}'", passed,
              f"status {expected['status']}, immediate actions {expected['n_immediate']}, "
              f"disagreement warning {warned}")

    at.sidebar.slider[2].set_value(1.5).run()             # move the DO slider by hand
    card = " ".join(m.value for m in at.markdown)
    check("manual slider change (DO 1.5 mg/L) gives Critical with an immediate action",
          not at.exception and ">Critical</div>" in card and int(at.metric[2].value) >= 1)

    print("\nDASHBOARD TEST " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
