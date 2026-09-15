"""
run_all.py -- every guard in one command.

These tests exist because each one corresponds to a defect that reached
production and stayed there. They are cheap, they need no GPU, and they need no
database. Run them in CI on every commit.

    python tests/run_all.py

Exit code is the number of failing files, so CI fails on any regression.

WHAT EACH ONE PROTECTS

  test_causality        the rolling means must not contain y_t. The original
                        leak; +250 kW on a 4-point mean for a +1000 perturbation.
  test_serving_parity   loads the REAL deployed builder and compares it with
                        the training pipeline. Fails when the two drift apart,
                        which is how the 3.9x under-prediction happened.
  test_feature_mode_guard  a model must refuse to serve against features it was
                        not fitted on.
  test_settlement       the KPI settlement arithmetic: u=0 saves nothing,
                        shifting into cheap hours saves exactly energy x price
                        difference, and the realised cost follows realised
                        demand rather than the plan's assumption.
  test_validation_v2    the promotion gate must reject a candidate that naive
                        persistence beats, however bad the incumbent is.
  test_telemetry_sanity PV must be dark at night -- the only check in the stack
                        that can catch a POD/label swap -- and it must use LOCAL
                        night, or it flags real dawn generation all summer.

test_serving_parity needs the deployed builder. Point PROD_BUILDER at it; if the
path is absent that single check skips and the rest still run, so CI on a
machine without the service repo is still useful.
"""
from __future__ import annotations
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = [
    "test_causality.py",
    "test_serving_parity.py",
    "test_feature_mode_guard.py",
    "test_settlement.py",
    "test_validation_v2.py",
    "test_telemetry_sanity.py",
]


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    failed = []
    for f in FILES:
        print("\n" + "=" * 72)
        print("  %s" % f)
        print("=" * 72)
        r = subprocess.run([sys.executable, os.path.join(HERE, f)],
                           cwd=ROOT, env=env)
        if r.returncode != 0:
            failed.append(f)

    print("\n" + "=" * 72)
    if failed:
        print("  FAILED: %s" % ", ".join(failed))
        print("=" * 72)
        return len(failed)
    print("  all %d guard files passed" % len(FILES))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
