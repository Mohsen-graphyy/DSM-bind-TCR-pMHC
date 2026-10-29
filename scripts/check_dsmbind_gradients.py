"""Run from the repository root; uses the existing environment and ESM cache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DSMBind"))
from bindenergy.apps.tcr_pmhc.gradient_check import main

if __name__ == "__main__":
    raise SystemExit(main())
