"""Run the benchmark without installing the project into the local Python environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DSMBind"))
from bindenergy.apps.tcr_pmhc.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
