"""Run E2E-DRO experiments controlled via a YAML config."""
import os
import sys
import runpy
import argparse
import yaml

# Ensure the package is importable when running this script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_from_config(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    experiments = cfg.get("experiments", [])
    hist_exps = [e for e in experiments if e in {"exp1", "exp2", "exp3", "exp4"}]
    synth = "exp5" in experiments

    if hist_exps:
        os.environ["EXP_LIST"] = ",".join(hist_exps)
        runpy.run_path(os.path.join(SCRIPT_DIR, "exp_hist.py"), run_name="__main__")

    if synth:
        runpy.run_path(os.path.join(SCRIPT_DIR, "exp_synthetic.py"), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E-DRO experiments")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"), help="Path to YAML config")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
