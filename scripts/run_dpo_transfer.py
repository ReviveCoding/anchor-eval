"""Optional GPU-only DPO runner. Requires manually prepared non-final preference pairs."""

from __future__ import annotations

import argparse

import yaml

from evalforge.dpo import run_optional_dpo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        run_optional_dpo(yaml.safe_load(handle))
