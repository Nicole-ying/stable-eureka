from __future__ import annotations

import argparse

from eg_rsa import EGRSARunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EG-RSA reward search experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eg_rsa_minimal.yml",
        help="Path to an EG-RSA YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = EGRSARunner(config_path=args.config)
    runner.run()
