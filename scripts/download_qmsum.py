#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    """Clone the official QMSum repository into a local data directory.

    The benchmark code expects users to provide their own dataset checkout rather
    than vendoring data into this project. Run this script from the command line
    when setting up QMSum data for healing or evaluation.
    """
    parser = argparse.ArgumentParser(description="Download QMSum from the official GitHub repository.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "QMSum"
    if target.exists():
        print(f"Already exists: {target}")
        return
    subprocess.run(
        ["git", "clone", "https://github.com/Yale-LILY/QMSum.git", str(target)],
        check=True,
    )
    print(f"Downloaded QMSum to {target}")


if __name__ == "__main__":
    main()
