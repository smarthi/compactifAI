#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    """Clone the UTTER repository that contains ELITR-Bench.

    ELITR-Bench data is kept outside the package so experiments can point at a
    local checkout. Run this script from the command line to create the expected
    data directory before benchmarking.
    """
    parser = argparse.ArgumentParser(description="Download the UTTER repository containing ELITR-Bench.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "UTTER-MS9-meetingdata"
    if target.exists():
        print(f"Already exists: {target}")
        return
    subprocess.run(
        ["git", "clone", "https://github.com/utter-project/UTTER-MS9-meetingdata.git", str(target)],
        check=True,
    )
    print(f"Downloaded ELITR-Bench repo to {target / 'ELITR-Bench'}")


if __name__ == "__main__":
    main()
