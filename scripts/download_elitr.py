#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
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

