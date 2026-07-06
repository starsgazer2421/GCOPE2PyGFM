#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pygfm.baseline_models.gcope.runner import main_from_cli


def main() -> None:
    main_from_cli(default_stage="prog")


if __name__ == "__main__":
    main()
