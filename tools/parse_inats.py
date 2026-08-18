#!/usr/bin/env python3

"""Back-compat alias for ``prepare_inats.py``."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("prepare_inats.py")),
        run_name="__main__",
    )
