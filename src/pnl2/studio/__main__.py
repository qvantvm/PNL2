"""python -m pnl2.studio"""

from __future__ import annotations

import sys
from pathlib import Path

from .app import main

if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(arg))
