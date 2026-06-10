"""Put 03_code/src and the vendored pykan on sys.path for the experiment scripts."""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

for _p in (PACKAGE_ROOT / "03_code" / "src",
           PACKAGE_ROOT / "03_code",
           PACKAGE_ROOT / "pykan"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
