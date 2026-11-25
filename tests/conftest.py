import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = os.environ.get("STPI_DATA")       # folder holding the public datasets, see README


def need_data(*parts):
    if not DATA:
        pytest.skip("set STPI_DATA to the folder with the public datasets")
    p = Path(DATA, *parts)
    if not p.exists():
        pytest.skip(f"missing {p}")
    return p
