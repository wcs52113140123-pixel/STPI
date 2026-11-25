"""Where the analysis scripts read and write.

STPI_DATA      folder holding the input data (layout in README.md); default: <repo>/data
STPI_DERIVED   derived inputs (MERFISH metadata, tendon sections); default: $STPI_DATA/derived
STPI_RESULTS   where tables are written and read; default: <repo>/results
STPI_DATA_DERIVED  where the two geometry-only files are written; default: <repo>/data_derived

The scripts use the four strings DATA_, DER_, RES_ and HERE_ as plain-text substitutes
for the corresponding absolute paths.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = Path(os.environ.get("STPI_DATA", ROOT / "data"))
DERIVED = Path(os.environ.get("STPI_DERIVED", DATA / "derived"))
RESULTS = Path(os.environ.get("STPI_RESULTS", ROOT / "results"))
DATA_DERIVED = Path(os.environ.get("STPI_DATA_DERIVED", ROOT / "data_derived"))   # geometry-only files kept in the repository

DATA_, DER_, RES_, HERE_ = str(DATA), str(DERIVED), str(RESULTS), str(HERE)
DD_ = str(DATA_DERIVED)


def data(*parts):
    p = DATA.joinpath(*parts)
    if not p.exists():
        sys.exit(f"missing input: {p}  (set STPI_DATA; see analysis/README.md)")
    return p
