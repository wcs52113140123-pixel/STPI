"""STPI: spatial transcriptomic positional information.

Measures how much positional information gene expression carries, in bits, under three
conditions that make the number checkable:

  1. the position is an expression-independent coordinate built from tissue geometry
     (stpi.coords);
  2. it is decoded from expression under spatial block cross-validation, optionally with a
     buffer between training and test units (stpi.decode, stpi.cv);
  3. the estimate is compared with an autocorrelation-preserving null (stpi.null).

stpi.audit combines the three for one section and returns the random-CV estimate, the
spatial-block estimate, the fraction of the random-CV estimate that is inflation, the
buffered estimates and the null P value.

Each module names the algorithm it implements.
"""
from .cv import checkerboard_folds, random_folds, spot_pitch
from .decode import recoverability, information_bits
from .features import log_hvg
from .coords import interiority, across_band
from .null import msr_surrogates, spin_surrogates, null_p
from .audit import audit_section, verdict
from ._seed import seed_of

__version__ = "1.0.1"

__all__ = [
    "checkerboard_folds", "random_folds", "spot_pitch",
    "recoverability", "information_bits", "log_hvg",
    "interiority", "across_band",
    "msr_surrogates", "spin_surrogates", "null_p",
    "audit_section", "verdict", "seed_of",
]
