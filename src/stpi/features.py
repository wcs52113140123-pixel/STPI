"""Expression features.

Used for both the buffered-pitch figure and the field-wide audit, so that both analyses operate on the same feature representation.
"""
import numpy as np


def log_hvg(M, hvg=2000):
    """Library-size normalisation to the median library, log1p, then the `hvg` features
    with the highest variance.

    M   units x genes counts, dense array or scipy sparse matrix
    """
    Xc = np.asarray(M.todense() if hasattr(M, "todense") else M, dtype=np.float64)
    lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1
    logn = np.log1p(Xc / lib * np.median(lib))
    return logn[:, np.argsort(logn.var(0))[-hvg:]]
