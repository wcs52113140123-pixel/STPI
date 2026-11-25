"""Deterministic seeds from names.

Produces a stable integer seed from an arbitrary string. Python's built-in hash() is
salted per process, so a seed derived from it changes between runs; SHA-256 does not.
"""
import hashlib


def seed_of(name: str) -> int:
    """32-bit seed from the first 8 hex digits of SHA-256(name)."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
