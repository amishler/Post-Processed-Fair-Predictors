"""
Utility functions for the counterfactualEO module.
"""
from collections.abc import Iterable


def to_iterable(x):
    """Utility to ensure input is iterable."""
    if isinstance(x, str):
        return [x]
    return x if isinstance(x, Iterable) else [x]
