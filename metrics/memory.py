from __future__ import annotations


def measure_memory() -> float:
    """Memory measurement is intentionally unsupported.

    The previous implementation returned the *whole process's* RSS via psutil,
    which is a constant process baseline (numpy/torch/numba caches + the full
    dataset + every prior estimator's leftovers) plus noise -- not a
    per-estimator footprint, and so meaningless for comparing estimators. Rather
    than report a misleading number (or silently return None), this raises per
    the "fail fast and loud" rule. A correct per-estimator metric (e.g. a
    tracemalloc peak delta around estimate(), or a fresh-subprocess RSS delta)
    can replace this when implemented.
    """
    raise NotImplementedError("Memory measurement is currently unsupported.")
