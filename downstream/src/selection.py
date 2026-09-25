"""Backward greedy condition selection from a response-similarity matrix."""

import itertools
import math

DEFAULT_ORDER = ("rain", "snow", "fog", "dust", "low_light")


def validate_matrix(matrix, order):
    """Read a symmetric, unit-diagonal cosine matrix in the supplied order."""
    if not isinstance(order, (list, tuple)):
        raise ValueError("order must be a list of condition names")
    order = list(order)
    if not order or any(not isinstance(name, str) or not name.strip() for name in order):
        raise ValueError("order must contain nonempty condition names")
    if len(set(order)) != len(order):
        raise ValueError("condition names must be unique")
    try:
        values = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("matrix must be a square array of numbers") from error
    size = len(order)
    if len(values) != size or any(len(row) != size for row in values):
        raise ValueError("matrix dimensions must match order")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise ValueError("similarities must be finite and between -1 and 1")
            if abs(value - values[j][i]) > 1e-12:
                raise ValueError("matrix must be symmetric")
        if row[i] != 1.0:
            raise ValueError("selection matrix must have a unit diagonal")
    return values, order


def objective(matrix, selected, weight):
    """Return coverage F, signed pairwise overlap R, and J = F - weight * R."""
    # Accumulate in fixed order, independently of Python's sum implementation.
    coverage = 0.0
    for row in matrix:
        coverage += max(float(row[index]) for index in selected)
    overlap = 0.0
    for a, b in itertools.combinations(selected, 2):
        overlap += float(matrix[a][b])
    return {"F": coverage, "R": overlap, "J": coverage - weight * overlap}


def select(matrix, weight, order=DEFAULT_ORDER):
    """Restart from all conditions; resolve exact ties in matrix order."""
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("lambda must be finite and nonnegative")
    matrix, order = validate_matrix(matrix, order)
    selected = list(range(len(order)))
    steps, deleted = [], []
    while len(selected) > 1:
        before = objective(matrix, selected, weight)
        candidates = []
        for index in selected:
            after = objective(matrix, [i for i in selected if i != index], weight)
            candidates.append({
                "condition": order[index], "index": index, **after,
                "gain": after["J"] - before["J"],
            })
        best = max(candidates, key=lambda candidate: candidate["gain"])
        steps.append({
            "retained": [order[i] for i in selected], **before,
            "candidates": candidates,
            "deleted": best["condition"] if best["gain"] > 0 else None,
        })
        if best["gain"] <= 0:
            break
        selected.remove(best["index"])
        deleted.append(best["condition"])
    return {
        "lambda": weight, "steps": steps, "deleted": deleted,
        "retained": [order[i] for i in selected],
        "final": objective(matrix, selected, weight),
        "stop": "singleton" if len(selected) == 1 else "no strictly positive deletion gain",
    }
