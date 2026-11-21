"""Select a first-interval rule from sampled initial data on a Dirichlet grid."""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dst

from .heat import evolve, gain, rate


@dataclass(frozen=True)
class StartupPlan:
    method: str
    substeps: int
    predicted_error: float

    @property
    def solves(self):
        return 2 if self.method == "Rannacher" else self.substeps


def choose(initial, dt, tolerance):
    """Use weighted Parseval error to screen candidates in increasing solve count.

    For an equal two-solve budget, try a damped start before split CN.
    The tolerance controls the discrete L2 time error at the first output.
    """
    u = np.asarray(initial, dtype=float)
    if u.ndim != 1 or not u.size or not np.isfinite(u).all():
        raise ValueError("expected a finite interior profile")
    if not np.isfinite(dt) or not np.isfinite(tolerance) or min(dt, tolerance) <= 0:
        raise ValueError("dt and tolerance must be positive and finite")
    m = len(u) + 1
    spectrum = dst(u, type=1, norm="ortho") / np.sqrt(m)
    z = dt * rate(m, np.arange(1, m))
    exact = np.exp(-z)
    candidates = [("CN", 1), ("Rannacher", 1)]
    candidates.extend(("CN", 2**j) for j in range(1, 13))
    for name, parts in candidates:
        error = np.linalg.norm(spectrum * (gain(name, z / parts)**parts - exact))
        if error <= tolerance:
            return StartupPlan(name, parts, float(error))
    raise ValueError("first-step tolerance unattainable within the substep budget")


def first_interval(initial, dt, plan):
    return evolve(initial, dt / plan.substeps, plan.substeps, plan.method)[plan.substeps]
