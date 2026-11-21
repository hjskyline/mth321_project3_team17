"""Fixed-step heat solvers on the interior nodes of the unit interval.

M denotes the number of equal spatial intervals. Both endpoint values are zero.
All methods use the same second-order centred spatial operator with kappa = 1.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def grid(m):
    return np.arange(1, m, dtype=float) / m


def mode(m, k):
    if not 1 <= k < m:
        raise ValueError("mode must be resolved on the interior grid")
    return np.sin(np.pi * k * grid(m))


def rate(m, k):
    return 4 * m**2 * np.sin(np.pi * k / (2 * m))**2


def gain(method, z):
    if method == "FTCS":
        return 1 - z
    if method == "CN":
        return (1 - z / 2) / (1 + z / 2)
    if method == "Rannacher":
        return (1 + z / 2)**-2  # first full interval only
    raise ValueError(method)


def l2(u):
    return np.sqrt(np.dot(u, u) / (len(u) + 1))


def amplitude(u, k):
    m = len(u) + 1
    return 2 * np.dot(u, mode(m, k)) / m


def evolve(initial, dt, steps, method, save=()):
    """Return selected full-step states, including the final state.

    'Rannacher' replaces interval one with two backward Euler half-steps.
    Sparse LU is constructed once per implicit run. No iterative tolerance.
    """
    u = np.asarray(initial, dtype=float).copy()
    if u.ndim != 1 or not u.size or not np.isfinite(u).all():
        raise ValueError("expected a finite, nonempty interior profile")
    if not np.isfinite(dt) or dt <= 0 or not isinstance(steps, int) or steps < 1:
        raise ValueError("expected a positive step size and integer step count")
    if method not in ("FTCS", "CN", "Rannacher"):
        raise ValueError(method)
    wanted = set(save) | {steps}
    if any(not isinstance(i, int) or not 0 <= i <= steps for i in wanted):
        raise ValueError("snapshot index outside integration interval")
    states = {0: u.copy()} if 0 in wanted else {}
    r = dt * (len(u) + 1)**2
    if method != "FTCS":
        n = len(u)
        off = np.full(n - 1, -r / 2)
        matrix = diags((off, np.full(n, 1 + r), off), (-1, 0, 1), format="csc")
        lu = splu(matrix)
    weight = r if method == "FTCS" else r / 2
    for step in range(1, steps + 1):
        if method == "Rannacher" and step == 1:
            u = lu.solve(lu.solve(u))
        else:
            rhs = (1 - 2 * weight) * u
            rhs[1:] += weight * u[:-1]
            rhs[:-1] += weight * u[1:]
            u = rhs if method == "FTCS" else lu.solve(rhs)
        if step in wanted:
            states[step] = u.copy()
    return states
