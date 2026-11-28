"""Fixed-step heat solvers on the interior nodes of the unit interval.

M denotes the number of equal spatial intervals. Both endpoint values are zero.
All methods use the same second-order centred spatial operator with kappa = 1.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .ode_solvers import FixedStepTheta


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

    r = (len(u) + 1) ** 2

    def fun(_t, u):
        rhs = -2.0 * u
        rhs[1:] += u[:-1]
        rhs[:-1] += u[1:]
        return r * rhs

    n = len(u)
    off = np.full(n - 1, r)
    matrix = diags((off, np.full(n, -2.0 * r), off), (-1, 0, 1), format="csc")

    soln = solve_ivp(
        fun,
        (0.0, dt * steps),
        initial,
        method=FixedStepTheta,
        num_steps=steps,
        jac=matrix,
        theta=0.0 if method == "FTCS" else 0.5 if method == "CN" else "Rannacher",
    )
    ret = {i: soln.y.T[i] for i in wanted}
    return ret
