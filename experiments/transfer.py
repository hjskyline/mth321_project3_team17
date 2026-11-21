"""Held-out profiles and a continuous heat-pulse reference for presentation."""
from statistics import median
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from .heat import evolve, grid, l2
from .startup_control import choose, first_interval

PROFILES = ("Tent", "Square pulse", "Grid spike")
GRIDS = (64, 128, 256)
STEPS = (1/8192, 1/2048, 1/512)
TOLERANCES = (0.0025, 0.01, 0.04)
FINAL = 1/128


def initial_data(m, name):
    x = grid(m)
    if name == "Tent":
        return 1 - abs(2*x - 1)
    if name == "Square pulse":
        # Symmetric half-values at jumps match the continuous Fourier convention.
        return np.where((x == 3/8) | (x == 5/8), 0.5,
                        ((x > 3/8) & (x < 5/8)).astype(float))
    if name == "Grid spike":
        return (x == 0.5).astype(float)
    raise ValueError(name)


def operator(m):
    diag = np.ones(m - 2)
    return m**2 * diags((diag, -2*np.ones(m-1), diag), (-1, 0, 1), format="csr")


def transfer():
    rows = []
    for m in GRIDS:
        lap = operator(m)
        for name in PROFILES:
            u = initial_data(m, name)
            final_reference = expm_multiply(FINAL*lap, u)
            for dt in STEPS:
                n = round(FINAL / dt)
                reference = expm_multiply(dt*lap, u)
                ordinary = evolve(u, dt, 1, "CN")[1]
                damped = evolve(u, dt, 1, "Rannacher")[1]
                err_cn, err_damped = l2(ordinary-reference), l2(damped-reference)
                for tol in TOLERANCES:
                    rule = choose(u, dt, tol)
                    first = first_interval(u, dt, rule)
                    err = l2(first-reference)
                    if abs(rule.predicted_error-err) > 2e-11 or err > tol + 2e-11:
                        raise RuntimeError(f"controller missed its target: {name}, M={m}, dt={dt}")
                    final = evolve(first, dt, n-1, "CN")[n-1]
                    end_error = l2(final-final_reference)
                    rows.append(dict(profile=name, intervals=m, dt=dt, tolerance=tol,
                                     selected=rule.method, first_parts=rule.substeps,
                                     first_solves=rule.solves, predicted_l2=rule.predicted_error,
                                     first_l2=err, cn_first_l2=err_cn, damped_first_l2=err_damped,
                                     min_first=float(min(0, min(first))), T=FINAL,
                                     final_l2_grid=end_error, final_within_tolerance=end_error <= tol))
    return rows


def continuous_pulse(x, t, terms):
    k = np.arange(1, terms+1)
    coeff = 2*(np.cos(3*np.pi*k/8) - np.cos(5*np.pi*k/8))/(np.pi*k)
    return np.sin(np.pi*x[:, None]*k) @ (coeff*np.exp(-(np.pi*k)**2*t))


def pulse():
    m, dt = 128, 1/1024
    u, x = initial_data(m, "Square pulse"), grid(m)
    lap = operator(m)
    observations = (1, 2, 4, 8)
    snapshots = {name:evolve(u, dt, max(observations), name, observations)
                 for name in ("CN", "Rannacher")}
    points, diagnostics = [], []
    for step in observations:
        t = step*dt
        exact = continuous_pulse(x, t, 128)
        # |b_k| <= 4/(pi*k); bound the Gaussian tail by a geometric series.
        j = 129
        tail = 4*np.exp(-(np.pi*j)**2*t)/(np.pi*j*(-np.expm1(-np.pi**2*t*(2*j+1))))
        if tail > 1e-12 or max(abs(exact-continuous_pulse(x,t,256))) > 1e-12:
            raise RuntimeError("continuous pulse reference too short")
        grid_ref = expm_multiply(t*lap, u)
        for method, states in snapshots.items():
            values = states[step]
            diagnostics.append(dict(method=method, step=step, t=t, dt=dt, intervals=m,
                                    linf_pde=max(abs(values-exact)), l2_time=l2(values-grid_ref),
                                    linf_space=max(abs(grid_ref-exact)), tail_bound=tail,
                                    total_variation=sum(abs(np.diff(np.r_[0,values,0])))))
        for i in range(len(x)):
            points.append(dict(x=x[i], step=step, t=t, initial=u[i], exact=exact[i],
                               cn=snapshots["CN"][step][i], damped=snapshots["Rannacher"][step][i]))
    return points, diagnostics


def positivity():
    rows = []
    m = 128
    u = initial_data(m, "Grid spike")
    for r in (0.25, 0.5, 1, 2, 4, 8, 16):
        dt = r / m**2
        exact = expm_multiply(dt*operator(m), u)
        for method in ("CN", "Rannacher"):
            out = evolve(u, dt, 1, method)[1]
            if l2(out) > l2(u)+1e-13:
                raise RuntimeError("implicit method increased heat energy")
            rows.append(dict(method=method, intervals=m, r=r, dt=dt,
                             centre=out[m//2-1], minimum=min(0.0, min(out)),
                             reference_centre=exact[m//2-1], energy_ratio=(l2(out)/l2(u))**2))
    return rows


def controller_cost():
    """One shared rule versus a profile-specific rule at the same accuracy target."""
    cases = []
    for m in GRIDS:
        for dt in STEPS:
            for name in PROFILES:
                u = initial_data(m, name)
                cases.append((u, dt, expm_multiply(dt*operator(m), u)))
    rows = []
    for tol in TOLERANCES:
        # Strong uniform comparator: select the smallest tested subdivision
        # meeting the target on every case, giving it a free retrospective search.
        for parts in (1, 2, 4, 8, 16, 32, 64):
            if all(l2(evolve(u,dt/parts,parts,"CN")[parts]-ref) <= tol
                   for u,dt,ref in cases):
                break
        else:
            raise RuntimeError("no verified fixed comparator for this tolerance")
        for policy in ("Uniform CN", "Controlled"):
            samples = []
            for repeat in range(6):
                out = []
                solves = 0
                tick = perf_counter()
                for u, dt, ref in cases:
                    if policy == "Uniform CN":
                        out.append(evolve(u, dt/parts, parts, "CN")[parts])
                        solves += parts
                    else:
                        plan = choose(u, dt, tol)
                        out.append(first_interval(u, dt, plan))
                        solves += plan.solves
                elapsed_ms = 1000*(perf_counter()-tick)
                if repeat:
                    samples.append(elapsed_ms)
            worst = max(l2(v-case[2]) for v,case in zip(out,cases))
            if worst > tol + 2e-11:
                raise RuntimeError("cost comparison failed its accuracy requirement")
            rows.append(dict(tolerance=tol, policy=policy, cases=len(cases),
                             uniform_parts=parts if policy=="Uniform CN" else None,
                             total_solves=solves, worst_l2=worst, median_ms=median(samples),
                             min_ms=min(samples), max_ms=max(samples),
                             **{f"run_{i}_ms":s for i,s in enumerate(samples,1)}))
    return rows
