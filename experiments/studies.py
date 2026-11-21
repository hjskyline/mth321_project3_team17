"""Numerical experiments. Constants below define the protocol before timing."""
from math import ceil, log2
from statistics import median
from time import perf_counter

import numpy as np

from .heat import amplitude, evolve, gain, l2, mode, rate

BASE = ("FTCS", "CN")
ALL = (*BASE, "Rannacher")
COST_TARGETS = (2e-3, 5e-4)
STARTUP_TOL = 0.01  # absolute discrete L2 error after one full interval


def profile(m, components, t=0.0, discrete=False):
    u = np.zeros(m - 1)
    for k, weight in components:
        decay = rate(m, k) if discrete else (np.pi * k)**2
        u += weight * np.exp(-decay * t) * mode(m, k)
    return u


def convergence():
    rows = []
    for study in ("space", "time"):
        for method in BASE:
            prev = None
            sizes = (16, 32, 64, 128) if study == "space" else (128, 256, 512, 1024)
            for size in sizes:
                if study == "space":
                    m, k, end = size, 1, 0.025
                    steps = ceil(end * m**2 / 0.005)
                else:
                    m, k, end, steps = 32, 3, 0.05, size
                dt = end / steps
                u = mode(m, k)
                ref = profile(m, [(k, 1)], end, discrete=(study == "time"))
                result = evolve(u, dt, steps, method)[steps]
                error = l2(result - ref)
                shift = None
                if study == "space":
                    fine = evolve(u, dt/2, 2*steps, method)[2*steps]
                    shift = l2(fine - result) / error
                    if shift > 0.025:
                        raise RuntimeError("temporal error contaminates the spatial study")
                order = log2(prev/error) if prev is not None else None
                rows.append(dict(study=study, method=method, intervals=m, mode=k,
                                 steps=steps, T=end, h=1/m, dt=dt, r=dt*m*m,
                                 l2_error=error, linf_error=max(abs(result-ref)),
                                 pairwise_order=order, half_dt_change_over_error=shift))
                prev = error
            expected = 1 if (study, method) == ("time", "FTCS") else 2
            if abs(order - expected) > 0.05:
                raise RuntimeError(f"unexpected asymptotic order: {study} {method} {order}")
    return rows


def stability():
    rows = []
    m, steps = 64, 12
    for z_high in (1.8, 1.99, 2.0, 2.01, 2.2):
        dt = z_high / rate(m, m-1)
        for k in (1, m-1):
            z = dt * rate(m, k)
            for method in BASE:
                states = evolve(mode(m, k), dt, steps, method, range(steps+1))
                for i, u in states.items():
                    observed = amplitude(u, k)
                    theory = gain(method, z)**i
                    if not np.isclose(theory, observed, atol=2e-12, rtol=1e-12):
                        raise RuntimeError("signed mode disagrees with amplification analysis")
                    rows.append(dict(method=method, intervals=m, mode=k, z_high=z_high,
                                     z_mode=z, r=dt*m*m, dt=dt, step=i, t=i*dt,
                                     signed_amplitude=observed, predicted=theory,
                                     semidiscrete_exact=np.exp(-i*z), norm_l2=l2(u)))
    return rows


def startup():
    """Predict first-step temporal error using orthogonality of two sine modes."""
    rows = []
    m, horizon = 64, 8
    for k in (15, 31):
        for weight in (0.002, 0.05, 0.25):
            components = [(1, 1.0), (k, weight)]
            for z_high in (0.5, 1.5, 2.0, 3.0, 6.0, 12.0, 24.0):
                dt = z_high / rate(m, k)
                ref = profile(m, components, dt, discrete=True)
                end_ref = profile(m, components, horizon*dt)
                for name in ("CN", "Rannacher"):
                    terms = []
                    for j, a in components:
                        z = dt * rate(m, j)
                        terms.append(a * (gain(name, z) - np.exp(-z)))
                    predicted = np.linalg.norm(terms) / np.sqrt(2)
                    states = evolve(profile(m, components), dt, horizon, name, [1])
                    observed = l2(states[1] - ref)
                    if abs(observed - predicted) > 1e-12:
                        raise RuntimeError("mixed-mode error predictor failed")
                    rows.append(dict(method=name, intervals=m, high_mode=k, weight=weight,
                                     z_high=z_high, dt=dt, first_l2=observed,
                                     predicted_l2=predicted, threshold=STARTUP_TOL,
                                     meets_threshold=bool(observed <= STARTUP_TOL),
                                     first_high_amplitude=amplitude(states[1], k),
                                     final_time=horizon*dt,
                                     final_linf_pde=max(abs(states[horizon]-end_ref))))
    return rows


def cost():
    """Fixed grid, common output times and error metric; setup included in timing."""
    rows, winners = [], []
    m, end = 128, 1/32
    components = [(1, 1.0), (15, 0.1)]
    u = profile(m, components)
    fractions = (1/16, 1/8, 1/4, 1/2, 1.0)
    refs = [profile(m, components, end*f) for f in fractions]
    for name in ALL:
        for steps in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            dt = end / steps
            if name == "FTCS" and dt * rate(m, m-1) > 2:
                continue
            saves = [int(steps*f) for f in fractions]
            evolve(u, dt, steps, name, saves)  # warm up, outside timer
            samples = []
            for _ in range(5):
                start = perf_counter()
                states = evolve(u, dt, steps, name, saves)
                samples.append(perf_counter() - start)
            err = max(max(abs(states[i] - ref)) for i, ref in zip(saves, refs))
            rows.append(dict(method=name, intervals=m, steps=steps, dt=dt, T=end,
                             max_sampled_linf=err, median_ms=1000*median(samples),
                             min_ms=1000*min(samples), max_ms=1000*max(samples),
                             **{f"run_{i}_ms":1000*s for i,s in enumerate(samples, 1)}))
        for target in COST_TARGETS:
            eligible = [r for r in rows if r["method"] == name and r["max_sampled_linf"] <= target]
            if eligible:
                best = min(eligible, key=lambda r: r["median_ms"])
                winners.append(dict(target=target, method=name, eligible=len(eligible),
                                    steps=best["steps"], error=best["max_sampled_linf"],
                                    median_ms=best["median_ms"]))
            else:
                winners.append(dict(target=target, method=name, eligible=0, steps=None,
                                    error=None, median_ms=None))
    return rows, winners
