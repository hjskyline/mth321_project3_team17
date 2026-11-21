"""Figures from recorded measurements; GIF panels use the same fixed axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .heat import evolve, grid, rate
from .studies import ALL, BASE, STARTUP_TOL, profile

COLORS = {"FTCS": "#246490", "CN": "#a94b26", "Rannacher": "#147466"}


def finish(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def convergence_figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, study, variable in zip(axes, ("space", "time"), ("h", "dt")):
        for name, marker in zip(BASE, ("o", "x")):
            group = [r for r in rows if r["study"] == study and r["method"] == name]
            ax.loglog([r[variable] for r in group], [r["l2_error"] for r in group],
                      marker=marker, color=COLORS[name],
                      label=f"{name} (p={group[-1]['pairwise_order']:.2f})")
        ax.set(xlabel={"h":"Grid spacing h","dt":"Time step Δt"}[variable],
               ylabel="Discrete L2 error",
               title={"space":"Against exact PDE solution","time":"Against exact grid evolution"}[study])
        ax.grid(alpha=0.25)
        ax.legend()
    finish(fig, path)


def stability_figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, name in zip(axes, BASE):
        for z in (1.8, 2.0, 2.2):
            group = [r for r in rows if r["z_high"] == z and r["method"] == name and r["mode"] == 63]
            ax.plot([r["step"] for r in group], [r["signed_amplitude"] for r in group],
                    ".-", label=f"z = {z:g}")
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.set(xlabel="Full steps", ylabel="Signed high-mode amplitude", title=name)
        ax.legend()
    finish(fig, path)


def startup_figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
    for ax, a in zip(axes, (0.002, 0.25)):
        for name in ("CN","Rannacher"):
            group = [r for r in rows if r["method"] == name and r["high_mode"] == 31 and r["weight"] == a]
            ax.loglog([r["z_high"] for r in group], [r["first_l2"] for r in group], "o-",
                      color=COLORS[name], label=name)
        ax.axhline(STARTUP_TOL, ls="--", color="0.35", label="Tolerance")
        ax.set(xlabel="High-mode step size z", title=f"Initial high-mode amplitude {a:g}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("First-step temporal L2 error")
    axes[-1].legend()
    finish(fig, path)


def cost_figure(rows, targets, path):
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for name in ALL:
        vals = [r for r in rows if r["method"] == name]
        x = [r["max_sampled_linf"] for r in vals]
        y = np.array([r["median_ms"] for r in vals])
        spread = [y - [r["min_ms"] for r in vals], [r["max_ms"] for r in vals] - y]
        ax.errorbar(x, y, yerr=spread, fmt="o-", capsize=2, color=COLORS[name], label=name)
    for target in targets:
        ax.axvline(target, ls="--", color="0.5", linewidth=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Maximum error over five sampled times",
           ylabel="Median solver time (ms)", title="Time to advance the same mixed profile")
    ax.legend()
    finish(fig, path)


def transient(path):
    m, k, weight, steps = 64, 31, 0.25, 8
    dt = 12.0 / rate(m, k)
    components = [(1, 1), (k, weight)]
    states = {n: evolve(profile(m, components), dt, steps, n, range(steps+1))
              for n in ("CN", "Rannacher")}
    # Remove the exact semidiscrete low mode so the high-frequency transient is visible.
    x = np.r_[0, grid(m), 1]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    lines = {n: ax.plot([], [], color=c, label=n)[0] for n,c in
             (("Exact grid evolution", "0.15"), ("CN", COLORS["CN"]), ("Rannacher", COLORS["Rannacher"]))}
    lines["Exact grid evolution"].set_linestyle("--")
    ax.set(xlim=(0,1), ylim=(-0.28,0.28), xlabel="Position x", ylabel="Temperature minus low mode")
    ax.legend(loc="lower right", fontsize=8)

    def frame(i):
        t = i*dt
        low = profile(m, [(1,1)], t, discrete=True)
        exact = profile(m, components, t, discrete=True)
        for name,line in lines.items():
            u = exact if name == "Exact grid evolution" else states[name][i]
            line.set_data(x, np.r_[0, u-low, 0])
        ax.set_title(f"High-frequency transient  |  step {i}  |  t = {t:.5f}")
        return list(lines.values())

    animation = FuncAnimation(fig, frame, frames=steps+1, interval=500, repeat=True)
    animation.save(path, writer=PillowWriter(fps=2), dpi=90)
    plt.close(fig)
