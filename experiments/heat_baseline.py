"""Week 10 reference run for the dimensionless Dirichlet heat equation."""

from math import exp, pi, sin


def ftcs_step(values, r):
    """Advance the interior nodes, imposing zero at both endpoints."""
    interior = [
        centre + r * (left - 2 * centre + right)
        for left, centre, right in zip(values, values[1:], values[2:])
    ]
    return [0.0, *interior, 0.0]


def main():
    intervals = 64
    h = 1 / intervals
    steps = 1024
    final_time = 0.1
    dt = final_time / steps
    r = dt / h**2
    initial = [0.0, *(sin(pi * j * h) for j in range(1, intervals)), 0.0]
    values = initial.copy()
    for _ in range(steps):
        values = ftcs_step(values, r)

    decay = exp(-pi**2 * final_time)
    gain = 1 - 4 * r * sin(pi * h / 2)**2
    exact_error = max(abs(u - decay * u0) for u, u0 in zip(values, initial))
    scheme_error = max(abs(u - gain**steps * u0) for u, u0 in zip(values, initial))
    print(f"intervals={intervals}, steps={steps}, T={final_time}, r={r:g}")
    print(f"centre: FTCS={values[intervals // 2]:.9f}, PDE={decay:.9f}")
    print(f"max error vs PDE: {exact_error:.3e}")
    print(f"max difference vs discrete mode: {scheme_error:.3e}")
    if exact_error > 2e-4 or scheme_error > 1e-12 or values[0] != 0 or values[-1] != 0:
        raise SystemExit("FAIL: check the stencil, boundary values and time step.")
    print("PASS")


if __name__ == "__main__":
    main()
