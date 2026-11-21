# mth321_project3_team17
MTH321 Project 3 — 1D heat equation: FTCS, Crank–Nicolson and Rannacher startup.

## Week 10 starting problem

All variables are dimensionless. Solve `u_t = u_xx` on `0 < x < 1` up to `T = 0.1` with zero temperature at both endpoints. The smooth initial profile is `sin(pi*x)`, which satisfies both boundary values. Its exact temperature is `exp(-pi²*t) sin(pi*x)`. The heat energy estimate gives uniqueness and continuous dependence on the initial data. The centred second difference approximates the positive-sign second derivative and has negative eigenvalues under these boundary conditions.

Run the initial reference experiment from the repository root with Python 3:

```sh
python3 experiments/heat_baseline.py
```

The script advances FTCS on 64 equal intervals for 1024 equal steps. It checks every node against the exact PDE solution and independently against the discrete sine mode after 1024 updates. The PDE comparison includes discretisation error. The discrete-mode comparison checks the implemented update to round-off.

## Next experiment

Use the same centred spatial stencil and endpoint values for FTCS and Crank–Nicolson (CN). For each discrete sine mode with decay rate `mu > 0`, put `z = mu*dt`. The predicted one-step amplitude multipliers are `1-z` for FTCS and `(1-z/2)/(1+z/2)` for CN. The exact semidiscrete multiplier is `exp(-z)`. Select the highest grid mode and measure its signed amplitude on both sides of `z = 2`. Growth for FTCS and sign reversal for CN above the threshold are distinct predictions. A disagreement well above round-off with the predicted multiplier on this grid calls for checking the implementation or the derivation.

For spatial convergence, refine the grid and show that halving the time step again has negligible effect. For temporal convergence, keep the grid fixed and compare with its exact semidiscrete sine-mode evolution. Expected orders are two in space for both methods, one in time for FTCS and two in time for CN. Keep FTCS inside its stability limit during convergence tests.

After both baseline schemes pass, replace the first CN step with two backward Euler half-steps. On mixed low- and high-frequency initial data, compare the early signed high-mode amplitude, final maximum error and measured cost at a common error target. Agree on the tolerance before running the comparison. Weekly role assignments and the peer challenge remain for the team.
