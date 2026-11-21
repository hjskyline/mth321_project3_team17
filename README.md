# MTH321 Project 3 · Team 17

When is a damping step worth its cost in a heat equation simulation? We compare FTCS and Crank–Nicolson (CN), then replace the first CN step with two backward Euler half-steps (Rannacher startup). The experiment distinguishes stability, accuracy at the first measured time and final-time accuracy.

## Reproduce the experiments

Use Python 3.12. In a fresh virtual environment, run `python -m pip install -r requirements.txt` once. From the repository root run:

```sh
python reproduce.py
```

This runs four unit tests and the original standard-library reference case, recomputes the five CSV tables and renders four plots and a nine-frame animation. Expect timing columns to change between runs. The numerical experiments use no random inputs. [Runtime metadata](results/environment.json) records the reference machine's OS, architecture and package versions.

## Model and numerical methods

All quantities are dimensionless. On the unit interval we solve $u_t=u_{xx}$, with zero temperature at both endpoints. The initial profiles are finite sums of sine modes and satisfy the boundary conditions. Integration by parts gives $d\|u\|_2^2/dt=-2\|u_x\|_2^2$; the sine-series solution gives existence and this estimate gives uniqueness and continuous dependence. Mode $k$ decays by $\exp[-(k\pi)^2 t]$. The sine baseline uses $T=0.1$. The other end times are specified below or recorded per configuration.

With $M$ equal intervals and $h=1/M$, store only the $M-1$ interior values. The centred difference $L_h$ has diagonal $-2/h^2$ and off-diagonals $1/h^2$. Dropping the known zero endpoints gives the boundary-adjacent rows. Forward Euler on $U'=L_hU$ gives $U^{n+1}=(I+\Delta t L_h)U^n$. Averaging the spatial term at the two time levels gives $(I-\Delta t L_h/2)U^{n+1}=(I+\Delta t L_h/2)U^n$. Each backward Euler half-step solves with the same left-hand matrix as CN. Two half-steps cover the first full interval; subsequent intervals use CN.

For these homogeneous Dirichlet boundaries the grid sine vectors diagonalise $L_h$ with eigenvalues $-\mu_k$, where $\mu_k=4\sin^2(k\pi/(2M))/h^2$ for $1\le k<M$. With $z=\mu_k\Delta t$ the amplification factors are $1-z$ (FTCS), $(1-z/2)/(1+z/2)$ (CN), and $(1+z/2)^{-2}$ (the startup interval). Testing the largest $\mu_k$ gives the exact finite-grid FTCS stability limit $\mu_{M-1}\Delta t\le2$; $r=\Delta t/h^2\le1/2$ suffices for every grid. CN is contractive for all modes, yet its gain changes sign at $z=2$ and approaches $-1$ as $z$ increases. Track signed mode amplitude as well as the norm.

[Solver implementation](experiments/heat.py) uses NumPy array operations for our FTCS and CN right-hand sides. SciPy sparse LU (`splu`, defaults) factors the constant implicit matrix once per run; there is no iterative stopping tolerance. Tests check the boundary rows against a separately assembled dense matrix. All runs use an integer number of full steps to end at their stated physical time.

## Experiments and evidence

| Study | Design | Evidence |
|---|---|---|
| Space | $M=16,32,64,128$, $T=0.025$, mode 1, $r=0.005$. Compare to the continuous exact solution. At each grid halve the time step again and measure the change relative to the original error. | [CSV](results/convergence.csv) · [Convergence plot](results/convergence.png) |
| Time | Fix $M=32$, $T=0.05$, mode 3. Double the step count from 128 to 1024 and compare with the exact grid evolution $\exp(-\mu_3 t)$ to isolate time error. | [CSV](results/convergence.csv) |
| Stability | At $M=64$ put the highest mode just below, at and above $z=2$ and track twelve signed updates. Repeat with mode 1 at identical step sizes. Compare each observed amplitude to its predicted discrete gain raised to the step count. | [CSV](results/stability.csv) · [Stability plot](results/stability.png) |
| Startup | For $M=64$, add mode 15 or 31 to mode 1 with weight 0.002, 0.05 or 0.25. Sweep seven high-mode step sizes from 0.5 to 24. Test a first-step absolute discrete $L^2$ tolerance of 0.01. Also measure continuous-PDE maximum error after eight steps. Within a configuration both methods share the same end time; across step sizes the end time varies. | [CSV](results/startup.csv) · [Startup plot](results/startup.png) · [Transient animation](results/startup.gif) |
| Cost | Fix $M=128$, $T=1/32$, and initial profile mode 1 plus 0.1 times mode 15. Measure maximum nodal error against the exact PDE at $T/16,T/8,T/4,T/2,T$. Choose the smallest median runtime among tested stable configurations meeting the same target. | [All trials](results/cost_trials.csv) · [Matched costs](results/matched_cost.csv) · [Cost plot](results/cost.png) |

**Using the startup criterion.** For initial modal weights $a_k$ and a candidate gain $g$, grid orthogonality predicts a first-step squared temporal error of $\tfrac12\sum_k a_k^2[g(z_k)-\exp(-z_k)]^2$. If CN meets the tolerance, ordinary startup suffices. If only the damped version meets it, use damping. If both exceed the threshold, retest with a smaller step. Across the 42 configurations, both met the tolerance in 24, only the damped version in 10, and neither in 8. For mode 31, weight 0.25 and $z=12$, damping reduced first-step error from 0.12627 to 0.003607. With weight 0.002 at the same $z$, ordinary CN was already within tolerance. These decisions concern first-step temporal error on the stated grid. Spatial error and a different quantity of interest may change the choice.

**Reading the other results.** The finest-pair spatial orders were 1.99995 (FTCS) and 1.99994 (CN). Halving the time step changed FTCS spatial-study errors by at most 1.55% of the measured error. The finest-pair time orders were 0.99727 and 2.00000. In the recorded cost trial at target 0.002, FTCS needed at least 1024 steps for stability among tested configurations; CN met the error target with 32 steps and had the shortest recorded median. At 0.0005 both implicit variants qualified at 64 steps and their measured times were close. Timing includes matrix assembly, factorisation, updates and copying the five sample states; reference generation, error calculation, plotting and file output are excluded. Each configuration used one untimed warm-up and five timed runs. Error bars show the measured range. Timings depend on hardware and load, and the fastest configuration is selected only from the tested grid of step counts.

**Scope for discussion.** These results apply to constant diffusivity, homogeneous fixed-temperature boundaries and resolved sine-mode data. The cost metric samples five times and can miss earlier transients. The startup cutoff is a chosen absolute tolerance on dimensionless data. Team review should decide whether that cutoff and the sampled times answer the intended physical question.

## Provenance and independent checks

The brief, heat problem in the course problem pack and heat scaffold motivate the setup. The startup procedure is from M. B. Giles and R. Carter, *Convergence analysis of Crank–Nicolson and Rannacher time-marching*, supplied in the course's Inspiring Materials. The mixed-mode tolerance experiment is our application of that procedure. Code and experiment design on this branch were AI-assisted. For review, [unit tests](tests/test_heat.py) recompute implicit updates using an independently assembled dense system; [the baseline](experiments/heat_baseline.py) compares a separate standard-library FTCS implementation with the continuous and discrete exact solutions. The team still needs to independently check claims, document its decisions for the Claim Audit, and record its own roles and contributions.
