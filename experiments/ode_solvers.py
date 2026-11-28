import numpy as np
import scipy.integrate as itg
from scipy.integrate._ivp.rk import RungeKutta, rk_step
from scipy.sparse import issparse, eye_array
from scipy.sparse.linalg import splu


class RK21(RungeKutta):
    order = 1
    error_estimator_order = 1
    n_stages = 1
    C = np.array([0.0])
    A = np.array([[0.0]])
    B = np.array([1.0])
    E = np.array([-0.5, 0.5])
    P = np.array([[1.0, 1, -1], [0, -1, 1]])


class FixedStepRK(RungeKutta):
    def __init__(
        self, fun, t0, y0, t_bound, num_steps=50, vectorized=False, **extraneous
    ):
        itg._ivp.common.warn_extraneous(extraneous)

        self.num_steps = num_steps
        self.steps_taken = 0
        self.step_ts = np.linspace(t0, t_bound, num_steps + 1)
        max_step = np.max(np.abs(self.step_ts[1:] - self.step_ts[:-1]))
        first_step = np.abs(self.step_ts[1] - self.step_ts[0])

        super().__init__(
            fun,
            t0,
            y0,
            t_bound,
            max_step=max_step,
            rtol=1e-3,
            atol=1e-6,
            vectorized=vectorized,
            first_step=first_step,
        )

    def _calc_y_f_new(self, t_new, h, h_abs):
        return rk_step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)

    def _step_impl(self):
        t = self.t
        y = self.y

        t_new = self.step_ts[self.steps_taken + 1]
        h = t_new - t
        h_abs = np.abs(h)

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if h_abs < min_step:
            return False, self.TOO_SMALL_STEP

        y_new, f_new = self._calc_y_f_new(t_new, h, h_abs)

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new
        self.steps_taken += 1

        return True, None


class FixedStepRK1(FixedStepRK, RK21):
    pass


class FixedStepRK3(FixedStepRK, itg.RK23):
    pass


class FixedStepRK5(FixedStepRK, itg.RK45):
    pass


class FixedStepRK8(FixedStepRK, itg.DOP853):
    pass


class FixedStepTheta(FixedStepRK1):
    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        jac=None,
        jac_sparsity=None,
        theta=0.0,
        num_steps=50,
        vectorized=False,
        **extraneous,
    ):
        itg._ivp.common.warn_extraneous(extraneous)
        self.theta = theta
        super().__init__(
            fun,
            t0,
            y0,
            t_bound,
            num_steps=num_steps,
            vectorized=vectorized,
        )

        self.jac_factor = None
        self.jac, self.J = itg.Radau._validate_jac(self, jac, jac_sparsity)
        if issparse(self.J):

            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            # TODO: use I = eye_array(self.n, format="csc") after spmatrix removed
            I = self.J.__class__(eye_array(self.n)).tocsc()
        else:

            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I
        self.LU = None

    def _calc_y_f_new(self, t_new, h, h_abs):
        # was lookin at this https://geochum.github.io/posts/theta-method-backward-euler-crank-nicolson/
        theta = self.theta
        y = self.y
        f = self.f
        J = self.J

        def _ensure_LU():
            LU = self.LU
            if LU is None:
                LU = self.lu(self.I - theta * h * J)
                self.LU = LU
            return LU

        if theta == 0.0:
            y_new = y + h * f
        elif theta == "Rannacher":
            theta = 0.5
            self.theta = theta
            LU = _ensure_LU()
            y_new = self.solve_lu(LU, self.solve_lu(LU, y))
        else:
            LU = _ensure_LU()
            y_new = self.solve_lu(LU, y + (1.0 - theta) * h * f)

        f_new = self.fun(t_new, y_new)
        return y_new, f_new
