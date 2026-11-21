import unittest

import numpy as np
from scipy.linalg import expm

from experiments.heat import l2
from experiments.startup_control import choose, first_interval
from experiments.transfer import continuous_pulse, grid, initial_data


class StartupTests(unittest.TestCase):
    def test_irregular_profile_against_dense_matrix_exponential(self):
        u = np.array([0.7, 0.0, 1.0, 0.3, 0.5, 0.0, 0.01])
        m, dt = len(u)+1, 0.06
        lap = m*m*(np.diag(-2*np.ones(m-1)) + np.diag(np.ones(m-2),1)
                   + np.diag(np.ones(m-2),-1))
        ref = expm(dt*lap) @ u
        for tol in (0.1, 0.01, 0.001, 0.00001):
            plan = choose(u,dt,tol)
            actual = l2(first_interval(u,dt,plan)-ref)
            self.assertAlmostEqual(actual,plan.predicted_error,places=13)
            self.assertLessEqual(actual,tol)

    def test_pulse_symmetry_and_jump_convention(self):
        u = initial_data(64,"Square pulse")
        self.assertEqual(u[23],0.5)
        np.testing.assert_array_equal(u,u[::-1])
        x = grid(64)
        exact = continuous_pulse(x,0.001,128)
        np.testing.assert_allclose(exact,exact[::-1],atol=1e-14)
        self.assertLessEqual(max(exact),1+1e-14)
        self.assertGreaterEqual(min(exact),-1e-14)
