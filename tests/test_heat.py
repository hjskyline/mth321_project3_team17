import unittest

import numpy as np

from experiments.heat import amplitude, evolve, gain, mode, rate


class HeatTests(unittest.TestCase):
    def test_direct_small_grid(self):
        # Independent dense matrix, including the two boundary-adjacent rows.
        u = np.array([0.4, -0.3, 0.8])
        dt = 0.013
        a = 16 * np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
        ident = np.eye(3)
        cn = np.linalg.solve(ident - dt*a/2, ident + dt*a/2)
        be = np.linalg.inv(ident - dt*a/2)
        for name, predicted in [
            ("FTCS", (ident + dt*a) @ (ident + dt*a) @ u),
            ("CN", cn @ cn @ u), ("Rannacher", cn @ be @ be @ u),
        ]:
            np.testing.assert_allclose(evolve(u, dt, 2, name)[2], predicted, atol=1e-14)

    def test_signed_mode_near_threshold(self):
        m, k = 32, 31
        for z in (1.9, 2.1, 16.0):
            for name in ("FTCS", "CN", "Rannacher"):
                u = evolve(mode(m, k), z/rate(m, k), 1, name)[1]
                self.assertAlmostEqual(amplitude(u, k), gain(name, z), places=12)

    def test_zero_and_single_interior_node(self):
        for name in ("FTCS", "CN", "Rannacher"):
            self.assertEqual(evolve([0.0], 0.1, 2, name)[2][0], 0.0)
            result = evolve([1.0], 0.1, 1, name)[1][0]
            self.assertAlmostEqual(result, gain(name, 0.8))

    def test_initial_snapshot_and_input_preservation(self):
        initial = mode(8, 2)
        states = evolve(initial, 0.01, 3, "CN", save=[0, 2])
        self.assertEqual(set(states), {0, 2, 3})
        np.testing.assert_array_equal(initial, states[0])
        np.testing.assert_array_equal(initial, mode(8, 2))
