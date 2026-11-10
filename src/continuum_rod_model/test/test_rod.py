"""Validation tests for the Cosserat rod solver.

Analytic references:
1. Zero load  -> perfectly straight rod.
2. Constant tip moment -> exact circular arc (kappa = M / EI).
3. Small tip force -> Euler-Bernoulli cantilever: delta = F L^3 / (3 EI),
   tip rotation = F L^2 / (2 EI).
4. Uniformly distributed transverse load -> delta = q L^4 / (8 EI).
5. Strong uniform magnetic field -> rod magnetized along its tangent
   aligns the tip with the field (energy minimum U = -mu . B).
6. Field classes: LinearField gradient check.
"""
import numpy as np
import pytest

from continuum_rod_model import (
    CosseratRod,
    UniformField,
    LinearField,
    quat_to_matrix,
    matrix_to_quat,
)

EI = 2.0e-6      # N m^2, bending stiffness
GJ = 2.0e-6      # N m^2, torsional stiffness
L = 0.08         # m


def make_rod(**kw):
    return CosseratRod(length=L, stiffness=(EI, EI, GJ), **kw)


ZERO_FIELD = UniformField([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestQuaternionHelpers:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            # random rotation via QR of a random matrix
            A = rng.normal(size=(3, 3))
            R, _ = np.linalg.qr(A)
            if np.linalg.det(R) < 0:
                R[:, 0] = -R[:, 0]
            q = matrix_to_quat(R)
            np.testing.assert_allclose(quat_to_matrix(q), R, atol=1e-12)

    def test_identity(self):
        np.testing.assert_allclose(quat_to_matrix([1, 0, 0, 0]),
                                   np.eye(3), atol=1e-15)


class TestFieldClasses:
    def test_linear_field_gradient(self):
        G = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -2.0]]) / 10.0  # symmetric, trace-free
        fld = LinearField([0.01, 0.0, 0.02], G, p0=(0.1, 0.0, 0.0))
        pts = np.array([[0.2, 0.3, -0.1], [0.0, 0.0, 0.5]])
        B, J = fld.field_and_gradient(pts)
        expected = np.array([0.01, 0.0, 0.02]) + (pts - np.array([0.1, 0.0, 0.0])) @ G.T
        np.testing.assert_allclose(B, expected, rtol=1e-12)
        np.testing.assert_allclose(J, np.tile(G, (2, 1, 1)), rtol=1e-12)


# ---------------------------------------------------------------------------
# zero load: straight rod
# ---------------------------------------------------------------------------

class TestZeroLoad:
    def test_straight_rod(self):
        rod = make_rod()
        sol = rod.solve(ZERO_FIELD, base_position=(0.1, -0.2, 0.3))
        # straight along +z from the base
        np.testing.assert_allclose(
            sol.positions[:, 0], 0.1, atol=1e-10)
        np.testing.assert_allclose(
            sol.positions[:, 1], -0.2, atol=1e-10)
        np.testing.assert_allclose(
            sol.positions[:, 2], 0.3 + sol.s, atol=1e-10)
        np.testing.assert_allclose(
            sol.tip_tangent, [0, 0, 1], atol=1e-10)
        assert sol.residual_norm < 1e-9

    def test_rotated_base(self):
        # base rotated 90 deg about y -> rod along +x
        R0 = np.array([[0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0]])
        rod = make_rod()
        sol = rod.solve(ZERO_FIELD, base_rotation=R0)
        np.testing.assert_allclose(
            sol.positions[-1], [L, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(sol.tip_tangent, [1, 0, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# constant tip moment: exact circular arc
# ---------------------------------------------------------------------------

class TestTipMomentArc:
    def test_circular_arc(self):
        M = 1.0e-5            # N m, about +y
        rod = make_rod()
        sol = rod.solve(ZERO_FIELD, tip_torque=(0.0, M, 0.0))
        kappa = M / EI
        theta = kappa * L
        tip_expected = np.array([(1 - np.cos(theta)) / kappa,
                                 0.0,
                                 np.sin(theta) / kappa])
        np.testing.assert_allclose(sol.tip_position, tip_expected,
                                   rtol=1e-5, atol=1e-9)
        # tangent rotates in the xz-plane: t = (sin theta, 0, cos theta)
        np.testing.assert_allclose(
            sol.tip_tangent, [np.sin(theta), 0.0, np.cos(theta)],
            rtol=1e-5, atol=1e-9)
        # curvature is constant along the rod
        np.testing.assert_allclose(sol.curvature,
                                   np.tile([0.0, kappa, 0.0], (len(sol.s), 1)),
                                   rtol=1e-5, atol=1e-7)
        assert sol.residual_norm < 1e-8


# ---------------------------------------------------------------------------
# small tip force: Euler-Bernoulli cantilever
# ---------------------------------------------------------------------------

class TestEulerBernoulli:
    def test_tip_force_small_deflection(self):
        F = 3.0e-5           # N, along +x -> small rotation ~0.05 rad
        rod = make_rod()
        sol = rod.solve(ZERO_FIELD, tip_force=(F, 0.0, 0.0))
        delta_expected = F * L ** 3 / (3 * EI)
        rot_expected = F * L ** 2 / (2 * EI)
        assert sol.tip_position[0] == pytest.approx(delta_expected, rel=5e-3)
        # tip tangent: t_x ~ sin(tip rotation)
        assert sol.tip_tangent[0] == pytest.approx(rot_expected, rel=1e-2)

    def test_distributed_load_sag(self):
        # constant transverse load q [N/m]: delta = q L^4 / (8 EI)
        q = 1.0e-3
        rod = make_rod(body_force=(q, 0.0, 0.0))
        sol = rod.solve(ZERO_FIELD)
        delta_expected = q * L ** 4 / (8 * EI)
        assert sol.tip_position[0] == pytest.approx(delta_expected, rel=5e-3)
        rot_expected = q * L ** 3 / (6 * EI)
        assert sol.tip_tangent[0] == pytest.approx(rot_expected, rel=1e-2)


# ---------------------------------------------------------------------------
# magnetic loading
# ---------------------------------------------------------------------------

class TestMagnetic:
    def test_alignment_in_uniform_field(self):
        # rod magnetized along its tangent in a strong uniform field along x:
        # the rod turns to align the tangent with the field direction
        # (finite stiffness -> ~85% alignment at this field strength)
        mu_per_len = 0.02            # A m^2 / m
        B0 = 0.05                    # T
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        sol = rod.solve(UniformField([B0, 0.0, 0.0]))
        t = sol.tip_tangent
        assert t[0] > 0.8
        assert sol.residual_norm < 1e-8

    def test_elastica_first_integral(self):
        # Independent conservation law of the planar magnetic elastica
        # (theta measured from +z, field along +x):
        #   EI/2 * (dtheta/ds)^2 + mu B sin(theta) = mu B sin(theta_tip)
        # Validating this checks the magnetic torque implementation against
        # an analytic integral of motion, not just against itself.
        mu_per_len, B0 = 0.02, 0.05
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        sol = rod.solve(UniformField([B0, 0.0, 0.0]))
        tangents = np.einsum('nij,j->ni', sol.rotations, [0, 0, 1.0])
        theta = np.arctan2(tangents[:, 0], tangents[:, 2])
        # world-frame curvature y-component = dtheta/ds
        k_world = np.einsum('nij,nj->ni', sol.rotations, sol.curvature)
        lhs = 0.5 * EI * k_world[:, 1] ** 2 + mu_per_len * B0 * np.sin(theta)
        rhs = mu_per_len * B0 * np.sin(theta[-1])
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-12)

    def test_weak_field_matches_analytic(self):
        # uniform field -> distributed moment tau = mu B [N m/m] about y;
        # small-deflection cantilever: delta = tau L^3 / (3 EI),
        # tip rotation = tau L^2 / (2 EI)
        mu_per_len = 0.05
        B0 = 1e-5
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        sol = rod.solve(UniformField([B0, 0.0, 0.0]))
        tau = mu_per_len * B0
        delta_expected = tau * L ** 3 / (3 * EI)
        rot_expected = tau * L ** 2 / (2 * EI)
        assert sol.tip_position[0] == pytest.approx(delta_expected, rel=1e-3)
        assert sol.tip_tangent[0] == pytest.approx(rot_expected, rel=1e-2)

    def test_magnetic_torque_beats_stiffness(self):
        # magnetic bending number  mu B L^2 / EI  >> 1 must produce
        # a large deflection;  << 1 a tiny one
        mu_per_len = 0.05
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        strong = rod.solve(UniformField([0.05, 0.0, 0.0]))
        weak = rod.solve(UniformField([1e-5, 0.0, 0.0]))
        assert strong.tip_position[0] > 0.5 * L    # almost fully turned
        assert weak.tip_position[0] < 1e-4         # essentially straight

    def test_gradient_force_pulls_rod(self):
        # rod magnetized along its tangent in a field B_z = g * x:
        # energy U = -mu_z * B_z(x) -> distributed force q = mu * g [N/m]
        # along +x; small-deflection cantilever: delta = q L^4 / (8 EI).
        g = 0.5                     # T/m
        mu_per_len = 1e-3           # A m^2 / m along body z (tangent)
        G = np.zeros((3, 3))
        G[2, 0] = g                 # dB_z/dx = g
        fld = LinearField([0.0, 0.0, 0.0], G)
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        sol = rod.solve(fld)
        q = mu_per_len * g
        delta_expected = q * L ** 4 / (8 * EI)
        assert sol.tip_position[0] == pytest.approx(delta_expected, rel=0.02)


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_strong_load_converges(self):
        # large magnetic load must still converge thanks to continuation
        rod = make_rod(magnetization=(0.0, 0.0, 0.2))
        sol = rod.solve(UniformField([0.1, 0.05, 0.0]),
                        tip_force=(1e-4, 0.0, 0.0),
                        tip_torque=(0.0, 1e-5, 0.0),
                        n_points=401)
        assert sol.residual_norm < 1e-8
        # inextensibility: centerline length equals L (chord sum from the
        # sampled points slightly underestimates the arc length)
        seg = np.linalg.norm(np.diff(sol.positions, axis=0), axis=1)
        assert np.sum(seg) == pytest.approx(L, rel=1e-3)
        assert np.all(seg <= (sol.s[1] - sol.s[0]) * 1.001)

    def test_initial_guess_reuse(self):
        rod = make_rod(magnetization=(0.0, 0.0, 0.02))
        fld = UniformField([0.01, 0.0, 0.0])
        sol1 = rod.solve(fld)
        # extract f0, m0 and pass as a guess
        guess = np.concatenate([sol1.force_body[0], sol1.moment_body[0]])
        sol2 = rod.solve(fld, initial_guess=guess)
        np.testing.assert_allclose(sol2.tip_position, sol1.tip_position,
                                   rtol=1e-6, atol=1e-10)

    def test_energy_decreases_toward_field(self):
        # final orientation has lower magnetic energy than the initial one
        # U = -sum(mu . B) ds  (uniform field => -mu L (t_avg . B))
        mu_per_len = 0.03
        B0 = np.array([0.02, 0.0, 0.0])
        rod = make_rod(magnetization=(0.0, 0.0, mu_per_len))
        sol = rod.solve(UniformField(B0))
        t0 = sol.rotations[0] @ np.array([0, 0, 1.0])
        t_avg = np.mean(np.einsum('nij,j->ni', sol.rotations,
                                  np.array([0, 0, 1.0])), axis=0)
        u0 = -mu_per_len * L * np.dot(t0, B0)
        u1 = -mu_per_len * L * np.dot(t_avg, B0)
        assert u1 < u0