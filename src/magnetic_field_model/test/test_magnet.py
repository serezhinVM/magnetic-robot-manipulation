"""Unit tests for the magnetic field models.

Validated against independent analytic references:
- point dipole on-axis closed form,
- cylinder on-axis closed form (exact, all z),
- far-field limit of the cylinder == point dipole,
- Maxwell equations outside the magnet (div B = 0, curl B = 0),
- analytic gradients vs central finite differences,
- dipole force vs finite-diff of the potential energy.
"""
import numpy as np
import pytest

from magnetic_field_model import (
    MU0,
    CylinderMagnet,
    DipoleMagnet,
    dipole_field,
    dipole_gradient,
    cylinder_field_on_axis,
    force_on_dipole,
    torque_on_dipole,
    rpy_to_matrix,
    quaternion_to_matrix,
)

RADIUS = 0.025
LENGTH = 0.025
REMANENCE = 1.45


@pytest.fixture(scope='module')
def magnet():
    return CylinderMagnet(radius=RADIUS, length=LENGTH, remanence=REMANENCE)


# ---------------------------------------------------------------------------
# point dipole model
# ---------------------------------------------------------------------------

class TestDipole:
    def test_on_axis_field(self):
        # B_z(0,0,z) = mu0/(4pi) * 2 m / z^3 on the moment axis
        m = np.array([0.0, 0.0, 2.5])
        z = 0.37
        B = dipole_field(m, [[0.0, 0.0, z]])
        expected = MU0 / (4 * np.pi) * 2 * m[2] / z ** 3
        assert B.shape == (1, 3)
        np.testing.assert_allclose(B[0], [0, 0, expected], rtol=1e-12)
        # equatorial: B_z = -mu0/(4pi) * m / r^3
        B_eq = dipole_field(m, [[z, 0.0, 0.0]])
        expected_eq = -MU0 / (4 * np.pi) * m[2] / z ** 3
        np.testing.assert_allclose(B_eq[0], [0, 0, expected_eq], rtol=1e-12)

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        m = np.array([1.3, -0.7, 2.1])
        pts = rng.uniform(-0.4, 0.4, size=(20, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.05]  # away from singularity
        J = dipole_gradient(m, pts)
        eps = 1e-7
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = eps
            num = (dipole_field(m, pts + dp) - dipole_field(m, pts - dp)) / (2 * eps)
            np.testing.assert_allclose(J[:, :, k], num, rtol=1e-6, atol=1e-14)

    def test_divergence_and_curl_free(self):
        # div B = trace(J) = 0 and curl B = 0 (J symmetric) away from source
        rng = np.random.default_rng(7)
        m = np.array([0.4, 1.1, -0.9])
        pts = rng.uniform(-0.5, 0.5, size=(30, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.1]
        J = dipole_gradient(m, pts)
        scale = np.abs(J).max()
        np.testing.assert_allclose(np.trace(J, axis1=1, axis2=2), 0.0,
                                   atol=1e-9 * scale)
        asym = J - np.transpose(J, (0, 2, 1))
        np.testing.assert_allclose(asym, 0.0, atol=1e-9 * scale)


# ---------------------------------------------------------------------------
# cylinder model: exact analytic references
# ---------------------------------------------------------------------------

class TestCylinderOnAxis:
    def test_matches_closed_form_outside(self, magnet):
        # exact B_z(z) on axis, points outside the magnet
        z = np.array([0.02, 0.05, 0.1, 0.25, -0.03, -0.4])
        pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        B = magnet.field(pts)
        expected = cylinder_field_on_axis(z, RADIUS, LENGTH, magnet.magnetization)
        np.testing.assert_allclose(B[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(B[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(B[:, 2], expected, rtol=1e-6)

    def test_matches_closed_form_inside(self, magnet):
        # the formula is also exact inside (B = mu0 (H + M) there)
        z = np.array([0.0, 0.005, -0.008, 0.011])
        pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        B = magnet.field(pts)
        expected = cylinder_field_on_axis(z, RADIUS, LENGTH, magnet.magnetization)
        np.testing.assert_allclose(B[:, 2], expected, rtol=1e-5)

    def test_center_field_value(self, magnet):
        # B(0) = mu0 M L / sqrt(L^2 + 4 R^2)  (classic textbook value)
        M = magnet.magnetization
        expected = MU0 * M * LENGTH / np.sqrt(LENGTH ** 2 + 4 * RADIUS ** 2)
        B = magnet.field([[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(B[0, 2], expected, rtol=1e-6)

    def test_symmetry(self, magnet):
        # B_z even in z, B_x/B_y odd -> zero on axis already covered; check
        # radial antisymmetry off axis: B_r(-rho) = -B_r(rho) at same z
        p = np.array([[0.01, 0.0, 0.06], [-0.01, 0.0, 0.06]])
        B = magnet.field(p)
        np.testing.assert_allclose(B[0, 0], -B[1, 0], rtol=1e-10)
        np.testing.assert_allclose(B[0, 2], B[1, 2], rtol=1e-10)


class TestCylinderFarField:
    def test_converges_to_dipole(self, magnet):
        # at 20x magnet size the field must match the point dipole
        m_vec = np.array([0.0, 0.0, magnet.moment])
        rng = np.random.default_rng(3)
        dirs = rng.normal(size=(15, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        pts = dirs * 2.0  # 2 m away, magnet size 2.5 cm
        B_cyl = magnet.field(pts)
        B_dip = dipole_field(m_vec, pts)
        scale = np.abs(B_dip).max()
        np.testing.assert_allclose(B_cyl, B_dip, atol=5e-3 * scale)

    def test_dipole_magnet_matches_function(self):
        dm = DipoleMagnet(moment=[0.0, 0.0, 1.7])
        pts = np.array([[0.3, 0.1, -0.2], [0.0, 0.4, 0.1]])
        B, J = dm.field_and_gradient(pts)
        np.testing.assert_allclose(B, dipole_field([0, 0, 1.7], pts), rtol=1e-12)
        np.testing.assert_allclose(J, dipole_gradient([0, 0, 1.7], pts), rtol=1e-12)


class TestCylinderGradient:
    def test_matches_finite_difference(self, magnet):
        rng = np.random.default_rng(11)
        # exterior points, kept away from the caps for quadrature accuracy
        pts = rng.uniform(-0.2, 0.2, size=(12, 3))
        keep = np.linalg.norm(pts, axis=1) > 0.06
        pts = pts[keep]
        J = magnet.gradient(pts)
        eps = 1e-7
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = eps
            num = (magnet.field(pts + dp) - magnet.field(pts - dp)) / (2 * eps)
            np.testing.assert_allclose(J[:, :, k], num, rtol=1e-5, atol=1e-9)

    def test_maxwell_outside(self, magnet):
        # div B = 0 and curl B = 0 in the current-free exterior region
        rng = np.random.default_rng(5)
        pts = rng.uniform(-0.3, 0.3, size=(25, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.15]
        J = magnet.gradient(pts)
        scale = np.abs(J).max()
        np.testing.assert_allclose(np.trace(J, axis1=1, axis2=2), 0.0,
                                   atol=1e-6 * scale)
        asym = J - np.transpose(J, (0, 2, 1))
        np.testing.assert_allclose(asym, 0.0, atol=1e-6 * scale)


class TestPose:
    def test_translation_and_rotation(self, magnet):
        # field at the transformed point of a posed magnet equals the
        # rotated field of the unposed magnet
        rpy = (0.3, -0.5, 1.1)
        R = rpy_to_matrix(*rpy)
        t = np.array([0.4, -0.2, 0.3])
        magnet.set_pose(t, rotation=R)
        try:
            p_local = np.array([[0.02, -0.03, 0.15], [0.1, 0.05, -0.2]])
            p_world = p_local @ R.T + t
            B_world, J_world = magnet.field_and_gradient(p_world)
            magnet.set_pose([0, 0, 0], rotation=np.eye(3))
            B_local, J_local = magnet.field_and_gradient(p_local)
            np.testing.assert_allclose(B_world, B_local @ R.T, rtol=1e-10)
            np.testing.assert_allclose(
                J_world, np.einsum('ia,nab,jb->nij', R, J_local, R), rtol=1e-10)
        finally:
            magnet.set_pose([0, 0, 0], rotation=np.eye(3))

    def test_quaternion_equals_rpy(self):
        rpy = (0.2, 0.4, -0.7)
        R1 = rpy_to_matrix(*rpy)
        # convert rpy -> quaternion (x,y,z,w) via scipy-free formula
        cr, sr = np.cos(rpy[0] / 2), np.sin(rpy[0] / 2)
        cp, sp = np.cos(rpy[1] / 2), np.sin(rpy[1] / 2)
        cy, sy = np.cos(rpy[2] / 2), np.sin(rpy[2] / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        R2 = quaternion_to_matrix(x, y, z, w)
        np.testing.assert_allclose(R1, R2, atol=1e-12)

    def test_set_pose_validation(self, magnet):
        with pytest.raises(ValueError):
            magnet.set_pose([0, 0, 0])
        with pytest.raises(ValueError):
            magnet.set_pose([0, 0, 0], rpy=(0, 0, 0), rotation=np.eye(3))
        magnet.set_pose([0, 0, 0], rotation=np.eye(3))


# ---------------------------------------------------------------------------
# force / torque on a point dipole
# ---------------------------------------------------------------------------

class TestForceTorque:
    def test_force_matches_energy_gradient(self, magnet):
        # F = grad(m.B) must equal -grad(U) with U = -m.B, via finite diff
        rng = np.random.default_rng(21)
        m = np.array([1e-4, -2e-4, 3e-4])
        pts = rng.uniform(-0.15, 0.15, size=(8, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.08]
        B, J = magnet.field_and_gradient(pts)
        F = force_on_dipole(m, B, J)
        eps = 1e-7
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = eps
            u_plus = -np.einsum('j,nj->n', m, magnet.field(pts + dp))
            u_minus = -np.einsum('j,nj->n', m, magnet.field(pts - dp))
            num = -(u_plus - u_minus) / (2 * eps)
            np.testing.assert_allclose(F[:, k], num, rtol=1e-5, atol=1e-12)

    def test_torque_aligned_is_zero(self, magnet):
        # torque T = m x B vanishes when m is parallel to B (on the axis)
        m = np.array([0.0, 0.0, 1e-3])
        B = magnet.field([[0.0, 0.0, 0.2]])
        T = torque_on_dipole(m, B)
        np.testing.assert_allclose(T, 0.0, atol=1e-15)

    def test_torque_antiparallel_is_zero(self, magnet):
        # equatorial point: B is antiparallel to z, m along z -> T = m x B = 0
        m = np.array([0.0, 0.0, 1e-3])
        B = magnet.field([[0.15, 0.0, 0.0]])
        assert np.allclose(B[0, :2], 0.0, atol=1e-12) and B[0, 2] < 0
        T = torque_on_dipole(m, B)
        np.testing.assert_allclose(T, 0.0, atol=1e-15)

    def test_torque_matches_energy_derivative(self, magnet):
        # T . k = -dU/d(delta) with U = -m.B and m rotated about axis k
        rng = np.random.default_rng(33)
        m = np.array([1e-4, 2e-4, 1.5e-4])
        p = np.array([0.08, -0.05, 0.12])
        B = magnet.field([p])
        k = np.array([0.0, 1.0, 0.0])  # rotation axis

        def energy(delta):
            # rotate m about k by delta (Rodrigues)
            kd = np.array([k[1] * m[2] - k[2] * m[1],
                           k[2] * m[0] - k[0] * m[2],
                           k[0] * m[1] - k[1] * m[0]])
            m_rot = (m * np.cos(delta) + kd * np.sin(delta)
                     + k * np.dot(k, m) * (1 - np.cos(delta)))
            return -np.dot(m_rot, B[0])

        eps = 1e-6
        dU = -(energy(eps) - energy(-eps)) / (2 * eps)  # = T . k
        T = torque_on_dipole(m, B)
        assert T[0] @ k == pytest.approx(dU, rel=1e-6)
