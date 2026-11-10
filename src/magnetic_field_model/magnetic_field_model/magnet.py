"""Magnetic field models for permanent magnets (SI units).

Units: B [T], H [A/m], distances [m], magnetization [A/m], moment [A*m^2].

Models
------
* ``dipole_field`` / ``dipole_gradient``: point dipole (far-field) model.
* ``CylinderMagnet``: uniformly magnetized cylinder via the equivalent
  magnetic charge model (two charged end caps) evaluated with
  Gauss-Legendre quadrature.  Valid for all exterior points; accuracy
  degrades within ~5 mm of the magnet surface.
* ``DipoleMagnet``: posed point dipole with the same interface.

Note on B vs H: the charge model computes H = (1/4pi) * int sigma*R/r^3 dA;
the functions return the physical induction B, i.e. mu0*H outside the
magnet body and mu0*(H + M) inside it.

Both ``field`` and ``gradient`` accept batches of points with shape (N, 3).
The Jacobian convention is J[i, j] = d B_i / d x_j.
"""
import numpy as np

MU0 = 4.0e-7 * np.pi
FOUR_PI = 4.0 * np.pi

__all__ = [
    'MU0', 'CylinderMagnet', 'DipoleMagnet',
    'dipole_field', 'dipole_gradient', 'cylinder_field_on_axis',
    'force_on_dipole', 'torque_on_dipole',
    'rpy_to_matrix', 'quaternion_to_matrix',
]


# ---------------------------------------------------------------------------
# rotation helpers (for poses; used later by ROS integration)
# ---------------------------------------------------------------------------

def rpy_to_matrix(roll, pitch, yaw):
    """Rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed axes)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quaternion_to_matrix(x, y, z, w):
    """Rotation matrix from quaternion (x, y, z, w) as used by ROS/tf2."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError('zero-norm quaternion')
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


# ---------------------------------------------------------------------------
# point dipole model
# ---------------------------------------------------------------------------

def dipole_field(moment, points):
    """B field [T] of a point dipole ``moment`` [A m^2] at the origin.

    B(r) = mu0/4pi * (3 (m.r) r / r^5 - m / r^3)
    Returns array (N, 3) for ``points`` of shape (N, 3).
    """
    m = np.asarray(moment, dtype=float).reshape(3)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    r2 = np.einsum('ij,ij->i', pts, pts)
    if np.any(r2 < 1e-18):
        raise ValueError('field point coincides with dipole position')
    inv_r3 = r2 ** -1.5
    inv_r5 = inv_r3 / r2
    mr = pts @ m
    return (MU0 / FOUR_PI) * (3.0 * mr[:, None] * pts * inv_r5[:, None]
                              - m[None, :] * inv_r3[:, None])


def dipole_gradient(moment, points):
    """Jacobian J[i, j] = dB_i/dx_j of ``dipole_field``. Returns (N, 3, 3)."""
    m = np.asarray(moment, dtype=float).reshape(3)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    r2 = np.einsum('ij,ij->i', pts, pts)
    if np.any(r2 < 1e-18):
        raise ValueError('field point coincides with dipole position')
    inv_r5 = r2 ** -2.5
    inv_r7 = inv_r5 / r2
    mr = pts @ m
    eye = np.eye(3)
    # J_ij = C [3 m_j r_i + 3 m_i r_j] r^-5 + 3 (m.r) delta_ij r^-5
    #        - 15 (m.r) r_i r_j r^-7
    t1 = 3.0 * inv_r5[:, None, None] * (
        np.einsum('j,ni->nij', m, pts) + np.einsum('i,nj->nij', m, pts))
    t2 = 3.0 * (mr * inv_r5)[:, None, None] * eye[None, :, :]
    t3 = 15.0 * (mr * inv_r7)[:, None, None] * np.einsum('ni,nj->nij', pts, pts)
    return (MU0 / FOUR_PI) * (t1 + t2 - t3)


def cylinder_field_on_axis(z, radius, length, magnetization):
    """Closed-form B_z on the axis of a cylinder magnetized along z.

    Valid both inside and outside the magnet (B outside = mu0*H, inside
    B = mu0*(H + M) and the formula accounts for that).
    ``z`` may be a scalar or array of axial coordinates [m].
    """
    z = np.asarray(z, dtype=float)
    R, h = float(radius), 0.5 * float(length)
    M = float(magnetization)

    def f(d):
        return d / np.sqrt(d * d + R * R)

    return 0.5 * MU0 * M * (f(z + h) - f(z - h))


# ---------------------------------------------------------------------------
# force / torque on a point dipole (used later for point magnetic robots)
# ---------------------------------------------------------------------------

def force_on_dipole(moment, B, gradB):
    """F = grad(m.B) = J^T m  [N] for a dipole in field B with Jacobian J."""
    m = np.asarray(moment, dtype=float).reshape(3)
    J = np.asarray(gradB, dtype=float)
    return np.einsum('j,nji->ni', m, J)


def torque_on_dipole(moment, B):
    """T = m x B  [N m]."""
    m = np.asarray(moment, dtype=float).reshape(3)
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return np.cross(m, B)


# ---------------------------------------------------------------------------
# uniformly magnetized cylinder (magnetic charge model + quadrature)
# ---------------------------------------------------------------------------

class CylinderMagnet:
    """Uniformly magnetized cylinder (axis = local +z).

    Parameters
    ----------
    radius : cylinder radius [m]
    length : cylinder length [m] (axis along local z, centered at origin)
    remanence : B_r [T] of the magnet material (N52 ~ 1.45 T); the
        uniform magnetization is M = B_r / mu0 [A/m] along local +z.
    n_radial, n_azimuth : Gauss-Legendre quadrature resolution per end cap.
    """

    def __init__(self, radius=0.025, length=0.025, remanence=1.45,
                 n_radial=48, n_azimuth=96):
        self.radius = float(radius)
        self.length = float(length)
        self.remanence = float(remanence)
        self.magnetization = self.remanence / MU0        # [A/m] along +z
        self.volume = np.pi * self.radius ** 2 * self.length
        self.moment = self.magnetization * self.volume   # [A m^2] along +z
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self._build_quadrature(n_radial, n_azimuth)

    # -- geometry ---------------------------------------------------------
    def _build_quadrature(self, n_radial, n_azimuth):
        x, wx = np.polynomial.legendre.leggauss(n_radial)
        y, wy = np.polynomial.legendre.leggauss(n_azimuth)
        rho = 0.5 * self.radius * (x + 1.0)      # [0, R]
        wrho = 0.5 * self.radius * wx
        phi = np.pi * (y + 1.0)                  # [0, 2pi]
        wphi = np.pi * wy
        rr, pp = np.meshgrid(rho, phi, indexing='ij')
        ww = np.outer(wrho, wphi) * rr           # dA = rho drho dphi
        xs = (rr * np.cos(pp)).ravel()
        ys = (rr * np.sin(pp)).ravel()
        w = ww.ravel()
        h = 0.5 * self.length
        top = np.column_stack([xs, ys, np.full_like(xs, h)])
        bot = np.column_stack([xs, ys, np.full_like(xs, -h)])
        self._qp = np.vstack([top, bot])         # (Q, 3) charge points
        M = self.magnetization
        self._qw = np.concatenate([M * w, -M * w])  # sigma = +M top, -M bottom

    def set_pose(self, position, rotation=None, rpy=None, quaternion=None):
        """Set the magnet pose in the world frame.

        Exactly one of ``rotation`` (3x3 matrix), ``rpy`` (roll, pitch, yaw)
        or ``quaternion`` (x, y, z, w) must be given.
        """
        self.position = np.asarray(position, dtype=float).reshape(3)
        given = [a is not None for a in (rotation, rpy, quaternion)]
        if sum(given) != 1:
            raise ValueError('give exactly one of rotation, rpy, quaternion')
        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        elif rpy is not None:
            self.rotation = rpy_to_matrix(*rpy)
        else:
            self.rotation = quaternion_to_matrix(*quaternion)

    # -- field ------------------------------------------------------------
    def _kernel(self, local_points):
        """B and J in the magnet frame from the charged-caps quadrature."""
        n = local_points.shape[0]
        B = np.empty((n, 3))
        J = np.empty((n, 3, 3))
        chunk = 256
        eye = np.eye(3)
        for s in range(0, n, chunk):
            p = local_points[s:s + chunk]
            R = p[:, None, :] - self._qp[None, :, :]        # (n,Q,3)
            r2 = np.einsum('nqk,nqk->nq', R, R)
            inv_r3 = r2 ** -1.5
            inv_r5 = inv_r3 / r2
            qw = self._qw
            Bc = np.einsum('q,nq,nqk->nk', qw, inv_r3, R)
            s1 = np.einsum('q,nq->n', qw, inv_r3)
            s2 = np.einsum('q,nq,nqi,nqj->nij', qw, inv_r5, R, R,
                           optimize=True)
            Jc = -3.0 * s2
            Jc[:, range(3), range(3)] += s1[:, None]
            Bc *= MU0 / FOUR_PI
            Jc *= MU0 / FOUR_PI
            B[s:s + chunk] = Bc
            J[s:s + chunk] = Jc
        return B, J

    def field_and_gradient(self, points):
        """(B (N,3), J (N,3,3)) at world-frame points (N,3).

        Returns the physical induction B everywhere, including inside the
        magnet body where B = mu0 (H + M); the charged-caps quadrature
        computes H and the uniform M is added back for interior points.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        local = (pts - self.position) @ self.rotation   # = R^T (p - t)
        B_l, J_l = self._kernel(local)
        # inside the magnet: B = mu0 H + mu0 M (M along local +z)
        half = 0.5 * self.length
        inside = ((np.abs(local[:, 2]) < half)
                  & (local[:, 0] ** 2 + local[:, 1] ** 2 < self.radius ** 2))
        B_l[inside, 2] += MU0 * self.magnetization
        B = B_l @ self.rotation.T
        J = np.einsum('ia,nab,jb->nij', self.rotation, J_l, self.rotation)
        return B, J

    def field(self, points):
        return self.field_and_gradient(points)[0]

    def gradient(self, points):
        return self.field_and_gradient(points)[1]


class DipoleMagnet:
    """Point dipole with the same pose interface as :class:`CylinderMagnet`."""

    def __init__(self, moment):
        self.moment = np.asarray(moment, dtype=float).reshape(3)
        self.position = np.zeros(3)
        self.rotation = np.eye(3)

    def set_pose(self, position, rotation=None, rpy=None, quaternion=None):
        self.position = np.asarray(position, dtype=float).reshape(3)
        given = [a is not None for a in (rotation, rpy, quaternion)]
        if sum(given) != 1:
            raise ValueError('give exactly one of rotation, rpy, quaternion')
        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        elif rpy is not None:
            self.rotation = rpy_to_matrix(*rpy)
        else:
            self.rotation = quaternion_to_matrix(*quaternion)

    def field_and_gradient(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        local = (pts - self.position) @ self.rotation
        B_l = dipole_field(self.moment, local)
        J_l = dipole_gradient(self.moment, local)
        B = B_l @ self.rotation.T
        J = np.einsum('ia,nab,jb->nij', self.rotation, J_l, self.rotation)
        return B, J

    def field(self, points):
        return self.field_and_gradient(points)[0]

    def gradient(self, points):
        return self.field_and_gradient(points)[1]
