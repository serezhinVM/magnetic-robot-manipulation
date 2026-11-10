"""Quasi-static Cosserat rod solver for magnetic soft continuum robots.

Model
-----
Inextensible, unshearable rod of length L.  State along the arclength s:

* ``r(s)``  -- position of the centerline in the world frame [m]
* ``R(s)``  -- material-frame orientation (world <- body), column 3 = tangent
* ``f(s)``  -- internal force in the body frame [N]
* ``m(s)``  -- internal moment in the body frame [N m]

Governing ODEs (static equilibrium, body-frame form with curvature ``k``):

    dr/ds   = R e3
    dR/ds   = R k^
    df/ds   = -k x f - R^T f_d
    dm/ds   = -k x m - e3 x f - R^T tau_d

with the linear constitutive law  m = K (k - k0), i.e. k = k0 + K^-1 m.

Distributed magnetic loads per unit length (magnetization ``mu`` given in
the body frame [A m^2 / m], field B [T], Jacobian J[i,j] = dB_i/dx_j):

    f_d   = J^T (R mu)           [N/m]      (= grad(mu_w . B))
    tau_d = (R mu) x B           [N m/m]

Boundary conditions: clamped base r(0) = r0, R(0) = R0; free tip with
optional tip force/torque:  R(L) f(L) = F_tip,  R(L) m(L) = M_tip.

Solution: shooting method on (f(0), m(0)) via ``scipy.optimize.root``
with load continuation (lambda = 0 .. 1) for robustness.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

__all__ = ['CosseratRod', 'RodSolution']

_E3 = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# quaternion helpers (internal convention: q = (w, x, y, z))
# ---------------------------------------------------------------------------

def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q):
    """Rotation matrix from quaternion (w, x, y, z)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """Quaternion (w, x, y, z) from rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# solution container
# ---------------------------------------------------------------------------

class RodSolution:
    """Result of a rod solve: sampled kinematics and internal loads."""

    def __init__(self, s, positions, rotations, quaternions, curvature,
                 force_body, moment_body, residual_norm):
        self.s = s                       # (N,) arclength [m]
        self.positions = positions       # (N, 3) world [m]
        self.rotations = rotations       # (N, 3, 3) world <- body
        self.quaternions = quaternions   # (N, 4) (w, x, y, z)
        self.curvature = curvature       # (N, 3) body-frame [1/m]
        self.force_body = force_body     # (N, 3) [N]
        self.moment_body = moment_body   # (N, 3) [N m]
        self.residual_norm = residual_norm

    @property
    def tip_position(self):
        return self.positions[-1]

    @property
    def tip_tangent(self):
        return self.rotations[-1] @ _E3

    @property
    def base_position(self):
        return self.positions[0]

    @property
    def tip_rotation(self):
        return self.rotations[-1]

    def force_world(self):
        return np.einsum('nij,nj->ni', self.rotations, self.force_body)

    def moment_world(self):
        return np.einsum('nij,nj->ni', self.rotations, self.moment_body)

    def __repr__(self):
        return (f'RodSolution(N={len(self.s)}, tip={self.tip_position}, '
                f'residual={self.residual_norm:.2e})')


# ---------------------------------------------------------------------------
# rod solver
# ---------------------------------------------------------------------------

class CosseratRod:
    """Quasi-static magnetic Cosserat rod.

    Parameters
    ----------
    length : rod length L [m]
    stiffness : (EI1, EI2, GJ) or a full 3x3 matrix K [N m^2],
        relating body moment to curvature change: m = K (k - k0)
    natural_curvature : k0 [1/m] in the body frame (straight by default)
    magnetization : magnetic moment per unit length mu [A m^2 / m],
        expressed in the body frame (e.g. (0, 0, mu) for axial magnetization)
    body_force : constant distributed force per unit length in the world
        frame [N/m], e.g. effective weight (gravity - buoyancy)
    """

    def __init__(self, length=0.08, stiffness=(1.0, 1.0, 1.0),
                 natural_curvature=(0.0, 0.0, 0.0),
                 magnetization=(0.0, 0.0, 0.0),
                 body_force=(0.0, 0.0, 0.0)):
        self.length = float(length)
        K = np.asarray(stiffness, dtype=float)
        if K.shape == (3,):
            K = np.diag(K)
        elif K.shape != (3, 3):
            raise ValueError('stiffness must be (3,) or (3,3)')
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.k0 = np.asarray(natural_curvature, dtype=float).reshape(3)
        self.mu = np.asarray(magnetization, dtype=float).reshape(3)
        self.body_force = np.asarray(body_force, dtype=float).reshape(3)

    # -- ODE right-hand side ---------------------------------------------
    def _rhs(self, s, y, field, scale):
        R = quat_to_matrix(y[3:7])
        f = y[7:10]
        m = y[10:13]
        k = self.k0 + self.Kinv @ m

        dr = R @ _E3
        dq = 0.5 * _quat_mult(y[3:7], np.array([0.0, k[0], k[1], k[2]]))

        # distributed loads (magnetic scaled by lambda, body force constant)
        B, J = field.field_and_gradient(y[0:3].reshape(1, 3))
        B = B[0]
        J = J[0]
        mu_w = R @ self.mu
        f_d = scale * (J.T @ mu_w) + self.body_force
        tau_d = scale * np.cross(mu_w, B)
        f_b = R.T @ f_d
        tau_b = R.T @ tau_d

        df = -np.cross(k, f) - f_b
        dm = -np.cross(k, m) - np.cross(_E3, f) - tau_b
        return np.concatenate([dr, dq, df, dm])

    # -- public API -------------------------------------------------------
    def solve(self, field, base_position=(0.0, 0.0, 0.0), base_rotation=None,
              tip_force=(0.0, 0.0, 0.0), tip_torque=(0.0, 0.0, 0.0),
              n_points=101, initial_guess=None, rtol=1e-8, atol=1e-10,
              direct=False, jac0=None, return_jacobian=False):
        """Solve the rod equilibrium under ``field``.

        Parameters
        ----------
        field : object with ``field_and_gradient(points) -> (B, J)``
        base_position, base_rotation : clamped-end pose in the world frame
        tip_force, tip_torque : tip loads in the world frame
        n_points : number of samples along the rod in the returned solution
        initial_guess : optional (f0, m0) guess in the body frame at s = 0
        rtol, atol : ODE integration tolerances
        direct : if True, try the full load (lambda = 1) right away using
            the initial guess and fall back to continuation on failure.
            Useful for warm-started real-time loops.
        jac0 : optional (6, 6) shooting Jacobian from a previous solve
            (see ``return_jacobian``); enables fast warm Newton iterations.
        return_jacobian : if True, ``solve`` returns ``(solution, jacobian)``
            where ``jacobian`` can be passed back as ``jac0`` next time.
        """
        R0 = (np.eye(3) if base_rotation is None
              else np.asarray(base_rotation, dtype=float).reshape(3, 3))
        y0 = np.zeros(13)
        y0[0:3] = np.asarray(base_position, dtype=float).reshape(3)
        y0[3:7] = matrix_to_quat(R0)
        F_tip = np.asarray(tip_force, dtype=float).reshape(3)
        M_tip = np.asarray(tip_torque, dtype=float).reshape(3)

        def integrate(f0, m0, scale, t_eval=None):
            y_start = y0.copy()
            y_start[7:10] = f0
            y_start[10:13] = m0
            sol = solve_ivp(self._rhs, (0.0, self.length), y_start,
                            args=(field, scale), method='RK45',
                            rtol=rtol, atol=atol,
                            dense_output=t_eval is None, t_eval=t_eval)
            if not sol.success:
                raise RuntimeError(f'ODE integration failed: {sol.message}')
            return sol

        def residual(v, scale):
            sol = integrate(v[:3], v[3:], scale)
            yT = sol.y[:, -1]
            RT = quat_to_matrix(yT[3:7])
            f_w = RT @ yT[7:10]
            m_w = RT @ yT[10:13]
            return np.concatenate([f_w - scale * F_tip,
                                   m_w - scale * M_tip])

        guess = (np.zeros(6) if initial_guess is None
                 else np.asarray(initial_guess, dtype=float).reshape(6))

        # ---- shooting Jacobian helpers (for real-time warm starting) ----
        def numeric_jacobian(v, scale):
            """(6, 6) d(residual)/d(f0, m0) by forward differences."""
            r0 = residual(v, scale)
            J = np.empty((6, 6))
            for i in range(6):
                dv = np.zeros(6)
                dv[i] = 1e-6 * max(abs(v[i]), 1e-3)
                J[:, i] = (residual(v + dv, scale) - r0) / dv[i]
            return J

        def newton(v, scale, J, max_iter=12, tol=1e-8):
            """Damped Newton with Broyden Jacobian updates.

            Returns (v, J, ok); each iteration costs 1-2 integrations.
            """
            r = residual(v, scale)
            for _ in range(max_iter):
                rn = float(np.linalg.norm(r))
                if rn < tol:
                    return v, J, True
                if J is None:
                    J = numeric_jacobian(v, scale)
                try:
                    dv = np.linalg.solve(J, -r)
                except np.linalg.LinAlgError:
                    return v, J, False
                # damped line search
                step, ok_ls = 1.0, False
                for _ in range(8):
                    v_new = v + step * dv
                    try:
                        r_new = residual(v_new, scale)
                    except RuntimeError:
                        step *= 0.5
                        continue
                    if np.linalg.norm(r_new) < rn:
                        ok_ls = True
                        break
                    step *= 0.5
                if not ok_ls:
                    return v, J, False
                # Broyden rank-1 update: J += (dr - J dv) dv^T / (dv^T dv)
                dr = r_new - r
                denom = float(dv @ dv)
                if denom > 0.0:
                    J = J + np.outer(dr - J @ dv, dv) / denom
                v, r = v_new, r_new
            return v, J, float(np.linalg.norm(r)) < tol

        # ---- solve: warm Newton first, continuation as fallback ----
        lam, dlam = 0.0, 0.5
        if direct:
            try:
                guess, jac, ok = newton(guess, 1.0, jac0)
                if ok:
                    lam = 1.0
                else:
                    jac0 = jac  # keep progress for the continuation stage
            except RuntimeError:
                pass
        while lam < 1.0 - 1e-12:
            lam_next = min(lam + dlam, 1.0)
            try:
                root_sol = root(residual, guess, args=(lam_next,),
                                method='hybr', tol=1e-10)
                res = residual(root_sol.x, lam_next)
                ok = np.linalg.norm(res) < 1e-8
            except RuntimeError:
                ok = False
                res = np.full(6, np.inf)
            if ok:
                guess = root_sol.x
                lam = lam_next
                dlam = min(dlam * 2.0, 0.5)
            else:
                dlam *= 0.5
                if dlam < 1e-3:
                    raise RuntimeError(
                        f'shooting failed at lambda={lam_next:.4f} '
                        f'(residual {np.linalg.norm(res):.2e}); try a '
                        f'better initial_guess or weaker loads')

        # final integration on a uniform grid
        t_eval = np.linspace(0.0, self.length, int(n_points))
        sol = integrate(guess[:3], guess[3:], 1.0, t_eval=t_eval)
        y = sol.y.T                        # (N, 13)
        N = y.shape[0]
        positions = y[:, 0:3]
        quaternions = y[:, 3:7]
        rotations = np.empty((N, 3, 3))
        curvature = np.empty((N, 3))
        force_body = y[:, 7:10]
        moment_body = y[:, 10:13]
        for i in range(N):
            rotations[i] = quat_to_matrix(quaternions[i])
            curvature[i] = self.k0 + self.Kinv @ moment_body[i]

        res_norm = float(np.linalg.norm(residual(guess, 1.0)))
        solution = RodSolution(t_eval, positions, rotations, quaternions,
                               curvature, force_body, moment_body, res_norm)
        if return_jacobian:
            # one extra finite-difference Jacobian (7 integrations) to be
            # reused as jac0 by the next warm call
            return solution, numeric_jacobian(guess, 1.0)
        return solution