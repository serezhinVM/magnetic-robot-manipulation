"""Simple field classes with the magnet-model interface.

All classes provide ``field_and_gradient(points)`` returning
(B (N,3), J (N,3,3)) with J[i, j] = dB_i/dx_j, matching
``magnetic_field_model.magnet``.
"""
import numpy as np


class UniformField:
    """Constant field B0, zero gradient. Useful for analytic validation
    and as a reference field for rod tests."""

    def __init__(self, B):
        self.B = np.asarray(B, dtype=float).reshape(3)

    def field_and_gradient(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        B = np.tile(self.B, (pts.shape[0], 1))
        J = np.zeros((pts.shape[0], 3, 3))
        return B, J

    def field(self, points):
        return self.field_and_gradient(points)[0]

    def gradient(self, points):
        return self.field_and_gradient(points)[1]


class LinearField:
    """Field with constant gradient: B = B0 + G (p - p0).

    ``G`` is the Jacobian (G[i, j] = dB_i/dx_j, constant).  Note: an
    arbitrary constant G is not curl-free; for physical fields G must be
    symmetric with zero trace.  This class is mainly a test/analysis tool.
    """

    def __init__(self, B0, G, p0=(0.0, 0.0, 0.0)):
        self.B0 = np.asarray(B0, dtype=float).reshape(3)
        self.G = np.asarray(G, dtype=float).reshape(3, 3)
        self.p0 = np.asarray(p0, dtype=float).reshape(3)

    def field_and_gradient(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        B = self.B0 + (pts - self.p0) @ self.G.T
        J = np.tile(self.G, (pts.shape[0], 1, 1))
        return B, J

    def field(self, points):
        return self.field_and_gradient(points)[0]

    def gradient(self, points):
        return self.field_and_gradient(points)[1]
