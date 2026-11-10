"""Closed-loop simulation node: magnet pose (TF) -> field -> robots -> RViz.

Pipeline executed at ``rate`` Hz:

1. read the flange magnet pose from TF (``world`` -> ``magnet_link``) and
   the rod base pose (``world`` -> ``rod_link``),
2. build the magnetic field model (``magnetic_field_model.CylinderMagnet``),
3. solve the quasi-static Cosserat rod (``continuum_rod_model``),
   warm-started from the previous solution (~100 ms per solve),
4. integrate the point-robot dynamics (overdamped dipole in viscous fluid:
   F = grad(m.B), m || B,  v = F / 6*pi*eta*r),
5. publish MarkerArray (rod tube, rod tip, point robots, field arrow) and
   PoseStamped/PoseArray for later control use.

All geometry is expressed in the ``world`` frame (RViz fixed frame).
"""
import time

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformException, Buffer, TransformListener

from magnetic_field_model import CylinderMagnet, quaternion_to_matrix
from continuum_rod_model import CosseratRod

__all__ = ['ContinuumSimNode', 'main']


def quat_from_z_axis(direction):
    """Quaternion (x, y, z, w) rotating +z onto ``direction``."""
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    d = d / n
    cos_a = float(np.clip(d[2], -1.0, 1.0))
    if cos_a > 1.0 - 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    if cos_a < -1.0 + 1e-12:
        return (1.0, 0.0, 0.0, 0.0)     # 180 deg about x
    axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(axis, d)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(cos_a)
    s = np.sin(0.5 * angle)
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
            float(np.cos(0.5 * angle)))


class ContinuumSimNode(Node):
    """Magnetic manipulation simulation: TF -> field -> rod + point robots."""

    def __init__(self):
        super().__init__('continuum_sim_node')

        # ---------------- parameters ----------------
        self.declare_parameter('rate', 5.0)                  # Hz
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('magnet_frame', 'magnet_link')
        self.declare_parameter('rod_base_frame', 'rod_link')

        # magnet (matches the URDF: N52 cylinder D50x25)
        self.declare_parameter('magnet.radius', 0.025)
        self.declare_parameter('magnet.length', 0.025)
        self.declare_parameter('magnet.remanence', 1.45)

        # magnetic soft rod (medical scale)
        self.declare_parameter('rod.length', 0.08)
        self.declare_parameter('rod.diameter', 0.002)
        self.declare_parameter('rod.young_modulus', 5.0e6)   # Pa
        self.declare_parameter('rod.moment', 3.0e-4)         # A m^2 total

        # point robots (overdamped dipole in viscous fluid)
        self.declare_parameter('points.count', 3)
        self.declare_parameter('points.radius', 0.0015)      # m
        self.declare_parameter('points.moment', 1.0e-6)      # A m^2
        self.declare_parameter('points.viscosity', 0.004)   # Pa s (glycerol)
        self.declare_parameter('points.initial_positions',
                               [0.55, -0.02, 0.16,
                                0.555, 0.02, 0.14,
                                0.545, 0.0, 0.17])

        # vessel confinement (horizontal tube along y)
        self.declare_parameter('vessel.center', [0.55, 0.0, 0.15])
        self.declare_parameter('vessel.radius', 0.015)
        self.declare_parameter('vessel.half_length', 0.15)

        p = lambda name: self.get_parameter(name).value   # noqa: E731

        self.world_frame = p('world_frame')
        self.magnet_frame = p('magnet_frame')
        self.rod_base_frame = p('rod_base_frame')

        self.magnet = CylinderMagnet(radius=p('magnet.radius'),
                                     length=p('magnet.length'),
                                     remanence=p('magnet.remanence'),
                                     n_radial=16, n_azimuth=32)
        self.magnet_length = p('magnet.length')

        d = p('rod.diameter')
        E = p('rod.young_modulus')
        I = np.pi * d ** 4 / 64.0
        Jp = np.pi * d ** 4 / 32.0
        self.rod = CosseratRod(
            length=p('rod.length'),
            stiffness=(E * I, E * I, (E / 3.0) * Jp),
            magnetization=(0.0, 0.0, p('rod.moment') / p('rod.length')))
        self.rod_radius = 0.5 * d
        self._rod_guess = None
        self._rod_jac = None

        n = int(p('points.count'))
        self.n_points = n
        self.points_radius = p('points.radius')
        self.points_moment = p('points.moment')
        # Stokes drag coefficient 6*pi*eta*r [N s / m]
        self.points_drag = 6.0 * np.pi * p('points.viscosity') * \
            p('points.radius')
        init = np.asarray(p('points.initial_positions'), dtype=float)
        self.points_pos = init.reshape(-1, 3)[:n].copy()

        self.vessel_center = np.asarray(p('vessel.center'), dtype=float)
        self.vessel_radius = p('vessel.radius')
        self.vessel_half = p('vessel.half_length')

        # ---------------- ROS interfaces ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/magnetic_sim/markers', 10)
        self.pub_rod_tip = self.create_publisher(
            PoseStamped, '/magnetic_sim/rod_tip', 10)
        self.pub_points = self.create_publisher(
            PoseArray, '/magnetic_sim/point_robots', 10)

        rate = p('rate')
        self._last_time = time.monotonic()
        self._dt_nominal = 1.0 / rate
        self.create_timer(self._dt_nominal, self._on_timer)
        self.get_logger().info(
            f'simulation started: {n} point robots, rod L='
            f'{p("rod.length")*1e3:.0f} mm, update rate {rate} Hz')

    # ------------------------------------------------------------------
    def _lookup_pose(self, frame):
        """Return (position (3,), rotation matrix (3,3)) of frame in world."""
        tr = self.tf_buffer.lookup_transform(self.world_frame, frame,
                                             rclpy.time.Time())
        t = tr.transform.translation
        q = tr.transform.rotation
        R = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return np.array([t.x, t.y, t.z]), R

    def _on_timer(self):
        now = time.monotonic()
        dt = min(now - self._last_time, 0.25)   # clamp for stability
        self._last_time = now

        # 1. magnet + rod base poses from TF
        try:
            mag_pos, mag_R = self._lookup_pose(self.magnet_frame)
            base_pos, base_R = self._lookup_pose(self.rod_base_frame)
        except TransformException as exc:
            self.get_logger().warn(f'TF not ready: {exc}',
                                   throttle_duration_sec=5.0)
            return

        # magnet body center is offset from magnet_link origin along its +z
        self.magnet.set_pose(mag_pos + mag_R @ [0.0, 0.0,
                                                0.5 * self.magnet_length],
                             rotation=mag_R)

        # 2+3. rod equilibrium (warm-started: previous solution + Jacobian)
        try:
            sol, self._rod_jac = self.rod.solve(
                self.magnet, base_position=base_pos, base_rotation=base_R,
                n_points=51, initial_guess=self._rod_guess,
                jac0=self._rod_jac, direct=True, rtol=1e-6, atol=1e-8,
                return_jacobian=True)
        except RuntimeError:
            # fall back to full continuation for a hard configuration
            try:
                sol, self._rod_jac = self.rod.solve(
                    self.magnet, base_position=base_pos, base_rotation=base_R,
                    n_points=51, return_jacobian=True)
            except RuntimeError as exc:
                self.get_logger().warn(f'rod solve failed: {exc}',
                                       throttle_duration_sec=5.0)
                return
        self._rod_guess = np.concatenate([sol.force_body[0],
                                          sol.moment_body[0]])

        # 4. point-robot dynamics (overdamped)
        if self.n_points > 0:
            B, J = self.magnet.field_and_gradient(self.points_pos)
            bnorm = np.linalg.norm(B, axis=1, keepdims=True)
            bnorm[bnorm < 1e-12] = 1.0
            m = self.points_moment * B / bnorm          # m || B
            F = np.einsum('nj,nji->ni', m, J)           # F = grad(m.B)
            v = F / self.points_drag
            self.points_pos += v * dt
            self._clamp_points()

        # 5. publish
        self._publish(sol)

    def _clamp_points(self):
        """Keep point robots inside the vessel tube (axis along y)."""
        rel = self.points_pos - self.vessel_center
        # radial distance from the tube axis (x-z plane)
        r = np.hypot(rel[:, 0], rel[:, 2])
        r_max = self.vessel_radius - self.points_radius
        over = r > r_max
        if np.any(over):
            scale = np.where(over, r_max / np.maximum(r, 1e-12), 1.0)
            rel[:, 0] *= scale
            rel[:, 2] *= scale
        # axial limit
        y_max = self.vessel_half - self.points_radius
        rel[:, 1] = np.clip(rel[:, 1], -y_max, y_max)
        self.points_pos = self.vessel_center + rel

    # ------------------------------------------------------------------
    def _publish(self, sol):
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # rod tube: one cylinder marker per segment
        pts = sol.positions
        for i in range(len(pts) - 1):
            seg = pts[i + 1] - pts[i]
            length = float(np.linalg.norm(seg))
            mk = Marker()
            mk.header.frame_id = self.world_frame
            mk.header.stamp = stamp
            mk.ns = 'rod'
            mk.id = i
            mk.type = Marker.CYLINDER
            mk.action = Marker.ADD
            mk.pose.position.x = float(0.5 * (pts[i, 0] + pts[i + 1, 0]))
            mk.pose.position.y = float(0.5 * (pts[i, 1] + pts[i + 1, 1]))
            mk.pose.position.z = float(0.5 * (pts[i, 2] + pts[i + 1, 2]))
            qx, qy, qz, qw = quat_from_z_axis(seg)
            mk.pose.orientation.x = qx
            mk.pose.orientation.y = qy
            mk.pose.orientation.z = qz
            mk.pose.orientation.w = qw
            mk.scale.x = mk.scale.y = 2.0 * self.rod_radius
            mk.scale.z = max(length, 1e-6)
            t = i / max(len(pts) - 2, 1)
            mk.color.r = 0.1 + 0.1 * t
            mk.color.g = 0.25 + 0.35 * t
            mk.color.b = 0.9 - 0.3 * t
            mk.color.a = 1.0
            ma.markers.append(mk)

        # rod tip
        tip = Marker()
        tip.header.frame_id = self.world_frame
        tip.header.stamp = stamp
        tip.ns = 'rod_tip'
        tip.id = 0
        tip.type = Marker.SPHERE
        tip.action = Marker.ADD
        tip.pose.position.x = float(pts[-1, 0])
        tip.pose.position.y = float(pts[-1, 1])
        tip.pose.position.z = float(pts[-1, 2])
        tip.pose.orientation.w = 1.0
        tip.scale.x = tip.scale.y = tip.scale.z = 2.5 * self.rod_radius
        tip.color.r, tip.color.g, tip.color.b, tip.color.a = 0.1, 0.9, 0.9, 1.0
        ma.markers.append(tip)

        # point robots
        colors = [(0.9, 0.15, 0.1), (0.95, 0.55, 0.1), (0.1, 0.7, 0.2),
                  (0.6, 0.2, 0.9), (0.9, 0.2, 0.7), (0.2, 0.6, 0.9)]
        for i in range(self.n_points):
            mk = Marker()
            mk.header.frame_id = self.world_frame
            mk.header.stamp = stamp
            mk.ns = 'point_robots'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(self.points_pos[i, 0])
            mk.pose.position.y = float(self.points_pos[i, 1])
            mk.pose.position.z = float(self.points_pos[i, 2])
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 2.0 * self.points_radius
            c = colors[i % len(colors)]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = c[0], c[1], c[2], 1.0
            ma.markers.append(mk)

        # field direction arrow at the rod tip
        B_tip = self.magnet.field(pts[-1:])[0]
        bnorm = float(np.linalg.norm(B_tip))
        if bnorm > 1e-9:
            arrow = Marker()
            arrow.header.frame_id = self.world_frame
            arrow.header.stamp = stamp
            arrow.ns = 'field'
            arrow.id = 0
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = []
            arrow.pose.position.x = float(pts[-1, 0])
            arrow.pose.position.y = float(pts[-1, 1])
            arrow.pose.position.z = float(pts[-1, 2])
            qx, qy, qz, qw = quat_from_z_axis(B_tip)
            arrow.pose.orientation.x = qx
            arrow.pose.orientation.y = qy
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x = 0.02               # shaft length
            arrow.scale.y = arrow.scale.z = 0.002
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = \
                1.0, 1.0, 0.2, 0.8
            ma.markers.append(arrow)

        self.pub_markers.publish(ma)

        # rod tip pose (for future visual servoing)
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.header.stamp = stamp
        ps.pose.position.x = float(pts[-1, 0])
        ps.pose.position.y = float(pts[-1, 1])
        ps.pose.position.z = float(pts[-1, 2])
        w, x, y, z = sol.quaternions[-1]        # (w, x, y, z)
        ps.pose.orientation.x = float(x)
        ps.pose.orientation.y = float(y)
        ps.pose.orientation.z = float(z)
        ps.pose.orientation.w = float(w)
        self.pub_rod_tip.publish(ps)

        # point-robot poses
        pa = PoseArray()
        pa.header.frame_id = self.world_frame
        pa.header.stamp = stamp
        for i in range(self.n_points):
            pose = Pose()
            pose.position.x = float(self.points_pos[i, 0])
            pose.position.y = float(self.points_pos[i, 1])
            pose.position.z = float(self.points_pos[i, 2])
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pub_points.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = ContinuumSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()