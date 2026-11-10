#!/usr/bin/env python3
"""Demo: magnetic soft continuum rod bending under the flange magnet.

Physical setup (medical scale):
- magnetic soft rod: d = 2 mm, L = 80 mm, silicone composite (E = 5 MPa),
  magnetized along its tangent (total moment ~3e-4 A m^2)
- permanent magnet D50x25 N52 held 60 mm above the rod tip (robot flange)

Plots (saved as PNG into --outdir):
  1. rod shape in the xz-plane for several magnet tilt angles,
  2. tip deflection and tip rotation vs magnet tilt angle.

Usage:
  python3 scripts/plot_rod.py [--outdir ~/rod_figs]
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from continuum_rod_model import CosseratRod  # noqa: E402
from magnetic_field_model import CylinderMagnet  # noqa: E402

# --- rod parameters (medical scale) ---------------------------------------
DIAMETER = 2e-3        # m
LENGTH = 80e-3         # m
E_MODULUS = 5.0e6      # Pa (silicone composite)
I = np.pi * DIAMETER ** 4 / 64.0
J = np.pi * DIAMETER ** 4 / 32.0
EI = E_MODULUS * I     # 3.93e-6 N m^2
GJ = (E_MODULUS / 3.0) * J
MOMENT_TOTAL = 1.5e-3  # A m^2, total magnetic moment of the rod
MU = MOMENT_TOTAL / LENGTH   # A m^2 / m along the tangent

# --- magnet (matches the URDF: D50x25 N52) --------------------------------
MAG_RADIUS = 0.025
MAG_LENGTH = 0.025
MAG_POS = np.array([0.0, 0.0, 0.14])   # 60 mm above the rod tip


def make_magnet(tilt_deg):
    """Magnet tilted about the y-axis (field bends the rod in the xz-plane)."""
    m = CylinderMagnet(radius=MAG_RADIUS, length=MAG_LENGTH, remanence=1.45,
                       n_radial=16, n_azimuth=32)
    m.set_pose(MAG_POS, rpy=(0.0, np.radians(tilt_deg), 0.0))
    return m


def solve_rod(magnet):
    rod = CosseratRod(length=LENGTH, stiffness=(EI, EI, GJ),
                      magnetization=(0.0, 0.0, MU))
    return rod.solve(magnet, n_points=101)


def plot_shapes(outdir):
    angles = [0.0, 30.0, 60.0, 90.0]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(angles)))
    for ang, c in zip(angles, colors):
        sol = solve_rod(make_magnet(ang))
        ax.plot(sol.positions[:, 0] * 1e3, sol.positions[:, 2] * 1e3,
                '-', color=c, lw=2.5,
                label=f'magnet tilt {ang:.0f} deg')
        ax.plot(sol.tip_position[0] * 1e3, sol.tip_position[2] * 1e3,
                'o', color=c, ms=6)
    # magnet position + axis arrow
    ax.plot(MAG_POS[0] * 1e3, MAG_POS[2] * 1e3, 'rs', ms=12,
            label='magnet (D50x25)')
    ax.axhline(0.0, color='k', lw=1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title('Magnetic soft rod (2x80 mm) bending under the flange '
                 'magnet\nbase clamped at the origin')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    path = os.path.join(outdir, 'rod_shapes.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sweep(outdir):
    angles = np.linspace(0.0, 90.0, 13)
    tip_x, tip_angle, tip_z = [], [], []
    for ang in angles:
        sol = solve_rod(make_magnet(ang))
        tip_x.append(sol.tip_position[0] * 1e3)
        tip_z.append(sol.tip_position[2] * 1e3)
        t = sol.tip_tangent
        tip_angle.append(np.degrees(np.arctan2(t[0], t[2])))
    tip_x = np.array(tip_x)
    tip_z = np.array(tip_z)
    tip_angle = np.array(tip_angle)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax1.plot(angles, tip_x, 'b-o', lw=2, label='tip x')
    ax1.plot(angles, tip_z - LENGTH * 1e3, 'g--s', lw=1.5,
             label='tip z (shortening)')
    ax1.set_ylabel('tip position [mm]')
    ax1.set_title('Rod tip response vs magnet tilt angle')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(angles, tip_angle, 'r-o', lw=2)
    ax2.set_xlabel('magnet tilt [deg]')
    ax2.set_ylabel('tip tangent angle [deg]')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'rod_sweep.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=os.path.expanduser('~/rod_figs'),
                        help='directory for the PNG figures')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f'rod: d={DIAMETER*1e3:.0f} mm, L={LENGTH*1e3:.0f} mm, '
          f'E={E_MODULUS/1e6:.0f} MPa')
    print(f'  EI = {EI:.3e} N m^2,  GJ = {GJ:.3e} N m^2')
    print(f'  magnetic moment: {MOMENT_TOTAL:.1e} A m^2 '
          f'({MU:.4f} A m^2 / m)')
    print(f'magnet: D{2*MAG_RADIUS*1e3:.0f}x{MAG_LENGTH*1e3:.0f} mm N52 at '
          f'{MAG_POS} m')

    for ang in (0.0, 45.0, 90.0):
        sol = solve_rod(make_magnet(ang))
        t = sol.tip_tangent
        print(f'  tilt {ang:4.0f} deg -> tip ({sol.tip_position[0]*1e3:7.2f}, '
              f'{sol.tip_position[2]*1e3:6.2f}) mm, '
              f'tip angle {np.degrees(np.arctan2(t[0], t[2])):6.1f} deg, '
              f'residual {sol.residual_norm:.1e}')

    p1 = plot_shapes(args.outdir)
    p2 = plot_sweep(args.outdir)
    print('figures saved:')
    for p in (p1, p2):
        print(f'  {p}')


if __name__ == '__main__':
    main()