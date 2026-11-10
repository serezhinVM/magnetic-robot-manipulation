#!/usr/bin/env python3
"""Visualize the magnetic field of the flange magnet.

Plots (saved as PNG into --outdir):
  1. |B| map in the magnet's xz-plane with field arrows (quiver).
  2. On-axis B_z profile: cylinder model vs point dipole vs closed form.
  3. |grad B| map in the xz-plane (governs the force on point robots).

Usage:
  python3 scripts/plot_field.py [--outdir ~/magnetic_field_figs]
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from magnetic_field_model import (  # noqa: E402
    MU0,
    CylinderMagnet,
    dipole_field,
    cylinder_field_on_axis,
)

RADIUS = 0.025   # m, matches the URDF magnet (D50x25)
LENGTH = 0.025
REMANENCE = 1.45  # T, N52


def plot_field_map(magnet, outdir):
    """|B| heatmap + field arrows in the xz-plane through the magnet axis."""
    xs = np.linspace(-0.15, 0.15, 121)
    zs = np.linspace(-0.15, 0.15, 121)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    B = magnet.field(pts)
    mag = np.linalg.norm(B, axis=1).reshape(X.shape) * 1e3  # mT

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X * 1e3, Z * 1e3, mag, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='|B| [mT]')
    # magnet cross-section outline
    ax.add_patch(plt.Rectangle((-RADIUS * 1e3, -LENGTH / 2 * 1e3),
                               2 * RADIUS * 1e3, LENGTH * 1e3,
                               fill=False, edgecolor='r', linewidth=2))
    # field arrows
    step = 10
    Bx = B[:, 0].reshape(X.shape)[::step, ::step] * 1e3
    Bz = B[:, 2].reshape(X.shape)[::step, ::step] * 1e3
    ax.quiver(X[::step, ::step] * 1e3, Z[::step, ::step] * 1e3, Bx, Bz,
              color='w', scale=3000, width=0.002)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title('|B| and field lines in the xz-plane (red = magnet)')
    ax.set_aspect('equal')
    fig.tight_layout()
    path = os.path.join(outdir, 'field_map.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_on_axis(magnet, outdir):
    """B_z(z) along the magnet axis: cylinder vs dipole vs closed form."""
    z = np.linspace(-0.12, 0.12, 241)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    B_cyl = magnet.field(pts)[:, 2]
    # point dipole is singular at z = 0 -> show it only outside |z| > 5 mm
    z_dip = np.where(np.abs(z) < 5e-3, np.nan, z)
    pts_dip = np.column_stack([np.zeros_like(z), np.zeros_like(z),
                               np.nan_to_num(z_dip, nan=1.0)])
    with np.errstate(invalid='ignore'):
        B_dip = dipole_field([0, 0, magnet.moment], pts_dip)[:, 2]
    B_dip = np.where(np.isnan(z_dip), np.nan, B_dip)
    B_exact = cylinder_field_on_axis(z, RADIUS, LENGTH, magnet.magnetization)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z * 1e3, B_cyl * 1e3, 'b-', lw=2, label='cylinder model (quadrature)')
    ax.plot(z * 1e3, B_exact * 1e3, 'r--', lw=1.5, label='closed form')
    ax.plot(z * 1e3, B_dip * 1e3, 'g:', lw=1.5, label='point dipole (far field)')
    ax.axvspan(-LENGTH / 2 * 1e3, LENGTH / 2 * 1e3, color='r', alpha=0.15,
               label='magnet body')
    ax.set_xlabel('z along magnet axis [mm]')
    ax.set_ylabel('B_z [mT]')
    ax.set_title('On-axis field: cylinder vs dipole approximation')
    ax.set_ylim(-100, 1700)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'on_axis_field.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gradient_map(magnet, outdir):
    """|grad B| (Frobenius norm of the Jacobian) in the xz-plane."""
    xs = np.linspace(-0.12, 0.12, 121)
    zs = np.linspace(-0.12, 0.12, 121)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    J = magnet.gradient(pts)
    gnorm = np.linalg.norm(J, axis=(1, 2)).reshape(X.shape)  # T/m

    fig, ax = plt.subplots(figsize=(7, 6))
    gplot = np.where(gnorm > 1e-3, gnorm, 1e-3)  # floor for log scale
    im = ax.pcolormesh(X * 1e3, Z * 1e3, gplot, shading='auto', cmap='magma',
                       norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e3))
    fig.colorbar(im, ax=ax, label='|grad B| [T/m] (log scale)')
    ax.add_patch(plt.Rectangle((-RADIUS * 1e3, -LENGTH / 2 * 1e3),
                               2 * RADIUS * 1e3, LENGTH * 1e3,
                               fill=False, edgecolor='c', linewidth=2))
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title('|grad B| in the xz-plane (force map for point robots)')
    ax.set_aspect('equal')
    fig.tight_layout()
    path = os.path.join(outdir, 'gradient_map.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default=os.path.expanduser(
        '~/magnetic_field_figs'), help='directory for the PNG figures')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    magnet = CylinderMagnet(radius=RADIUS, length=LENGTH,
                            remanence=REMANENCE)

    print(f'magnet: D{2*RADIUS*1e3:.0f}x{LENGTH*1e3:.0f} mm, '
          f'Br = {REMANENCE} T')
    print(f'  magnetization M = {magnet.magnetization:.3e} A/m')
    print(f'  dipole moment  m = {magnet.moment:.4f} A*m^2')

    # quick numeric sanity printout
    for d in (0.05, 0.1, 0.2):
        B = magnet.field([[0.0, 0.0, d]])[0, 2]
        Bd = dipole_field([0, 0, magnet.moment], [[0.0, 0.0, d]])[0, 2]
        print(f'  B_z({d*1e3:5.0f} mm) = {B*1e3:8.3f} mT   '
              f'(dipole: {Bd*1e3:8.3f} mT)')

    p1 = plot_field_map(magnet, args.outdir)
    p2 = plot_on_axis(magnet, args.outdir)
    p3 = plot_gradient_map(magnet, args.outdir)
    print('figures saved:')
    for p in (p1, p2, p3):
        print(f'  {p}')


if __name__ == '__main__':
    main()
