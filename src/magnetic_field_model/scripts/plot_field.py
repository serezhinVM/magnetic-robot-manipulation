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
    """|B| heatmap + field lines + direction arrows in the xz-plane.

    The sampling grid is cell-centered so that NO point lands exactly on the
    magnet surfaces (x = +-R, z = +-L/2) or its cap edge, where the field is
    discontinuous / singular and the surface-charge quadrature is invalid.
    """
    n = 120
    half = 0.15
    cell = 2 * half / n
    xs = -half + (np.arange(n) + 0.5) * cell       # 2.5 mm cells, offset
    X, Z = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    B = magnet.field(pts)
    mag = np.linalg.norm(B, axis=1).reshape(X.shape) * 1e3  # mT

    # physics sanity: the magnitude map must be symmetric in x and z.
    # Points within 5 mm of the magnet boundary are excluded: the field is
    # physically singular at the cap edge and the quadrature noise there is
    # not a symmetry violation of the model.
    dx = np.maximum(np.abs(X) - RADIUS, 0.0)
    dz = np.maximum(np.abs(Z) - LENGTH / 2, 0.0)
    far = np.hypot(dx, dz) > 5e-3
    if (np.abs(mag - mag[:, ::-1])[far].max() > 1e-6 * mag.max() or
            np.abs(mag - mag[::-1, :])[far].max() > 1e-6 * mag.max()):
        print('WARNING: |B| grid is not symmetric -- check the field model')

    fig, ax = plt.subplots(figsize=(7, 6))
    # vmax = 700 mT: the physical maximum (|B| ~ 650-725 mT inside/at the
    # pole faces); values near the cap-edge singularity would otherwise
    # stretch the scale and flatten the contrast.
    im = ax.pcolormesh(X * 1e3, Z * 1e3, mag, shading='auto', cmap='viridis',
                       vmin=0.0, vmax=700.0)
    fig.colorbar(im, ax=ax, label='|B| [mT]')
    # magnet cross-section outline
    ax.add_patch(plt.Rectangle((-RADIUS * 1e3, -LENGTH / 2 * 1e3),
                               2 * RADIUS * 1e3, LENGTH * 1e3,
                               fill=False, edgecolor='r', linewidth=2))

    # field lines: every loop crosses the equator z = 0, so symmetric seeds
    # there produce continuous, closed, strictly symmetric field lines.
    # NOTE: start_points are in data coordinates (mm here, like the grid).
    seed_x = np.array([12., 22., 32., 45., 62., 85., 115.])
    start = np.array([[sx, 0.0] for sx in seed_x] +
                     [[-sx, 0.0] for sx in seed_x] +
                     [[0.0, 50.0]])                # axial line
    ax.streamplot(xs * 1e3, xs * 1e3, B[:, 0].reshape(X.shape),
                  B[:, 2].reshape(X.shape), color='w', linewidth=0.9,
                  density=2.0, arrowsize=0.9, start_points=start)

    # direction arrows: fixed length, direction only (the magnitude is the
    # background color), on a sparse symmetric grid
    aq = np.array([-125., -90., -55., -20., 20., 55., 90., 125.]) / 1e3
    XQ, ZQ = np.meshgrid(aq, aq)
    BQ = magnet.field(np.column_stack([XQ.ravel(), np.zeros(XQ.size),
                                       ZQ.ravel()]))
    U = BQ[:, 0].reshape(XQ.shape)
    V = BQ[:, 2].reshape(XQ.shape)
    nrm = np.hypot(U, V)
    nrm[nrm < 1e-12] = 1.0
    ax.quiver(XQ * 1e3, ZQ * 1e3, U / nrm, V / nrm, color='w',
              pivot='middle', angles='xy', scale_units='xy', scale=0.2,
              width=0.0025, alpha=0.85)

    ax.set_xlim(-half * 1e3, half * 1e3)
    ax.set_ylim(-half * 1e3, half * 1e3)
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
    """|grad B| (Frobenius norm of the Jacobian) in the xz-plane.

    Same cell-centered grid as plot_field_map: no sample point lands on the
    magnet surfaces or cap edge, where the gradient is singular.
    """
    n = 120
    half = 0.12
    cell = 2 * half / n
    xs = -half + (np.arange(n) + 0.5) * cell       # 2 mm cells, offset
    X, Z = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    J = magnet.gradient(pts)
    gnorm = np.linalg.norm(J, axis=(1, 2)).reshape(X.shape)  # T/m

    # physics sanity: the gradient map must be symmetric in x and z
    # (excluding the 5 mm band around the singular cap edge, see above)
    dx = np.maximum(np.abs(X) - RADIUS, 0.0)
    dz = np.maximum(np.abs(Z) - LENGTH / 2, 0.0)
    far = np.hypot(dx, dz) > 5e-3
    if (np.abs(gnorm - gnorm[:, ::-1])[far].max() > 1e-6 * gnorm.max() or
            np.abs(gnorm - gnorm[::-1, :])[far].max() > 1e-6 * gnorm.max()):
        print('WARNING: |grad B| grid is not symmetric -- check the field model')

    fig, ax = plt.subplots(figsize=(7, 6))
    gplot = np.clip(gnorm, 1e-2, 1e3)  # log-scale range
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
                            remanence=REMANENCE,
                            n_radial=64, n_azimuth=128)

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
