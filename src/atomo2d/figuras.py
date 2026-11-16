"""Figures for the orbitals and the FEM mesh."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


def plot_mesh(mesh: dict, a_star_nm: float, title: str,
              outfile: str | Path, zoom_nm: float = 35.0) -> None:
    """Plot the triangular mesh and a central zoom."""
    pts = mesh["pts"] * a_star_nm
    tris = mesh["tris"]
    tri = Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.triplot(tri, color="0.25", linewidth=0.15)
    ax.set_aspect("equal")
    ax.set_xlim(-1.02 * pts[:, 0].max(), 1.02 * pts[:, 0].max())
    ax.set_ylim(-1.02 * pts[:, 1].max(), 1.02 * pts[:, 1].max())
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(mesh_title(mesh, title))

    axins = ax.inset_axes([0.56, 0.56, 0.42, 0.42])
    axins.triplot(tri, color="0.25", linewidth=0.5)
    axins.set_xlim(-zoom_nm, zoom_nm)
    axins.set_ylim(-zoom_nm, zoom_nm)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("central zoom", fontsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="C3")

    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def mesh_title(mesh: dict, title: str) -> str:
    return (
        f"{title}: {mesh['nv']} nodes, {len(mesh['tris'])} triangles "
        f"(P2, {mesh['ndof']} dof)"
    )


def _etiqueta_energia(e_mev: float) -> str:
    """Energy label: eV for atomic scales, meV for semiconductor scales."""
    if abs(e_mev) >= 1000.0:
        return f"$E = {e_mev / 1000.0:.3f}$ eV"
    return f"$E = {e_mev:.4f}$ meV"


def _panel(densidades, energias, estados, r_nm, title, outfile):
    """2 x 4 grid of probability densities in the thesis style.

    Same look as the thesis figures: rainbow colour map on a light-blue
    background, linear scale, disk with a black outline and no axes.
    """
    from .fem import dibujar_orbital

    fig, axs = plt.subplots(2, 4, figsize=(16, 8.6))
    for ax, (name, _n, _m, _kind) in zip(axs.ravel(), estados):
        dibujar_orbital(ax, densidades[name], r_nm)
        ax.set_title(f"{name}    {_etiqueta_energia(energias[name])}",
                     fontsize=13, pad=8)
    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5, w_pad=1.0)
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return None
    return fig


def plot_orbitals(orb, energies, evaluator, a_star_nm, estados, r_zoom_nm,
                  title, outfile, npts: int = 501):
    """Probability density |psi|^2 of the FEM orbitals (thesis style)."""
    x = np.linspace(-r_zoom_nm, r_zoom_nm, npts)
    X, Y = np.meshgrid(x, x)
    dens = {
        name: evaluator(orb[name], X / a_star_nm, Y / a_star_nm)
        for name, *_ in estados
    }
    return _panel(dens, energies, estados, r_zoom_nm, title, outfile)


def plot_analytic_orbitals(material, estados, r_zoom_nm, title,
                           outfile=None, npts: int = 501):
    """Probability density |psi|^2 of the analytic orbitals (thesis style)."""
    from .analitica import energia, orbital_grid

    dens, ener = {}, {}
    for name, n, m, kind in estados:
        _, _, psi = orbital_grid(n, m, kind=kind, r_max_nm=r_zoom_nm,
                                 npts=npts, material=material)
        dens[name] = psi
        ener[name] = energia(n, material)
    return _panel(dens, ener, estados, r_zoom_nm, title, outfile)
