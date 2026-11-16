"""Figures for the orbitals and the FEM mesh."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
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


def plot_orbitals(
    orb: dict[str, np.ndarray],
    energies: dict[str, float],
    evaluator,
    a_star_nm: float,
    estados: list[tuple[str, int, int, str]],
    r_zoom_nm: float,
    title: str,
    outfile: str | Path,
    npts: int = 401,
) -> None:
    """2 x 4 grid of |psi|^2 for the first eight states."""
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0)
    im = None

    x = np.linspace(-r_zoom_nm, r_zoom_nm, npts)
    X, Y = np.meshgrid(x, x)

    for ax, (name, _n, _m, _kind) in zip(axs.ravel(), estados):
        Z = evaluator(orb[name], X / a_star_nm, Y / a_star_nm)
        W = np.abs(Z) ** 2
        W = W / np.nanmax(W)
        im = ax.imshow(
            W,
            extent=[-r_zoom_nm, r_zoom_nm, -r_zoom_nm, r_zoom_nm],
            origin="lower",
            cmap="inferno",
            norm=norm,
            interpolation="bilinear",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}   $E={energies[name]:.4f}$ meV", fontsize=11)

    fig.colorbar(
        im, ax=axs, shrink=0.75, pad=0.02, label=r"$|\psi|^2/|\psi|^2_{\max}$"
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_analytic_orbitals(
    material,
    estados,
    r_zoom_nm: float,
    title: str,
    outfile: str | Path | None = None,
    npts: int = 401,
):
    """2 x 4 grid of analytic |psi|^2 for the first eight states."""
    from .analitica import energia, orbital_grid

    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0)
    im = None

    for ax, (name, n, m, kind) in zip(axs.ravel(), estados):
        X, Y, psi = orbital_grid(
            n, m, kind=kind, r_max_nm=r_zoom_nm, npts=npts, material=material
        )
        W = np.abs(psi) ** 2
        W = W / np.nanmax(W)
        im = ax.imshow(
            W,
            extent=[-r_zoom_nm, r_zoom_nm, -r_zoom_nm, r_zoom_nm],
            origin="lower",
            cmap="inferno",
            norm=norm,
            interpolation="bilinear",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}   $E={energia(n, material):.4f}$ meV",
                     fontsize=11)

    fig.colorbar(
        im, ax=axs, shrink=0.75, pad=0.02, label=r"$|\psi|^2/|\psi|^2_{\max}$"
    )
    fig.suptitle(title, fontsize=14)
    if outfile is not None:
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
