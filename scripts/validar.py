"""Validate the in-house FEM solver for the 2D hydrogen atom.

Two cases are solved:

* ``vacuum``  : m* = m0, eps_r = 1  -> a* = 0.0529 nm, Ry* = 13.606 eV
* ``GaAs``    : m* = 0.067 m0, eps_r = 12.9 -> a* = 10.19 nm, Ry* = 5.478 meV

For each case the first eight levels are compared with the exact 2D
hydrogenic energies E_n = -Ry*/(n + 1/2)^2.  The script writes
``results/resultados.json``, ``results/energies.csv`` and the orbital and
mesh figures.

No external FEM library is used: the mesh is Delaunay with geometric
refinement, the elements are quadratic Lagrange (P2) and the eigensolver is
SciPy's shift-invert ``eigsh``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
sys.path.insert(0, str(REPO / "src"))

import atomo2d.analitica as an
import atomo2d.fem as fem
from atomo2d.fem import GAAS, VACIO
from atomo2d.figuras import plot_mesh, plot_orbitals


CASOS = [
    {
        "name": "vacuum",
        "material": VACIO,
        "R_nm": 40.0 * VACIO.a_star_nm,
        "R_check_nm": 60.0 * VACIO.a_star_nm,
        "r0_nm": 0.0001 * VACIO.a_star_nm,
        "hmax_nm": 0.2 * VACIO.a_star_nm,
        "zoom_nm": 1.0,
        "mesh_zoom_nm": 0.2,
        "title": "2D hydrogen atom in vacuum",
    },
    {
        "name": "gaas",
        "material": GAAS,
        "R_nm": 200.0,
        "R_check_nm": 300.0,
        "r0_nm": 0.001,
        "hmax_nm": 2.5,
        "zoom_nm": 100.0,
        "mesh_zoom_nm": 35.0,
        "title": "Hydrogenic donor in GaAs",
    },
]


def construir_y_resolver(caso: dict, R_nm: float):
    material = caso["material"]
    mesh = fem.construir_malla(
        R_nm=R_nm,
        r0_nm=caso["r0_nm"],
        n_theta=96,
        alpha=0.18,
        hmax_nm=caso["hmax_nm"],
        incluir_origen=True,
        conectividad="delaunay",
        offset_alternado=False,
        a_star_nm=material.a_star_nm,
    )
    K, M, D = fem.ensamblar(mesh)
    vals, vecs, free, Mff = fem.resolver_modos(mesh, K, M, k=16, sigma=-3.0)
    return mesh, K, M, D, vals, vecs, free, Mff


def fase_relativa(phi: np.ndarray, free: np.ndarray, iref: int) -> np.ndarray:
    """Fix the global phase using the coefficient of a reference node."""
    j = int(np.where(free == iref)[0][0])
    coef = phi[j]
    if abs(coef) < 1e-300:
        return phi
    return phi * np.exp(-1j * np.angle(coef))


def normalizar(v: np.ndarray, Mff) -> np.ndarray:
    return v / np.sqrt(np.real(v.conj() @ (Mff @ v)))


def construir_orbitales(mesh, D, vecs, free, Mff) -> dict[str, np.ndarray]:
    """Build the eight real orbitals with their (n, m) labels."""
    grupos = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]
    desc = fem.descomponer_angular(mesh, D, vecs, free, grupos)

    def phi_m(g, m):
        return g["phi"][:, int(np.argmin(np.abs(g["m"] - m)))]

    pos_free = {int(g): i for i, g in enumerate(free)}
    vertices_libres = [i for i in range(mesh["nv"]) if i in pos_free]
    r_ref = 0.5  # in units of a*
    iref_x = min(
        vertices_libres,
        key=lambda i: np.hypot(
            mesh["p2_pts"][i, 0] - r_ref, mesh["p2_pts"][i, 1]
        ),
    )

    def s_orbital(g):
        phi = fase_relativa(phi_m(g, 0), free, 0)
        return fem.incrustar(normalizar(np.real(phi), Mff), free, mesh["ndof"])

    def real_orbital(g, m, parte):
        phi = fase_relativa(phi_m(g, m), free, iref_x)
        v = np.real(phi) if parte == "re" else np.imag(phi)
        return fem.incrustar(normalizar(v, Mff), free, mesh["ndof"])

    g0, g1, g2 = desc
    return {
        "1s": s_orbital(g0),
        "2s": s_orbital(g1),
        "2px": real_orbital(g1, +1, "re"),
        "2py": real_orbital(g1, +1, "im"),
        "3s": s_orbital(g2),
        "3px": real_orbital(g2, +1, "re"),
        "3py": real_orbital(g2, +1, "im"),
        "3dxy": real_orbital(g2, +2, "im"),
    }


def energias_de_orbitales(orb, K, M, ry_star_mev: float) -> dict[str, float]:
    return {
        name: float((u @ (K @ u)) / (u @ (M @ u)) * ry_star_mev)
        for name, u in orb.items()
    }


def filas_tabla(caso, energias_R, energias_Rcheck):
    material = caso["material"]
    filas = []
    for nombre, n, m, _kind in an.ESTADOS:
        eteo = an.energia(n, material)
        eR = energias_R[nombre]
        eRc = energias_Rcheck[nombre]
        filas.append(
            [
                nombre,
                f"({n},{m:+d})" if m != 0 else f"({n},0)",
                f"{eteo:.4f}",
                f"{eR:.4f}",
                f"{eR - eteo:+.4f}",
                f"{eRc:.4f}",
                f"{eRc - eteo:+.4f}",
            ]
        )
    return filas


def procesar_caso(caso: dict) -> dict:
    material = caso["material"]
    print(f"\n=== {caso['title']} ===")
    print(f"a* = {material.a_star_nm:.6f} nm, Ry* = {material.ry_star_mev:.6f} meV")
    print(f"R = {caso['R_nm']:.4f} nm, r0 = {caso['r0_nm']:.6g} nm")

    print("  solving R ...")
    mesh, K, M, D, vals, vecs, free, Mff = construir_y_resolver(caso, caso["R_nm"])
    orb = construir_orbitales(mesh, D, vecs, free, Mff)
    ev = fem.evaluador_p2(mesh)
    energias = energias_de_orbitales(orb, K, M, material.ry_star_mev)

    print("  solving R_check ...")
    mesh_c, Kc, Mc, Dc, valsc, vecsc, freec, Mffc = construir_y_resolver(
        caso, caso["R_check_nm"]
    )
    orbc = construir_orbitales(mesh_c, Dc, vecsc, freec, Mffc)
    energias_c = energias_de_orbitales(orbc, Kc, Mc, material.ry_star_mev)

    fig_orbitals = REPO / "figures" / f"orbitals_{caso['name']}.png"
    fig_mesh = REPO / "figures" / f"mesh_{caso['name']}.png"
    plot_orbitals(
        orb,
        energias,
        ev,
        material.a_star_nm,
        an.ESTADOS,
        r_zoom_nm=caso["zoom_nm"],
        title=(
            f"{caso['title']} (FEM P2, R = {caso['R_nm']:.3g} nm, "
            f"a* = {material.a_star_nm:.3g} nm)"
        ),
        outfile=fig_orbitals,
    )
    plot_mesh(
        mesh,
        material.a_star_nm,
        caso["title"],
        fig_mesh,
        zoom_nm=caso["mesh_zoom_nm"],
    )

    filas = filas_tabla(caso, energias, energias_c)
    return {
        "name": caso["name"],
        "title": caso["title"],
        "parameters": fem.parametros(material, caso["R_nm"], caso["r0_nm"]),
        "mesh_R": {
            "R_nm": caso["R_nm"],
            "nodes": int(mesh["nv"]),
            "triangles": int(len(mesh["tris"])),
            "ndof": int(mesh["ndof"]),
            "quality": mesh["calidad"],
        },
        "mesh_Rcheck": {
            "R_nm": caso["R_check_nm"],
            "nodes": int(mesh_c["nv"]),
            "triangles": int(len(mesh_c["tris"])),
            "ndof": int(mesh_c["ndof"]),
            "quality": mesh_c["calidad"],
        },
        "energies_R": energias,
        "energies_Rcheck": energias_c,
        "table": filas,
    }


def main() -> None:
    REPO.joinpath("results").mkdir(exist_ok=True)
    REPO.joinpath("figures").mkdir(exist_ok=True)

    resultados = {}
    filas_csv = []
    for caso in CASOS:
        datos = procesar_caso(caso)
        resultados[caso["name"]] = datos
        for fila in datos["table"]:
            filas_csv.append([caso["name"]] + fila)

        print(f"\nTable — {caso['title']} (meV)")
        print(
            f"{'State':<8}{'(n,m)':<8}{'E_theory':>12}"
            f"{'E_FEM_R':>12}{'diff':>10}{'E_FEM_Rc':>12}{'diff':>10}"
        )
        for fila in datos["table"]:
            print(
                f"{fila[0]:<8}{fila[1]:<8}{fila[2]:>12}"
                f"{fila[3]:>12}{fila[4]:>10}{fila[5]:>12}{fila[6]:>10}"
            )

    with (REPO / "results" / "resultados.json").open("w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    with (REPO / "results" / "energies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "case",
                "state",
                "(n,m)",
                "E_theory_meV",
                "E_FEM_R_meV",
                "diff_R",
                "E_FEM_Rcheck_meV",
                "diff_Rcheck",
            ]
        )
        writer.writerows(filas_csv)

    print("\nResults written to", REPO / "results")


if __name__ == "__main__":
    main()
