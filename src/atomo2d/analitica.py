"""Analytic solution of the 2D hydrogen atom / hydrogenic impurity.

The derivation follows the thesis appendix B.  In polar coordinates the
Hamiltonian is

    H = -hbar^2/(2 mu) [d2/dr2 + (1/r) d/dr + (1/r^2) d2/dphi2]
        - e^2/(4 pi eps0 epsr r).

Separation of variables, psi(r, phi) = R_nm(r) Phi_m(phi), gives

    Phi_m(phi) = exp(i m phi) / sqrt(2 pi),      |m| <= n
    R_nm(r)   = N_nm rho^|m| exp(-rho/2) L_{n-|m|}^{2|m|}(rho),
    rho       = 2 r / ((n + 1/2) a*),
    E_n       = -Ry* / (n + 1/2)^2,             n = 0, 1, 2, ...

For a semiconductor, a* and Ry* are the effective Bohr radius and Rydberg.
The normalization N_nm is fixed by

    integral_0^inf |R_nm(r)|^2 r dr = 1.

The thesis leaves N_nm implicit; here it is computed by quadrature, so the
radial functions are normalized for any (n, m).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_genlaguerre

from .fem import GAAS, Material


def energia(n: int, material: Material = GAAS) -> float:
    """Energy of the 2D hydrogenic level ``n`` in meV."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    return -material.ry_star_mev / (n + 0.5) ** 2


def energia_ev(n: int, material: Material = GAAS) -> float:
    """Energy of the 2D hydrogenic level ``n`` in eV."""
    return energia(n, material) / 1000.0


def _rho(r_nm: np.ndarray | float, n: int, material: Material):
    return 2.0 * np.asarray(r_nm) / ((n + 0.5) * material.a_star_nm)


def _u(rho: np.ndarray | float, n: int, m: int):
    """Unnormalized radial function in the dimensionless variable rho."""
    k = n - abs(m)
    return rho ** abs(m) * np.exp(-rho / 2.0) * eval_genlaguerre(
        k, 2 * abs(m), rho
    )


@lru_cache(maxsize=128)
def _norm(n: int, m: int, material_name: str, a_star_nm: float,
          ry_star_mev: float) -> float:
    """Normalization constant N_nm such that int |R|^2 r dr = 1."""
    if a_star_nm <= 0:
        raise ValueError("a_star_nm must be positive")

    def integrand(x: float) -> float:
        rho = 2.0 * x / (n + 0.5)
        return float(_u(rho, n, m) ** 2) * x

    integral, _ = quad(integrand, 0.0, np.inf, limit=200)
    return 1.0 / (a_star_nm * np.sqrt(integral))


def normalizacion(n: int, m: int, material: Material = GAAS) -> float:
    """Normalization constant N_nm in 1/length."""
    return _norm(n, m, material.name, material.a_star_nm,
                 material.ry_star_mev)


def radial(r_nm: np.ndarray | float, n: int, m: int,
           material: Material = GAAS) -> np.ndarray:
    """Normalized radial function R_nm(r) in 1/nm."""
    rho = _rho(r_nm, n, material)
    return normalizacion(n, m, material) * _u(rho, n, m)


def orbital_grid(
    n: int,
    m: int,
    kind: str = "s",
    r_max_nm: float = 100.0,
    npts: int = 401,
    material: Material = GAAS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a real 2D orbital on a square grid.

    Parameters
    ----------
    n : principal index (0 -> 1s, 1 -> 2s/2p, ...).
    m : angular index (|m| <= n).
    kind : ``"s"`` for m = 0, ``"cos"`` or ``"sin"`` for real combinations.
    r_max_nm : half-width of the grid in nm.
    npts : number of grid points per axis.
    material : ``GAAS`` (default) or ``VACIO``.

    Returns
    -------
    X, Y, psi : arrays in nm and the real wavefunction.
    """
    x = np.linspace(-r_max_nm, r_max_nm, npts)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)

    R = radial(r, n, m, material)
    if kind == "s" or m == 0:
        psi = R / np.sqrt(2.0 * np.pi)
    elif kind == "cos":
        psi = R * np.cos(m * phi) / np.sqrt(np.pi)
    elif kind == "sin":
        psi = R * np.sin(m * phi) / np.sqrt(np.pi)
    else:
        raise ValueError("kind must be 's', 'cos' or 'sin'")
    return X, Y, psi


# Labels used by the validation tables and figures.
ESTADOS = [
    ("1s", 0, 0, "s"),
    ("2s", 1, 0, "s"),
    ("2px", 1, 1, "cos"),
    ("2py", 1, -1, "sin"),
    ("3s", 2, 0, "s"),
    ("3px", 2, 1, "cos"),
    ("3py", 2, -1, "sin"),
    ("3dxy", 2, -2, "sin"),
]
