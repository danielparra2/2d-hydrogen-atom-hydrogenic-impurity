"""P2 finite elements for the 2D hydrogen atom and hydrogenic impurities.

The physical problem is

    H = -hbar^2/(2 m*) laplacian - e^2/(4 pi eps0 epsr r)

on a disk of radius R with psi(R) = 0.  The impurity sits at the origin.
Internally the solver uses dimensionless units:

    x~ = x / a*,   E~ = E / Ry*,   H~ = -laplacian~ - 2/r~,

where a* is the effective Bohr radius and Ry* the effective Rydberg.
The material is selected with the `Material` dataclass (`VACIO` for the
vacuum atom, `GAAS` for a donor in GaAs).

The implementation is deliberately explicit: unstructured Delaunay mesh
with geometric refinement towards the origin, quadratic Lagrange elements
(P2), seven-point quadrature and sparse assembly.  No FEM library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.tri as mtri


# ---------------------------------------------------------------------------
# Constantes y materiales
# ---------------------------------------------------------------------------

M0 = 9.109383713928e-31          # kg
E_CHARGE = 1.602176634e-19       # C
HBAR = 1.0545718176461565e-34    # J s
EPS0 = 8.854187818826713e-12     # F/m


@dataclass(frozen=True)
class Material:
    """Effective-mass material parameters.

    The 2D Coulomb problem is solved in units of the effective Bohr radius
    ``a*`` and the effective Rydberg ``Ry*``.  Both are derived from the
    relative effective mass and the static relative permittivity.
    """

    name: str
    m_eff_rel: float
    eps_r: float

    @property
    def m_eff(self) -> float:
        return self.m_eff_rel * M0

    @property
    def a_star_m(self) -> float:
        return 4.0 * np.pi * EPS0 * self.eps_r * HBAR**2 / (
            self.m_eff * E_CHARGE**2
        )

    @property
    def a_star_nm(self) -> float:
        return self.a_star_m * 1e9

    @property
    def ry_star_j(self) -> float:
        return HBAR**2 / (2.0 * self.m_eff * self.a_star_m**2)

    @property
    def ry_star_mev(self) -> float:
        return self.ry_star_j / E_CHARGE * 1e3

    @property
    def c_kinetic(self) -> float:
        """Coefficient ``hbar^2/(2 e m*)`` used in the COMSOL PDE."""
        return HBAR**2 / (2.0 * E_CHARGE * self.m_eff)

    @property
    def permittivity(self) -> float:
        return self.eps_r * EPS0


VACIO = Material("vacuum", m_eff_rel=1.0, eps_r=1.0)
GAAS = Material("GaAs", m_eff_rel=0.067, eps_r=12.9)

# GaAs/AlGaAs parameters kept for traceability with the .mph model.
X_AL = 0.3
V_AL_MEV = 0.6 * (1047.0 * X_AL + 470.0 * X_AL**2)  # meV

# Backwards-compatible module constants (GaAs).
M_EFF_REL = GAAS.m_eff_rel
EPS_R = GAAS.eps_r
M_EFF = GAAS.m_eff
A_STAR = GAAS.a_star_m
RY_STAR = GAAS.ry_star_j
A_STAR_NM = GAAS.a_star_nm
RY_STAR_MEV = GAAS.ry_star_mev
C_GA = GAAS.c_kinetic
E_GA = GAAS.permittivity

R_NM = 200.0
R0_NM = 0.01


def parametros(material: Material = GAAS, R_nm: float = R_NM,
               r0_nm: float = R0_NM) -> dict:
    """Physical parameters in SI and in the units used by the notebooks."""
    return {
        "material": material.name,
        "m0": M0,
        "e": E_CHARGE,
        "hbar": HBAR,
        "eps0": EPS0,
        "m_eff_rel": material.m_eff_rel,
        "m_eff": material.m_eff,
        "eps_r": material.eps_r,
        "a_star_m": material.a_star_m,
        "a_star_nm": material.a_star_nm,
        "Ry_star_J": material.ry_star_j,
        "Ry_star_meV": material.ry_star_mev,
        "c": material.c_kinetic,
        "permittivity": material.permittivity,
        "V_Al_meV": V_AL_MEV if material is GAAS else 0.0,
        "R_nm": R_nm,
        "r0_nm": r0_nm,
    }


# ---------------------------------------------------------------------------
# Malla triangular no estructurada
# ---------------------------------------------------------------------------

def _radios(r0: float, R: float, alpha: float, hmax: float) -> np.ndarray:
    """Radios con espaciado h(r) = min(hmax, alpha*r)."""
    radios = [r0]
    r = r0
    while radios[-1] < R * (1.0 - 1e-13):
        h = min(hmax, alpha * r)
        r = min(R, r + h)
        if r - radios[-1] > 1e-15:
            radios.append(r)
    return np.asarray(radios)


def _calidad(pts: np.ndarray, tris: np.ndarray) -> dict:
    v = pts[tris]
    a = np.linalg.norm(v[:, 1] - v[:, 2], axis=1)
    b = np.linalg.norm(v[:, 2] - v[:, 0], axis=1)
    c = np.linalg.norm(v[:, 0] - v[:, 1], axis=1)
    angulos = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        cosang = (q * q + r * r - p * p) / (2.0 * q * r)
        angulos.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    amin = np.min(angulos, axis=0)
    area = 0.5 * np.abs(
        (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
        - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0])
    )
    return {
        "min_angle": float(amin.min()),
        "median_angle": float(np.median(amin)),
        "p01_angle": float(np.percentile(amin, 1)),
        "min_area": float(area.min()),
        "max_area": float(area.max()),
    }


def _finalizar_malla(
    pts,
    tris,
    radios,
    r0,
    R,
    R_nm,
    r0_nm,
    n_theta,
    incluir_origen,
    offset_alternado,
    conectividad,
    n_hueco=0,
):
    """Completa aristas, nodos P2, fronteras y metadatos de la malla."""
    # Orientacion positiva (antihoraria) para que det(J) > 0.
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (
        v1[:, 1] - v0[:, 1]
    ) * (v2[:, 0] - v0[:, 0])
    swap = area2 < 0
    if np.any(swap):
        tris[swap, 1], tris[swap, 2] = tris[swap, 2].copy(), tris[swap, 1].copy()

    # Aristas unicas.
    loc_edges = np.stack(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1
    )  # (ntri,3,2): e01, e12, e20
    claves = np.sort(loc_edges, axis=2)
    aristas, inv = np.unique(
        claves.reshape(-1, 2), axis=0, return_inverse=True
    )
    inv = inv.reshape(-1, 3)
    counts = np.bincount(inv.ravel(), minlength=len(aristas))
    borde = np.where(counts == 1)[0]

    nv = len(pts)
    ne = len(aristas)
    p2_pts = np.vstack(
        [pts, 0.5 * (pts[aristas[:, 0]] + pts[aristas[:, 1]])]
    )

    # Orden local: [v0,v1,v2, midpoint(e01), midpoint(e12), midpoint(e20)]
    edge_local = inv
    dofs = np.column_stack([tris, nv + edge_local])

    # Frontera exterior e interior.
    borde_r = np.linalg.norm(p2_pts[nv + borde], axis=1)
    r_medio = 0.5 * (r0 + R)
    borde_ext = borde[borde_r > r_medio]
    borde_int = borde[borde_r <= r_medio]
    vert_ext = np.unique(aristas[borde_ext].ravel())
    dirichlet = np.unique(
        np.concatenate([vert_ext, nv + borde_ext]).astype(np.int64)
    )

    return {
        "pts": pts,
        "tris": tris,
        "aristas": aristas,
        "p2_pts": p2_pts,
        "dofs": dofs,
        "nv": nv,
        "ne": ne,
        "ndof": nv + ne,
        "borde": borde,
        "borde_ext": borde_ext,
        "borde_int": borde_int,
        "dirichlet": dirichlet,
        "radios": radios,
        "n_theta": n_theta,
        "hueco_eliminado": n_hueco,
        "incluir_origen": bool(incluir_origen),
        "offset_alternado": bool(offset_alternado),
        "conectividad": conectividad,
        "calidad": _calidad(pts, tris),
        "r0": r0,
        "R": R,
        "R_nm": R_nm,
        "r0_nm": r0_nm,
    }


def _malla_polar(radios, n_theta, incluir_origen):
    """Conectividad estructurada por anillos (alternativa de diagnostico)."""
    tris = []
    nrings = len(radios)
    if incluir_origen:
        for j in range(n_theta):
            tris.append([0, 1 + j, 1 + (j + 1) % n_theta])
        base = 1
    else:
        base = 0
    for i in range(nrings - 1):
        bi = base + i * n_theta
        bo = bi + n_theta
        for j in range(n_theta):
            a = bi + j
            b = bi + (j + 1) % n_theta
            c = bo + (j + 1) % n_theta
            d = bo + j
            tris.append([a, b, c])
            tris.append([a, c, d])
    return np.asarray(tris, dtype=np.int64)


def construir_malla(
    R_nm: float = R_NM,
    r0_nm: float = R0_NM,
    n_theta: int = 96,
    alpha: float = 0.18,
    hmax_nm: float = 3.0,
    incluir_origen: bool = True,
    offset_alternado: bool = True,
    conectividad: str = "delaunay",
    a_star_nm: float = A_STAR_NM,
) -> dict:
    """Malla triangular de un disco con refinamiento geometrico central.

    Por defecto se usa Delaunay sobre anillos concentricos.  Si
    ``incluir_origen=True`` el origen es un nodo y el dominio es el disco
    completo; los triangulos que tocan el origen se integran con una
    cuadratura especial (Duffy) para tratar 1/r.  Si es ``False`` se deja
    un hueco interno pequeno con condicion natural.

    ``conectividad='polar'`` usa una conectividad estructurada por
    anillos; sirve para comparar y para preservar la simetria rotacional.
    """
    if a_star_nm <= 0:
        raise ValueError("a_star_nm must be positive")
    r0 = r0_nm / a_star_nm
    R = R_nm / a_star_nm
    hmax = hmax_nm / a_star_nm

    if conectividad == "polar":
        offset_alternado = False

    radios = _radios(r0, R, alpha, hmax)
    pts = []
    anillo = []
    if incluir_origen:
        pts.append(np.array([[0.0, 0.0]]))
        anillo.append(np.array([-1]))
    for i, r in enumerate(radios):
        offset = 0.5 * (i % 2) if offset_alternado else 0.0
        theta = 2.0 * np.pi * (np.arange(n_theta) + offset) / n_theta
        pts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        anillo.append(np.full(n_theta, i))
    pts = np.vstack(pts)
    anillo = np.concatenate(anillo)

    if conectividad == "delaunay":
        tri = Delaunay(pts)
        tris = tri.simplices.copy()
        if incluir_origen:
            n_hueco = 0
        else:
            hueco = np.all(anillo[tris] == 0, axis=1)
            tris = tris[~hueco]
            n_hueco = int(hueco.sum())
    elif conectividad == "polar":
        tris = _malla_polar(radios, n_theta, incluir_origen)
        n_hueco = 0
    else:
        raise ValueError("conectividad debe ser 'delaunay' o 'polar'")

    mesh = _finalizar_malla(
        pts,
        tris,
        radios,
        r0,
        R,
        R_nm,
        r0_nm,
        n_theta,
        incluir_origen,
        offset_alternado,
        conectividad,
        n_hueco,
    )
    mesh["a_star_nm"] = a_star_nm
    return mesh


# ---------------------------------------------------------------------------
# Elemento P2 y cuadratura
# ---------------------------------------------------------------------------

def p2_shape(xi: float, eta: float):
    """Funciones de forma P2 y sus gradientes en el triangulo de referencia.

    Vertices: 1=(0,0), 2=(1,0), 3=(0,1).
    Nodos 4,5,6: puntos medios de las aristas 1-2, 2-3, 3-1.
    """
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    N = np.array(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ]
    )
    dxi = np.array(
        [
            1.0 - 4.0 * l1,
            4.0 * l2 - 1.0,
            0.0,
            4.0 * (l1 - l2),
            4.0 * l3,
            -4.0 * l3,
        ]
    )
    deta = np.array(
        [
            1.0 - 4.0 * l1,
            0.0,
            4.0 * l3 - 1.0,
            -4.0 * l2,
            4.0 * l2,
            4.0 * (l1 - l3),
        ]
    )
    return N, np.column_stack([dxi, deta])


def p2_shape_array(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Version vectorizada de las funciones de forma P2."""
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    return np.column_stack(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ]
    )


def cuadratura_7():
    """Regla de Dunavant de 7 puntos, grado 5.

    Los pesos ya incluyen el area del triangulo de referencia (1/2),
    de modo que sum(w) = 1/2.
    """
    a1 = 0.059715871789770
    b1 = 0.470142064105115
    a2 = 0.797426985353087
    b2 = 0.101286507323456
    xi = np.array([1 / 3, a1, b1, b1, a2, b2, b2])
    eta = np.array([1 / 3, b1, a1, b1, b2, a2, b2])
    w = np.array(
        [
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
    )
    # Los pesos estandar suman 1 (promedio); se multiplican por el area 1/2.
    return xi, eta, 0.5 * w


# ---------------------------------------------------------------------------
# Ensamblaje
# ---------------------------------------------------------------------------

def ensamblar(mesh: dict, escala_coulomb: float = 1.0):
    """Ensambla K, M y la matriz del momento angular D.

    K_ij = int grad(N_i).grad(N_j) - int (2/r) N_i N_j
    M_ij = int N_i N_j
    D_ij = int N_i (x dN_j/dy - y dN_j/dx)
    """
    pts = mesh["pts"]
    tris = mesh["tris"]
    dofs = mesh["dofs"]
    ntri = len(tris)
    ndof = mesh["ndof"]

    v = pts[tris]  # (ntri,3,2)
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=2)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ = np.linalg.inv(J)
    invJt = np.swapaxes(invJ, 1, 2)

    xi_g, eta_g, w_g = cuadratura_7()
    Ke = np.zeros((ntri, 6, 6))
    Me = np.zeros((ntri, 6, 6))
    De = np.zeros((ntri, 6, 6))

    for xi, eta, w in zip(xi_g, eta_g, w_g):
        N, gN_ref = p2_shape(xi, eta)
        gN = np.einsum("eab,ib->eia", invJt, gN_ref)
        xq = v[:, 0, 0] + xi * J[:, 0, 0] + eta * J[:, 0, 1]
        yq = v[:, 0, 1] + xi * J[:, 1, 0] + eta * J[:, 1, 1]
        r = np.hypot(xq, yq)
        a = -2.0 * escala_coulomb / r
        NN = np.outer(N, N)
        Ke += w * detJ[:, None, None] * (
            np.einsum("eia,eja->eij", gN, gN) + a[:, None, None] * NN
        )
        Me += w * detJ[:, None, None] * NN
        vec = xq[:, None] * gN[:, :, 1] - yq[:, None] * gN[:, :, 0]
        De += w * detJ[:, None, None] * np.einsum("i,ej->eij", N, vec)

    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()

    def _global(Ke):
        return coo_matrix(
            (Ke.ravel(), (rows, cols)), shape=(ndof, ndof)
        ).tocsr()

    K = _global(Ke)
    M = _global(Me)
    D = _global(De)
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    return K, M, D


# ---------------------------------------------------------------------------
# Autovalores
# ---------------------------------------------------------------------------

def resolver_modos(
    mesh: dict,
    K,
    M,
    k: int = 12,
    sigma: float = -1.0,
):
    """Resuelve K u = E M u con shift-invert y devuelve E~ y vectores full."""
    fixed = mesh["dirichlet"]
    free = np.setdiff1d(np.arange(mesh["ndof"]), fixed)
    Kff = K[free][:, free]
    Mff = M[free][:, free]
    vals, vecs = eigsh(
        Kff,
        k=k,
        M=Mff,
        sigma=sigma,
        which="LM",
        tol=1e-10,
        maxiter=20000,
    )
    orden = np.argsort(vals)
    vals = vals[orden]
    vecs = vecs[:, orden]
    for j in range(vecs.shape[1]):
        norma = np.sqrt(vecs[:, j] @ (Mff @ vecs[:, j]))
        vecs[:, j] /= norma
    full = np.zeros((mesh["ndof"], k))
    full[free] = vecs
    return vals, full, free, Mff


# ---------------------------------------------------------------------------
# Postproceso
# ---------------------------------------------------------------------------

def evaluador_p2(mesh: dict):
    """Devuelve una funcion que evalua psi_h en puntos (X,Y) arbitrarios."""
    pts = mesh["p2_pts"]
    tris = mesh["tris"]
    dofs = mesh["dofs"]
    tri_obj = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    finder = tri_obj.get_trifinder()

    def evaluar(u: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Un desplazamiento minimo evita fallos del localizador justo en
        # vertices (por ejemplo, el origen).
        eps = 1e-12
        X = np.asarray(X) + eps
        Y = np.asarray(Y) + eps
        idx = finder(X, Y)
        dtype = np.result_type(u.dtype, np.float64)
        Z = np.full(X.shape, np.nan, dtype=dtype)
        valido = idx >= 0
        if not np.any(valido):
            return Z
        t = idx[valido]
        v = pts[tris[t, :3]]
        x0 = v[:, 0]
        d1 = v[:, 1] - v[:, 0]
        d2 = v[:, 2] - v[:, 0]
        dx = X[valido] - x0[:, 0]
        dy = Y[valido] - x0[:, 1]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        xi = (dx * d2[:, 1] - dy * d2[:, 0]) / det
        eta = (d1[:, 0] * dy - d1[:, 1] * dx) / det
        N = p2_shape_array(xi, eta)
        Z[valido] = np.sum(N * u[dofs[t]], axis=1)
        return Z

    return evaluar


def descomponer_angular(
    mesh: dict,
    D,
    vecs: np.ndarray,
    free: np.ndarray,
    grupos: list[list[int]],
) -> dict:
    """Diagonaliza L_z en cada subespacio degenerado.

    Devuelve, para cada grupo, los autovalores m y los coeficientes
    complejos de los autovectores con m definido.
    """
    Lff = -1j * D[free][:, free]
    salida = []
    for grupo in grupos:
        U = vecs[np.ix_(free, grupo)]
        Lsub = U.conj().T @ (Lff @ U)
        mvals, w = np.linalg.eigh(Lsub)
        phi = U @ w
        salida.append({"m": np.real(mvals), "phi": phi, "grupo": grupo})
    return salida


def combinacion_real(
    phi_p: np.ndarray,
    phi_m: np.ndarray,
    tipo: str,
    Mff,
) -> np.ndarray:
    """Combina e^{i m phi} y e^{-i m phi} en una funcion real.

    tipo='cos' -> cos(m phi); tipo='sin' -> sin(m phi).
    """
    if tipo == "cos":
        f = (phi_p + phi_m) / np.sqrt(2.0)
    elif tipo == "sin":
        f = (phi_p - phi_m) / (1j * np.sqrt(2.0))
    else:
        raise ValueError("tipo debe ser 'cos' o 'sin'")
    f = np.real(f)
    norma = np.sqrt(f @ (Mff @ f))
    return f / norma


def incrustar(v_free: np.ndarray, free: np.ndarray, ndof: int) -> np.ndarray:
    u = np.zeros(ndof, dtype=float)
    u[free] = np.real(v_free)
    return u


# ---------------------------------------------------------------------------
# Estilo de las figuras
# ---------------------------------------------------------------------------

FONDO = (205 / 255, 225 / 255, 245 / 255)


def mapa_orbitales():
    """Colour map of the thesis figures.

    Light-blue background that fades continuously into blue, cyan, green,
    yellow and red, as in the COMSOL exports used in the thesis.
    """
    from matplotlib.colors import LinearSegmentedColormap

    puntos = [
        (0.00, FONDO),
        (0.06, (0.62, 0.66, 0.93)),
        (0.16, (0.20, 0.20, 0.86)),
        (0.30, (0.13, 0.33, 1.00)),
        (0.44, (0.20, 0.85, 1.00)),
        (0.58, (0.55, 1.00, 0.35)),
        (0.72, (1.00, 0.92, 0.00)),
        (0.86, (0.93, 0.10, 0.00)),
        (1.00, (0.45, 0.00, 0.00)),
    ]
    cmap = LinearSegmentedColormap.from_list("tesis", puntos, N=512)
    cmap.set_bad("white")
    return cmap


def dibujar_orbital(
    ax,
    Z: np.ndarray,
    R_nm: float,
    titulo: str = "",
    vmin_frac: float = 0.0,
):
    """Dibuja |psi|^2 en el disco, con el estilo de la tesis."""
    import matplotlib.pyplot as plt

    cmap = mapa_orbitales()
    W = np.abs(Z) ** 2
    W = W / np.nanmax(W)
    X = np.linspace(-R_nm, R_nm, Z.shape[1])
    Y = np.linspace(-R_nm, R_nm, Z.shape[0])
    XX, YY = np.meshgrid(X, Y)
    W = np.where(XX**2 + YY**2 <= R_nm**2, W, np.nan)
    ax.imshow(
        W,
        extent=[-R_nm, R_nm, -R_nm, R_nm],
        origin="lower",
        cmap=cmap,
        vmin=vmin_frac,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.add_patch(
        plt.Circle((0, 0), R_nm, fill=False, color="black", linewidth=1.0)
    )
    ax.set_xlim(-R_nm, R_nm)
    ax.set_ylim(-R_nm, R_nm)
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=9)
