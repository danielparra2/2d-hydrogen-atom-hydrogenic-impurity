# Method — P2 finite elements from scratch

This document describes the numerical solver in `src/atomo2d/fem.py` and the
validation in `scripts/validar.py`.

## 1. Problem

A hydrogenic donor at the origin of a disk of radius `R`, with Dirichlet
boundary `psi(R) = 0`:

$$
\left[-\frac{\hbar^2}{2m^*}\nabla^2
-\frac{e^2}{4\pi\varepsilon_0\varepsilon_r r}\right]\psi = E\psi.
$$

Two cases are solved:

| Case | `m*` | `eps_r` | `a*` | `Ry*` |
|---|---|---|---|---|
| Vacuum | `m0` | 1 | 0.052918 nm | 13.606 eV |
| GaAs | `0.067 m0` | 12.9 | 10.1886 nm | 5.4779 meV |

Internally the solver is dimensionless:

```
x~ = x/a*,   E~ = E/Ry*,   H~ = -laplacian~ - 2/r~.
```

The material enters only through `a*` and `Ry*`. This is why the same code
describes the hydrogen atom and the donor in GaAs.

## 2. Mesh

- Unstructured **Delaunay** triangulation of concentric rings.
- Geometric refinement towards the origin: `h(r) = min(h_max, alpha r)`.
- The origin is a node; triangles touching it are integrated with a special
  quadrature to handle the `1/r` singularity.
- Dirichlet boundary on the outer circle.
- Two domain sizes are used per case to check finite-size effects:

| Case | `R` | `R_check` | `r0` | `h_max` | nodes (`R`) |
|---|---|---|---|---|---|
| Vacuum | 40 a* = 2.117 nm | 60 a* | 1e-4 a* | 0.2 a* | 24 193 |
| GaAs | 200 nm | 300 nm | 0.001 nm | 2.5 nm | 12 865 |

## 3. Discretization

- Quadratic Lagrange elements (P2), six degrees of freedom per triangle.
- Seven-point Dunavant quadrature of degree 5.
- Sparse assembly of `K`, `M` and the angular-momentum matrix `D`:

```
K_ij = int grad(N_i).grad(N_j) - int (2/r) N_i N_j
M_ij = int N_i N_j
D_ij = int N_i (x dN_j/dy - y dN_j/dx)
```

- Generalized eigenvalue problem `K u = E M u`, solved with SciPy
  `eigsh(..., sigma=-3.0, which="LM")` (shift-invert).
- The eigenvectors are normalized with the mass matrix.
- Degenerate subspaces are rotated to eigenstates of `L_z`; real orbitals are
  built as `cos(m phi)` / `sin(m phi)` combinations.

No external FEM library is used.

## 4. Validation

The reference is the exact 2D hydrogenic spectrum
`E_n = -Ry*/(n + 1/2)^2`.

### GaAs

| State | `(n,m)` | `E_theory` (meV) | `E_FEM` R=200 | diff | `E_FEM` R=300 | diff |
|---|---|---|---|---|---|---|
| 1s | (0,0) | -21.9117 | -21.9105 | +0.0012 | -21.9105 | +0.0012 |
| 2s | (1,0) | -2.4346 | -2.4346 | +0.0001 | -2.4346 | +0.0001 |
| 2px | (1,+1) | -2.4346 | -2.4346 | +0.0000 | -2.4346 | +0.0000 |
| 2py | (1,-1) | -2.4346 | -2.4346 | +0.0000 | -2.4346 | +0.0000 |
| 3s | (2,0) | -0.8765 | -0.8675 | +0.0090 | -0.8764 | +0.0001 |
| 3px | (2,+1) | -0.8765 | -0.8695 | +0.0070 | -0.8764 | +0.0000 |
| 3py | (2,-1) | -0.8765 | -0.8695 | +0.0070 | -0.8764 | +0.0000 |
| 3dxy | (2,-2) | -0.8765 | -0.8737 | +0.0028 | -0.8765 | +0.0000 |

**GaAs:** at `R = 200 nm` the largest deviation is **0.0090 meV** (3s, about 1 % of that
level): the `n = 2` shell feels the boundary. At `R = 300 nm` that shell agrees to
≤ 0.0001 meV and the largest remaining error is **0.0012 meV** on the 1s level, which is
independent of the domain size and comes from the origin.

### Vacuum

| State | `E_theory` (meV) | `E_FEM` R=40 a* | diff | `E_FEM` R=60 a* | diff |
|---|---|---|---|---|---|
| 1s | -54422.7725 | -54419.8204 | +2.9521 | -54419.8206 | +2.9519 |
| 2s | -6046.9747 | -6046.8566 | +0.1182 | -6046.8566 | +0.1181 |
| 2px | -6046.9747 | -6046.9639 | +0.0108 | -6046.9639 | +0.0108 |
| 2py | -6046.9747 | -6046.9639 | +0.0109 | -6046.9639 | +0.0108 |
| 3s | -2176.9109 | -2176.8842 | +0.0267 | -2176.8843 | +0.0266 |
| 3px | -2176.9109 | -2176.9070 | +0.0039 | -2176.9071 | +0.0038 |
| 3py | -2176.9109 | -2176.9070 | +0.0039 | -2176.9071 | +0.0038 |
| 3dxy | -2176.9109 | -2176.9065 | +0.0044 | -2176.9065 | +0.0044 |

**Vacuum:** the absolute error is a few meV on a 54 eV scale, i.e. a
**relative error of ~5 × 10^-5**. The error is dominated by the inner hole
`r0` and the `1/r` singularity, not by the domain size. The nodal structure
of every orbital is correct.

> Scope: the thesis reports ≤ 0.008 meV for the COMSOL model on the 200 nm domain. This
> independent Python solver gives 0.009 meV on the same domain. The vacuum case is an extra
> check that is not in the thesis.

## 5. COMSOL model

`models/AH2D_material_base.mph` is the COMSOL model used as reference for the
GaAs case. It uses pure GaAs (`c_Ga`, `E_Ga`); the AlGaAs potential `V_Al` is
defined but not used in the eigenvalue equation of this validation. The Python
solver does not read the `.mph`; it is included for provenance.

## 6. Caveats

- The model is **2D**; no z-direction, strain or band non-parabolicity.
- A single donor at the origin.
- Dirichlet boundary at finite `R`; checked with a second domain size.
- The vacuum case uses `R = 40 a*`; the 1s orbital is already negligible at
  the boundary, but the inner singularity dominates the error.
- The angular decomposition assumes the degenerate subspaces are separated
  from the rest of the spectrum; this is true for the first three shells.
