# Analytic solution of the 2D hydrogen atom

This note reproduces the derivation used in the thesis (appendix B) and
explains how it is implemented in `src/atomo2d/analitica.py`.

## 1. Hamiltonian in polar coordinates

The two-body problem is restricted to a plane. In polar coordinates
`(r, phi)` the effective Hamiltonian is

$$
\hat H = -\frac{\hbar^2}{2\mu}
\left(
\frac{\partial^2}{\partial r^2}
+\frac{1}{r}\frac{\partial}{\partial r}
+\frac{1}{r^2}\frac{\partial^2}{\partial \phi^2}
\right)
-\frac{e^2}{4\pi\varepsilon_0 r}.
$$

## 2. Separation of variables

Write `psi(r, phi) = R_nm(r) Phi_m(phi)`. The angular equation gives

$$
\Phi_m(\phi) = \frac{1}{\sqrt{2\pi}}e^{im\phi},
\qquad m = 0, \pm1, \dots, \pm n
\quad (|m| \le n).
$$

The radial equation is solved by

$$
R_{nm}(r) = N_{nm}\,\rho^{|m|}e^{-\rho/2}
L_{n-|m|}^{2|m|}(\rho),
\qquad
\rho = \frac{2r}{\left(n+\tfrac12\right)a_0},
$$

where `L` is the associated Laguerre polynomial and `N_nm` is fixed by

$$
\int_0^\infty |R_{nm}(r)|^2 r\,dr = 1.
$$

## 3. Energies

### Vacuum

$$
E_n = -\frac{Ry}{\left(n+\tfrac12\right)^2},
\qquad n = 0,1,2,\dots
$$

### Semiconductor

With the effective mass `m*` and the static relative permittivity `eps_r`,

$$
Ry^* = Ry\left(\frac{m^*}{m_e}\right)\left(\frac{1}{\varepsilon_r^2}\right),
\qquad
E_n = -\frac{Ry^*}{\left(n+\tfrac12\right)^2}.
$$

### GaAs example

With `m* = 0.067 m0` and `eps_r = 12.9`, the thesis quotes `Ry*_GaAs ≈ 5.475 meV` and the table
below. The code uses CODATA constants and obtains `Ry* = 5.4779 meV`, so its levels differ from
this table in the third decimal (for example `E_0 = -21.912 meV`).

| `n` | `E_n` (eV) | `E_n` (meV) |
|---|---|---|
| 0 | −2.190 × 10⁻² | −21.903 |
| 1 | −2.434 × 10⁻³ | −2.434 |
| 2 | −8.761 × 10⁻⁴ | −0.876 |
| 3 | −4.470 × 10⁻⁴ | −0.447 |
| 4 | −2.704 × 10⁻⁴ | −0.270 |
| 5 | −1.810 × 10⁻⁴ | −0.181 |
| 6 | −1.296 × 10⁻⁴ | −0.130 |
| 7 | −9.734 × 10⁻⁵ | −0.097 |

## 4. Implementation notes

- The associated Laguerre polynomials are evaluated with
  `scipy.special.eval_genlaguerre(k, 2|m|, rho)`.
- The thesis leaves `N_nm` implicit. Here it is computed by adaptive
  quadrature,

  ```
  N_nm = 1 / sqrt( integral_0^inf |u_nm(rho)|^2 (r/a*) d(r/a*) ),
  ```

  so every radial function is normalized for any `(n, m)`.
- Real orbitals are built from the complex ones:

  ```
  s      : R_n0 / sqrt(2 pi)
  p_x    : R_n1 cos(phi) / sqrt(pi)
  p_y    : R_n1 sin(phi) / sqrt(pi)
  d_xy   : R_n2 sin(2 phi) / sqrt(pi)
  ```

  (and similarly for other `m`).

## 5. Original thesis text (appendix B, verbatim)

This is the LaTeX source of appendix B exactly as it appears in the thesis, kept for provenance.
Sections 1 to 3 above are the same content rendered for GitHub.

```latex
\subsection*{B. Átomo de Hidrógeno 2D}
\phantomsection\label{anexo_B}
Análogamente al caso tridimensional, partiendo del Hamiltoniano del sistema de dos cuerpos y restringiendo el movimiento al plano, se obtiene el Hamiltoniano efectivo en coordenadas polares $(r,\phi)$:

\begin{equation*}
\hat{H} = -\frac{\hbar^2}{2\mu} \left( \frac{\partial^2}{\partial r^2} + \frac{1}{r}\frac{\partial}{\partial r} + \frac{1}{r^2}\frac{\partial^2}{\partial \phi^2} \right) - \frac{e^2}{4\pi\varepsilon_0 r}.
\end{equation*}

Aplicando el método de separación de variables $\psi(r,\phi) = R_{nm}(r)\Phi_m(\phi)$, se determina la solución angular y la parte radial:

\begin{equation*}
\Phi_m(\phi) = \frac{1}{\sqrt{2\pi}} e^{im\phi}, \quad m=0,\pm1,\ldots,\pm n \quad (|m| \leq n)
\end{equation*}

\begin{equation*}
R_{nm}(r) = N_{nm} \rho^{|m|} e^{-\rho/2} L_{n-|m|}^{2|m|}(\rho), \quad \rho = \frac{2r}{(n+\tfrac{1}{2})a_{0}}.
\end{equation*}


\subsubsection*{Energía en el vacío}

 Los niveles de energía resultantes para el átomo bidimensional son:

\begin{tcolorbox}[boxsep=8pt,colback=white,colframe=black,boxrule=0.75pt,arc=0pt,left=8pt,right=8pt,top=6pt,bottom=6pt]
\vspace{-0.3cm}
\begin{equation*}
E_n = -\frac{Ry}{(n+\tfrac{1}{2})^2}, \quad (n=0,1,2,\ldots)
\end{equation*}
\vspace{-0.5cm}
\end{tcolorbox}

\subsubsection*{Energía dentro de semiconductores}

Para un medio semiconductor, se emplean la masa efectiva $m^*$ y la permitividad relativa $\varepsilon_r$. Definimos la constante de Rydberg efectiva $Ry^*$ y los niveles de energía corregidos como:

\begin{tcolorbox}[boxsep=8pt,colback=white,colframe=black,boxrule=0.75pt,arc=0pt,left=8pt,right=8pt,top=6pt,bottom=6pt]
\vspace{-0.3cm}
\begin{equation*}
Ry^* = Ry \left(\frac{m^*}{m_e}\right)\left(\frac{1}{\varepsilon_r^2}\right), \quad E_n = -\frac{Ry^*}{(n+\tfrac{1}{2})^2}
\end{equation*}
\vspace{-0.5cm}
\end{tcolorbox}



\noindent \textbf{Ejemplo para una impureza de GaAs:} Con $m^* = 0.067\,m_e$ y $\varepsilon_r = 12.9$, se tiene $Ry^*_{\mathrm{GaAs}} \approx 5.475\ \text{meV}$. Los niveles se tabulan a continuación:


\begin{table}[H]
\centering
\begin{tabular}{|c|c|c|}
\hline
\rule{0pt}{3ex}\textbf{$n$} & \textbf{$E_n$ (eV)} & \textbf{$E_n$ (meV)} \\
\hline
\rule{0pt}{3ex}0 & $-2.190\times10^{-2}$ & $-21.903$ \\
\hline
1 & $-2.434\times10^{-3}$ & $-2.434$ \\
\hline
2 & $-8.761\times10^{-4}$ & $-0.876$ \\
\hline
3 & $-4.470\times10^{-4}$ & $-0.447$ \\
\hline
4 & $-2.704\times10^{-4}$ & $-0.270$ \\
\hline
5 & $-1.810\times10^{-4}$ & $-0.181$ \\
\hline
6 & $-1.296\times10^{-4}$ & $-0.130$ \\
\hline
7 & $-9.734\times10^{-5}$ & $-0.097$ \\
\hline
\end{tabular}
\caption[Niveles de energía del hidrógeno 2D en GaAs]{Niveles de energía de una impureza hidrogenoide bidimensional de GaAs, calculados a partir de la solución analítica $E_n = -Ry^{*}_{\mathrm{GaAs}}/(n+\tfrac{1}{2})^2$. La primera columna corresponde al número cuántico principal $n$; las dos siguientes expresan el mismo nivel energético en eV y en meV, respectivamente.}
\end{table}
```
