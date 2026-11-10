"""Velocity matrix elements and intersubband transitions for light
polarized in the plane of the layers (new in v0.16.0).

`dipole_matrix` and `level_transitions` give the coupling to light
polarized along the growth axis z. Light polarized in the plane (for
example at normal incidence) couples through the in-plane velocity
operator of the envelope model,

    hbar v_e = dH/dk_e,   k_e = k_t . e,  e = (cos angle, sin angle),

where H(kx, ky) is the assembled matrix that the solvers diagonalize.
For one parabolic band this operator is a constant times the identity,
so it cannot connect two different subbands: in-plane polarized light
does not drive intersubband transitions of a simple band. In the
six-band model the A5 and A6 terms couple the heavy-, light- and
split-off components, so it can.

The matrix is a polynomial of second degree in (kx, ky) (asserted in
the tests: the third finite difference vanishes to rounding), so its
derivative is exact here: each 6 x 6 bulk block is differentiated as
[B(k + e) - B(k - e)] / 2, which is exact for a polynomial of second
degree, and the derivative blocks are placed on the grid with the same
discretization as the solvers' own assembly (the kz^2 terms do not
depend on k_t and drop out; strain, potential and band edge do not
depend on k_t either).

Oscillator strength for in-plane polarization e, from level I to level
L, per hole in level I (the standard definition with the free-electron
mass m0):

    f = (2 / (m0 dE)) sum |<f| p_e |i>|^2 = sum |<f| dH/dk_e |i>|^2
        / ((hbar^2 / 2 m0) dE),     dE = E_I - E_L.

For light polarized along z the same definition, with
hbar v_z = i [H, z], gives f = dE |z_fi|^2 / (hbar^2 / 2 m0), which is
`level_transitions` with ``mass_ratio=1``, so the two can be compared
directly.

Anchors asserted in the tests: on the demo set the velocity matrix is
2 c A k_e times the identity (no in-plane intersubband coupling, a
closed form); its diagonal equals dE_n/dk_e from finite differences of
the solver's energies (the Hellmann-Feynman theorem); it obeys the
effective-mass sum rule of second-order perturbation theory,
d^2 E_n / dk_e^2 = <n| d^2 H / dk_e^2 |n>
+ 2 sum_{m != n} |<m| dH/dk_e |n>|^2 / (E_n - E_m), an exact identity
for the discrete matrix, checked against finite differences of the
energies; the level sums do not depend on the polarization angle at
k_t = 0 and are unchanged by a common rotation of k_t and e (the
subbands are isotropic in the plane); and they do not depend on the
basis the solver picks inside a level.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import HBAR2_OVER_2M0, bulk_blocks, check_grid, place_blocks
from .isb import degenerate_levels
from .params import WurtziteParameters

__all__ = ["velocity_matrix", "inplane_transitions"]


def _unit_vector(angle):
    a = float(angle)
    if not np.isfinite(a):
        raise ValueError("angle must be finite (radians)")
    return np.cos(a), np.sin(a)


def _derivative_blocks(p, kx, ky, ex, ey):
    """d/dk_e of the k-dependent bulk blocks (H0, H1); H2 does not
    depend on k_t. Exact for blocks of second degree in (kx, ky)."""
    P0, P1, _ = bulk_blocks(p, kx + ex, ky + ey)
    M0, M1, _ = bulk_blocks(p, kx - ex, ky - ey)
    return 0.5 * (P0 - M0), 0.5 * (P1 - M1)


def velocity_derivative_matrix(z, params, kx: float = 0.0,
                               ky: float = 0.0,
                               angle: float = 0.0) -> np.ndarray:
    """The 6N x 6N matrix dH/dk_e (eV nm) at in-plane momentum (kx, ky)
    for the in-plane direction e = (cos angle, sin angle).

    params : one `WurtziteParameters` (the hard-wall matrix of
    `solve_subbands`) or one per grid point (the matrix of
    `solve_heterostructure`, with the same midpoint averages)."""
    z = check_grid(z)
    n = z.size
    dz = z[1] - z[0]
    kx, ky = float(kx), float(ky)
    if not (np.isfinite(kx) and np.isfinite(ky)):
        raise ValueError("kx and ky must be finite (nm^-1)")
    ex, ey = _unit_vector(angle)
    inv_2dz = 1.0 / (2.0 * dz)
    if isinstance(params, WurtziteParameters):
        d0, d1 = _derivative_blocks(params, kx, ky, ex, ey)
        off = (-1j) * d1 * inv_2dz
        return place_blocks(n, d0, off, off.conj().T)
    plist = list(params)
    if len(plist) != n:
        raise ValueError(f"one parameter set per grid point required "
                         f"({n}); got {len(plist)}")
    if not all(isinstance(p, WurtziteParameters) for p in plist):
        raise ValueError("params must be WurtziteParameters, one set or "
                         "one per grid point")
    cache = {}
    blocks = []
    for p in plist:
        if id(p) not in cache:
            cache[id(p)] = _derivative_blocks(p, kx, ky, ex, ey)
        blocks.append(cache[id(p)])
    d0s = np.array([b[0] for b in blocks])
    d1s = np.array([b[1] for b in blocks])
    off = (-1j) * (0.5 * (d1s[:-1] + d1s[1:])) * inv_2dz
    return place_blocks(n, d0s, off, off.conj().transpose(0, 2, 1))


def _state_matrix(z, envelopes):
    z = np.asarray(z, dtype=float)
    F = np.asarray(envelopes)
    if F.ndim != 3 or F.shape[1] != 6 or F.shape[2] != z.size:
        raise ValueError(
            "envelopes must have shape (n_states, 6, N) on the grid z "
            f"(N = {z.size}), as the solvers return them; got {F.shape}")
    # point-major columns, the ordering of the assembled matrix
    return F.transpose(0, 2, 1).reshape(F.shape[0], -1).T


def velocity_matrix(z, params, envelopes, kx: float = 0.0,
                    ky: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """In-plane velocity matrix elements <f| dH/dk_e |i> (eV nm; divide
    by hbar for a velocity) between solver states (new in v0.16.0).

    z : the uniform grid of the solve (nm). params : the parameter set
        of `solve_subbands`, or the per-point list of
        `solve_heterostructure`, used for the solve.
    envelopes : (n, 6, N) envelopes from that solve, at in-plane
        momentum (kx, ky) (nm^-1), which must be the momentum passed
        here: the operator depends on it.
    angle : direction e = (cos angle, sin angle) of the polarization
        (radians), measured from kx.

    Returns the (n, n) Hermitian matrix V with V[f, i] =
    <f| dH/dk_e |i>. Its diagonal is the group velocity dE_n/dk_e
    (times hbar) of each nondegenerate state, exactly (the
    Hellmann-Feynman theorem for the discrete matrix), with no
    finite-difference step. Inside a degenerate level the individual
    elements depend on the basis the solver chose; sums over the level
    (as in `inplane_transitions`) do not.
    """
    z = check_grid(z)
    X = _state_matrix(z, envelopes)
    D = velocity_derivative_matrix(z, params, kx, ky, angle)
    dz = z[1] - z[0]
    V = (X.conj().T @ (D @ X)) * dz
    return 0.5 * (V + V.conj().T)


def inplane_transitions(z, params, energies, envelopes, kx: float = 0.0,
                        ky: float = 0.0, angle: float = 0.0,
                        initial: int = 0, tol_ev: float = 1e-9) -> dict:
    """Intersubband transitions for light polarized in the plane,
    between whole levels (new in v0.16.0); the in-plane counterpart of
    `level_transitions`.

    z, params : as in `velocity_matrix`. energies, envelopes : (n,) and
        (n, 6, N) from one solve at in-plane momentum (kx, ky), which
        must be passed here too. angle : polarization direction
        (radians from kx). initial : index of the initial level (0 =
        the highest, where the holes sit). tol_ev : the largest
        spacing within a level (eV), as in `degenerate_levels`.

    For every level L, with I the initial level and g_I its number of
    states:
      transition_energy[L] = E_I - E_L (eV),
      v2[L] = (1/g_I) sum_{i in I} sum_{f in L} |<f| dH/dk_e |i>|^2
        (eV^2 nm^2),
      oscillator_strength[L] = v2[L] / ((hbar^2/2m0) (E_I - E_L)),
        per hole in level I, with the free-electron mass (module
        docstring); comparable with `level_transitions(...,
        mass_ratio=1)` for light polarized along z. Negative for the
        levels above I (emission), as there.
    The entries of the initial level itself are 0.

    These sums do not depend on the basis the solver picks inside a
    level (asserted). Pass complete levels (an even n_states at
    k_t = 0). At k_t = 0 they also do not depend on the angle, since
    the subbands are isotropic in the plane (asserted). Returns
    dict(levels, level_energies, degeneracy, initial,
    transition_energy, v2, oscillator_strength).
    """
    e = np.asarray(energies, dtype=float)
    F = np.asarray(envelopes)
    if F.ndim != 3 or F.shape[0] != e.size:
        raise ValueError("envelopes must have shape (n_states, 6, N), "
                         "one state per energy")
    levels = degenerate_levels(e, tol_ev)
    I = int(initial)
    if I != initial or not 0 <= I < len(levels):
        raise ValueError(f"initial must be a level index from 0 to "
                         f"{len(levels) - 1}; got {initial!r}")
    V = velocity_matrix(z, params, F, kx, ky, angle)
    e_lev = np.array([e[c].mean() for c in levels])
    g_i = levels[I].size
    v2 = np.zeros(len(levels))
    for L, c in enumerate(levels):
        if L == I:
            continue
        v2[L] = float((np.abs(V[np.ix_(c, levels[I])]) ** 2).sum()) / g_i
    de = e_lev[I] - e_lev
    de[I] = 0.0
    osc = np.zeros(len(levels))
    nz = de != 0.0
    osc[nz] = v2[nz] / (HBAR2_OVER_2M0 * de[nz])
    return {"levels": levels, "level_energies": e_lev,
            "degeneracy": np.array([c.size for c in levels]),
            "initial": I, "transition_energy": de, "v2": v2,
            "oscillator_strength": osc}
