"""Bir-Pikus strain terms for the six-band wurtzite Hamiltonian.

Strain enters the valence Hamiltonian through the deformation-potential
matrix of the same symmetry structure as the k-quadratic terms (Bir and
Pikus; in the six-band wurtzite form of Chuang and Chang, Phys. Rev. B
54, 2491 (1996), the same paper the kinetic assembly already follows):
each k_i k_j bilinear is replaced by the strain component eps_ij and
each A_i by the deformation potential D_i,

    lambda_eps = D1 eps_zz + D2 (eps_xx + eps_yy)
    theta_eps  = D3 eps_zz + D4 (eps_xx + eps_yy)
    K_eps      = D5 (eps_xx - eps_yy + 2 i eps_xy)
    H_eps      = D6 (eps_xz + i eps_yz)

occupying exactly the matrix positions of lambda, theta, K and H_t.
That structural identity is not just documentation here -- it is the
module's strongest test: `strain_blocks` evaluated at eps_ij = k_i k_j
with D_i set equal to A_i (times hbar^2/2m0) must reproduce the kinetic
blocks H0 + H1 kz + H2 kz^2 entry for entry, at machine precision.

No deformation-potential values are shipped, on purpose: D1..D6 differ
between parameterizations and materials, so they must be supplied with
a citation through the optional D1..D6 fields of WurtziteParameters,
exactly like every other number in this package.  A strained assembly
on a parameter set without them is refused, never defaulted.

Further exact facts the test suite asserts: zero strain changes nothing
at machine precision; the assembled matrix stays exactly Hermitian for
arbitrary (symmetric) strain tensors; at k = 0 with the splittings
switched off, diagonal strain gives the closed-form eigenvalues
lambda_eps + theta_eps (four-fold) and lambda_eps (two-fold); and a
pure shear eps_xz mixes exactly the H_t positions.
"""
from __future__ import annotations

import numpy as np

__all__ = ["strain_blocks"]


def _check_strain(strain):
    e = np.asarray(strain, dtype=float)
    if e.shape != (3, 3):
        raise ValueError("strain must be a 3 x 3 tensor")
    if np.abs(e - e.T).max() > 1e-12 * max(1.0, np.abs(e).max()):
        raise ValueError("strain tensor must be symmetric")
    return e


def _check_potentials(p):
    ds = [getattr(p, f"D{i}", None) for i in range(1, 7)]
    if any(d is None for d in ds):
        raise ValueError(
            "this parameter set carries no deformation potentials "
            "(D1..D6 are None). Supply cited values before building a "
            "strained Hamiltonian; none are shipped by default, on "
            "purpose.")
    return [float(d) for d in ds]


def strain_blocks(p, strain):
    """The 6 x 6 Bir-Pikus strain matrix H_eps for parameter set ``p``
    (which must carry cited D1..D6) and a symmetric 3 x 3 strain tensor
    (dimensionless).  Added to the k-dependent Hamiltonian wherever a
    ``strain`` argument is accepted; energies in eV.
    """
    e = _check_strain(strain)
    D1, D2, D3, D4, D5, D6 = _check_potentials(p)
    exx, eyy, ezz = e[0, 0], e[1, 1], e[2, 2]
    exy, exz, eyz = e[0, 1], e[0, 2], e[1, 2]

    lam = D1 * ezz + D2 * (exx + eyy)
    th = D3 * ezz + D4 * (exx + eyy)
    K = D5 * (exx - eyy + 2j * exy)
    Ht = D6 * (exz + 1j * eyz)

    F = lam + th
    G = lam + th
    H = np.array([
        [F, -np.conj(K), 0, 0, 0, 0],
        [-K, G, 0, 0, 0, 0],
        [0, 0, lam, 0, 0, 0],
        [0, 0, 0, F, -K, 0],
        [0, 0, 0, -np.conj(K), G, 0],
        [0, 0, 0, 0, 0, lam],
    ], dtype=complex)
    # the H_t positions, exactly as in the kinetic H1 pattern
    H[0, 2] += -np.conj(Ht)
    H[2, 0] += -Ht
    H[1, 2] += Ht
    H[2, 1] += np.conj(Ht)
    H[3, 5] += np.conj(Ht)
    H[5, 3] += Ht
    H[4, 5] += -Ht
    H[5, 4] += -np.conj(Ht)
    return 0.5 * (H + H.conj().T)
