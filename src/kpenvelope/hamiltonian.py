"""Assembly of the six-band wurtzite envelope Hamiltonian on a 1D grid.

The bulk Hamiltonian follows the standard wurtzite six-band form (Chuang
and Chang, Phys. Rev. B 54, 2491 (1996)), written here as

    H(k) = H0(kx, ky) + H1(kx, ky) * kz + H2 * kz^2

The envelope operator is obtained by kz -> -i d/dz with the symmetrized
(Ben Daniel-Duke) discretization for the kz^2 terms and the symmetrized
first-difference for terms linear in kz, which keeps the discrete matrix
exactly Hermitian, including for position-dependent parameters.

Off-diagonal phase conventions differ between papers; eigenvalues are
invariant under those phase choices. The test suite checks the bulk
eigenvalues against the block-diagonal 3 x 3 form of Chuang and Chang
(their Eq. (45), whose entries depend only on |k_t|): every bulk level
is two-fold degenerate and independent of the in-plane direction.

Energies in eV, lengths in nm. Valence-electron energy convention: holes
occupy the HIGHEST eigenvalues.
"""
from __future__ import annotations

import numpy as np

# hbar^2 / (2 m0) in eV nm^2
HBAR2_OVER_2M0 = 0.0380998


def check_grid(z):
    """Return z as a float array after checking it is a usable grid:
    one-dimensional, at least three finite points, strictly increasing
    and uniform (new checks in v0.12.0: a decreasing grid used to give
    NaN envelopes through a negative step)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 3:
        raise ValueError("z must be a 1D grid of at least 3 points")
    if not np.all(np.isfinite(z)):
        raise ValueError("z grid must be finite")
    dz = z[1] - z[0]
    if not dz > 0.0:
        raise ValueError("z grid must be strictly increasing")
    if not np.allclose(np.diff(z), dz):
        raise ValueError("z grid must be uniform")
    return z


def check_potential(potential, n):
    """None, or a finite array with one value per grid point (new check
    in v0.12.0: a longer array used to be silently truncated)."""
    if potential is None:
        return None
    v = np.asarray(potential, dtype=float)
    if v.shape != (n,):
        raise ValueError(f"potential must have one value per grid point "
                         f"({n}); got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("potential must be finite")
    return v


def check_n_states(n_states, n):
    """1 <= n_states <= 6 N (new check in v0.12.0)."""
    k = int(n_states)
    if k != n_states or k < 1 or k > 6 * n:
        raise ValueError(f"n_states must be an integer from 1 to 6 N = "
                         f"{6 * n}; got {n_states!r}")
    return k


def bulk_blocks(p, kx: float, ky: float):
    """Return (H0, H1, H2): 6x6 complex blocks for the given in-plane k.

    H(k) = H0 + H1*kz + H2*kz^2, with kz in nm^-1 and energies in eV.
    """
    c = HBAR2_OVER_2M0
    kt2 = kx * kx + ky * ky
    kplus = kx + 1j * ky

    lam0 = c * p.A2 * kt2                 # kz-independent part of lambda
    th0 = c * p.A4 * kt2                  # kz-independent part of theta
    F0 = p.delta1 + p.delta2 + lam0 + th0
    G0 = p.delta1 - p.delta2 + lam0 + th0
    K0 = c * p.A5 * kplus * kplus         # K term, no kz dependence
    D = np.sqrt(2.0) * p.delta3

    H0 = np.array([
        [F0, -np.conj(K0), 0, 0, 0, 0],
        [-K0, G0, 0, 0, 0, D],
        [0, 0, lam0, 0, D, 0],
        [0, 0, 0, F0, -K0, 0],
        [0, 0, D, -np.conj(K0), G0, 0],
        [0, D, 0, 0, 0, lam0],
    ], dtype=complex)

    # terms linear in kz: H = A6 k+ kz. Upper block (rows 1-3):
    # -H* at (1,3), H at (2,3); lower block (rows 4-6): H at (4,6),
    # -H* at (5,6) (Chuang-Chang). Before v0.12.0 the lower block had
    # H and H* swapped, which is invisible along kx (H real there) but
    # broke the bulk Kramers degeneracy and the in-plane isotropy in
    # every other direction; see CHANGELOG 0.12.0.
    Ht = c * p.A6 * kplus
    H1 = np.zeros((6, 6), dtype=complex)
    H1[0, 2] = -np.conj(Ht)
    H1[2, 0] = -Ht
    H1[1, 2] = Ht
    H1[2, 1] = np.conj(Ht)
    H1[3, 5] = Ht
    H1[5, 3] = np.conj(Ht)
    H1[4, 5] = -np.conj(Ht)
    H1[5, 4] = -Ht

    # kz^2 coefficients: lambda gains A1 kz^2, theta gains A3 kz^2
    lam2 = c * p.A1
    th2 = c * p.A3
    H2 = np.diag([lam2 + th2, lam2 + th2, lam2, lam2 + th2, lam2 + th2, lam2]).astype(complex)
    return H0, H1, H2


def assemble_hamiltonian(p, z, kx: float = 0.0, ky: float = 0.0,
                         potential=None, strain=None):
    """Dense 6N x 6N envelope Hamiltonian on grid z (uniform, nm).

    potential : optional array of the HOLE potential energy V_h(z) in eV;
        it enters the valence-electron Hamiltonian as -V_h on the diagonal.
    strain : optional symmetric 3 x 3 strain tensor (dimensionless);
        adds the Bir-Pikus matrix of :func:`~kpenvelope.strain.strain_blocks`
        to every grid point. The parameter set must carry cited D1..D6.
    Hard-wall boundaries at both ends of the grid (for finite barriers
    use `kpenvelope.heterostructure.assemble_heterostructure`).
    """
    z = check_grid(z)
    n = z.size
    dz = z[1] - z[0]
    potential = check_potential(potential, n)

    H0, H1, H2 = bulk_blocks(p, kx, ky)
    if strain is not None:
        from .strain import strain_blocks
        H0 = H0 + strain_blocks(p, strain)
    H = np.zeros((6 * n, 6 * n), dtype=complex)

    idx = lambda i: slice(6 * i, 6 * i + 6)
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dz = 1.0 / (2.0 * dz)

    for i in range(n):
        diag = H0.copy()
        # kz^2 term: -d/dz H2 d/dz -> +2 H2 / dz^2 on the diagonal
        diag += 2.0 * H2 * inv_dz2
        if potential is not None:
            diag -= potential[i] * np.eye(6)
        H[idx(i), idx(i)] = diag
        if i + 1 < n:
            # kz^2 off-diagonal: -H2/dz^2 ; kz linear: -i H1 * (D_central)
            off = -H2 * inv_dz2 + (-1j) * H1 * inv_2dz
            H[idx(i), idx(i + 1)] = off
            H[idx(i + 1), idx(i)] = off.conj().T
    return H
