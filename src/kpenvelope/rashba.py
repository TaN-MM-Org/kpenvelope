"""The k-linear Rashba spin splitting of wurtzite structures (new in
v0.11) -- with YOUR measured coefficient, never a shipped default.

Wurtzite crystals lack inversion symmetry along the c axis, so even
without any interface field the carriers near the zone centre feel a
k-linear spin-orbit term of the universal form

    H_so = alpha * c_hat . (sigma x k)  =  alpha (sigma_x k_y - sigma_y k_x)

for in-plane momentum (W. Stefanowicz et al., Phys. Rev. B 89,
205201 (2014), who measured alpha = 4.5 +/- 1 meV A for bulk
wurtzite n-GaN:Si; heterostructure two-dimensional electron gases
show larger values, e.g. about 5.5-6 meV A from weak-antilocalization
measurements in GaN/AlGaN structures, Phys. Rev. B 74, 033302 (2006)
and Phys. Rev. B 74, 113308 (2006)). This addresses a stated limit
of earlier releases -- "no k-linear bulk-inversion-asymmetry terms"
-- for the conduction-band companion problem, with exact structure:

* the splitting between the two spin branches is EXACTLY
  2 |alpha| k at every in-plane momentum;
* the eigenstates' spins lie in the plane, perpendicular to k
  (the chiral Rashba texture), exactly;
* at k = 0 the two branches are exactly degenerate (Kramers).

All three are asserted in the tests as identities, not tolerances of
convenience. The valence-band k-linear terms of the six-band model
have a different, parameter-set-specific structure whose vetted
coefficients this package's sources do not provide -- so they remain
deliberately not shipped, and the six-band `splitting_vs_k` remains
the tool for hole spin splittings.

Following the standing rule, no alpha value is shipped as a default:
you pass your sample's coefficient (in eV nm, this package's units;
1 meV A = 1e-4 eV nm) together with a `reference` naming where it
comes from.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rashba_hamiltonian", "rashba_splitting", "rashba_spins"]


def _check(alpha_evnm, reference):
    a = float(alpha_evnm)
    if not np.isfinite(a):
        raise ValueError("alpha_evnm must be finite (eV nm; "
                         "1 meV A = 1e-4 eV nm)")
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError(
            "a real `reference` string is required: no Rashba "
            "coefficient ships with this package, so yours must "
            "carry its source (your measurement or a paper)")
    return a


def rashba_hamiltonian(alpha_evnm, kx, ky, reference):
    """The 2x2 k-linear Rashba Hamiltonian (eV) at in-plane momentum
    (kx, ky) in nm^-1: alpha (sigma_x k_y - sigma_y k_x), the
    c_hat . (sigma x k) form of wurtzite structures."""
    a = _check(alpha_evnm, reference)
    kx, ky = float(kx), float(ky)
    # sigma_x * ky - sigma_y * kx, with sigma_y = [[0, -i], [i, 0]]
    return a * np.array([[0.0, ky + 1j * kx],
                         [ky - 1j * kx, 0.0]])


def rashba_splitting(alpha_evnm, kt, reference):
    """The exact spin splitting 2 |alpha| k_t (eV) at in-plane
    momentum magnitude kt (nm^-1, scalar or array)."""
    a = _check(alpha_evnm, reference)
    k = np.asarray(kt, dtype=float)
    if np.any(k < 0.0) or not np.all(np.isfinite(k)):
        raise ValueError("kt must be finite and >= 0 (nm^-1)")
    out = 2.0 * abs(a) * k
    return float(out) if out.ndim == 0 else out


def rashba_spins(alpha_evnm, kx, ky, reference):
    """Eigenvalues and in-plane spin directions of the two branches.

    Returns dict(energies, spins): energies (2,) in eV ascending, and
    spins (2, 3) -- the Pauli expectation values <sigma> of each
    eigenstate. The Rashba texture is exact: each spin lies in the
    plane, perpendicular to k, with the two branches opposite; the
    tests assert spin . k = 0 and |spin| = 1 to machine precision.
    Refuses k = 0, where the branches are exactly degenerate and no
    direction is defined (Kramers).
    """
    a = _check(alpha_evnm, reference)
    kx, ky = float(kx), float(ky)
    if kx == 0.0 and ky == 0.0:
        raise ValueError("at k = 0 the two branches are exactly "
                         "degenerate (Kramers) and no spin direction "
                         "is defined; evaluate at finite k")
    h = rashba_hamiltonian(a, kx, ky, reference)
    e, v = np.linalg.eigh(h)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1j], [1j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    spins = np.empty((2, 3))
    for i in range(2):
        psi = v[:, i]
        spins[i] = [np.real(psi.conj() @ s @ psi)
                    for s in (sx, sy, sz)]
    return {"energies": e, "spins": spins}
