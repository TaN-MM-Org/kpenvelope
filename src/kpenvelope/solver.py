"""Eigen-solution of the assembled envelope Hamiltonian.

How the top states are found (new in v0.14.0). The assembled 6N x 6N
matrix is never changed, but it is diagonalized in independent pieces
whenever it falls apart into them, which is exact in exact arithmetic:

* At k_t = 0 (no in-plane momentum) the six components split into
  groups that no matrix element connects, for example (HH up),
  (HH down), (LH up, CH down) and (LH down, CH up) without strain.
  The groups are read off the zero pattern of the assembled matrix
  itself, so any strain or profile that couples them simply joins the
  groups. Each group is diagonalized on its own, as a real matrix when
  its elements are real. When the time-reversal partner of a group
  (components 1 <-> 4, 2 <-> 5, 3 <-> 6) gives exactly the same
  matrix, element for element, its eigenvalues and eigenvectors are
  reused, so the two members of a Kramers pair have bit-for-bit equal
  energies and come out in a fixed order: first the member that lives
  on the group holding the lowest component number.
* At k_t > 0 the Chuang-Chang block diagonalization (Phys. Rev. B 54,
  2491 (1996), the 3 x 3 "upper" and "lower" blocks of their Eq. (45))
  is applied: at in-plane angle phi = atan2(ky, kx), the operator
  Q = [[0, W], [W^H, 0]] with W = diag(-i e^{-3 i phi}, -i e^{i phi},
  i e^{-i phi}) commutes with every k-dependent block of every
  material, so the matrix splits into two 3N x 3N blocks
  (Q = +1 and Q = -1). The splitting is used only after checking that
  the coupling it neglects is at rounding level (at most 1e-14 times
  the largest matrix element); a strain that breaks in-plane isotropy
  makes the coupling large, and the full matrix is then diagonalized
  as before.
* Since v0.15.0 each of the two blocks is made real before it is
  diagonalized: one phase factor per component (the same at every grid
  point) turns it into a real symmetric matrix, whose diagonalization
  is about three times faster than that of the complex one. The phases
  are read off the block and accepted only if what is left of the
  imaginary part is at rounding level (at most 1e-14 times the largest
  element); otherwise the complex block is diagonalized as in v0.14.0.

The test suite checks the energies against the dense diagonalization
of the full matrix (the pre-0.14 path), and the real blocks against
the complex ones of v0.14.0, and the eigenvectors through the
projector onto each degenerate level, which does not depend on the
arbitrary basis inside a Kramers pair.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import assemble_hamiltonian, check_grid, check_n_states

# time reversal maps component c to _TR[c] (Chuang-Chang ordering:
# 1 <-> 4, 2 <-> 5, 3 <-> 6, counting from 0 here)
_TR = (3, 4, 5, 0, 1, 2)

# largest neglected U/L coupling, relative to the largest element
_SPLIT_TOL = 1e-14


def _groups(H, n):
    """Connected groups of the six components: two components are in
    one group when any element of H joins them (at any grid points)."""
    nz = (H.reshape(n, 6, n, 6) != 0).any(axis=(0, 2))
    nz = nz | nz.T
    parent = list(range(6))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(6):
        for b in range(a + 1, 6):
            if nz[a, b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    out = {}
    for c in range(6):
        out.setdefault(find(c), []).append(c)
    return [out[r] for r in sorted(out)]


def _eig(M, vectors):
    # eigh also when only the eigenvalues are wanted: eigvalsh takes a
    # different LAPACK path whose eigenvalues differ in the last bits,
    # and the energies-only helpers must equal solve_subbands exactly
    if not np.iscomplexobj(M) or not M.imag.any():
        M = M.real
    vals, vecs = np.linalg.eigh(M)
    return vals, (vecs if vectors else None)


def _split_groups(H, n, groups, vectors):
    """Diagonalize each component group on its own. Returns a list of
    (eigenvalues, eigenvectors or None, index array into the 6n
    space)."""
    done = {}
    pieces = []
    keys = {tuple(sorted(g)) for g in groups}
    for g in groups:
        key = tuple(sorted(g))
        if key in done:
            continue
        twin = [_TR[c] for c in g]
        tkey = tuple(sorted(twin))
        # the twin is listed right after its group, in the component
        # order time reversal maps it to, if it is a group of its own
        both = (g, twin) if (tkey != key and tkey in keys) else (g,)
        for comps in both:
            if tuple(sorted(comps)) in done:
                continue
            idx = (6 * np.arange(n)[:, None]
                   + np.asarray(comps)[None, :]).ravel()
            pieces.append(idx)
            done[tuple(sorted(comps))] = len(pieces) - 1
    results = []
    cache = []
    for idx in pieces:
        M = H[np.ix_(idx, idx)]
        hit = None
        for (M0, vals0, vecs0) in cache:
            if M0.shape == M.shape:
                if np.array_equal(M0, M):
                    hit = (vals0, vecs0)
                    break
                if np.array_equal(np.conj(M0), M):
                    hit = (vals0, None if vecs0 is None else np.conj(vecs0))
                    break
        if hit is None:
            vals, vecs = _eig(M, vectors)
            cache.append((M, vals, vecs))
        else:
            vals, vecs = hit
        results.append((vals, vecs, idx))
    return results


def _omega(kx, ky):
    phi = np.arctan2(ky, kx)
    return np.array([-1j * np.exp(-3j * phi), -1j * np.exp(1j * phi),
                     1j * np.exp(-1j * phi)])


def _split_upper_lower(H, n, kx, ky):
    """The Chuang-Chang upper and lower blocks (3n x 3n each) and the
    largest element of the coupling between them."""
    w = _omega(kx, ky)
    H4 = H.reshape(n, 2, 3, n, 2, 3)
    aa, ab = H4[:, 0, :, :, 0, :], H4[:, 0, :, :, 1, :]
    ba, bb = H4[:, 1, :, :, 0, :], H4[:, 1, :, :, 1, :]
    wl = w[None, :, None, None]
    wr = np.conj(w)[None, None, None, :]
    p = ab * wr
    q = wl * ba
    r = wl * bb * wr
    up = 0.5 * (aa + p + q + r)
    lo = 0.5 * (aa - p - q + r)
    coupling = float(np.abs(0.5 * (aa - p + q - r)).max())
    up = up.reshape(3 * n, 3 * n)
    lo = lo.reshape(3 * n, 3 * n)
    up = 0.5 * (up + up.conj().T)
    lo = 0.5 * (lo + lo.conj().T)
    return up, lo, coupling, w


def _real_gauge(M, n):
    """Diagonal phases that make a Chuang-Chang block real (new in
    v0.15.0). Returns (d, R) with M = diag(d) R diag(d)^H and R real
    symmetric up to rounding, or None when no such phases exist.

    One phase per component (the same at every grid point) is read off
    the largest element of each coupling between two components; the
    result is accepted only if the imaginary part left over is at
    rounding level (at most _SPLIT_TOL times the largest element), so
    a block that cannot be made real is never treated as real."""
    B = M.reshape(n, 3, n, 3)
    scale = float(np.abs(M).max())
    if scale == 0.0:
        return None
    chi = [0.0, None, None]
    for _ in range(2):
        for a in range(3):
            for b in range(3):
                if a == b or chi[a] is None or chi[b] is not None:
                    continue
                blk = B[:, a, :, b]
                mag = np.abs(blk)
                k = int(np.argmax(mag))
                if mag.flat[k] > 0.0:
                    # e^{-i chi_a} M_ab e^{i chi_b} real
                    chi[b] = chi[a] - float(np.angle(blk.flat[k]))
    chi = [0.0 if c is None else c for c in chi]
    d = np.tile(np.exp(1j * np.asarray(chi)), n)
    Mg = (np.conj(d)[:, None] * M) * d[None, :]
    if float(np.abs(Mg.imag).max()) > _SPLIT_TOL * scale:
        return None
    return d, Mg.real


def _eig_block(M, n, vectors):
    """Eigenpairs of one Chuang-Chang block: as a real symmetric
    matrix after the diagonal phase change of `_real_gauge` when one
    exists (about three times faster than the complex
    diagonalization; new in v0.15.0), otherwise as before."""
    g = _real_gauge(M, n)
    if g is not None:
        d, R = g
        vals, vecs = np.linalg.eigh(R)
        return vals, (d[:, None] * vecs if vectors else None)
    return _eig(M, vectors)


def top_eigenpairs(H, n: int, n_states: int, kx: float = 0.0,
                   ky: float = 0.0, vectors: bool = True) -> tuple:
    """The n_states highest eigenvalues (descending) of the assembled
    6n x 6n Hermitian matrix H, and, with ``vectors``, their
    eigenvectors as the columns of a (6n, n_states) array.

    Exact block structure is used where it exists (module docstring);
    otherwise this is the dense diagonalization of earlier versions.
    """
    groups = _groups(H, n)
    if len(groups) > 1:
        res = _split_groups(H, n, groups, vectors)
        vals = np.concatenate([r[0] for r in res])
        order = np.argsort(-vals, kind="stable")[:n_states]
        if not vectors:
            return vals[order], None
        out = np.zeros((6 * n, order.size),
                       dtype=complex)
        starts = np.cumsum([0] + [r[0].size for r in res])
        for col, o in enumerate(order):
            k = int(np.searchsorted(starts, o, side="right") - 1)
            vals_k, vecs_k, idx = res[k]
            out[idx, col] = vecs_k[:, o - starts[k]]
        return vals[order], out
    if kx != 0.0 or ky != 0.0:
        up, lo, coupling, w = _split_upper_lower(H, n, kx, ky)
        if coupling <= _SPLIT_TOL * float(np.abs(H).max()):
            vu, xu = _eig_block(up, n, vectors)
            vl, xl = _eig_block(lo, n, vectors)
            vals = np.concatenate([vu, vl])
            order = np.argsort(-vals, kind="stable")[:n_states]
            if not vectors:
                return vals[order], None
            out = np.zeros((n, 6, order.size), dtype=complex)
            s = 1.0 / np.sqrt(2.0)
            for col, o in enumerate(order):
                if o < vu.size:
                    u, sign = xu[:, o].reshape(n, 3), 1.0
                else:
                    u, sign = xl[:, o - vu.size].reshape(n, 3), -1.0
                out[:, :3, col] = s * u
                out[:, 3:, col] = sign * s * np.conj(w)[None, :] * u
            return vals[order], out.reshape(6 * n, order.size)
    # dense: exactly the path of earlier versions
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1][:n_states]
    return vals[order], (vecs[:, order] if vectors else None)


def envelopes_from_vectors(vecs, n, dz):
    """(6n, m) eigenvectors -> (m, 6, n) envelopes normalized so that
    sum_m integral |F_m|^2 dz = 1."""
    m = vecs.shape[1]
    envelopes = np.empty((m, 6, n), dtype=complex)
    for j in range(m):
        v = vecs[:, j].reshape(n, 6).T  # (6, N)
        norm = np.sqrt((np.abs(v) ** 2).sum() * dz)
        envelopes[j] = v / norm
    return envelopes


def solve_subbands(p, z, kx: float = 0.0, ky: float = 0.0,
                   potential=None, n_states: int = 8,
                   strain=None) -> tuple:
    """Solve for the top valence states (the ones holes occupy).

    Returns (energies, envelopes):
      energies : (n_states,) eV, sorted descending (highest first).
      envelopes : (n_states, 6, N) complex envelope functions, normalized
          such that sum_m integral |F_m|^2 dz = 1.
    strain : optional symmetric 3 x 3 tensor, passed to the assembly
        (requires cited D1..D6 on the parameter set).

    Since v0.14.0 the matrix is diagonalized in exact independent blocks
    where it has them (see the module docstring): the energies agree
    with the full diagonalization of earlier versions to rounding
    (1e-12 eV is asserted in the tests), several times faster. At
    k_t = 0 the two members of each Kramers pair now have exactly equal
    energies, and each lies on one spin group (the first on the group
    with the lowest component number), on every computer. Which
    combination of a pair you get is a convention, as before: only
    quantities summed over a pair are fixed by the physics.
    """
    z = check_grid(z)
    n_states = check_n_states(n_states, z.size)
    H = assemble_hamiltonian(p, z, kx, ky, potential, strain=strain)
    vals, vecs = top_eigenpairs(H, z.size, n_states, kx, ky)
    return vals, envelopes_from_vectors(vecs, z.size, z[1] - z[0])


def subband_energies(p, z, kx: float = 0.0, ky: float = 0.0,
                     potential=None, n_states: int = 8,
                     strain=None) -> np.ndarray:
    """The energies of `solve_subbands` (bit for bit) without building
    the envelopes (new in v0.14.0). Descending, eV."""
    z = check_grid(z)
    n_states = check_n_states(n_states, z.size)
    H = assemble_hamiltonian(p, z, kx, ky, potential, strain=strain)
    return top_eigenpairs(H, z.size, n_states, kx, ky, vectors=False)[0]
