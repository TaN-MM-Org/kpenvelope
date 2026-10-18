"""v0.15 anchors.

Fix (each test fails on v0.14.0; the docstrings give the old numbers):

1. The check that no state below the computed ones holds holes looked
   only at k_t = 0 (`fill_subbands_kgrid` and the loops with
   ``filling="kgrid"``), and so missed a branch that rises above its
   k_t = 0 energy before it falls.

Speed (each result checked against the v0.14.0 code, which the tests
keep as the reference, with a stated tolerance):

2. The Chuang-Chang blocks at finite in-plane momentum are made real by
   a diagonal phase change and diagonalized as real symmetric matrices.
3. The assemblies place their blocks without a Python loop.

New, each against an independent calculation:

4. `parameter_sensitivities`: closed forms of the discrete demo well,
   central finite differences of the solvers' energies (Richardson
   extrapolated), `offset_sensitivity`, and a Monte Carlo sample of
   parameter sets for the propagated errors.
5. `level_transitions`: closed forms of the discrete demo well (the
   sine eigenvectors of the finite-difference matrix), the
   multiplet sums of `oscillator_strengths`, the published 0.10378 nm
   of the tilted GaN well, and invariance under any unitary mixing of
   the states inside each level.

Tutorial:

6. `examples/tutorial_isb_mass_calibration.py` runs, recovers the mass
   its synthetic line was made from within its error bar, and prints
   what its docstring shows.

No test here depends on which member of a Kramers pair the solver
returns first, or on the phase of an eigenvector.
"""
import dataclasses
import pathlib
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

import kpenvelope.selfconsistent as sc_mod
from kpenvelope import (HBAR2_OVER_2M0 as C, PARAMETER_NAMES,
                        aln_rinke2008, assemble_hamiltonian,
                        assemble_heterostructure, degenerate_levels,
                        demo_single_band, dipole_matrix,
                        fill_subbands_kgrid, gan_rinke2008,
                        heterostructure_energies, layered_profile,
                        level_transitions, offset_sensitivity,
                        oscillator_strengths, overlap_geometry_integral,
                        parameter_sensitivities, solve_self_consistent,
                        solve_subbands, subband_energies)
from kpenvelope.hamiltonian import bulk_blocks
from kpenvelope.solver import _omega, _real_gauge, top_eigenpairs

GAN = gan_rinke2008()
ALN = aln_rinke2008()
# deformation potentials: arbitrary TEST numbers (not material
# constants), only there to switch the strain terms on
GAN_D = dataclasses.replace(GAN, D1=-3.7, D2=4.5, D3=8.2, D4=-4.1,
                            D5=-4.0, D6=-5.5)


# ---------------------------------------------------------------- 1

M_TEST = 0.5          # synthetic masses (m0) and energies, not material
M_2 = 0.3             # numbers: two parabolic branches, and a third
E2_EDGE = -0.02       # that rises by A_LIN^2 M_2 / (4 C) = 1.2 meV to
A_LIN = 0.0247        # its top at k = A_LIN M_2 / (2 C) = 0.097 nm^-1
                      # and stays below the first two everywhere


def _three_branches(k):
    """Two parabolic branches from 0 and a third from E2_EDGE that rises
    linearly first (as a spin-split branch of an asymmetric well does),
    then a far lower fourth; descending at every k."""
    e01 = -C * k * k / M_TEST
    e2 = E2_EDGE + A_LIN * k - C * k * k / M_2
    return np.array([e01, e01, e2, E2_EDGE - 0.5 - k])


# E_F of the two parabolic branches alone is -ps / (2 dos) with
# dos = M/(4 pi C); ps puts it 0.5 meV above the third branch's edge
PS_TEST = -2.0 * (M_TEST / (4 * np.pi * C)) * (E2_EDGE + 0.0005)


def test_fill_kgrid_warns_when_the_next_branch_rises_above_its_edge():
    """Branch 2 starts at -20 meV, 0.5 meV below the Fermi level of the
    two branches filled (-19.50 meV), but rises to -18.8 meV at
    0.097 nm^-1. With n_states = 2 and three energies per point,
    v0.14.0 compared only the k_t = 0 energy with E_F and gave no
    warning, although branch 2 holds 1.11e-3 nm^-2 (2.7 % of ps) when
    it is filled too (n_states = 3, E_F = -18.97 meV); now it warns."""
    def solve_at_k(kx, ky):
        return _three_branches(np.hypot(kx, ky))

    kw = dict(ps=PS_TEST, kmax=1.0, nk=81, ntheta=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        occ2, ef2 = fill_subbands_kgrid(solve_at_k, n_states=2, **kw)
    msgs = [str(x.message) for x in w if x.category is RuntimeWarning]
    assert len(msgs) == 1 and "highest point on the momentum grid" \
        in msgs[0]
    assert ef2 > E2_EDGE                    # the edge alone looks empty
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # complete: no warning
        occ3, ef3 = fill_subbands_kgrid(solve_at_k, n_states=3, **kw)
    assert abs(occ3[2] - 1.11e-3) < 1e-5 and abs(ef3 + 0.018967) < 1e-6
    # the edge check alone (the v0.14.0 test) would have passed
    assert solve_at_k(0.0, 0.0)[2] < ef2


def test_kgrid_loop_checks_the_next_branch_on_the_whole_grid(
        monkeypatch):
    """The same three branches inside `solve_self_consistent` with
    ``filling="kgrid"`` (the solver is replaced by the synthetic one,
    whose envelopes do not depend on the potential, so the loop
    settles at once). v0.14.0 checked the next state at k_t = 0 only
    and reported n_states_sufficient = True; now it warns and reports
    False. Without the rising branch nothing changes."""
    z = np.linspace(0.0, 6.0, 30)
    env = np.zeros((6, z.size))
    env[0] = np.sin(np.pi * (z + 0.2) / 6.4)
    env /= np.sqrt((env ** 2).sum() * (z[1] - z[0]))

    def fake_solve(p, z_, kx=0.0, ky=0.0, potential=None, n_states=8):
        e = _three_branches(np.hypot(kx, ky))[:n_states]
        return e, np.repeat(env[None], n_states, axis=0).astype(complex)

    def fake_energies(p, z_, kx=0.0, ky=0.0, potential=None,
                      n_states=8):
        return _three_branches(np.hypot(kx, ky))[:n_states]

    monkeypatch.setattr(sc_mod, "solve_subbands", fake_solve)
    monkeypatch.setattr(sc_mod, "subband_energies", fake_energies)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = solve_self_consistent(demo_single_band(), z, PS_TEST,
                                  n_states=2, filling="kgrid", kmax=1.0,
                                  nk=81, mixing=0.5, tol=1e-6)
    assert r.converged and r.fermi_level > E2_EDGE
    assert r.n_states_sufficient is False
    assert any("highest point on the momentum grid" in str(x.message)
               for x in w)
    # with three states nothing is missing
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r3 = solve_self_consistent(demo_single_band(), z, PS_TEST,
                                   n_states=3, filling="kgrid", kmax=1.0,
                                   nk=81, mixing=0.5, tol=1e-6)
    assert r3.n_states_sufficient and abs(r3.occupations[2] - 1.11e-3) \
        < 1e-5


def test_kgrid_loop_results_unchanged_by_the_extra_state():
    """The final state of the k-grid loop now solves for one more state
    at every grid momentum; the first n_states of such a solve are those
    of an n_states solve, so the results are the same as when that
    extra state is not asked for (the solver's prefix property,
    checked directly, energies to 0 and envelopes to 0)."""
    z = np.linspace(0.0, 5.0, 31)
    for kx, ky in ((0.0, 0.0), (0.3, 0.1)):
        e4, f4 = solve_subbands(GAN, z, kx, ky, potential=0.1 * z,
                                n_states=4)
        e5, f5 = solve_subbands(GAN, z, kx, ky, potential=0.1 * z,
                                n_states=5)
        assert np.array_equal(e4, e5[:4]) and np.array_equal(f4, f5[:4])


# ---------------------------------------------------------------- 2

def _split_v014(H, n, kx, ky):
    """The upper/lower split of v0.14.0, as it was."""
    w = _omega(kx, ky)
    H4 = H.reshape(n, 2, 3, n, 2, 3)
    aa, ab = H4[:, 0, :, :, 0, :], H4[:, 0, :, :, 1, :]
    ba, bb = H4[:, 1, :, :, 0, :], H4[:, 1, :, :, 1, :]
    wl = w[None, :, None, None]
    wr = np.conj(w)[None, None, None, :]
    p = ab * wr
    q = wl * ba
    r = wl * bb * wr
    up = (0.5 * (aa + p + q + r)).reshape(3 * n, 3 * n)
    lo = (0.5 * (aa - p - q + r)).reshape(3 * n, 3 * n)
    return 0.5 * (up + up.conj().T), 0.5 * (lo + lo.conj().T), w


def _top_v014(H, n, m, kx, ky):
    """Eigenpairs of the two blocks by the complex diagonalization of
    v0.14.0, assembled into 6n-vectors as v0.14.0 did."""
    up, lo, w = _split_v014(H, n, kx, ky)
    vu, xu = np.linalg.eigh(up)
    vl, xl = np.linalg.eigh(lo)
    vals = np.concatenate([vu, vl])
    order = np.argsort(-vals, kind="stable")[:m]
    out = np.zeros((n, 6, m), dtype=complex)
    s = 1.0 / np.sqrt(2.0)
    for col, o in enumerate(order):
        if o < vu.size:
            u, sign = xu[:, o].reshape(n, 3), 1.0
        else:
            u, sign = xl[:, o - vu.size].reshape(n, 3), -1.0
        out[:, :3, col] = s * u
        out[:, 3:, col] = sign * s * np.conj(w)[None, :] * u
    return vals[order], out.reshape(6 * n, m)


def _gauge_cases():
    z = np.linspace(0.0, 9.0, 31)
    pl, edge = layered_profile(z, [(3.0, ALN, -0.5), (3.0, GAN, 0.0),
                                   (3.0, ALN, -0.5)])
    biax = np.diag([0.01, 0.01, -0.005])
    for kx, ky in ((0.3, 0.0), (0.2, 0.25), (0.0, 0.4), (-0.3, 0.1),
                   (-0.05, -0.6), (1.2, -0.7)):
        yield z, kx, ky, assemble_hamiltonian(GAN, z, kx, ky,
                                              potential=0.05 * z)
        yield z, kx, ky, assemble_hamiltonian(ALN, z, kx, ky)
        yield z, kx, ky, assemble_heterostructure(z, pl, edge, kx, ky,
                                                  potential=0.03 * z)
        yield z, kx, ky, assemble_hamiltonian(GAN_D, z, kx, ky,
                                              potential=0.05 * z,
                                              strain=biax)


def test_real_gauge_blocks_match_the_v014_complex_blocks():
    """At finite in-plane momentum each Chuang-Chang block is made real
    by one phase per component and diagonalized as a real symmetric
    matrix (24 cases: GaN, AlN, a layered stack with a potential and a
    biaxially strained GaN, six directions and sizes of k_t). Against
    the complex diagonalization of v0.14.0 (kept above as the
    reference): energies to 1e-12 eV (the largest difference seen is
    about 1e-14 eV; eigenvalues of a matrix whose largest element is
    about 50 eV are fixed to a few times 1e-16 of that), projectors
    onto the top 8 states (a gap in every case) to 1e-10, residuals
    |H v - E v| below 1e-12 eV and orthonormality to 1e-12."""
    n_states = 8
    for z, kx, ky, H in _gauge_cases():
        n = z.size
        e_ref, v_ref = _top_v014(H, n, n_states, kx, ky)
        e, v = top_eigenpairs(H, n, n_states, kx, ky)
        assert np.abs(e - e_ref).max() < 1e-12
        allv = np.linalg.eigvalsh(H)[::-1]
        assert allv[n_states - 1] - allv[n_states] > 1e-6   # a gap
        P = v @ v.conj().T
        P_ref = v_ref @ v_ref.conj().T
        assert np.abs(P - P_ref).max() < 1e-10
        assert np.abs(H @ v - v * e[None, :]).max() < 1e-12
        assert np.abs(v.conj().T @ v - np.eye(n_states)).max() < 1e-12


def test_real_gauge_is_refused_for_a_block_that_cannot_be_real():
    """A random complex Hermitian matrix has no such phases: the gauge
    returns None (and the complex diagonalization is used). The phases
    it finds for a solver block leave an imaginary part at rounding
    level only (below 1e-15 of the largest element here)."""
    n = 5
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3 * n, 3 * n)) + 1j * rng.normal(size=(3 * n,
                                                                3 * n))
    assert _real_gauge(A + A.conj().T, n) is None
    z = np.linspace(0.0, 4.0, n)
    H = assemble_hamiltonian(GAN, z, 0.3, 0.2)
    up = _split_v014(H, n, 0.3, 0.2)[0]
    d, R = _real_gauge(up, n)
    back = (d[:, None] * R) * np.conj(d)[None, :]
    assert np.abs(back - up).max() < 1e-15 * np.abs(up).max()


def _assemble_v014(p, z, kx=0.0, ky=0.0, potential=None):
    """The per-point loop of v0.14.0 (hard walls, no strain)."""
    n = z.size
    dz = z[1] - z[0]
    H0, H1, H2 = bulk_blocks(p, kx, ky)
    H = np.zeros((6 * n, 6 * n), dtype=complex)
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dz = 1.0 / (2.0 * dz)
    for i in range(n):
        diag = H0.copy()
        diag += 2.0 * H2 * inv_dz2
        if potential is not None:
            diag -= potential[i] * np.eye(6)
        H[6 * i:6 * i + 6, 6 * i:6 * i + 6] = diag
        if i + 1 < n:
            off = -H2 * inv_dz2 + (-1j) * H1 * inv_2dz
            H[6 * i:6 * i + 6, 6 * i + 6:6 * i + 12] = off
            H[6 * i + 6:6 * i + 12, 6 * i:6 * i + 6] = off.conj().T
    return H


def _assemble_hetero_v014(z, plist, band_edge, kx, ky, potential):
    """The per-point loop of v0.14.0 (layered stack, no strain)."""
    n = z.size
    dz = z[1] - z[0]
    blocks = [bulk_blocks(p, kx, ky) for p in plist]
    H = np.zeros((6 * n, 6 * n), dtype=complex)
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dz = 1.0 / (2.0 * dz)
    half = [0.5 * (blocks[i][2] + blocks[i + 1][2]) for i in range(n - 1)]
    for i in range(n):
        H0_i, H1_i, H2_i = blocks[i]
        diag = H0_i.copy()
        left = half[i - 1] if i > 0 else blocks[i][2]
        right = half[i] if i + 1 < n else blocks[i][2]
        diag += (left + right) * inv_dz2
        diag += band_edge[i] * np.eye(6)
        diag -= potential[i] * np.eye(6)
        H[6 * i:6 * i + 6, 6 * i:6 * i + 6] = diag
        if i + 1 < n:
            H1_half = 0.5 * (H1_i + blocks[i + 1][1])
            off = -half[i] * inv_dz2 + (-1j) * H1_half * inv_2dz
            H[6 * i:6 * i + 6, 6 * i + 6:6 * i + 12] = off
            H[6 * i + 6:6 * i + 12, 6 * i:6 * i + 6] = off.conj().T
    return H


def test_assembly_without_the_python_loop_matches_v014():
    """The blocks are computed by the same operations as in the loop of
    v0.14.0 (kept above) and only placed differently, so the matrices
    agree to rounding of single additions and multiplications; the
    tolerance is 1e-15 of the largest element (they were identical on
    the development machine)."""
    rng = np.random.default_rng(7)
    for n in (3, 4, 17, 40):
        z = np.linspace(0.0, 0.25 * n, n)
        for kx, ky in ((0.0, 0.0), (0.4, -0.3)):
            v = rng.normal(size=n) * 0.1
            for p in (GAN, ALN, demo_single_band()):
                A = _assemble_v014(p, z, kx, ky, v)
                B = assemble_hamiltonian(p, z, kx, ky, potential=v)
                assert np.abs(A - B).max() <= 1e-15 * np.abs(A).max()
            pl = [GAN if rng.random() < 0.5 else ALN for _ in range(n)]
            edge = rng.normal(size=n) * 0.2
            A = _assemble_hetero_v014(z, pl, edge, kx, ky, v)
            B = assemble_heterostructure(z, pl, edge, kx, ky, potential=v)
            assert np.abs(A - B).max() <= 1e-15 * np.abs(A).max()


# ---------------------------------------------------------------- 4

def test_sensitivities_equal_the_closed_forms_of_the_demo_well():
    """Demo set (six identical bands E = A c k^2) between hard walls,
    N = 41 points: the finite-difference matrix has the eigenvalues
    A c (2 - 2 cos(l pi / (N + 1))) / dz^2 (+ A c k_t^2), each six-fold,
    so dE/dA1 = c (2 - 2 cos(l pi / (N + 1))) / dz^2 for all six states
    of level l and dE/dA2 = c k_t^2 (A1 and A2 act on all six
    components). To 1e-12 relative. A3 and delta1 act on four of the
    six components (A4 and A5 too at finite k_t): they split the level,
    which is flagged, and the value is the level mean, 4/6 of the full
    shift for A3 and exactly 4/6 for delta1."""
    p = demo_single_band(-2.0)
    N = 41
    z = np.linspace(0.0, 4.0, N)
    dz = z[1] - z[0]
    lev = np.repeat(np.arange(1, 4), 6)
    shift = C * (2.0 - 2.0 * np.cos(lev * np.pi / (N + 1))) / dz ** 2
    r = parameter_sensitivities(z, p, n_states=18)
    d = r["derivatives"][:, 0, :]
    assert np.abs(d[:, 0] / shift - 1.0).max() < 1e-12
    assert np.abs(d[:, 2] / (4.0 / 6.0 * shift) - 1.0).max() < 1e-12
    assert np.abs(d[:, 6] - 4.0 / 6.0).max() < 1e-12
    for k in (1, 3, 4, 5, 7, 8):      # nothing (A2, A4..A6 need k_t) or
        assert np.abs(d[:, k]).max() < 1e-12   # +/- in equal parts
    assert r["level_split"].all()
    assert np.array_equal(r["energies"],
                          subband_energies(p, z, n_states=18))
    kt = 0.5
    r = parameter_sensitivities(z, p, kx=0.3, ky=0.4, n_states=12)
    d = r["derivatives"][:, 0, :]
    assert np.abs(d[:, 0] / shift[:12] - 1.0).max() < 1e-12
    assert np.abs(d[:, 1] / (C * kt * kt) - 1.0).max() < 1e-12


def _richardson(f, h):
    """Central difference extrapolated from steps h and 2h; its error
    is O(h^4) plus rounding."""
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(2 * h) - f(-2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


def test_sensitivities_against_finite_differences_on_gan():
    """Hellmann-Feynman derivatives against central finite differences
    of the solvers' own energies, for every parameter: a tilted GaN
    well at k_t = 0 (Kramers pairs, not split by any parameter) and at
    k = (0.25, 0.1) nm^-1, and a GaN well with AlN barriers at k_t = 0.2
    (both materials and the band-edge scale). Steps of 2e-4 (A's) and
    2e-5 eV (deltas), Richardson extrapolated. Tolerance 2e-8 eV per
    unit of the parameter: the extrapolated truncation error was below
    4e-10 here, and the rounding term is about (5/3) dE / (2 h) with
    dE ~ 1e-13 eV (eigenvalue rounding for elements up to ~100 eV), at
    most 4e-9. The plain central difference converges to the same
    values at second order (checked once in development: the
    difference falls fourfold per halving of the step)."""
    z = np.linspace(0.0, 5.0, 41)
    v = 0.04 * z

    def check(r, energies_of, materials, n_states):
        for j, mat in enumerate(materials):
            for k, name in enumerate(PARAMETER_NAMES):
                h = 2e-4 if name.startswith("A") else 2e-5

                def f(dh, name=name, mat=mat):
                    new = dataclasses.replace(
                        mat, **{name: getattr(mat, name) + dh})
                    return energies_of(mat, new)[:n_states]

                fd = _richardson(f, h)
                assert np.abs(fd - r["derivatives"][:, j, k]).max() \
                    < 2e-8, (j, name)

    for kx, ky, ns in ((0.0, 0.0, 6), (0.25, 0.1, 4)):
        r = parameter_sensitivities(z, GAN, kx=kx, ky=ky, potential=v,
                                    n_states=ns)
        assert not r["level_split"].any()
        if kx == 0.0:      # both members of each pair: the same number
            d = r["derivatives"]
            assert np.abs(d[0::2] - d[1::2]).max() < 1e-12
        check(r, lambda old, new: subband_energies(new, z, kx, ky, v, ns),
              [GAN], ns)

    zz = np.linspace(0.0, 6.0, 41)
    pl, shape = layered_profile(zz, [(1.5, ALN, -1.0), (3.0, GAN, 0.0),
                                     (1.5, ALN, -1.0)])
    edge = 0.2 * shape
    r = parameter_sensitivities(zz, pl, band_edge=edge, kx=0.2,
                                n_states=4)
    assert r["materials"][0] is ALN and r["materials"][1] is GAN

    def het(old, new):
        swapped = [new if p is old else p for p in pl]
        return heterostructure_energies(zz, swapped, edge, 0.2,
                                        n_states=4)
    check(r, het, r["materials"], 4)

    def edge_f(ds):
        return heterostructure_energies(zz, pl, (1.0 + ds) * edge, 0.2,
                                        n_states=4)
    fd = _richardson(edge_f, 1e-4)
    assert np.abs(fd - r["d_band_edge_scale"]).max() < 2e-8


def test_band_edge_derivative_equals_offset_sensitivity():
    """README example 6 stack (AlN | GaN | AlN, 97 points) at an offset
    of 0.1 eV: the exact derivative of the 0 -> 2 spacing with respect
    to the offset, (dE0/ds - dE2/ds) / offset, equals the central
    difference of `offset_sensitivity` (relative step 1e-4) to 1e-6
    relative."""
    z = np.linspace(0.0, 12.0, 97)
    params, shape = layered_profile(z, [(4.0, ALN, -1.0), (4.0, GAN, 0.0),
                                        (4.0, ALN, -1.0)])
    r = parameter_sensitivities(z, params, band_edge=0.1 * shape,
                                n_states=4)
    d = r["d_band_edge_scale"]
    exact = (d[0] - d[2]) / 0.1
    fd = offset_sensitivity(z, params, shape, 0.1)
    assert abs(exact / fd - 1.0) < 1e-6


def test_propagated_errors_match_a_monte_carlo_sample():
    """Tilted GaN well (31 points, k_t = 0) with 2 % errors on A1..A6
    and 1 meV on each splitting (test numbers, not cited errors). The
    linear propagation gives sigma of E0 and of the spacing E0 - E2;
    400 parameter sets drawn from those errors (fixed seed) give a
    sample spread that agrees to 12 % (the sampling error of a standard
    deviation from 400 draws is 3.5 %). The covariance is J diag(s^2)
    J^T, checked here entry by entry against that formula too."""
    z = np.linspace(0.0, 4.0, 31)
    v = 0.05 * z
    sig = {name: (0.02 * abs(getattr(GAN, name)) if name.startswith("A")
                  else 0.001) for name in PARAMETER_NAMES}
    r = parameter_sensitivities(z, GAN, potential=v, n_states=4,
                                sigmas=sig)
    J = r["derivatives"][:, 0, :]
    s = np.array([sig[n] for n in PARAMETER_NAMES])
    assert np.allclose(r["covariance"], (J * s ** 2) @ J.T, rtol=1e-12,
                       atol=0.0)
    cov = r["covariance"]
    s_e0 = np.sqrt(cov[0, 0])
    s_gap = np.sqrt(cov[0, 0] + cov[2, 2] - 2 * cov[0, 2])
    rng = np.random.default_rng(11)
    samples = []
    for _ in range(400):
        q = dataclasses.replace(GAN, **{n: getattr(GAN, n) + sig[n]
                                        * rng.normal()
                                        for n in PARAMETER_NAMES})
        samples.append(subband_energies(q, z, potential=v, n_states=4))
    e = np.array(samples)
    assert abs(e[:, 0].std() / s_e0 - 1.0) < 0.12
    assert abs((e[:, 0] - e[:, 2]).std() / s_gap - 1.0) < 0.12


def test_sensitivity_refusals():
    z = np.linspace(0.0, 4.0, 21)
    with pytest.raises(ValueError, match="unknown parameter"):
        parameter_sensitivities(z, GAN, sigmas={"A7": 0.1})
    with pytest.raises(ValueError, match="finite and >= 0"):
        parameter_sensitivities(z, GAN, sigmas={"A1": -0.1})
    with pytest.raises(ValueError, match="needs a band_edge"):
        parameter_sensitivities(z, GAN, sigma_band_edge_scale=0.1)
    pl = [GAN] * 10 + [ALN] * 11
    with pytest.raises(ValueError, match="one mapping of parameter"):
        parameter_sensitivities(z, pl, sigmas={"A1": 0.1})
    with pytest.raises(ValueError, match="one parameter set per grid"):
        parameter_sensitivities(z, pl[:-1])
    with pytest.raises(ValueError, match="n_states"):
        parameter_sensitivities(z, GAN, n_states=0)
    with pytest.raises(ValueError, match="one mapping of parameter"):
        parameter_sensitivities(z, pl, sigmas=[{"A1": 0.1}])
    with pytest.raises(ValueError, match="as a mapping"):
        parameter_sensitivities(z, pl, sigmas=[{"A1": 0.1}, 0.2])
    with pytest.raises(ValueError, match="WurtziteParameters"):
        parameter_sensitivities(z, [GAN] * 20 + ["GaN"])
    with pytest.raises(ValueError, match="finite"):
        parameter_sensitivities(z, GAN, kx=float("nan"))
    r = parameter_sensitivities(z, pl, sigmas=[{"A1": 0.1}, None])
    assert r["sigma_energies"].shape == (8,)
    # the band-edge scale error adds s^2 d d^T to the covariance
    edge = np.where(np.arange(21) < 10, -0.3, 0.0)
    a = parameter_sensitivities(z, pl, band_edge=edge, sigmas=[{}, {}])
    b = parameter_sensitivities(z, pl, band_edge=edge, sigmas=[{}, {}],
                                sigma_band_edge_scale=0.1)
    d = a["d_band_edge_scale"]
    assert np.abs(a["covariance"]).max() == 0.0
    assert np.allclose(b["covariance"], 0.01 * np.outer(d, d), rtol=1e-14,
                       atol=0.0)
    with pytest.raises(ValueError, match="sigma_band_edge_scale"):
        parameter_sensitivities(z, pl, band_edge=edge,
                                sigma_band_edge_scale=-1.0)


def test_heavy_hole_mass_along_c_is_one_over_a1_plus_a3():
    """The tutorial calibrates A1 + A3 as the heavy-hole mass along c,
    m0 / |A1 + A3|. In the bulk matrix at k_t = 0 the heavy-hole
    components (1 and 4) are decoupled with the energy
    delta1 + delta2 + c (A1 + A3) kz^2: two of the six bulk eigenvalues
    equal it, for GaN and AlN at several kz (1e-14 eV)."""
    for p in (GAN, ALN):
        for kz in (0.0, 0.3, 1.1):
            H0, H1, H2 = bulk_blocks(p, 0.0, 0.0)
            ev = np.linalg.eigvalsh(H0 + H1 * kz + H2 * kz * kz)
            hh = p.delta1 + p.delta2 + C * (p.A1 + p.A3) * kz * kz
            assert np.sort(np.abs(ev - hh))[1] < 1e-14


# ---------------------------------------------------------------- 5

def test_level_transitions_equal_the_discrete_closed_forms():
    """Demo set, 8 nm hard-wall well on 201 points (README example 4).
    The finite-difference matrix has the eigenvectors
    phi_l(j) = sqrt(2/(N+1)) sin(l pi j / (N+1)), so the level 1 -> 2
    dipole is z_12 = sum_j z_j phi_1 phi_2 and the depolarization
    integral is S = dz sum_j (sum_{i<=j} phi_1 phi_2)^2, both from the
    closed-form vectors, not the solver's. Level sums (six states each)
    equal them to 1e-12 relative; the oscillator strength equals the
    multiplet sum of `oscillator_strengths` (README: 0.9606) to 1e-12;
    the same-parity level 3 has z2 below 1e-20 nm^2."""
    p = demo_single_band(A=-2.0)
    L = 8.0
    z = np.linspace(0.0, L, 201)
    E, F = solve_subbands(p, z, n_states=18)
    r = level_transitions(z, E, F, mass_ratio=0.5)
    assert list(r["degeneracy"]) == [6, 6, 6]
    N = z.size
    j = np.arange(1, N + 1)

    def phi(m):
        return np.sqrt(2.0 / (N + 1)) * np.sin(m * np.pi * j / (N + 1))

    z12 = (z * phi(1) * phi(2)).sum()
    S12 = ((np.cumsum(phi(1) * phi(2))) ** 2).sum() * (z[1] - z[0])
    assert abs(r["z2"][1] / z12 ** 2 - 1.0) < 1e-12
    assert abs(r["S"][1] / S12 - 1.0) < 1e-12
    assert r["z2"][2] < 1e-20 and r["z2"][0] == 0.0
    f = oscillator_strengths(E, dipole_matrix(z, F), mass_ratio=0.5)
    assert abs(r["oscillator_strength"][1] / f[6:12].sum() - 1.0) < 1e-12
    assert abs(r["oscillator_strength"][1] - 0.9606) < 1e-4
    e_cf = -2.0 * C * (2 - 2 * np.cos(np.pi * np.arange(1, 3) / (N + 1))) \
        / (z[1] - z[0]) ** 2
    assert abs(r["transition_energy"][1] / (e_cf[0] - e_cf[1]) - 1) < 1e-10


def test_level_transitions_do_not_depend_on_the_basis_inside_a_level():
    """Tilted GaN well of README example 5 (6 nm, 61 points, potential
    0.08 z), k_t = 0. One HH2 member gives S = 0.10378 nm and the other
    0 (a convention of the solver); the level value is 0.10378 nm (the
    number pinned since v0.13.0, from the textbook form summed over the
    pair). Mixing each Kramers pair by a random unitary matrix changes
    the single-member values but not the level values (z2 and S to
    1e-10 relative). At finite k_t, where every level is one state,
    the level values are the single-state values (1e-12 of the largest
    value)."""
    z = np.linspace(0.0, 6.0, 61)
    e, F = solve_subbands(GAN, z, potential=0.08 * z, n_states=6)
    r = level_transitions(z, e, F)
    assert [len(c) for c in r["levels"]] == [2, 2, 2]
    assert abs(r["S"][2] - 0.10378) < 1e-5
    single = [overlap_geometry_integral(z, F[0], F[f]) for f in (4, 5)]
    assert abs(sum(single) / r["S"][2] - 1.0) < 1e-12
    rng = np.random.default_rng(5)
    changed = False
    for _ in range(4):
        G = F.copy()
        for a in (0, 2, 4):
            U = np.linalg.qr(rng.normal(size=(2, 2))
                             + 1j * rng.normal(size=(2, 2)))[0]
            G[a:a + 2] = np.tensordot(U, F[a:a + 2], 1)
        q = level_transitions(z, e, G)
        assert np.abs(q["S"] - r["S"]).max() < 1e-10 * r["S"].max()
        assert np.abs(q["z2"] - r["z2"]).max() < 1e-10 * r["z2"].max()
        one = overlap_geometry_integral(z, G[0], G[4])
        changed |= abs(one - single[0]) > 1e-3
    assert changed
    e, F = solve_subbands(GAN, z, kx=0.3, potential=0.08 * z, n_states=4)
    r = level_transitions(z, e, F)
    assert list(r["degeneracy"]) == [1, 1, 1, 1]
    d = dipole_matrix(z, F)
    for f in (1, 2, 3):
        S1 = overlap_geometry_integral(z, F[0], F[f])
        assert abs(r["S"][f] - S1) <= 1e-12 * r["S"].max()
        assert abs(r["z2"][f] - abs(d[0, f]) ** 2) <= 1e-12 * r["z2"].max()


def test_level_refusals():
    z = np.linspace(0.0, 4.0, 21)
    e, F = solve_subbands(GAN, z, n_states=4)
    with pytest.raises(ValueError, match="sorted"):
        degenerate_levels(e[::-1])
    with pytest.raises(ValueError, match="shape"):
        level_transitions(z, e, F[:3])
    with pytest.raises(ValueError, match="initial"):
        level_transitions(z, e, F, initial=2)
    with pytest.raises(ValueError, match="mass_ratio"):
        level_transitions(z, e, F, mass_ratio=-1.0)
    with pytest.raises(ValueError, match="grid"):
        level_transitions(z[:-1], e, F)
    assert [len(c) for c in degenerate_levels(e)] == [2, 2]
    r = level_transitions(z, e, F, initial=1)
    assert r["transition_energy"][0] < 0.0 and r["z2"][1] == 0.0


# ---------------------------------------------------------------- 6

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def test_tutorial_isb_mass_calibration():
    """The tutorial recovers the heavy-hole mass along c its synthetic
    line was made from (2.0 m0) within its total error bar, its exact
    and finite-difference sensitivities agree, and the printed output
    is the one in the script's docstring."""
    out = subprocess.run([sys.executable,
                          str(EXAMPLES / "tutorial_isb_mass_calibration.py")],
                         capture_output=True, text=True, check=True,
                         timeout=600).stdout
    m = re.search(r"m_hh\^c = ([\d.]+) \+/- ([\d.]+) \(line\) "
                  r"\+/- ([\d.]+) \(other parameters\) m0", out)
    val, s_line, s_par = (float(x) for x in m.groups())
    assert abs(val - 2.0) < 2 * np.hypot(s_line, s_par) + 1e-3
    hf, fd = (float(x) for x in re.search(
        r"exact ([\d.-]+) eV, finite difference ([\d.-]+) eV",
        out).groups())
    assert abs(hf - fd) < 2e-6
    doc = (EXAMPLES / "tutorial_isb_mass_calibration.py").read_text()
    assert "\n".join("    " + line if line else line
                     for line in out.strip().split("\n")) in doc
