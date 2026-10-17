"""v0.14 anchors.

Speed (each result checked against the dense diagonalization of
v0.13.1, which the tests keep as the reference):

1. The block eigen-solver gives the energies of the dense
   diagonalization to 1e-12 eV and the same states, compared through
   the projector onto each complete degenerate level (independent of
   the basis and phase inside a Kramers pair), for hard walls, layered
   stacks, potentials, strains that keep and that break the block
   structure, at k_t = 0 and in several in-plane directions.
2. The self-consistent loops and the momentum-grid filler give the
   same results with the old and the new eigen-solver.

Fixes (each test fails on v0.13.1; the docstrings give the old
numbers):

3. `dos_from_dispersion` returned 0 and a spike at the start of a path
   that does not begin at k_t = 0.
4. `fill_subbands_thermal` filled negative masses with negative
   occupations.

New, each against an independent calculation:

5. `cyclotron_mass`: exact for parabolic branches, second order
   against a closed-form nonparabolic branch, and against a direct
   difference of two six-band solves around k_F.
6. `fit_sdh_dingle`: constants against scipy.constants, the Dingle
   temperature against the Lifshitz-Kosevich form, noise-free data,
   numpy.polyfit (weighted and unweighted), a Monte Carlo check of the
   error bars and a finite-difference check of the mass propagation.

Tutorials:

7. Both scripts in examples/ run and print their key numbers, which
   are checked against the true values they were made from (analysis)
   and against the Onsager relation (prediction).

No test here depends on which member of a Kramers pair the solver
returns first, or on the phase of an eigenvector.
"""
import dataclasses
import math
import pathlib
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest
import scipy.constants as sc

import kpenvelope
import kpenvelope.heterostructure as het_mod
import kpenvelope.solver as solver_mod
from kpenvelope import (HBAR2_OVER_2M0 as C, aln_rinke2008,
                        assemble_hamiltonian, assemble_heterostructure,
                        cyclotron_mass, demo_single_band,
                        dos_from_dispersion, fill_subbands_kgrid,
                        fill_subbands_thermal, fit_sdh_dingle,
                        gan_rinke2008, heterostructure_energies,
                        layered_profile, lk_thermal_factor,
                        solve_heterostructure, solve_self_consistent,
                        solve_self_consistent_hetero, solve_subbands,
                        subband_dispersion, subband_energies)
from kpenvelope.lab import (DINGLE_KELVIN_PS, DINGLE_TESLA_PS,
                            LK_TESLA_PER_KELVIN)
from kpenvelope.solver import top_eigenpairs

GAN = gan_rinke2008()
ALN = aln_rinke2008()
# deformation potentials: arbitrary TEST numbers (not material
# constants), only there to switch the strain terms on
GAN_D = dataclasses.replace(GAN, D1=-3.7, D2=4.5, D3=8.2, D4=-4.1,
                            D5=-4.0, D6=-5.5)


def _dense(H, n, n_states, kx=0.0, ky=0.0, vectors=True):
    """The eigen-solution of v0.13.1, kept as the reference."""
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1][:n_states]
    return vals[order], (vecs[:, order] if vectors else None)


def _levels(e, tol=1e-9):
    """Index ranges of degenerate levels (energies descending)."""
    out, i = [], 0
    while i < e.size:
        j = i
        while j + 1 < e.size and abs(e[j + 1] - e[i]) < tol:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


# ---------------------------------------------------------------- 1

def _cases():
    z = np.linspace(0.0, 9.0, 31)
    pl, edge = layered_profile(z, [(3.0, ALN, -0.5), (3.0, GAN, 0.0),
                                   (3.0, ALN, -0.5)])
    strains = {
        "biaxial": np.diag([0.01, 0.01, -0.005]),
        "uniaxial in-plane": np.diag([0.01, 0.002, -0.005]),
        "shear xy": np.array([[0.01, 0.003, 0.0], [0.003, 0.01, 0.0],
                              [0.0, 0.0, -0.005]]),
        "shear xz": np.array([[0.01, 0.0, 0.002], [0.0, 0.01, 0.0],
                              [0.002, 0.0, -0.005]]),
    }
    for kx, ky in ((0.0, 0.0), (0.3, 0.0), (0.2, 0.25), (0.0, 0.4),
                   (-0.3, 0.1)):
        yield ("tilted GaN", z, kx, ky,
               assemble_hamiltonian(GAN, z, kx, ky, potential=0.05 * z))
        yield ("symmetric GaN", z, kx, ky,
               assemble_hamiltonian(GAN, z, kx, ky))
        yield ("AlN/GaN/AlN", z, kx, ky,
               assemble_heterostructure(z, pl, edge, kx, ky,
                                        potential=0.03 * z))
        yield ("demo", z, kx, ky,
               assemble_hamiltonian(demo_single_band(), z, kx, ky))
        for name, s in strains.items():
            yield (name, z, kx, ky,
                   assemble_hamiltonian(GAN_D, z, kx, ky,
                                        potential=0.05 * z, strain=s))


def test_block_solver_matches_the_dense_diagonalization():
    """Energies to 1e-12 eV; projector of every complete level among
    the top 8 states to 1e-8; orthonormal eigenvectors and residual
    |H v - E v| below 1e-12 eV. Cases with in-plane anisotropic strain
    whose axes differ from the momentum direction cannot be split and
    must give the dense result bit for bit."""
    n_states = 8
    for name, z, kx, ky, H in _cases():
        n = z.size
        e_ref, v_ref = _dense(H, n, 6 * n)
        e, v = top_eigenpairs(H, n, n_states, kx, ky)
        assert np.abs(e - e_ref[:n_states]).max() < 1e-12, (name, kx, ky)
        for i, j in _levels(e_ref):
            if j > n_states:
                break
            p_ref = v_ref[:, i:j] @ v_ref[:, i:j].conj().T
            p = v[:, i:j] @ v[:, i:j].conj().T
            assert np.abs(p - p_ref).max() < 1e-8, (name, kx, ky, i)
        assert np.abs(v.conj().T @ v - np.eye(n_states)).max() < 1e-12
        assert np.abs(H @ v - v * e).max() < 1e-12
        # strains whose in-plane axes differ from the momentum direction
        # break the block structure: then the dense path is used
        if ((name == "shear xy" and (kx, ky) != (0.0, 0.0))
                or (name in ("uniaxial in-plane", "shear xz")
                    and (kx, ky) in ((0.2, 0.25), (-0.3, 0.1)))):
            assert np.array_equal(e, e_ref[:n_states]), (name, kx, ky)


def test_solvers_and_energy_helpers_agree():
    """solve_subbands and solve_heterostructure equal the dense
    reference to 1e-12 eV, their envelopes are normalized, and the
    energies-only helpers equal them bit for bit."""
    z = np.linspace(0.0, 6.0, 41)
    for kx, ky in ((0.0, 0.0), (0.35, 0.1)):
        e, F = solve_subbands(GAN, z, kx, ky, potential=0.06 * z,
                              n_states=6)
        H = assemble_hamiltonian(GAN, z, kx, ky, potential=0.06 * z)
        assert np.abs(e - _dense(H, z.size, 6)[0]).max() < 1e-12
        norms = (np.abs(F) ** 2).sum(axis=(1, 2)) * (z[1] - z[0])
        assert np.abs(norms - 1.0).max() < 1e-12
        assert np.array_equal(
            subband_energies(GAN, z, kx, ky, potential=0.06 * z,
                             n_states=6), e)
    pl, edge = layered_profile(z, [(2.0, ALN, -0.4), (4.0, GAN, 0.0)])
    for kx, ky in ((0.0, 0.0), (0.2, -0.3)):
        e, _ = solve_heterostructure(z, pl, edge, kx, ky, n_states=5)
        H = assemble_heterostructure(z, pl, edge, kx, ky)
        assert np.abs(e - _dense(H, z.size, 5)[0]).max() < 1e-12
        assert np.array_equal(
            heterostructure_energies(z, pl, edge, kx, ky, n_states=5), e)
    # the dispersion path equals solve_subbands exactly, as documented
    d = subband_dispersion(GAN, z, [0.4], theta=0.3, n_states=4)
    e, _ = solve_subbands(GAN, z, 0.4 * np.cos(0.3),
                          0.4 * np.sin(0.3), n_states=4)
    assert np.array_equal(d[0], e)


def test_kramers_partners_at_zero_momentum_have_equal_energies():
    """At k_t = 0 the two members of each Kramers pair now have
    bit-for-bit equal energies (time-reversal twins are diagonalized
    once). Energies only: no eigenvector basis enters."""
    z = np.linspace(0.0, 6.0, 41)
    e, _ = solve_subbands(GAN, z, potential=0.08 * z, n_states=8)
    assert np.array_equal(e[0::2], e[1::2])
    pl, edge = layered_profile(z, [(2.0, ALN, -0.4), (4.0, GAN, 0.0)])
    e, _ = solve_heterostructure(z, pl, edge, potential=0.03 * z,
                                 n_states=8)
    assert np.array_equal(e[0::2], e[1::2])


def test_per_layer_strain_assembly_and_block_solver():
    """The per-point strain path of assemble_heterostructure (untested
    before 0.14.0): one tensor, or the same tensor at every point, on a
    uniform stack equals the uniform assembly with that strain exactly;
    a stack whose two layers carry different biaxial strains splits into
    blocks, and its energies equal the dense reference to 1e-12 eV."""
    z = np.linspace(0.0, 5.0, 21)
    s = np.array([[0.01, 0.002, 0.001], [0.002, -0.003, 0.0005],
                  [0.001, 0.0005, -0.005]])
    pl = [GAN_D] * z.size
    for kx, ky in ((0.0, 0.0), (0.3, 0.2)):
        a = assemble_hamiltonian(GAN_D, z, kx, ky, potential=0.02 * z,
                                 strain=s)
        for sl in (s, np.array([s] * z.size)):
            b = assemble_heterostructure(z, pl, None, kx, ky,
                                         potential=0.02 * z,
                                         strain_list=sl)
            assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="strain_list"):
        assemble_heterostructure(z, pl, None, strain_list=np.zeros((2, 3, 3)))
    pl, edge = layered_profile(z, [(2.0, dataclasses.replace(
        GAN_D, A1=-4.0), -0.3), (3.0, GAN_D, 0.0)])
    strains = np.array([np.diag([0.01, 0.01, -0.006]) if p is pl[0]
                        else np.diag([-0.004, -0.004, 0.002]) for p in pl])
    for kx, ky in ((0.0, 0.0), (0.25, -0.15)):
        H = assemble_heterostructure(z, pl, edge, kx, ky,
                                     strain_list=strains)
        e = solve_heterostructure(z, pl, edge, kx, ky, n_states=6,
                                  strain_list=strains)[0]
        assert np.abs(e - _dense(H, z.size, 6)[0]).max() < 1e-12


# ---------------------------------------------------------------- 2

@pytest.fixture
def dense_solver(monkeypatch):
    """Route every solve through the v0.13.1 dense diagonalization."""
    def use():
        monkeypatch.setattr(solver_mod, "top_eigenpairs", _dense)
        monkeypatch.setattr(het_mod, "top_eigenpairs", _dense)
    return use


def test_self_consistent_loops_match_the_old_solver(dense_solver):
    """Hard-wall GaN loop (parabolic filling, 4.6e13 cm^-2 on a 6 nm,
    41-point grid) and finite-barrier loop: potential, energies,
    occupations and density agree with the old solver to 1e-9 (eV and
    nm^-2), with the same iteration count."""
    z = np.linspace(0.0, 6.0, 41)
    pl, edge = layered_profile(z, [(1.5, ALN, -0.5), (4.5, GAN, 0.0)])

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = solve_self_consistent(GAN, z, 0.46, n_states=8,
                                      mixing=0.5, tol=1e-6)
            b = solve_self_consistent_hetero(z, pl, edge, ps=0.2,
                                             eps_r=10.4, n_states=6,
                                             mixing=0.4, tol=1e-6,
                                             sheet_z=1.5)
        return a, b

    new = run()
    dense_solver()
    old = run()
    for r_new, r_old in zip(new, old):
        assert r_new.iterations == r_old.iterations
        assert r_new.n_states_sufficient == r_old.n_states_sufficient
        for f in ("potential", "energies", "occupations", "density"):
            assert np.abs(getattr(r_new, f) - getattr(r_old, f)).max() \
                < 1e-9, f
        assert np.allclose(r_new.masses, r_old.masses, rtol=1e-9)


def test_kgrid_filling_matches_the_old_solver(dense_solver):
    """fill_subbands_kgrid on a tilted GaN well, four directions: the
    occupations and Fermi level agree with the old solver to 1e-10."""
    z = np.linspace(0.0, 5.0, 31)
    v = 0.1 * z

    def run():
        return fill_subbands_kgrid(
            lambda kx, ky: solve_subbands(GAN, z, kx, ky, potential=v,
                                          n_states=4)[0],
            ps=0.2, n_states=4, kmax=1.5, nk=8, ntheta=4)

    occ, ef = run()
    dense_solver()
    occ_old, ef_old = run()
    assert np.abs(occ - occ_old).max() < 1e-10
    assert abs(ef - ef_old) < 1e-10


# ---------------------------------------------------------------- 3

def test_dos_is_nan_before_a_path_that_starts_above_zero():
    """Demo set (A = -2, constant DOS 1/(|A| 4 pi c) = 1.0443 nm^-2
    eV^-1) on a path from 0.2 nm^-1, where E = -3.048 meV. v0.13.1
    returned 0 at -2 ... +4 meV, where the band continues, and 2.596
    and 2.076 at -4 and -3 meV. Now those are NaN, and the inside of the
    path keeps the constant (to 1.5 %)."""
    A = -2.0
    kts = np.linspace(0.2, 2.0, 46)
    E = (A * C * kts ** 2)[:, None]
    eg = np.linspace(-0.12, 0.004, 125)
    d = dos_from_dispersion(kts, E, eg)[:, 0]
    exact = 1.0 / abs(A) / (4.0 * np.pi * C)
    assert abs(exact - 1.0443) < 1e-4
    assert np.all(np.isnan(d[eg > -4.5e-3]))
    inside = (eg < -8e-3) & (eg > E[-1, 0] + 2e-3)
    assert np.all(np.abs(d[inside] / exact - 1.0) < 0.015)
    # a path from 0 is unchanged: zero above the edge, no NaN there
    k0 = np.linspace(0.0, 2.0, 46)
    d0 = dos_from_dispersion(k0, (A * C * k0 ** 2)[:, None], eg)[:, 0]
    assert np.all(d0[eg > 1.5e-3] == 0.0)


# ---------------------------------------------------------------- 4

def test_thermal_filler_refuses_negative_masses():
    """v0.13.1 returned [0.234, -0.134] nm^-2 (negative holes) for
    edges [0, -0.01] eV, masses [0.5, -0.3] and ps = 0.1, at T = 0 and
    at 77 K."""
    for T in (0.0, 77.0):
        with pytest.raises(ValueError, match="must not be negative"):
            fill_subbands_thermal(np.array([0.0, -0.01]),
                                  np.array([0.5, -0.3]), 0.1, T)
    # an infinite mass is still allowed and simply excluded
    occ = fill_subbands_thermal(np.array([0.0, -0.01]),
                                np.array([0.5, np.inf]), 0.1, 0.0)
    assert occ[1] == 0.0 and abs(occ.sum() - 0.1) < 1e-15


def test_heterostructure_refuses_a_non_finite_band_edge():
    """v0.13.1 returned NaN energies (with a RuntimeWarning from the
    normalization) for a band edge with one NaN; now refused."""
    z = np.linspace(0.0, 6.0, 31)
    pl, edge = layered_profile(z, [(6.0, GAN, 0.0)])
    edge[3] = np.nan
    with pytest.raises(ValueError, match="band_edge must be finite"):
        solve_heterostructure(z, pl, edge, n_states=2)


# ---------------------------------------------------------------- 5

def test_cyclotron_mass_exact_for_parabolic_branches():
    """E = A c k^2 gives m_c = 1/|A| at every density and grid spacing
    (1e-12), and k_F = sqrt(4 pi n)."""
    for A, nk in ((-2.0, 11), (-0.7, 5), (-5.0, 40)):
        k = np.linspace(0.0, 2.0, nk)
        E = np.column_stack([A * C * k ** 2, A * C * k ** 2 - 0.01])
        n = np.array([0.03, 0.2])
        r = cyclotron_mass(k, E, n)
        assert np.abs(r["mass_ratio"] - 1.0 / abs(A)).max() < 1e-12
        assert np.allclose(r["k_fermi"], np.sqrt(4 * np.pi * n),
                           rtol=1e-15)
        s = cyclotron_mass(k, A * C * k ** 2, 0.1)
        assert isinstance(s["mass_ratio"], float)
        assert abs(s["mass_ratio"] - 1.0 / abs(A)) < 1e-12


def test_cyclotron_mass_second_order_on_a_nonparabolic_branch():
    """E = -(Eg/2)(sqrt(1 + 4 c k^2 / (m Eg)) - 1) (m = 0.2, Eg = 0.5
    eV; a test function, not a material) has the closed form
    m_c(k) = m sqrt(1 + 4 c k^2 / (m Eg)). Over 15 densities and 5
    spacings the error stays below 0.5 / (nk - 1)^2 (the observed
    constant is 0.28)."""
    m, Eg = 0.2, 0.5
    for n in np.linspace(0.02, 0.3, 15):
        for nk in (11, 21, 41, 81, 161):
            k = np.linspace(0.0, 2.2, nk)
            E = -(Eg / 2) * (np.sqrt(1 + 4 * C * k ** 2 / (m * Eg)) - 1)
            r = cyclotron_mass(k, E, n)
            exact = m * np.sqrt(1 + 4 * C * r["k_fermi"] ** 2 / (m * Eg))
            assert abs(r["mass_ratio"] - exact) * (nk - 1) ** 2 < 0.5


def test_cyclotron_mass_on_gan_against_a_direct_difference():
    """Top two levels of a 5 nm hard-wall GaN well: the mass from a
    path with 0.05 nm^-1 spacing agrees with c d(k^2)/dE from two
    direct solves at k_F -/+ 1e-3 nm^-1 to 1 %, and to 0.3 % with
    0.025 nm^-1 spacing."""
    z = np.linspace(0.0, 5.0, 41)
    direct = {}
    for nk, tol in ((21, 1e-2), (41, 3e-3)):
        kts = np.linspace(0.0, 1.0, nk)
        E = subband_dispersion(GAN, z, kts, n_states=4)[:, [0, 2]]
        for n in (0.02, 0.05, 0.07):
            r = cyclotron_mass(kts, E, n)
            for b, col in ((0, 0), (1, 2)):
                kf, d = r["k_fermi"][b], 1e-3
                if (n, b) not in direct:
                    ep = subband_energies(GAN, z, kx=kf + d,
                                          n_states=4)[col]
                    em = subband_energies(GAN, z, kx=kf - d,
                                          n_states=4)[col]
                    direct[n, b] = C * 4 * kf * d / abs(ep - em)
                assert abs(r["mass_ratio"][b] / direct[n, b] - 1) < tol


def test_cyclotron_mass_refusals():
    k = np.linspace(0.0, 1.0, 11)
    E = -0.1 * k ** 2
    with pytest.raises(ValueError, match="start at 0"):
        cyclotron_mass(k + 0.1, E, 0.01)
    with pytest.raises(ValueError, match="beyond the path"):
        cyclotron_mass(k, E, 0.2)          # k_F = 1.59 nm^-1
    # maximum at 0.3 nm^-1: E(0) lies below E(k_F = 0.354), so the
    # occupied region is a ring, not a disk
    with pytest.raises(ValueError, match="not the disk"):
        cyclotron_mass(k, -0.1 * (k - 0.3) ** 2, 0.01)
    # a branch that rises a little before it falls still fills a disk
    # (E(0) above E_F): accepted, m_c = c 2k / |dE/dk| to 1e-3
    kk = np.linspace(0.0, 1.0, 201)
    r = cyclotron_mass(kk, -0.1 * kk ** 2 + 0.002 * kk, 0.05)
    kf = r["k_fermi"]
    assert abs(r["mass_ratio"] / (C * 2 * kf / abs(-0.2 * kf + 0.002))
               - 1) < 1e-3
    with pytest.raises(ValueError, match="n_branch"):
        cyclotron_mass(k, E, 0.0)
    with pytest.raises(ValueError, match="at least three"):
        cyclotron_mass(k[:2], E[:2], 0.01)
    with pytest.raises(ValueError, match="one n_branch per branch"):
        cyclotron_mass(k, np.column_stack([E, E]), [0.01, 0.02, 0.03])


# ---------------------------------------------------------------- 6

def _dingle_data(B, A0, m, tau, T):
    return (A0 * lk_thermal_factor(T, B, m)
            * np.exp(-math.pi * sc.m_e * m / (sc.e * B * tau * 1e-12)))


def test_dingle_constants():
    """pi m0 / e against scipy.constants (relative 1e-6: the package's
    6-digit hbar^2/2m0 sets m0), hbar/(2 pi k_B) to 1e-15, and the
    Dingle temperature reproduces the damping of exp(-pi/(omega_c
    tau)) through the Lifshitz-Kosevich form
    exp(-2 pi^2 k_B T_D/(hbar omega_c)) (Shoenberg, chapter 8)."""
    assert abs(DINGLE_TESLA_PS / (math.pi * sc.m_e / sc.e * 1e12)
               - 1.0) < 1e-6
    assert abs(DINGLE_KELVIN_PS / (sc.hbar / (2 * math.pi * sc.k) * 1e12)
               - 1.0) < 1e-15
    m, tau = 0.43, 0.17
    B = np.linspace(1.0, 15.0, 8)
    td = DINGLE_KELVIN_PS / tau
    lk = np.exp(-LK_TESLA_PER_KELVIN * m * td / B)
    dingle = np.exp(-DINGLE_TESLA_PS * m / (B * tau))
    assert np.abs(lk / dingle - 1.0).max() < 1e-12


def test_dingle_fit_recovers_noise_free_lifetime():
    B = np.linspace(2.5, 9.0, 9)
    for T, tau in ((1.5, 0.12), (np.linspace(1.4, 1.6, 9), 0.4)):
        y = _dingle_data(B, 4.0, 0.5, tau, T)
        r = fit_sdh_dingle(B, y, T, 0.5)
        assert abs(r["tau_q_ps"] / tau - 1.0) < 2e-6   # scipy m_e vs m0
        assert abs(r["A0"] / 4.0 - 1.0) < 1e-5
        assert abs(r["dingle_temperature_K"]
                   - sc.hbar / (2 * math.pi * sc.k * tau * 1e-12)) \
            < 1e-5 * r["dingle_temperature_K"]


def test_dingle_fit_against_numpy_polyfit():
    """Weighted and unweighted: slope, intercept and their errors equal
    those of numpy.polyfit on ln(A / R_T) against 1/B (relative 1e-9),
    and tau_q = -pi m0 m / (e slope)."""
    rng = np.random.default_rng(7)
    B = np.linspace(3.0, 10.0, 8)
    m, T = 0.45, 2.0
    clean = _dingle_data(B, 3.7, m, 0.2, T)
    sig = 0.03 * clean
    y = clean + sig * rng.standard_normal(B.size)
    lny = np.log(y) - np.log(lk_thermal_factor(T, B, m))
    for weighted in (True, False):
        r = fit_sdh_dingle(B, y, T, m,
                           sigma_amplitudes=sig if weighted else None)
        w = y / sig if weighted else np.ones_like(y)
        p, cov = np.polyfit(1.0 / B, lny, 1, w=w, cov="unscaled")
        if not weighted:
            res = lny - np.polyval(p, 1.0 / B)
            cov = cov * (res ** 2).sum() / (B.size - 2)
        assert abs(r["slope_T"] / p[0] - 1) < 1e-9
        assert abs(math.log(r["A0"]) / p[1] - 1) < 1e-9
        assert abs(r["sigma_slope_T"] / math.sqrt(cov[0, 0]) - 1) < 1e-9
        assert abs(r["sigma_A0"] / r["A0"] / math.sqrt(cov[1, 1]) - 1) \
            < 1e-9
        assert abs(r["tau_q_ps"] / (-DINGLE_TESLA_PS * m / p[0]) - 1) \
            < 1e-9
        assert (r["chi2_red"] is None) != weighted


def test_dingle_fit_error_bars_match_the_scatter():
    """Monte Carlo: 400 data sets with 3 % Gaussian noise. The spread
    of tau_q equals the reported error to 12 %, the mean chi2_red is
    1 to 0.1, and the mean tau_q is the true one to 1 %."""
    rng = np.random.default_rng(2026)
    B = np.linspace(2.5, 9.0, 9)
    m, tau, T = 0.5, 0.12, 2.0
    clean = _dingle_data(B, 4.0, m, tau, T)
    sig = 0.03 * clean
    t, s, chi = [], [], []
    for _ in range(400):
        r = fit_sdh_dingle(B, clean + sig * rng.standard_normal(B.size),
                           T, m, sigma_amplitudes=sig)
        t.append(r["tau_q_ps"])
        s.append(r["sigma_tau_q_ps"])
        chi.append(r["chi2_red"])
    assert abs(np.std(t) / np.mean(s) - 1.0) < 0.12
    assert abs(np.mean(t) / tau - 1.0) < 0.01
    assert abs(np.mean(chi) - 1.0) < 0.1


def test_dingle_mass_error_propagation_against_a_finite_difference():
    """sigma_tau_q_mass = |d tau_q / d m| sigma_m, with the derivative
    checked against refits at m -/+ 1e-5 (relative 1e-5), and the total
    is the quadrature sum."""
    B = np.linspace(2.5, 9.0, 9)
    y = _dingle_data(B, 4.0, 0.5, 0.12, 3.0) * (1 + 0.01 * np.sin(B))
    sig = 0.01 * y
    r = fit_sdh_dingle(B, y, 3.0, 0.5, sigma_amplitudes=sig,
                       sigma_mass_ratio=0.02)
    h = 1e-5
    tp = fit_sdh_dingle(B, y, 3.0, 0.5 + h, sigma_amplitudes=sig)
    tm = fit_sdh_dingle(B, y, 3.0, 0.5 - h, sigma_amplitudes=sig)
    deriv = (tp["tau_q_ps"] - tm["tau_q_ps"]) / (2 * h)
    assert abs(r["sigma_tau_q_mass_ps"] / (abs(deriv) * 0.02) - 1) < 1e-5
    assert abs(r["sigma_tau_q_ps"] - math.hypot(
        r["sigma_tau_q_stat_ps"], r["sigma_tau_q_mass_ps"])) < 1e-15
    assert r["sigma_tau_q_mass_ps"] > 0.0


def test_dingle_fit_refusals():
    B = np.linspace(3.0, 9.0, 6)
    y = _dingle_data(B, 4.0, 0.5, 0.15, 2.0)
    with pytest.raises(ValueError, match="does not fall with 1/B"):
        # grows towards low field after the R_T correction
        fit_sdh_dingle(B, lk_thermal_factor(2.0, B, 0.5)
                       * np.exp(1.0 / B), 2.0, 0.5)
    with pytest.raises(ValueError, match="3 or more distinct"):
        fit_sdh_dingle([3.0, 3.0, 4.0], [1.0, 1.0, 1.2], 2.0, 0.5)
    with pytest.raises(ValueError, match="fields_T"):
        fit_sdh_dingle(-B, y, 2.0, 0.5)
    with pytest.raises(ValueError, match="amplitudes must be"):
        fit_sdh_dingle(B, -y, 2.0, 0.5)
    with pytest.raises(ValueError, match="mass_ratio"):
        fit_sdh_dingle(B, y, 2.0, 0.0)
    with pytest.raises(ValueError, match="temperature_K"):
        fit_sdh_dingle(B, y, -1.0, 0.5)
    with pytest.raises(ValueError, match="sigma_amplitudes"):
        fit_sdh_dingle(B, y, 2.0, 0.5, sigma_amplitudes=y[:3])
    with pytest.raises(ValueError, match="sigma_mass_ratio"):
        fit_sdh_dingle(B, y, 2.0, 0.5, sigma_mass_ratio=-0.1)
    # nearly undamped over a narrow range, with 5 % scatter
    Bn = np.array([8.0, 8.5, 9.0, 9.5])
    yn = _dingle_data(Bn, 4.0, 0.5, 5.0, 0.3) * np.array(
        [1.0, 1.05, 0.95, 1.0])
    with pytest.raises(ValueError, match="do not determine"):
        fit_sdh_dingle(Bn, yn, 0.3, 0.5)


# ---------------------------------------------------------------- 7

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _run(name):
    out = subprocess.run([sys.executable, str(EXAMPLES / name)],
                         capture_output=True, text=True, check=True,
                         timeout=600)
    return out.stdout


def _num(pattern, text):
    return [float(x) for x in re.search(pattern, text).groups()]


def test_tutorial_sdh_analysis():
    """Recovers the synthetic truth (m* = 0.5, tau_q = 0.15 ps) within
    two error bars; the density is (h/e)^-1 * 2 * 150 T exactly; the
    printed output is the one in the script's docstring."""
    out = _run("tutorial_sdh_analysis.py")
    n, sn = _num(r"n = ([\d.e+]+) \+/- ([\d.e+]+) cm", out)
    assert abs(n / (2 * 150.0 * sc.e / sc.h * 1e-4) - 1) < 1e-3
    assert abs(sn / n - 2.0 / 150.0) < 1e-3
    m, sm = _num(r"m\* = ([\d.]+) \+/- ([\d.]+) m0", out)
    assert abs(m - 0.5) < 2 * sm + 1e-3
    tau, st = _num(r"tau_q = ([\d.]+) \+/- ([\d.]+) ps", out)
    assert abs(tau - 0.15) < 2 * st + 1e-3
    td, _ = _num(r"T_D = ([\d.]+) \+/- ([\d.]+) K", out)
    assert abs(td - DINGLE_KELVIN_PS / tau) < 0.1
    doc = (EXAMPLES / "tutorial_sdh_analysis.py").read_text()
    assert "\n".join("    " + l if l else l
                     for l in out.strip().split("\n")) in doc


def test_tutorial_sdh_prediction():
    """Four occupied branches holding 4.6e13 cm^-2 in total; each
    printed frequency is (h/e) times the printed density (Onsager), and
    the two heavy branches are much heavier than the two light ones."""
    out = _run("tutorial_sdh_prediction.py")
    assert "converged True" in out
    tot, = _num(r"total ([\d.e+]+) cm\^-2", out)
    assert abs(tot / 4.6e13 - 1) < 1e-4
    rows = re.findall(r"^\s+(\d)\s+([\d.]+) \+/- [\d.]+\s+([\d.]+) "
                      r"\+/- [\d.]+\s+([\d.]+) \+/- [\d.]+$", out, re.M)
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3]
    for _, n13, f, _ in rows:
        f_onsager = float(n13) * 1e13 * 1e4 * sc.h / sc.e
        assert abs(float(f) - f_onsager) < 0.1
    masses = [float(r[3]) for r in rows]
    assert min(masses[:2]) > 3 * max(masses[2:])
    assert abs(sum(float(r[1]) for r in rows) - 4.6) < 5e-4
