"""v0.6 roadmap closures against exact statements: full k-grid filling
agrees with the closed-form parabolic filling on an exactly parabolic
model, the finite-barrier self-consistent loop reproduces the
hard-wall loop exactly on a uniform stack, spin splittings vanish where
time reversal and symmetry demand and appear under an asymmetric
potential, and the transport groundwork (group velocity, 2D DOS) hits
its parabolic closed forms."""
import numpy as np
import pytest

from kpenvelope import (HBAR2_OVER_2M0 as C, demo_single_band,
                        dos_from_dispersion, fill_subbands_kgrid,
                        gan_rinke2008, group_velocity,
                        solve_self_consistent, solve_self_consistent_hetero,
                        solve_subbands, spin_splitting, splitting_vs_k,
                        subband_dispersion)
from kpenvelope.selfconsistent import _fill_subbands

Z = np.linspace(0.0, 8.0, 61)


def test_kgrid_filling_matches_parabolic_filling_on_a_parabolic_model():
    """The demo set is exactly parabolic with mass 1/|A| m0, so the
    non-parabolic k-grid filler must agree with the closed-form
    parabolic filler to grid resolution, and conserve charge exactly."""
    p = demo_single_band(A=-2.0)
    ps = 0.05
    e0, _ = solve_subbands(p, Z, n_states=4)
    occ_par = _fill_subbands(e0, np.full(4, 0.5), ps)
    occ_kg, ef = fill_subbands_kgrid(
        lambda kx, ky: solve_subbands(p, Z, kx=kx, ky=ky, n_states=4)[0],
        ps, 4, kmax=1.4, nk=41, ntheta=1)
    assert np.abs(occ_kg - occ_par).max() < 4e-3
    assert abs(occ_kg.sum() - ps) < 1e-12


def test_kgrid_filling_refuses_an_undersized_kmax():
    p = demo_single_band(A=-2.0)
    with pytest.raises(ValueError):
        fill_subbands_kgrid(
            lambda kx, ky: solve_subbands(p, Z, kx=kx, ky=ky,
                                          n_states=4)[0],
            0.05, 4, kmax=0.2, nk=8, ntheta=1)


def test_hetero_self_consistency_reproduces_hard_wall_on_uniform_stack():
    """A uniform stack with no offset is the same Hamiltonian as the
    hard-wall assembly, so the two self-consistent loops must produce
    identical iterates -- exactly, not approximately."""
    p = demo_single_band(A=-2.0)
    z = np.linspace(0.0, 10.0, 51)
    r_hw = solve_self_consistent(p, z, ps=0.05, n_states=3, tol=1e-6)
    r_ht = solve_self_consistent_hetero(z, [p] * z.size, None, ps=0.05,
                                        eps_r=p.eps_r, n_states=3,
                                        tol=1e-6)
    assert np.abs(r_hw.energies - r_ht.energies).max() == 0.0
    assert np.abs(r_hw.occupations - r_ht.occupations).max() == 0.0
    assert abs(r_ht.occupations.sum() - 0.05) == 0.0


def test_spin_splitting_vanishes_where_symmetry_demands():
    """Kramers at k = 0 (any potential) and at every k in a symmetric
    well; Rashba-type splitting appears only under an asymmetric
    potential at finite k."""
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 41)
    e_sym = solve_subbands(p, z, kx=0.3, n_states=4)[0]
    assert np.abs(spin_splitting(e_sym)).max() < 1e-9
    tilt = 0.05 * (z - z[0]) / (z[-1] - z[0])
    e_k0 = solve_subbands(p, z, kx=0.0, potential=tilt, n_states=4)[0]
    assert np.abs(spin_splitting(e_k0)).max() < 1e-9
    e_k = solve_subbands(p, z, kx=0.3, potential=tilt, n_states=4)[0]
    assert np.abs(spin_splitting(e_k)).min() > 1e-5


def test_splitting_vs_k_grows_from_exact_zero():
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 41)
    tilt = 0.05 * (z - z[0]) / (z[-1] - z[0])
    kts = np.array([0.0, 0.15, 0.3])
    s = splitting_vs_k(p, z, kts, potential=tilt, n_pairs=1)
    assert abs(s[0, 0]) < 1e-9
    assert s[2, 0] > s[1, 0] > 1e-6


def test_group_velocity_and_dos_parabolic_closed_forms():
    """E = E0 + c A k^2 exactly, so v = 2 c A k exactly (to machine
    precision with second-order edges) and the 2D DOS is the constant
    (1/|A|) / (4 pi c) per spin-resolved branch."""
    A = -2.0
    p = demo_single_band(A=A)
    kts = np.linspace(0.0, 0.8, 41)
    disp = subband_dispersion(p, Z, kts, n_states=2)
    v = group_velocity(kts, disp)
    assert np.abs(v[:, 0] - 2.0 * C * A * kts).max() < 1e-10
    eg = np.linspace(disp[:, 0].min() + 0.005, disp[0, 0] - 0.005, 30)
    rho = dos_from_dispersion(kts, disp, eg)
    expect = (1.0 / abs(A)) / (4.0 * np.pi * C)
    assert np.abs(rho[:, 0] - expect).max() / expect < 0.01


def test_dos_refuses_non_monotone_dispersions():
    kts = np.linspace(0.0, 1.0, 21)
    fake = np.stack([np.cos(6 * kts)], axis=1)   # turns up past 3k = pi
    with pytest.raises(ValueError):
        dos_from_dispersion(kts, fake, np.linspace(-0.5, 0.5, 11))
