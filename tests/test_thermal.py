"""v0.8 experimental-conditions anchors: finite-temperature subband
filling held to its closed form by independent numerical quadrature
and to the T = 0 filler in the cold limit, the k-grid filler at the
same temperature, the convergence flag, and the exact lab-unit
conversions -- every claim a checked identity, not a stored number."""
import numpy as np
import pytest

from kpenvelope import (KB_EV_PER_K, demo_single_band,
                        fill_subbands_kgrid, fill_subbands_thermal,
                        gan_rinke2008, sheet_density_from_cm2,
                        sheet_density_to_cm2, solve_self_consistent,
                        solve_subbands)
from kpenvelope.selfconsistent import _fill_subbands

Z = np.linspace(0.0, 8.0, 61)


def test_boltzmann_constant_from_exact_si_definitions():
    """k_B/e from the exact 2019 SI defining constants; the quoted
    CODATA value of k_B in eV/K is the independent cross-check."""
    assert abs(KB_EV_PER_K - 8.617333262e-5) < 1e-14


def test_closed_form_matches_direct_quadrature():
    """n = dos kT ln(1 + exp((E - EF)/kT)) against numerical
    integration of the Fermi-Dirac hole factor over the constant 2D
    DOS -- two independent code paths."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    dos, E, kT = 0.7, -0.05, KB_EV_PER_K * 300.0
    for ef in (-0.10, -0.05, 0.01):
        closed = dos * kT * np.logaddexp(0.0, (E - ef) / kT)
        val, err = scipy_integrate.quad(
            lambda e: dos / (1.0 + np.exp((ef - e) / kT)),
            E - 60.0 * kT, E, limit=200)
        assert abs(closed - val) < 1e-10


def test_cold_limit_recovers_the_t0_filler_and_t0_routes_exactly():
    energies = np.array([-0.02, -0.05, -0.09, -0.20])
    masses = np.array([1.9, 0.4, 1.9, np.inf])
    ps = 0.05
    occ0 = _fill_subbands(energies, masses, ps)
    # temperature_K = 0 is literally the same code path
    assert np.array_equal(fill_subbands_thermal(energies, masses, ps, 0.0),
                          occ0)
    # and the T -> 0 limit converges to it
    occ_cold = fill_subbands_thermal(energies, masses, ps, 0.05)
    assert np.abs(occ_cold - occ0).max() < 1e-6
    # neutrality is exact at any temperature
    for T in (4.2, 77.0, 300.0):
        occ = fill_subbands_thermal(energies, masses, ps, T)
        assert abs(occ.sum() - ps) < 1e-15
        assert np.all(occ >= 0.0)
    with pytest.raises(ValueError):
        fill_subbands_thermal(energies, masses, ps, -1.0)


def test_temperature_spreads_holes_downward():
    """Warming moves holes from the top subband into lower ones -- the
    physics the experimentalist's temperature knob controls."""
    energies = np.array([-0.02, -0.05, -0.09])
    masses = np.array([1.9, 1.9, 1.9])
    ps = 0.02
    occ_cold = fill_subbands_thermal(energies, masses, ps, 4.2)
    occ_warm = fill_subbands_thermal(energies, masses, ps, 300.0)
    assert occ_warm[0] < occ_cold[0]
    assert occ_warm[-1] > occ_cold[-1]


def test_kgrid_thermal_matches_parabolic_closed_form():
    """On the exactly parabolic demo set the finite-T k-grid filler
    must agree with the closed-form thermal parabolic filler."""
    p = demo_single_band(A=-2.0)
    ps = 0.05
    T = 150.0
    e0, _ = solve_subbands(p, Z, n_states=4)
    occ_par = fill_subbands_thermal(e0, np.full(4, 0.5), ps, T)
    occ_kg, ef = fill_subbands_kgrid(
        lambda kx, ky: solve_subbands(p, Z, kx=kx, ky=ky, n_states=4)[0],
        ps, 4, kmax=2.2, nk=61, ntheta=1, temperature_K=T)
    assert np.abs(occ_kg - occ_par).max() < 4e-3
    assert abs(occ_kg.sum() - ps) < 1e-12
    # the Fermi tail reaching kmax is refused, as at T = 0
    with pytest.raises(ValueError, match="edge of the k-grid"):
        fill_subbands_kgrid(
            lambda kx, ky: solve_subbands(p, Z, kx=kx, ky=ky,
                                          n_states=4)[0],
            ps, 4, kmax=0.6, nk=13, ntheta=1, temperature_K=T)


def test_self_consistent_loop_at_temperature_and_converged_flag():
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 49)
    res300 = solve_self_consistent(p, z, ps=0.2, n_states=4,
                                   temperature_K=300.0)
    assert res300.converged
    assert abs(res300.occupations.sum() - 0.2) < 1e-12
    # T = 0 default reproduces the historical path bit for bit
    res0a = solve_self_consistent(p, z, ps=0.2, n_states=4)
    res0b = solve_self_consistent(p, z, ps=0.2, n_states=4,
                                  temperature_K=0.0)
    assert np.array_equal(res0a.occupations, res0b.occupations)
    assert np.array_equal(res0a.potential, res0b.potential)
    # warming spreads the gas: more subbands carry occupation
    n_occ_0 = int((res0a.occupations > 1e-6).sum())
    n_occ_300 = int((res300.occupations > 1e-6).sum())
    assert n_occ_300 >= n_occ_0
    # an iteration-starved run reports its failure instead of hiding it
    starved = solve_self_consistent(p, z, ps=0.2, n_states=4,
                                    max_iter=1, tol=1e-12)
    assert not starved.converged


def test_sheet_density_conversions_are_exact():
    assert sheet_density_from_cm2(4.6e13) == 0.46
    assert sheet_density_to_cm2(0.46) == 4.6e13
    x = 1.2345e13
    assert sheet_density_to_cm2(sheet_density_from_cm2(x)) == x
    arr = np.array([1e12, 1e13, 1e14])
    assert np.array_equal(sheet_density_from_cm2(arr),
                          arr * 1e-14)
