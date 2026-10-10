"""v0.13 anchors.

Fixes (each test failed on v0.12.0; the docstrings give the old
numbers):

1. `layered_profile` assigned an interface grid point to the wrong
   layer whenever the running sum of thicknesses rounded up past it.
2. The depolarization geometry integral depended on the arbitrary
   global phase of an eigenvector.
3. The self-consistent loops (and `fill_subbands_kgrid`) silently left
   out the holes of occupied states beyond ``n_states``.
4. `fill_subbands_thermal` returned all-zero occupations (total 0, not
   ps) when no subband had a finite mass, and for a negative ps.
5. `dos_from_dispersion` returned a DOS of exactly 0 below the end of
   the momentum path.

New lab-data features, each against an independent calculation:

6. Shubnikov-de Haas frequency <-> density (Onsager relation), against
   the Fermi wavevector of exactly parabolic subbands and
   scipy.constants.
7. Lifshitz-Kosevich mass fit, against scipy.optimize.curve_fit and a
   Monte Carlo check of its error bars.
8. Wavenumber <-> energy, against scipy.constants.
"""
import dataclasses
import math
import pathlib
import re
import warnings
from fractions import Fraction

import numpy as np
import pytest
import scipy.constants as sc
from scipy.optimize import curve_fit

import kpenvelope
from kpenvelope import (HBAR2_OVER_2M0 as C, aln_rinke2008,
                        demo_single_band, depolarization_shift,
                        dos_from_dispersion, energy_from_wavenumber,
                        energy_to_wavenumber, fill_subbands_kgrid,
                        fill_subbands_thermal, fit_sdh_mass,
                        gan_rinke2008, layered_profile, lk_thermal_factor,
                        overlap_geometry_integral, sdh_frequency,
                        sheet_density_from_sdh, solve_heterostructure,
                        solve_self_consistent, solve_subbands,
                        subband_dispersion)
from kpenvelope.lab import H_OVER_E_T_NM2, LK_TESLA_PER_KELVIN

GAN = gan_rinke2008()
ALN = aln_rinke2008()


# ---------------------------------------------------------------- 1

def _exact_layer_index(i, dz, thicknesses):
    """Layer of grid point i (at i * dz) by exact rational arithmetic:
    a point on an interface belongs to the layer on its right."""
    zi = i * dz
    edge = Fraction(0)
    for k, t in enumerate(thicknesses[:-1]):
        edge += t
        if zi < edge:
            return k
    return len(thicknesses) - 1


def test_layered_profile_matches_exact_assignment():
    """Over a sweep of decimal stacks, every grid point lands in the
    layer exact arithmetic puts it in. v0.12.0 got 10 of the points of
    this sweep wrong (interface points sent to the left layer)."""
    a_list = ["1.9", "2.1", "2.3", "2.7", "3.3", "4.1", "4.7", "0.7"]
    w_list = ["3.4", "5.4", "2.6", "4.4"]
    wrong = 0
    for sa in a_list:
        for sw in w_list:
            ts = [Fraction(sa), Fraction(sw), Fraction("3.3")]
            dz = Fraction(1, 10)
            n = int(sum(ts) / dz) + 1
            z = np.linspace(0.0, float(sum(ts)), n)
            tags = [demo_single_band(-1.0), demo_single_band(-2.0),
                    demo_single_band(-3.0)]
            pl, edge = layered_profile(
                z, [(float(t), p, float(k)) for k, (t, p)
                    in enumerate(zip(ts, tags))])
            for i in range(n):
                k = _exact_layer_index(i, dz, ts)
                wrong += (pl[i] is not tags[k]) + (edge[i] != k)
    assert wrong == 0


def test_translated_well_has_the_same_levels():
    """A 3.4 nm GaN well between AlN barriers, moved along a 9.4 nm grid
    (dz = 0.1 nm): the well always has 34 points and the top level does
    not move. v0.12.0 gave the well 33 points at a = 2.7 nm (2.7 + 3.4
    rounds to 6.1000000000000005) and moved the top level from 1.978 to
    1.234 meV. Offset illustrative, not a cited GaN/AlN value."""
    tops = []
    for a in (2.0, 2.7):
        b = 6.0 - a
        z = np.linspace(0.0, a + 3.4 + b, 95)
        params, edge = layered_profile(
            z, [(a, ALN, -0.5), (3.4, GAN, 0.0), (b, ALN, -0.5)])
        assert sum(p is GAN for p in params) == 34
        e, _ = solve_heterostructure(z, params, edge, n_states=1)
        tops.append(e[0])
    assert abs(tops[1] - tops[0]) < 1e-6
    assert abs(tops[0] - 1.978e-3) < 1e-6


def test_layered_profile_refuses_bad_layers():
    z = np.linspace(0.0, 4.0, 41)
    with pytest.raises(ValueError, match="not negative"):
        layered_profile(z, [(5.0, GAN, 0.0), (-1.0, ALN, -0.5)])
    with pytest.raises(ValueError, match="at least one layer"):
        layered_profile(z, [])


# ---------------------------------------------------------------- 2

def _textbook_S(z, f, g):
    """S = int dz [int^z f g]^2 for real one-component functions."""
    dz = z[1] - z[0]
    return float(((np.cumsum(f * g) * dz) ** 2).sum() * dz)


def test_geometry_integral_does_not_depend_on_eigenvector_phase():
    """README example 5 states. Multiplying the second state by i (an
    equally valid eigenvector) gave S = 0 and alpha = 0 in v0.12.0,
    against S = 0.4503 nm and alpha = 2.311 here. The real case is the textbook formula
    bit for bit."""
    L = 8.0
    z = np.linspace(0.0, L, 801)
    f1 = np.sqrt(2 / L) * np.sin(np.pi * z / L)
    f2 = np.sqrt(2 / L) * np.sin(2 * np.pi * z / L)
    S_real = overlap_geometry_integral(z, f1[None], f2[None])
    assert S_real == _textbook_S(z, f1, f2)
    for phase in (1j, np.exp(0.7j), -1.0):
        S = overlap_geometry_integral(z, f1[None], phase * f2[None])
        assert abs(S / S_real - 1.0) < 1e-12
    e1 = -C * (np.pi / L) ** 2 / 0.5
    e2 = -C * (2 * np.pi / L) ** 2 / 0.5
    a = depolarization_shift(z, f1[None], f2[None], e1, e2, 0.05, 10.0)
    b = depolarization_shift(z, f1[None], 1j * f2[None], e1, e2, 0.05,
                             10.0)
    assert abs(b["alpha"] / a["alpha"] - 1.0) < 1e-12
    assert abs(S_real - 0.4503) < 1e-4 and abs(a["alpha"] - 2.311) < 1e-3


def test_geometry_integral_phase_invariance_on_solver_states():
    """HH1 -> HH2 of a tilted GaN well at k = 0 (the solver's states are
    real there). Each level is a Kramers pair, and which member of a
    pair the eigensolver returns first can differ between computers
    (one CI machine returned the partner with S = 0). So the test uses
    S summed over the HH2 pair, which does not depend on that choice:
    0.10378 nm, the same for any unitary mix of either pair. Each term
    equals the textbook real-function form evaluated component by
    component, and the sum is unchanged by a phase on either state.
    v0.12.0 gave 0 with i on the HH2 states."""
    z = np.linspace(0.0, 6.0, 61)
    e, F = solve_subbands(GAN, z, potential=0.08 * z, n_states=6)
    assert np.abs(F.imag).max() == 0.0
    assert abs(e[0] - e[1]) < 1e-9 and abs(e[4] - e[5]) < 1e-9
    dz = z[1] - z[0]

    def textbook(fi, fj):
        lam = (fi * fj).sum(axis=0)
        return float(((np.cumsum(lam) * dz) ** 2).sum() * dz)

    ref = textbook(F[0].real, F[4].real) + textbook(F[0].real, F[5].real)
    assert abs(ref - 0.10378) < 1e-5
    for pi_, pj in ((1.0, 1.0), (1.0, 1j), (np.exp(2.1j), np.exp(-0.4j))):
        S = sum(overlap_geometry_integral(z, pi_ * F[0], pj * F[j])
                for j in (4, 5))
        assert abs(S / ref - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(3):
        U = np.linalg.qr(rng.normal(size=(2, 2))
                         + 1j * rng.normal(size=(2, 2)))[0]
        V = np.linalg.qr(rng.normal(size=(2, 2))
                         + 1j * rng.normal(size=(2, 2)))[0]
        A = np.tensordot(U, F[0:2], 1)
        B = np.tensordot(V, F[4:6], 1)
        S = sum(overlap_geometry_integral(z, A[0], B[j]) for j in (0, 1))
        assert abs(S / ref - 1.0) < 1e-10


# ---------------------------------------------------------------- 3

# demo set with a synthetic, deliberately large permittivity (a test
# number, not a material constant): a weak field keeps the hard-wall
# levels close, so 0.6 nm^-2 fills two of them
DEMO_WEAK = dataclasses.replace(demo_single_band(A=-2.0), eps_r=1000.0)


def _demo_loop(n_states, **kw):
    z = np.linspace(0.0, 6.0, 30)
    return solve_self_consistent(DEMO_WEAK, z, ps=0.6, n_states=n_states,
                                 mixing=0.6, tol=1e-9, max_iter=200, **kw)


def test_loop_warns_when_occupied_states_were_not_computed():
    """Demo well (levels six-fold degenerate, mass 0.5 m0) at 0.6 nm^-2,
    where the second level is occupied. With n_states = 6 v0.12.0 put
    0.1 nm^-2 into each first-level state without notice (0.0779 with
    12 states, and 0.0221 in each second-level state); now the loop
    warns and flags it. With 12 states the closed-form parabolic filling
    of the returned edges (dos = 0.5 / (4 pi c) per state) reproduces
    the occupations, and the Fermi level it implies lies below the
    second level and above the third, which is therefore rightly
    empty."""
    with pytest.warns(RuntimeWarning, match="Increase n_states"):
        short = _demo_loop(6)
    assert not short.n_states_sufficient
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = _demo_loop(12)
    assert full.n_states_sufficient and full.converged
    e = full.energies
    dos = 0.5 / (4.0 * np.pi * C)
    # closed form: 6 dos (E1 - EF) + 6 dos (E2 - EF) = ps
    ef = (6 * dos * (e[0] + e[6]) - 0.6) / (12 * dos)
    assert ef < e[6]
    assert abs(full.occupations[0] - 0.0779) < 1e-4
    expect = dos * np.clip(e - ef, 0.0, None)
    assert np.abs(full.occupations - expect).max() < 1e-10
    assert np.abs(short.occupations - 0.1).max() < 1e-9
    third = solve_subbands(DEMO_WEAK, full.z,
                           potential=full.potential, n_states=13)[0][12]
    assert third < ef
    # an odd n_states cuts a Kramers pair (here a degenerate level)
    with pytest.warns(RuntimeWarning):
        _demo_loop(11)


def test_loop_warns_with_kgrid_filling_too():
    with pytest.warns(RuntimeWarning, match="Increase n_states"):
        r = _demo_loop(6, filling="kgrid", kmax=1.6, nk=4)
    assert not r.n_states_sufficient


def test_gan_hard_wall_example_needs_more_than_four_states():
    """The GaN gas of README example 2 (0.46 nm^-2, hard walls), on a
    coarser grid (49 points): with 4 states the parabolic Fermi level
    lies below the uncomputed third pair (8 states are not enough
    either on this grid); with 12 it does not."""
    z = np.linspace(0.0, 6.0, 49)
    kw = dict(mixing=0.5, tol=2e-5)
    with pytest.warns(RuntimeWarning):
        r4 = solve_self_consistent(GAN, z, 0.46, n_states=4, **kw)
    assert not r4.n_states_sufficient
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r12 = solve_self_consistent(GAN, z, 0.46, n_states=12, **kw)
    assert r12.n_states_sufficient
    assert r12.occupations[4:].sum() > 0.05    # third to fifth pairs


def test_fill_kgrid_warns_when_given_more_energies_than_it_fills():
    p = demo_single_band(A=-2.0)
    z = np.linspace(0.0, 8.0, 41)

    def at_k(kx, ky):
        return solve_subbands(p, z, kx=kx, ky=ky, n_states=12)[0]

    with pytest.warns(RuntimeWarning, match="state 6"):
        fill_subbands_kgrid(at_k, 0.3, 6, kmax=2.6, nk=5, ntheta=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        occ, _ = fill_subbands_kgrid(at_k, 0.3, 12, kmax=2.6, nk=5,
                                     ntheta=1)
    assert abs(occ.sum() - 0.3) < 1e-14


# ---------------------------------------------------------------- 4

def test_thermal_filler_refuses_what_it_cannot_fill():
    """v0.12.0 returned [0, 0] for the infinite masses (a total of 0
    instead of 0.1 nm^-2), [-0.1, -0] for ps = -0.1, and [500.5, 499.5]
    for 1e3 nm^-2, whose Fermi level the bisection could not reach
    (the rescaling to the total hid it)."""
    e = np.array([0.0, -0.01])
    with pytest.raises(ValueError, match="finite positive mass"):
        fill_subbands_thermal(e, np.array([np.inf, np.inf]), 0.1, 0.0)
    with pytest.raises(ValueError, match="finite positive mass"):
        fill_subbands_thermal(e, np.array([np.inf, np.inf]), 0.1, 77.0)
    with pytest.raises(ValueError, match="ps must be"):
        fill_subbands_thermal(e, np.array([0.5, 0.5]), -0.1, 0.0)
    with pytest.raises(ValueError, match="one mass per"):
        fill_subbands_thermal(e, np.array([0.5]), 0.1, 0.0)
    with pytest.raises(ValueError, match="unphysically large"):
        fill_subbands_thermal(e, np.array([0.5, 0.5]), 1e3, 0.0)
    # zero density with no finite mass is a valid empty filling
    assert np.all(fill_subbands_thermal(
        e, np.array([np.inf, np.inf]), 0.0, 0.0) == 0.0)


def test_loop_input_checks():
    z = np.linspace(0.0, 6.0, 25)
    for bad in (dict(ps=-0.1), dict(ps=float("nan")),
                dict(ps=0.1, mixing=0.0), dict(ps=0.1, tol=0.0)):
        with pytest.raises(ValueError):
            solve_self_consistent(GAN, z, **bad)


# ---------------------------------------------------------------- 5

def test_dos_is_nan_beyond_the_path():
    """Demo set, path to 0.8 nm^-1. Inside the path the DOS is the
    closed-form constant (1/|A|)/(4 pi c); below E(0.8) the path says
    nothing. v0.12.0 returned 0.0 there, a fake gap."""
    A = -2.0
    p = demo_single_band(A=A)
    z = np.linspace(0.0, 6.0, 41)
    kts = np.linspace(0.0, 0.8, 41)
    disp = subband_dispersion(p, z, kts, n_states=1)
    e_end = disp[-1, 0]
    eg = np.linspace(e_end - 0.03, disp[0, 0] - 0.005, 60)
    rho = dos_from_dispersion(kts, disp, eg)[:, 0]
    inside = eg > e_end + 2 * (eg[1] - eg[0])
    expect = (1.0 / abs(A)) / (4.0 * np.pi * C)
    assert np.abs(rho[inside] - expect).max() / expect < 0.01
    assert np.all(np.isnan(rho[eg < e_end]))


# ---------------------------------------------------------------- 6

def test_sdh_constants_against_scipy():
    assert abs(H_OVER_E_T_NM2 / (sc.h / sc.e * 1e18) - 1.0) < 1e-15
    lk = 2.0 * np.pi ** 2 * sc.k * sc.m_e / (sc.hbar * sc.e)
    # differs only through the 6-digit hbar^2/2m0 of the package
    assert abs(LK_TESLA_PER_KELVIN / lk - 1.0) < 1e-6
    assert abs(energy_from_wavenumber(1000.0)
               - sc.h * sc.c / sc.e * 1e5) < 1e-17
    nu = np.array([500.0, 1234.5, 3000.0])
    assert np.allclose(energy_to_wavenumber(energy_from_wavenumber(nu)),
                       nu, rtol=1e-15, atol=0.0)


def test_onsager_frequency_against_the_fermi_wavevector():
    """Exactly parabolic demo subbands (two six-fold levels, mass 0.5)
    filled by the k-grid filler: each state's SdH frequency must equal
    hbar pi kF^2 / (2 pi e) (Onsager), with kF from the closed-form
    Fermi level, computed here in SI units with scipy.constants."""
    p = demo_single_band(A=-2.0)
    z = np.linspace(0.0, 8.0, 41)
    e0, _ = solve_subbands(p, z, n_states=12)

    def at_k(kx, ky):
        return solve_subbands(p, z, kx=kx, ky=ky, n_states=12)[0]

    occ, ef = fill_subbands_kgrid(at_k, 0.3, 12, kmax=2.6, nk=5,
                                  ntheta=1)
    kf2 = np.clip(e0 - ef, 0.0, None) * 0.5 / C          # nm^-2
    f_onsager = sc.hbar * np.pi * kf2 * 1e18 / (2 * np.pi * sc.e)
    assert f_onsager[6] > 10.0                     # second level filled
    f = sdh_frequency(occ, branches=1)
    assert np.abs(f / f_onsager - 1.0).max() < 1e-9
    # a pair shown as one spin-degenerate frequency, and the inverse
    pair = occ[0] + occ[1]
    F2 = sdh_frequency(pair, branches=2)
    assert abs(F2 / f_onsager[0] - 1.0) < 1e-9
    inv = sheet_density_from_sdh(F2, branches=2, sigma_frequency_T=0.5)
    assert abs(inv["n_sheet"] / pair - 1.0) < 1e-14
    assert abs(inv["sigma_n_sheet"] - 2 * 0.5 / H_OVER_E_T_NM2) < 1e-18
    assert abs(inv["n_sheet_cm2"] - inv["n_sheet"] * 1e14) < 1e-3


def test_sdh_refusals():
    with pytest.raises(TypeError):
        sheet_density_from_sdh(100.0)               # branches required
    for b in (0, 1.5, True, None):
        with pytest.raises(ValueError, match="branches"):
            sheet_density_from_sdh(100.0, branches=b)
    with pytest.raises(ValueError, match="frequency_T"):
        sheet_density_from_sdh(-5.0, branches=2)
    with pytest.raises(ValueError, match="n_sheet"):
        sdh_frequency(-0.1, branches=1)


# ---------------------------------------------------------------- 7

T_DATA = np.array([1.6, 3.0, 5.0, 8.0, 12.0, 16.0])


def _lk_scipy(T, A0, m, B):
    X = LK_TESLA_PER_KELVIN * m * T / B
    return A0 * X / np.sinh(X)


def test_lk_thermal_factor_closed_form():
    X = np.array([1e-6, 1e-4, 0.3, 1.0, 5.0, 40.0])
    T = X / LK_TESLA_PER_KELVIN            # with m = 1, B = 1
    R = lk_thermal_factor(T, 1.0, 1.0)
    assert np.allclose(R, X / np.sinh(X), rtol=1e-12, atol=0.0)
    assert lk_thermal_factor(0.0, 5.0, 0.3) == 1.0
    assert lk_thermal_factor(1e4, 1.0, 1.0) == 0.0   # no overflow


def test_lk_fit_recovers_noise_free_mass():
    B, m, A0 = 12.0, 0.48, 3.2
    y = _lk_scipy(T_DATA, A0, m, B)
    r = fit_sdh_mass(T_DATA, y, B)
    assert abs(r["mass_ratio"] / m - 1.0) < 1e-9
    assert abs(r["A0"] / A0 - 1.0) < 1e-9
    assert r["x_max"] > 5.0


@pytest.mark.parametrize("absolute", [False, True])
def test_lk_fit_against_scipy_curve_fit(absolute):
    """Fixed, hand-perturbed amplitudes (no random numbers): the mass,
    A0 and both error bars agree with scipy.optimize.curve_fit, an
    independent optimizer with a finite-difference Jacobian."""
    B = 12.0
    y = _lk_scipy(T_DATA, 3.2, 0.48, B) * np.array(
        [1.02, 0.97, 1.01, 1.04, 0.95, 1.03])
    sig = 0.03 * y if absolute else None
    r = fit_sdh_mass(T_DATA, y, B, sigma_amplitudes=sig)
    popt, pcov = curve_fit(lambda T, A0, m: _lk_scipy(T, A0, m, B),
                           T_DATA, y, p0=(3.0, 0.5), sigma=sig,
                           absolute_sigma=absolute, xtol=1e-14,
                           ftol=1e-14, gtol=1e-14)
    assert abs(r["A0"] / popt[0] - 1.0) < 1e-7
    assert abs(r["mass_ratio"] / popt[1] - 1.0) < 1e-7
    assert abs(r["sigma_A0"] / math.sqrt(pcov[0, 0]) - 1.0) < 1e-4
    assert abs(r["sigma_mass_ratio"] / math.sqrt(pcov[1, 1]) - 1.0) < 1e-4
    corr = pcov[0, 1] / math.sqrt(pcov[0, 0] * pcov[1, 1])
    assert abs(r["correlation"] - corr) < 1e-4
    if absolute:
        resid = (y - _lk_scipy(T_DATA, *popt, B)) / sig
        assert abs(r["chi2_red"] - (resid ** 2).sum() / 4) < 1e-6
    else:
        assert r["chi2_red"] is None


def test_lk_fit_error_bars_match_the_scatter():
    """Monte Carlo: 400 data sets with 3 % Gaussian noise. The spread of
    the fitted masses equals the reported 1-sigma (to 12 %), and the
    mean chi2_red is 1 (to 0.1)."""
    rng = np.random.default_rng(2026)
    B, m, A0 = 12.0, 0.48, 3.2
    clean = _lk_scipy(T_DATA, A0, m, B)
    sig = 0.03 * clean
    ms, sms, chis = [], [], []
    for _ in range(400):
        y = clean + sig * rng.standard_normal(T_DATA.size)
        r = fit_sdh_mass(T_DATA, y, B, sigma_amplitudes=sig)
        ms.append(r["mass_ratio"])
        sms.append(r["sigma_mass_ratio"])
        chis.append(r["chi2_red"])
    assert abs(np.std(ms) / np.mean(sms) - 1.0) < 0.12
    assert abs(np.mean(ms) / m - 1.0) < 0.01
    assert abs(np.mean(chis) - 1.0) < 0.1


def test_lk_fit_refusals():
    B = 12.0
    with pytest.raises(ValueError, match="edge of the searched"):
        fit_sdh_mass(T_DATA, np.ones(T_DATA.size), B)        # flat
    with pytest.raises(ValueError, match="3 or more distinct"):
        fit_sdh_mass([2.0, 4.0], [1.0, 0.8], B)
    with pytest.raises(ValueError, match="3 or more distinct"):
        fit_sdh_mass([2.0, 2.0, 4.0], [1.0, 1.0, 0.8], B)
    with pytest.raises(ValueError, match="amplitudes must be"):
        fit_sdh_mass(T_DATA, -np.ones(T_DATA.size), B)
    with pytest.raises(ValueError, match="field_T"):
        fit_sdh_mass(T_DATA, np.ones(T_DATA.size), 0.0)
    # barely damped (X up to 0.12 for the true 0.48 m0 at 0.5 K and
    # 30 T) with 1 % scatter: the best fit is 0.93 +/- 0.61 m0, and
    # chi-square rises by 0.46 and 1.74 at -/+ sigma, not by 1
    T = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = _lk_scipy(T, 1.0, 0.48, 30.0) * np.array(
        [1.0, 1.01, 0.99, 1.005, 0.995])
    with pytest.raises(ValueError, match="error bar would not be honest"):
        fit_sdh_mass(T, y, 30.0)
    # the same scatter up to 4 K is damped enough and is accepted
    T4 = np.linspace(0.1, 4.0, 5)
    y4 = _lk_scipy(T4, 1.0, 0.48, 30.0) * np.array(
        [1.0, 1.01, 0.99, 1.005, 0.995])
    r = fit_sdh_mass(T4, y4, 30.0)
    assert abs(r["mass_ratio"] - 0.48) < 2 * r["sigma_mass_ratio"]


# ---------------------------------------------------------------- release

def test_version_is_consistent():
    root = pathlib.Path(__file__).resolve().parents[1]
    py = (root / "pyproject.toml").read_text()
    cff = (root / "CITATION.cff").read_text()
    v = re.search(r'^version = "([^"]+)"', py, re.M).group(1)
    assert kpenvelope.__version__ == v
    assert re.search(r"^version: " + re.escape(v) + r"\s*$", cff, re.M)
