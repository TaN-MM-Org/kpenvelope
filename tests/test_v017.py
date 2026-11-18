"""v0.17 anchors: interface charges of relaxed nitride stacks, electron
gases, error propagation and fits.

Independent references used here:

* Dreyer et al., Phys. Rev. X 6, 021038 (2016), Tables I and II (HSE
  constants), copied number by number.
* The archived results of T. M. Mahim, nitride-misfit-polarization v1.3.1
  (doi:10.5281/zenodo.24315036), computed with a separate code (SciPy
  eigensolvers, Brent root search, and the labplan fitting package):
  results/prediction.json (interface charges and the sum rule, Table S6 of
  its Supplemental Material), results/shen_analysis.json (Schroedinger-
  Poisson densities of the Shen et al. series and the mobility bounds),
  results/shen_uncertainty.json (the joint fit). The numbers are copied
  with all their digits.
* Closed forms: Eq. 1 and 2 of the misfit paper, the 3/4 quantum of the
  zinc-blende reference, the two-carrier Hall relations, a straight-line
  least-squares fit and the Deming (orthogonal-distance) slope.
* SciPy quadrature (for F_1/2) and a dense linear solve (for the
  tridiagonal solver).
"""
import math

import numpy as np
import pytest

from kpenvelope import (E_PER_NM2_PER_C_M2, POLARIZATION_SETS, LineCharge,
                        NitrideMaterial, StackLayer, algan,
                        charge_change_per_relaxation, fermi_dirac_half,
                        fit_interface_series, fit_least_squares,
                        interface_charges, layer_strains,
                        line_charge_from_mobility_drop, line_charge_limits,
                        line_charge_mobility_bound, misfit_line_density,
                        polarization_quantum, propagate, solve_electrons,
                        split_two_carrier, strained_polarization,
                        trapped_from_sum_rule, two_carrier_hall,
                        zinc_blende_formal_polarization)
from kpenvelope.electrons import _thomas

CM2 = 1e14          # e/nm^2 -> e/cm^2
F_SHEN = {"pbesol-mahim2026": 0.31325546873917676,
          "hse-dreyer2016": 0.2104741823844428}


# ------------------------------------------------------------- materials
def test_hse_set_is_dreyer_tables_I_and_II():
    s = POLARIZATION_SETS["hse-dreyer2016"]
    assert (s["GaN"]["P_H"], s["GaN"]["e31"], s["GaN"]["e33"]) == (1.312, -0.551, 1.020)
    assert (s["AlN"]["P_H"], s["AlN"]["e31"], s["AlN"]["e33"]) == (1.351, -0.676, 1.569)
    assert (s["InN"]["P_H"], s["InN"]["e31"], s["InN"]["e33"]) == (1.026, -0.604, 1.238)
    assert all(s[m]["c13_over_c33"] is None for m in ("GaN", "AlN", "InN"))


def test_algan_end_points_and_vegard():
    g, a = algan(0.0), algan(1.0)
    assert (g.a, a.a) == (0.3189, 0.3112)
    assert (g.me, a.me, g.eps_r, a.eps_r) == (0.20, 0.32, 10.4, 10.7)
    assert g.ec == 0.0
    # Varshni at 300 K and the 0.7 offset rule
    eg_g = 3.510 - 0.909e-3 * 300 ** 2 / (300 + 830)
    eg_a = 6.25 - 1.799e-3 * 300 ** 2 / (300 + 1462)
    assert g.eg == pytest.approx(eg_g, rel=1e-15)
    m = algan(0.22)
    eg = 0.22 * eg_a + 0.78 * eg_g - 0.7 * 0.22 * 0.78
    assert m.eg == pytest.approx(eg, rel=1e-14)
    assert m.ec == pytest.approx(0.7 * (eg - eg_g), rel=1e-14)
    assert m.P_H == pytest.approx(0.22 * a.P_H + 0.78 * g.P_H, rel=1e-15)
    # the mismatch used throughout the misfit analysis
    assert (g.a - m.a) / m.a == pytest.approx(0.0053403781769575465, rel=1e-14)


def test_hse_set_borrows_and_declares_the_pbesol_elastic_ratio():
    m = algan(0.5, polarization="hse-dreyer2016")
    p = POLARIZATION_SETS["pbesol-mahim2026"]
    assert m.c13_over_c33 == pytest.approx(
        0.5 * (p["GaN"]["c13_over_c33"] + p["AlN"]["c13_over_c33"]), rel=1e-15)
    assert "C13/C33 not given by that source" in m.reference
    assert "supplied by the caller" in algan(0.5, polarization="hse-dreyer2016",
                                             c13_over_c33=0.27).reference


def test_zinc_blende_reference_is_three_quarters_of_a_quantum():
    for a in (0.3112, 0.3189, 0.354):
        assert zinc_blende_formal_polarization(a) / polarization_quantum(a) \
            == pytest.approx(0.75, rel=1e-15)


def test_material_refusals():
    with pytest.raises(ValueError):
        algan(1.2)
    with pytest.raises(ValueError):
        algan(0.2, polarization="lda")
    with pytest.raises(ValueError):
        NitrideMaterial("X", 0.3, 1.3, -0.5, 1.0, 0.25, reference="")
    with pytest.raises(ValueError):
        NitrideMaterial("X", -0.3, 1.3, -0.5, 1.0, 0.25, reference="r")


# ----------------------------------------------------- interface charges
def _cases(pol):
    g, a, a30, a22 = (algan(x, polarization=pol) for x in (0.0, 1.0, 0.3, 0.22))
    return {"GaN on AlN": [a, g], "AlN on GaN": [g, a],
            "Al0.30Ga0.70N on GaN": [g, a30], "Al0.22Ga0.78N on GaN": [g, a22],
            "Al0.30Ga0.70N on GaN strained to AlN": [a, g, a30]}


def _top(mats, r, lc):
    L = [StackLayer(mats[0], 100.0)] + [StackLayer(m, 10.0) for m in mats[1:-1]] \
        + [StackLayer(mats[-1], 20.0, relaxation=r)]
    return interface_charges(L, line_charge=lc)[-1]


# results/prediction.json: coherent charge and relaxed charges (cm^-2)
PRED = {
    ("pbesol-mahim2026", "GaN on AlN"): (-61482836103203.42, {
        "bare": -2410470804556.8794, "electron-counting": -43533755352599.78,
        "neutral": -42704433740860.73, "shen": -30082129493889.844,
        "trapped": -55326737987831.61}),
    ("hse-dreyer2016", "GaN on AlN"): (-39243529183566.98, {
        "bare": 24341885390396.86, "electron-counting": -16781399157646.053,
        "neutral": -15202938236967.443, "shen": -6879773816460.946,
        "trapped": -23526102657473.945}),
    ("pbesol-mahim2026", "Al0.30Ga0.70N on GaN strained to AlN"): (14975561884540.84, {
        "bare": 59795506540013.63, "electron-counting": 30374480529610.938,
        "neutral": 31408552170990.242, "shen": 40300920867936.28,
        "trapped": 22516183474044.2}),
}


@pytest.mark.parametrize("key", list(PRED))
def test_interface_charges_match_the_archived_table_S6(key):
    pol, case = key
    mats = _cases(pol)[case]
    coh, relaxed = PRED[key]
    assert _top(mats, 0.0, None).sigma_cm2 == pytest.approx(coh, rel=1e-12)
    film = mats[-1]
    eps_c = _top(mats, 0.0, None).eps_c
    q = F_SHEN[pol] * abs(line_charge_limits(film, eps_c)["bare"])
    specs = {"bare": "bare", "electron-counting": "electron-counting",
             "neutral": "neutral", "shen": LineCharge(bound_fraction=F_SHEN[pol]),
             "trapped": LineCharge(trapped=q)}
    for k, spec in specs.items():
        assert _top(mats, 1.0, spec).sigma_cm2 == pytest.approx(relaxed[k], rel=1e-12), k


def test_bare_and_electron_counting_line_charges():
    g, b = algan(0.0), algan(0.22)
    lim = line_charge_limits(b, (g.a - b.a) / b.a)
    # results/shen_analysis.json, predictions_t100 (e/nm)
    assert lim["bare"] == pytest.approx(-2.645081574331336, rel=1e-12)
    assert lim["electron-counting"] == pytest.approx(0.08508558450058583, rel=1e-10)
    assert line_charge_limits(b, -0.01)["bare"] == -lim["bare"]


def test_line_density_and_spacing_of_the_relaxed_shen_barrier():
    g, b = algan(0.0, 77.0), algan(0.22, 77.0)
    eps_c = (g.a - b.a) / b.a
    LA = misfit_line_density(0.78, eps_c, b.a)
    assert LA * 1e7 == pytest.approx(262636.5817813589, rel=1e-12)      # per cm
    assert 3.0 / LA == pytest.approx(114.22628103260405, rel=1e-12)     # nm


def test_equation_2_decomposition_is_exact_for_both_references():
    g, b = algan(0.0), algan(0.3)
    for ref in ("H", "ZB"):
        for r in (0.0, 0.3, 1.0):
            for lc in ("bare", "neutral", 0.42):
                L = [StackLayer(g, 50.0), StackLayer(b, 20.0, relaxation=r)]
                c = interface_charges(L, line_charge=lc, reference=ref)[0]
                assert c.sigma == pytest.approx(c.coherent + c.relaxation_term + c.lines,
                                                abs=1e-15)


def test_references_differ_only_by_an_r_independent_constant():
    """Dreyer et al.: H and ZB, used consistently, agree to first order; the
    whole r dependence is the same once the line charge is fixed."""
    g, b = algan(0.0), algan(0.4)
    L = lambda r: [StackLayer(g, 50.0), StackLayer(b, 20.0, relaxation=r)]
    d = [interface_charges(L(r), "neutral", "H")[0].sigma
         - interface_charges(L(r), "neutral", "ZB")[0].sigma for r in (0, 0.25, 0.5, 1)]
    assert np.ptp(d) < 1e-15
    eps = (g.a - b.a) / b.a
    expect = (zinc_blende_formal_polarization(g.a) - zinc_blende_formal_polarization(b.a)) \
        + 2 * eps * b.P_H
    assert d[0] == pytest.approx(expect, rel=1e-10)
    assert abs(d[0]) < 1e-3          # second order in the strain


def test_the_two_recipes_differ_by_the_bare_end_charge():
    """H recipe (bare ends) minus ZB recipe (neutral lines), beyond their
    coherent difference, is -2 r eps_c P_H: the formula of Wang et al.,
    Appl. Phys. Lett. 124, 263502 (2024)."""
    g, a = algan(0.0), algan(1.0)
    eps = (g.a - a.a) / a.a
    for r in (0.25, 0.5, 1.0):
        L = lambda rr: [StackLayer(g, 50.0), StackLayer(a, 5.0, relaxation=rr)]
        h = interface_charges(L(r), "bare", "H")[0].sigma \
            - interface_charges(L(0.0), None, "H")[0].sigma
        z = interface_charges(L(r), "neutral", "ZB")[0].sigma \
            - interface_charges(L(0.0), None, "ZB")[0].sigma
        assert h - z == pytest.approx(-2 * r * eps * a.P_H, rel=1e-12)


def test_relaxed_interface_without_a_line_charge_is_refused():
    g, b = algan(0.0), algan(0.22)
    with pytest.raises(ValueError, match="misfit dislocations"):
        interface_charges([StackLayer(g, 50.0), StackLayer(b, 100.0, relaxation=0.78)])
    with pytest.raises(ValueError, match="bottom layer"):
        interface_charges([StackLayer(g, 50.0, relaxation=0.5), StackLayer(b, 10.0)])
    with pytest.raises(ValueError):
        interface_charges([StackLayer(g, 50.0), StackLayer(b, 10.0, relaxation=0.2)],
                          line_charge="halfway")
    with pytest.raises(ValueError):
        StackLayer(g, 10.0, relaxation=1.5)


def test_strain_carries_through_a_multilayer_stack():
    """GaN 8 percent relaxed on AlN (the 20 nm channel of Elwaradi et al.)
    leaves Al0.3Ga0.7N on it compressed by about 1.5 percent."""
    aln, gan, a30 = algan(1.0), algan(0.0), algan(0.3)
    st = layer_strains([StackLayer(aln, 500.0), StackLayer(gan, 20.0, relaxation=0.08),
                        StackLayer(a30, 19.0)])
    assert st[1][0] == pytest.approx((aln.a - gan.a) / gan.a, rel=1e-14)
    a_gan = gan.a * (1 + 0.92 * st[1][0])
    assert st[2][0] == pytest.approx((a_gan - a30.a) / a30.a, rel=1e-14)
    assert -0.0155 < st[2][0] < -0.0145


def test_sum_rule_against_the_archived_values():
    """results/prediction.json, odd_even_Al0.30 (PBEsol): Al0.3Ga0.7N
    stretched on GaN and compressed on GaN held at the lattice of AlN."""
    gan, aln, a30 = algan(0.0), algan(1.0), algan(0.3)
    e_t, e_c = (gan.a - a30.a) / a30.a, (aln.a - a30.a) / a30.a
    f = F_SHEN["pbesol-mahim2026"]
    q = f * abs(line_charge_limits(a30, e_t)["bare"])
    assert q == pytest.approx(0.8267859466377652, rel=1e-12)
    yp = charge_change_per_relaxation(a30, e_t, LineCharge(trapped=q))
    ym = charge_change_per_relaxation(a30, e_c, LineCharge(trapped=q))
    assert yp == pytest.approx(-1.1916382523092466, rel=1e-12)
    assert ym == pytest.approx(0.3548100991631598, rel=1e-12)
    assert yp + ym == pytest.approx(-0.8368281531460868, rel=1e-12)
    assert trapped_from_sum_rule(yp, ym, a30.a) == pytest.approx(q, rel=1e-12)
    # any bound charge cancels in the sum, whatever the constants
    for lc in ("bare", "neutral", "electron-counting", LineCharge(bound_fraction=0.6)):
        s = charge_change_per_relaxation(a30, e_t, lc) + \
            charge_change_per_relaxation(a30, e_c, lc)
        assert abs(s) < 1e-14
    # and the slope matches the stack calculation
    L = [StackLayer(gan, 50.0), StackLayer(a30, 20.0, relaxation=0.4)]
    c0 = interface_charges([StackLayer(gan, 50.0), StackLayer(a30, 20.0)])[0]
    c = interface_charges(L, LineCharge(bound_fraction=f, trapped=0.1))[0]
    assert (c.sigma - c0.sigma) / (2 * 0.4 * abs(e_t)) == pytest.approx(
        charge_change_per_relaxation(a30, e_t, LineCharge(bound_fraction=f, trapped=0.1)),
        rel=1e-12)


def test_strained_polarization_branches():
    m = algan(0.0)
    assert strained_polarization(m, 0.0, "H") == m.P_H
    assert strained_polarization(m, 0.0, "ZB") == pytest.approx(
        m.P_H - zinc_blende_formal_polarization(m.a), rel=1e-15)
    # the improper term
    e = 1e-3
    dH = strained_polarization(m, e, "H") - strained_polarization(m, 0.0, "H")
    dZ = strained_polarization(m, e, "ZB") - strained_polarization(m, 0.0, "ZB")
    assert dH - dZ == pytest.approx(-2 * e * m.P_H, rel=1e-12)


# ------------------------------------------------------------- electrons
def test_fermi_dirac_half_against_quadrature():
    quad = pytest.importorskip("scipy.integrate").quad
    for e in (-30.3, -3.1, -0.4, 0.0, 0.7, 5.5, 22.2, 39.9):
        f = quad(lambda x: np.sqrt(x) / (1 + np.exp(min(x - e, 700))), 0, max(e, 0) + 80,
                 points=[max(e, 0)] if e > 0 else None, limit=400, epsabs=0,
                 epsrel=1e-12)[0] * 2 / np.sqrt(np.pi)
        F, _ = fermi_dirac_half(e)
        assert F == pytest.approx(f, rel=1e-8)
    # derivative against a central difference of F itself
    for e in (-5.0, 0.3, 12.0):
        h = 1e-5
        d = (fermi_dirac_half(e + h)[0] - fermi_dirac_half(e - h)[0]) / (2 * h)
        assert fermi_dirac_half(e)[1] == pytest.approx(d, rel=1e-6)
    # limits beyond the table
    assert fermi_dirac_half(-60.0)[0] == pytest.approx(np.exp(-60.0), rel=1e-12)


def test_tridiagonal_solver():
    rng = np.random.default_rng(3)
    n = 40
    a, c = rng.normal(size=n - 1), rng.normal(size=n - 1)
    b = 4 + rng.random(n)
    d = rng.normal(size=n)
    M = np.diag(b) + np.diag(c, 1) + np.diag(a, -1)
    assert np.allclose(_thomas(a, b, c, d), np.linalg.solve(M, d), rtol=1e-12, atol=1e-12)


def _shen(T=77.0):
    return algan(0.0, T), algan(0.22, T)


DSIG = -0.004430561884400016 * E_PER_NM2_PER_C_M2    # archived calibration
SHEN_KW = dict(phi_B=0.5, temperature_K=77.0, dz=0.1, quantum_window=(40.0, 84.0),
               n_states=40, classical_outside=True)


def _shen_layers(t, R=0.0):
    g, b = _shen()
    return [StackLayer(g, 80.0), StackLayer(b, 3.0, relaxation=R),
            StackLayer(b, t, donors=1.26e-3)]


def test_electrons_reproduce_the_archived_shen_densities():
    """results/shen_analysis.json, baseline 77K_mu2_0: 2DEG and barrier
    electron densities (cm^-2), computed there with SciPy eigensolvers."""
    ref = [((25, 0.0, None), 11694328992521.723, 303664736137.3699),
           ((100, 0.0, None), 11745673878362.277, 9697821249300.553),
           ((100, 0.78, "bare"), 2228240249866.323, 8460853010481.266),
           ((100, 0.78, "electron-counting"), 8577693621522.084, 9281818168622.287),
           ((100, 0.78, "neutral"), 8380056159215.483, 9255989758194.1)]
    for (t, R, lc), n1, n2 in ref:
        r = solve_electrons(_shen_layers(t, R), line_charge=lc, extra_sheet={0: DSIG},
                            **SHEN_KW)
        assert r.converged
        assert r.sheet_quantum * CM2 == pytest.approx(n1, rel=1e-9)
        assert r.sheet_classical * CM2 == pytest.approx(n2, rel=1e-9)


def test_electron_solver_physics_checks():
    g, b = algan(0.0), algan(0.25)
    sheets = []
    for t in (10.0, 20.0, 40.0):
        r = solve_electrons([StackLayer(g, 60.0), StackLayer(b, t)], phi_B=1.0,
                            quantum_window=(30.0, 60.0 + t))
        assert r.converged
        sheets.append(r.sheet_quantum)
        # the electrons sit in the GaN, next to the interface
        zc = (r.z * r.n).sum() / r.n.sum()
        assert 55.0 < zc < 60.0
        # the lowest level lies below the Fermi level (an occupied 2DEG)
        assert r.subband_energies[0] < 0.0
    assert sheets[0] < sheets[1] < sheets[2]
    # thick undoped barrier: the 2DEG approaches the interface charge
    # (Gauss's law), short of it by the surface field
    sig = interface_charges([StackLayer(g, 60.0), StackLayer(b, 40.0)])[0].sigma_nm2
    assert 0.5 * sig < sheets[-1] < sig


def test_electron_solver_refusals():
    g = algan(0.0)
    bare = NitrideMaterial("bare", 0.31, 1.33, -0.6, 1.5, 0.3, reference="test only")
    with pytest.raises(ValueError, match="no eps_r"):
        solve_electrons([StackLayer(g, 50.0), StackLayer(bare, 10.0)], phi_B=1.0)
    with pytest.raises(ValueError, match="interface index"):
        solve_electrons([StackLayer(g, 50.0), StackLayer(algan(0.2), 10.0)], phi_B=1.0,
                        extra_sheet={3: 0.01})


def test_two_carrier_hall_round_trip():
    nH, muH = two_carrier_hall(6.45e12, 1108.0, 9.7e12, 83.0)
    n1, mu1 = split_two_carrier(nH, muH, 9.7e12, 83.0)
    assert (n1, mu1) == (pytest.approx(6.45e12, rel=1e-12), pytest.approx(1108.0, rel=1e-12))
    with pytest.raises(ValueError):
        split_two_carrier(1e12, 100.0, 1e13, 100.0)


def test_mobility_bounds_match_the_archived_values():
    """results/shen_analysis.json, baseline 77K_mu2_0, mobility block."""
    LA = 262636.5817813589 * 1e-7          # nm^-1
    ns = 6449999999999.999 * 1e-14
    strict = line_charge_mobility_bound(ns, 1108.0, LA, 0.20, 10.4)
    assert strict == pytest.approx(1.277806399828856, rel=1e-12)
    ref = [2727.0, 2899.0, 3019.0, 2711.0]
    est = [0.9845684487250481, 1.004358464817271, 1.0166321645985266, 0.9825780572651635]
    for m, e in zip(ref, est):
        assert line_charge_from_mobility_drop(m, 1108.0, ns, LA, 0.20, 10.4) \
            == pytest.approx(e, rel=1e-12)


# ---------------------------------------------------------- uncertainty
def test_linear_propagation_is_exact_for_a_linear_function():
    f = lambda a, b, c: 3 * a - 2 * b + 0.5 * c
    p = propagate(f, {"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 0.1, "b": 0.2})
    assert p.sigma == pytest.approx(math.hypot(0.3, 0.4), rel=1e-10)
    assert sum(p.budget.values()) == pytest.approx(1.0, rel=1e-12)
    assert p.budget["b"] == pytest.approx(0.64, rel=1e-10)
    pc = propagate(f, {"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 0.1, "b": 0.2},
                   correlation={("a", "b"): 0.5})
    assert pc.sigma == pytest.approx(math.sqrt(0.09 + 0.16 - 2 * 0.5 * 0.3 * 0.4), rel=1e-10)


def test_monte_carlo_agrees_with_linear_and_respects_bounds():
    f = lambda a, b: 2 * a + b
    mc = propagate(f, {"a": 0.0, "b": 1.0}, {"a": 0.5, "b": 1.0},
                   method="monte-carlo", n_samples=4000, seed=1)
    exact = math.hypot(1.0, 1.0)
    assert abs(mc.sigma - exact) < 4 * exact / math.sqrt(2 * 4000)
    assert abs(mc.mean - 1.0) < 4 * exact / math.sqrt(4000)
    tr = propagate(lambda r: r, {"r": 0.9}, {"r": 0.2}, method="monte-carlo",
                   n_samples=500, seed=2, bounds={"r": (0.0, 1.0)})
    assert tr.samples.max() <= 1.0 and tr.rejected_fraction > 0.2
    with pytest.raises(ValueError):
        propagate(f, {"a": 0.0}, {"b": 1.0})


def test_propagation_through_an_interface_charge():
    """sigma depends linearly on Lambda, so the linear error is |dsigma/dLambda|
    times the error of Lambda, and dsigma/dLambda is L_A."""
    g, b = algan(0.0), algan(0.22)

    def sig(lam):
        L = [StackLayer(g, 50.0), StackLayer(b, 100.0, relaxation=0.78)]
        return interface_charges(L, line_charge=lam)[0].sigma_nm2

    p = propagate(sig, {"lam": -0.84}, {"lam": 0.53})
    LA = misfit_line_density(0.78, (g.a - b.a) / b.a, b.a)
    assert p.sigma == pytest.approx(0.53 * LA, rel=1e-9)


# --------------------------------------------------------------- fitting
def test_weighted_straight_line_fit_is_exact():
    x = np.linspace(0, 4, 9)
    sy = 0.1 + 0.05 * x
    y = 1.5 - 0.7 * x + np.array([0.03, -0.02, 0.01, 0.0, -0.04, 0.02, 0.05, -0.01, 0.0])
    fr = fit_least_squares(lambda t, r: t[0] + t[1] * r[0], x, y, (0.0, 0.0), y_sigma=sy)
    W = np.diag(1 / sy ** 2)
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ W @ A)
    th = cov @ A.T @ W @ y
    assert np.allclose(fr.theta, th, rtol=1e-8, atol=1e-10)
    assert np.allclose(fr.cov, cov, rtol=1e-6)
    # without reading errors the covariance is scaled by chi2 / dof
    fu = fit_least_squares(lambda t, r: t[0] + t[1] * r[0], x, y, (0.0, 0.0))
    A0 = np.linalg.inv(A.T @ A)
    s2 = ((y - A @ (A0 @ A.T @ y)) ** 2).sum() / (len(x) - 2)
    assert np.allclose(fu.cov, A0 * s2, rtol=1e-6)


def test_errors_in_variables_fit_matches_the_deming_slope():
    """Equal and constant errors: the orthogonal-distance slope has the
    closed Deming form."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.1, 1.2, 1.9, 3.2, 3.9, 5.1])
    s = 0.2
    fr = fit_least_squares(lambda t, r: t[0] + t[1] * r[0], x, y, (0.0, 1.0),
                           y_sigma=s, x_sigma=s)
    sxx = ((x - x.mean()) ** 2).sum()
    syy = ((y - y.mean()) ** 2).sum()
    sxy = ((x - x.mean()) * (y - y.mean())).sum()
    b = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    assert fr.theta[1] == pytest.approx(b, rel=1e-7)
    assert fr.theta[0] == pytest.approx(y.mean() - b * x.mean(), rel=1e-6)


def test_fit_refusals():
    with pytest.raises(ValueError, match="y_sigma"):
        fit_least_squares(lambda t, r: t[0], [1, 2, 3], [1, 2, 3], (0.0,), x_sigma=0.1)
    with pytest.raises(ValueError):
        fit_least_squares(lambda t, r: t[0] + t[1], [1.0], [1.0], (0.0, 0.0))


def test_line_charge_fit_reproduces_the_archived_value():
    """Lambda alone, with the archived calibration fixed, from the relaxed
    Shen sample (results/shen_analysis.json: -0.8285862684205223 e/nm, found
    there by a Brent search with xtol 1e-3)."""
    def build(row):
        return _shen_layers(row[0], row[1])

    fr = fit_least_squares(
        lambda t, row: solve_electrons(build(row), line_charge=float(t[0]),
                                       extra_sheet={0: DSIG}, **SHEN_KW).sheet_quantum,
        np.array([[100.0, 0.78]]), np.array([0.645e13 * 1e-14]), (-0.8,),
        y_sigma=0.01)
    assert fr.theta[0] == pytest.approx(-0.8285862684205223, abs=1e-3)
    assert fit_interface_series is not None
