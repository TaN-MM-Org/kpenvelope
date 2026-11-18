"""v0.18 anchors: InGaN, InAlN and ScAlN records, the Vurgaftman-Meyer
polarization set, parameter-set checks, and the line-charge tools.

Independent references used here (numbers copied by hand from the PDFs):

* I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003), Table I
  (wurtzite binaries), Table III (bowings), Sec. II F (band offsets).
* O. Ambacher et al., J. Appl. Phys. 130, 045102 (2021): Eqs. (1a),
  (3a), (8), (18a)-(18c), (23c), (23d), (39b), (39c), and the percentages
  quoted in its abstract and text, which test the fits independently.
* The archived results of nitride-misfit-polarization v1.3.1 for the
  line-charge constraints and planners (as in test_v017).
"""
import math
import warnings

import numpy as np
import pytest

from kpenvelope import (AMBACHER2021_SCALN, POLARIZATION_SETS,
                        PUBLISHED_CONSTRAINTS, VURGAFTMAN2003, Constraint,
                        NitrideMaterial, StackLayer, algan,
                        combine_line_charge, fit_least_squares, gan,
                        inaln, ingan,
                        interface_charges, line_charge_error_from_density,
                        plan_holography, plan_sum_rule, scaln,
                        solve_electrons, strained_polarization,
                        with_numbers, zinc_blende_formal_polarization)
from test_v017 import DSIG, SHEN_KW, _shen_layers  # noqa: F401

QE = 1.602176634e-19


def _q(a_nm):
    """e sqrt(3) / (2 a^2) in C/m^2, written out again here."""
    return QE * math.sqrt(3.0) / (2.0 * (a_nm * 1e-9) ** 2)


def _charge(sub, film, ref="H", r=0.0, line=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = interface_charges([StackLayer(sub, 100.0),
                               StackLayer(film, 20.0, relaxation=r)],
                              line_charge=line, reference=ref)[0]
    return c, w


# ------------------------------------------------ Vurgaftman-Meyer tables
def test_vurgaftman_table_one_as_printed():
    t = VURGAFTMAN2003
    assert t["GaN"] == {"P_sp": -0.034, "c11": 390.0, "c12": 145.0,
                        "c13": 106.0, "c33": 398.0, "d13": -1.6, "d33": 3.1}
    assert t["AlN"] == {"P_sp": -0.090, "c11": 396.0, "c12": 137.0,
                        "c13": 108.0, "c33": 373.0, "d13": -2.1, "d33": 5.4}
    assert t["InN"] == {"P_sp": -0.042, "c11": 223.0, "c12": 115.0,
                        "c13": 92.0, "c33": 224.0, "d13": -3.5, "d33": 7.6}
    inn = ingan(1.0, temperature_K=1e-9)
    assert inn.a == 0.3545 and inn.me == 0.07
    assert inn.eg == pytest.approx(0.78, abs=1e-12)          # E_g at 0 K
    # Varshni 0.245 meV/K, 624 K at 300 K:
    assert ingan(1.0).eg == pytest.approx(0.78 - 0.245e-3 * 300 ** 2 / 924)


def test_piezo_constants_from_moduli_by_hand():
    ps = POLARIZATION_SETS["vurgaftman2003"]
    # e31 = (c11 + c12) d13 + c13 d33, e33 = 2 c13 d13 + c33 d33 (GPa pm/V)
    hand = {"GaN": ((535 * -1.6 + 106 * 3.1) / 1e3, (2 * 106 * -1.6 + 398 * 3.1) / 1e3),
            "AlN": ((533 * -2.1 + 108 * 5.4) / 1e3, (2 * 108 * -2.1 + 373 * 5.4) / 1e3),
            "InN": ((338 * -3.5 + 92 * 7.6) / 1e3, (2 * 92 * -3.5 + 224 * 7.6) / 1e3)}
    for m, (e31, e33) in hand.items():
        assert ps[m]["e31"] == pytest.approx(e31, abs=1e-12)
        assert ps[m]["e33"] == pytest.approx(e33, abs=1e-12)
    assert ps["GaN"]["e31"] == pytest.approx(-0.5274, abs=1e-12)
    assert ps["AlN"]["e33"] == pytest.approx(1.5606, abs=1e-12)
    assert ps["InN"]["c13_over_c33"] == pytest.approx(92 / 224)


def test_vurgaftman_set_keeps_its_zb_polarization_and_bowing():
    for x, m in ((0.0, "GaN"), (1.0, "AlN")):
        rec = algan(x, polarization="vurgaftman2003")
        assert rec.P_ZB == pytest.approx(VURGAFTMAN2003[m]["P_sp"], abs=1e-12)
        assert rec.P_H == pytest.approx(VURGAFTMAN2003[m]["P_sp"] + _q(rec.a))
    # Table III, C(P_sp) = -0.021, -0.037, -0.070 in E = ... - x(1-x) C
    a5 = algan(0.5, polarization="vurgaftman2003")
    assert a5.P_ZB == pytest.approx((-0.034 - 0.090) / 2 + 0.021 / 4, abs=1e-12)
    i5 = ingan(0.5, polarization="vurgaftman2003")
    assert i5.P_ZB == pytest.approx((-0.034 - 0.042) / 2 + 0.037 / 4, abs=1e-12)
    n5 = inaln(0.5, polarization="vurgaftman2003")
    assert n5.P_ZB == pytest.approx((-0.090 - 0.042) / 2 + 0.070 / 4, abs=1e-12)
    # an explicit bowing replaces the recommendation and is recorded
    z = algan(0.5, polarization="vurgaftman2003", psp_bowing=0.0)
    assert z.P_ZB == pytest.approx((-0.034 - 0.090) / 2, abs=1e-12)
    b = ingan(0.5, psp_bowing=0.01)
    assert "0.01 C/m^2 supplied by the caller" in b.reference
    # the elastic ratio of this set is c13(x) / c33(x)
    assert a5.c13_over_c33 == pytest.approx(107.0 / 385.5)


def test_default_algan_is_unchanged():
    # the v0.17 numbers (PBEsol set, no bowing) are what algan still gives
    m = algan(0.22)
    pb = POLARIZATION_SETS["pbesol-mahim2026"]
    assert m.P_H == 0.22 * pb["AlN"]["P_H"] + 0.78 * pb["GaN"]["P_H"]
    assert m.parameter_set == "pbesol-mahim2026"


# ------------------------------------------------------- InGaN and InAlN
def test_indium_alloy_bands_follow_the_cited_rules():
    g, inn = gan("hse-dreyer2016"), ingan(1.0)
    for x in (0.1, 0.3, 0.8):
        m = ingan(x)
        eg = x * inn.eg + (1 - x) * g.eg - 1.4 * x * (1 - x)
        assert m.eg == pytest.approx(eg, abs=1e-12)
        assert m.ec == pytest.approx(0.5 * x + eg - g.eg, abs=1e-12)
        assert m.eps_r == pytest.approx(10.4 + 4.0 * x)
        assert m.me == pytest.approx(0.20 - 0.13 * x)
    # the AlN end of InAlN has the band edge of algan(1) (the 0.7 rule)
    assert inaln(0.0).ec == pytest.approx(algan(1.0).ec, abs=1e-12)
    assert inaln(1.0).ec == pytest.approx(ingan(1.0).ec, abs=1e-12)
    n = inaln(0.4)
    eg = 0.4 * inn.eg + 0.6 * algan(1.0).eg - 2.5 * 0.24
    ev = 0.4 * 0.5 + 0.6 * (algan(1.0).ec - (algan(1.0).eg - g.eg))
    assert n.ec == pytest.approx(ev + eg - g.eg, abs=1e-12)
    assert "transitivity" in n.reference and "provisional" in n.reference


def test_hse_indium_alloys_take_the_elastic_ratio_from_vurgaftman():
    m = ingan(0.2)
    assert m.c13_over_c33 == pytest.approx((0.2 * 92 + 0.8 * 106) / (0.2 * 224 + 0.8 * 398))
    assert m.P_H == pytest.approx(0.2 * 1.026 + 0.8 * 1.312)
    assert m.e31 == pytest.approx(0.2 * -0.604 + 0.8 * -0.551)
    assert "Table I (each linear in x)" in m.reference
    with pytest.raises(ValueError, match="has no"):
        ingan(0.2, polarization="pbesol-mahim2026")


def test_lattice_matched_inaln():
    x = (0.3189 - 0.3112) / (0.3545 - 0.3112)
    assert inaln(x).a == pytest.approx(0.3189, abs=1e-15)
    assert x == pytest.approx(0.1778, abs=1e-4)


# -------------------------------------------------------------- ScAlN
def test_scaln_fits_are_the_printed_equations():
    t = AMBACHER2021_SCALN
    assert t["a_angstrom"] == (3.741, 3.110, -0.242)
    assert t["P_SP"] == (-0.874, -0.089, 0.741)
    assert t["C13_GPa"] == (141.70, 108.00, 51.95)
    assert t["C33_GPa"] == (-155.17, 373.00, 95.49)
    assert t["e31"] == (-1.353, -0.593, 0.576)
    assert t["e33"] == (9.125, 1.471, -6.625)
    assert t["eps33"] == (89.93, 10.31, -62.48)
    x = 0.3
    m = scaln(x)
    assert m.a == pytest.approx((3.741 * x + 3.110 * (1 - x) - 0.242 * x * (1 - x)) / 10)
    assert m.P_ZB == pytest.approx(-0.874 * x - 0.089 * (1 - x) + 0.741 * x * (1 - x), abs=1e-12)
    assert m.c13_over_c33 == pytest.approx(
        (141.70 * x + 108.00 * (1 - x) + 51.95 * x * (1 - x))
        / (-155.17 * x + 373.00 * (1 - x) + 95.49 * x * (1 - x)))
    assert np.isnan(m.eg) and np.isnan(m.ec) and np.isnan(m.me)
    with pytest.raises(ValueError):
        scaln(0.55)


def test_scaln_fits_match_the_numbers_in_the_text():
    s0, s5 = scaln(0.0), scaln(0.5)
    # "e33(x = 0.5) = 3.641 > e33(x = 0) = 1.471" (Sec. VI)
    assert s5.e33 == pytest.approx(3.641, abs=1e-3)
    assert s0.e33 == 1.471
    # abstract: C33, e33 and P_SP change "up to 65%, 150%, and 230%"
    c33 = lambda x: -155.17 * x + 373.0 * (1 - x) + 95.49 * x * (1 - x)
    assert 1 - c33(0.5) / c33(0.0) == pytest.approx(0.644, abs=0.01)
    assert s5.e33 / s0.e33 - 1 == pytest.approx(1.48, abs=0.03)
    assert s5.P_ZB / s0.P_ZB - 1 == pytest.approx(2.33, abs=0.03)
    # Sec. III: eps33 "enhancement by 235%" from x = 0 to 0.5
    assert s5.eps_r / s0.eps_r - 1 == pytest.approx(2.35, abs=0.01)


def test_ambacher_gan_and_lattice_match():
    g = gan("ambacher2021")
    assert g.a == pytest.approx(0.31986, abs=1e-15)
    assert g.P_ZB == pytest.approx(-0.034, abs=1e-12)
    assert not g.has_piezo and g.ec == 0.0
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if scaln(mid).a < g.a else (lo, mid)
    assert lo == pytest.approx(0.2023, abs=1e-4)
    # unstrained use is fine, strained use is refused
    assert strained_polarization(g, 0.0) == g.P_H
    with pytest.raises(ValueError, match="no piezoelectric"):
        strained_polarization(g, 0.01)
    with pytest.raises(ValueError, match="no piezoelectric"):
        interface_charges([StackLayer(scaln(0.1), 50.0), StackLayer(g, 10.0)])


def test_scaln_on_gan_charge_is_the_zb_recipe_of_the_paper():
    g, f = gan("ambacher2021"), scaln(0.12)
    c, w = _charge(g, f, "ZB")
    assert not w
    eps = (g.a - f.a) / f.a
    pe = 2 * eps * (f.e31 - f.e33 * f.c13_over_c33)      # Ambacher Eq. (46)
    assert c.sigma == pytest.approx(-0.034 - (f.P_ZB + pe), abs=1e-14)


# ---------------------------------------------------- set mixing and H/ZB
def test_mixing_sets_warns_and_same_set_does_not():
    _, w = _charge(gan(), scaln(0.18))
    assert len(w) == 1 and "one set" in str(w[0].message)
    for ps in ("pbesol-mahim2026", "hse-dreyer2016", "vurgaftman2003"):
        _, w = _charge(gan(ps), algan(0.25, polarization=ps))
        assert not w
    _, w = _charge(gan("hse-dreyer2016"), ingan(0.1))
    assert not w
    mine = NitrideMaterial("X", 0.315, 1.3, -0.5, 1.0, 0.25, reference="test")
    _, w = _charge(gan(), mine)
    assert not w


def test_h_minus_zb_is_two_eps_pzb_plus_second_order():
    for sub, film in ((gan(), algan(0.25)),
                      (gan("hse-dreyer2016"), ingan(0.15)),
                      (gan("ambacher2021"), scaln(0.1))):
        h, _ = _charge(sub, film, "H")
        z, _ = _charge(sub, film, "ZB")
        eps = h.eps_c
        first = 2 * eps * film.P_ZB
        q = _q(film.a)
        assert abs((h.sigma - z.sigma) - first) < 4 * q * eps * eps
    h, _ = _charge(gan(), algan(0.25))
    z, _ = _charge(gan(), algan(0.25), "ZB")
    assert abs(h.sigma - z.sigma) / h.sigma == pytest.approx(0.015, abs=0.002)
    h, _ = _charge(gan("hse-dreyer2016"), ingan(0.15))
    z, _ = _charge(gan("hse-dreyer2016"), ingan(0.15), "ZB")
    assert abs(h.sigma - z.sigma) / abs(h.sigma) == pytest.approx(0.10, abs=0.01)


def test_with_numbers_records_the_source_and_drives_the_solver():
    s = scaln(0.18)
    with pytest.raises(ValueError):
        with_numbers(s, "")
    with pytest.raises(ValueError, match="cannot replace"):
        with_numbers(s, "x", parameter_set="y")
    # made-up TEST numbers (not data): an offset, a mass, a gap
    b = with_numbers(s, "test values", ec=1.5, me=0.3, eg=5.0)
    assert b.parameter_set == "ambacher2021"
    assert b.reference.endswith("; ec, eg, me from test values")
    p = with_numbers(s, "test values", P_H=1.2)
    assert p.parameter_set == ""
    g = gan("ambacher2021")
    r = solve_electrons([StackLayer(g, 60.0), StackLayer(b, 8.0)], phi_B=1.0,
                        temperature_K=300.0, quantum_window=(40.0, 68.0))
    assert r.converged and r.sheet_quantum > 0.0
    with pytest.raises(ValueError):
        solve_electrons([StackLayer(g, 60.0), StackLayer(s, 8.0)], phi_B=1.0)


def test_material_validation():
    with pytest.raises(ValueError, match="finite number or NaN"):
        NitrideMaterial("X", 0.315, 1.3, math.inf, 1.0, 0.25, reference="t")
    with pytest.raises(ValueError, match="finite number;"):
        NitrideMaterial("X", 0.315, math.nan, -0.5, 1.0, 0.25, reference="t")
    m = NitrideMaterial("X", 0.315, 1.3, math.nan, 1.0, 0.25, reference="t")
    assert not m.has_piezo
    with pytest.raises(ValueError):
        algan(1.2)
    with pytest.raises(ValueError):
        ingan(-0.1)
    with pytest.raises(ValueError, match="unknown polarization set"):
        inaln(0.2, polarization="nope")
    assert zinc_blende_formal_polarization(0.3189) == pytest.approx(_q(0.3189))

# --------------------------------------------------- line-charge knowledge


def _truncnorm(mu, s, a, b):
    from math import erf, exp, pi, sqrt
    phi = lambda t: exp(-t * t / 2) / sqrt(2 * pi)
    Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
    al, be = (a - mu) / s, (b - mu) / s
    Z = Phi(be) - Phi(al)
    m = mu + s * (phi(al) - phi(be)) / Z
    v = s * s * (1 + (al * phi(al) - be * phi(be)) / Z - ((phi(al) - phi(be)) / Z) ** 2)
    return m, sqrt(v), 1 - Z


def test_published_constraints_are_the_archived_numbers():
    c = PUBLISHED_CONSTRAINTS
    assert (c["shen-density-pbesol"].value, c["shen-density-pbesol"].sigma) == \
        (-0.8448566143029012, 0.5323733875570534)
    assert (c["shen-density-hse"].value, c["shen-density-hse"].sigma) == \
        (-0.5729500135068806, 0.5323732285916173)
    assert c["shen-mobility"].value == 1.277806399828856
    assert c["elwaradi-mobility"].value == 1.3862680293570193
    assert all(k.source and k.assumptions for k in c.values())


def test_combination_is_the_truncated_normal():
    p = combine_line_charge(["shen-density-pbesol", "shen-mobility"])
    m, s, removed = _truncnorm(-0.8448566143029012, 0.5323733875570534,
                               -1.277806399828856, 1.277806399828856)
    assert p.mean == pytest.approx(m, rel=1e-12)
    assert p.sd == pytest.approx(s, rel=1e-12)
    assert p.removed == pytest.approx(removed, rel=1e-12)
    # the quantiles invert the CDF: the 68 percent interval holds 68 percent
    from math import erf, sqrt
    Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
    mu, sg, U = -0.8448566143029012, 0.5323733875570534, 1.277806399828856
    cdf = lambda v: (Phi((v - mu) / sg) - Phi((-U - mu) / sg)) / (1 - removed)
    assert cdf(p.interval68[1]) - cdf(p.interval68[0]) == pytest.approx(0.6826894921370859, abs=1e-12)
    assert cdf(p.median) == pytest.approx(0.5, abs=1e-12)
    assert p.interval95[0] >= -1.277806399828856 - 1e-3
    # a limit that does not bind changes nothing
    p2 = combine_line_charge(["shen-density-pbesol", "shen-mobility",
                              "elwaradi-mobility"])
    assert p2.mean == pytest.approx(p.mean, abs=1e-12)
    # alone, the Gaussian comes back
    g = combine_line_charge(["shen-density-pbesol"])
    assert g.mean == pytest.approx(-0.8448566143029012, rel=1e-14)
    assert g.sd == pytest.approx(0.5323733875570534, rel=1e-14)
    assert g.removed == pytest.approx(0.0, abs=1e-12)


def test_two_gaussians_combine_by_inverse_variance():
    a = Constraint("a", "gaussian", -1.0, 0.5, "x", "test", "test")
    b = Constraint("b", "gaussian", -0.4, 0.25, "x", "test", "test")
    p = combine_line_charge([a, b])
    w = np.array([4.0, 16.0])
    assert p.mean == pytest.approx((w @ [-1.0, -0.4]) / w.sum(), rel=1e-14)
    assert p.sd == pytest.approx(1 / math.sqrt(w.sum()), rel=1e-14)


def test_combination_refusals():
    with pytest.raises(ValueError):
        combine_line_charge(["shen-mobility"])
    with pytest.raises(ValueError):
        combine_line_charge(["no-such-thing"])
    tight = Constraint("u", "upper", 1e-6, math.nan, "x", "test", "test")
    far = Constraint("g", "gaussian", -5.0, 0.01, "x", "test", "test")
    with pytest.raises(ValueError, match="exclude"):
        combine_line_charge([far, tight])


def test_planners_reproduce_the_archived_numbers():
    """results/prediction.json, odd_even_Al0.30 (PBEsol): Al0.3Ga0.7N
    compressed on GaN held at AlN, and the stretched/compressed pair, for an
    error of 1e-3 C/m^2 on each charge change."""
    gan, aln, a30 = algan(0.0), algan(1.0), algan(0.3)
    e_t, e_c = (gan.a - a30.a) / a30.a, (aln.a - a30.a) / a30.a
    assert plan_holography(a30, e_c, 0.25) == pytest.approx(0.23212676686914735, rel=1e-12)
    assert plan_holography(a30, e_c, 0.5) == pytest.approx(0.11606338343457367, rel=1e-12)
    assert plan_holography(a30, e_c, 0.5, n_layers=4) == pytest.approx(
        0.11606338343457367 / 2, rel=1e-12)
    assert plan_sum_rule(a30, e_t, e_c, 0.25, 0.25) == pytest.approx(0.2946374647121935, rel=1e-12)
    assert plan_sum_rule(a30, e_t, e_c, 0.5, 0.5) == pytest.approx(0.14731873235609674, rel=1e-12)
    with pytest.raises(ValueError):
        plan_sum_rule(a30, e_c, e_t, 0.5, 0.5)


def test_density_planner_matches_the_fit_error():
    """One relaxed sample, one parameter: the planned error equals the
    error bar the fit would give for the same reading error."""
    kw = dict(SHEN_KW)
    row = [100.0, 0.78]
    plan = line_charge_error_from_density(
        lambda r: _shen_layers(r[0], r[1]), row, -0.83, density_error_nm2=0.001,
        extra_sheet={0: DSIG}, **kw)
    fr = fit_least_squares(
        lambda t, r: solve_electrons(_shen_layers(r[0], r[1]), line_charge=float(t[0]),
                                     extra_sheet={0: DSIG}, **kw).sheet_quantum,
        np.array([row]), np.array([0.645e13 * 1e-14]), (-0.8286,), y_sigma=0.001)
    assert plan["sigma_line_charge"] == pytest.approx(fr.sigma[0], rel=1e-3)
    # an error on the relaxation adds through the slope
    p2 = line_charge_error_from_density(
        lambda r: _shen_layers(r[0], r[1]), row, -0.83, density_error_nm2=0.001,
        settings_error=[0.0, 0.05], extra_sheet={0: DSIG}, **kw)
    expect = math.hypot(0.001, p2["dn_dsettings"][1] * 0.05) / abs(p2["dn_dLambda"])
    assert p2["sigma_line_charge"] == pytest.approx(expect, rel=1e-12)
    assert p2["sigma_line_charge"] > plan["sigma_line_charge"]


# ------------------------------------------------------- drift-diffusion
from kpenvelope import (Contact, Transport, bernoulli, cv_curve,  # noqa: E402
                        iv_curve, solve_drift_diffusion,
                        solve_quasi_equilibrium)
from kpenvelope.driftdiffusion import _solve_banded_numpy  # noqa: E402
from kpenvelope.electrons import HBAR, KB, M0  # noqa: E402

# A made-up TEST material and TEST transport numbers (silicon-like, not
# data): the checks below are against closed forms, not against a material.
_TM = NitrideMaterial("T", 0.3189, 1.33, -0.5, 0.9, 0.27, reference="test",
                      eg=1.12, ec=0.0, me=1.08, eps_r=11.7)
_TT = Transport(1000.0, 400.0, 0.81, reference="test values")
_Q = 1.602176634e-19
_EPS0 = 8.8541878128e-12


def _nc(m, T=300.0):
    return 2 * (m * M0 * KB * T * _Q / (2 * math.pi * HBAR ** 2)) ** 1.5 * 1e-6


def test_bernoulli():
    x = np.array([-800.0, -30.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 30.0, 800.0])
    B = bernoulli(x)
    assert B[4] == 1.0
    assert B[6] == pytest.approx(1.0 / (math.e - 1.0), rel=1e-14)
    assert np.allclose(bernoulli(-x), B + x, rtol=1e-13, atol=1e-13)
    assert B[0] == 800.0 and B[-1] == 0.0


def test_banded_numpy_solver_matches_dense():
    rng = np.random.default_rng(3)
    n, l, u = 40, 5, 5
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - l), min(n, i + u + 1)):
            A[i, j] = rng.normal()
    b = rng.normal(size=n)
    ab = np.zeros((l + u + 1, n))
    for i in range(n):
        for j in range(max(0, i - l), min(n, i + u + 1)):
            ab[u + i - j, j] = A[i, j]
    assert np.allclose(_solve_banded_numpy(l, u, ab, b), np.linalg.solve(A, b),
                       rtol=1e-10, atol=1e-10)


def test_dd_ohmic_resistor_is_ohms_law():
    L = [StackLayer(_TM, 1000.0, donors=1e-5)]
    for V in (0.01, 0.1):
        r = solve_drift_diffusion(L, _TT, top=Contact("ohmic", voltage=V),
                                  statistics="boltzmann", dz=5.0)
        assert r.current_density == pytest.approx(
            -_Q * 1e16 * 1000.0 * V / 1000e-7, rel=1e-9)
    r = solve_drift_diffusion(L, _TT, top=Contact("ohmic"), dz=5.0)
    assert abs(r.current_density) < 1e-12


def test_dd_pn_junction_built_in_and_short_diode():
    NA = ND = 1e17
    L = [StackLayer(_TM, 2000.0, donors=ND * 1e-21),
         StackLayer(_TM, 2000.0, acceptors=NA * 1e-21)]
    r0 = solve_drift_diffusion(L, _TT, top=Contact("ohmic"),
                               statistics="boltzmann", dz=4.0)
    kT = KB * 300.0
    ni2 = _nc(1.08) * _nc(0.81) * math.exp(-1.12 / kT)
    Vbi = kT * math.log(NA * ND / ni2)
    assert r0.psi[0] - r0.psi[-1] == pytest.approx(Vbi, abs=1e-10)
    assert np.allclose(r0.efn, 0.0) and np.allclose(r0.efp, 0.0)
    eps = 11.7 * _EPS0 * 1e-2
    J = {}
    for V in (0.4, 0.5):
        r = solve_drift_diffusion(L, _TT, top=Contact("ohmic", voltage=V),
                                  statistics="boltzmann", dz=4.0, guess=r0)
        W = math.sqrt(2 * eps * (Vbi - V) / _Q * (2 / NA))
        Wn = Wp = 2000e-7 - W / 2
        Js = _Q * ni2 * (1000 * kT / (NA * Wp) + 400 * kT / (ND * Wn))
        # forward current flows from the p side (top) down: negative
        assert -r.current_density == pytest.approx(Js * math.expm1(V / kT), rel=5e-3)
        J[V] = (-r.current_density, Js)
        r0 = r
    # ideality 1: the ratio follows exp(dV/kT) once the small change of
    # the neutral widths (in Js) is taken out
    ratio = (J[0.5][0] / J[0.5][1]) / (J[0.4][0] / J[0.4][1])
    assert ratio == pytest.approx(math.expm1(0.5 / kT) / math.expm1(0.4 / kT),
                                  rel=1e-3)


def test_dd_schottky_cv_slope_is_the_depletion_formula():
    ND = 1e17
    L = [StackLayer(algan(0.0), 500.0, donors=ND * 1e-21)]
    tr = Transport(1000.0, 30.0, 1.0, reference="test values")
    eps = 10.4 * _EPS0 * 1e-2
    for method in ("drift-diffusion", "quasi-equilibrium"):
        V, C, Q, res = cv_curve(L, [-1.5, -1.0, -0.5],
                                top=Contact("schottky", barrier=1.0), dz=1.0,
                                transport=tr, method=method)
        slope = np.polyfit(V, 1 / C ** 2, 1)[0]
        assert slope == pytest.approx(-2 / (_Q * eps * ND), rel=1e-3)
        assert np.all(np.diff(Q) > 0)          # less negative gate charge


def test_quasi_equilibrium_cv_of_a_transistor_matches_the_electron_solver():
    # an undoped buffer: no vertical current path, so C-V is quasi-static
    L = [StackLayer(algan(0.0), 1000.0), StackLayer(algan(0.25), 25.0)]
    V, C, Q, res = cv_curve(L, [-3.0, -1.0, 0.0],
                            top=Contact("schottky", barrier=1.0),
                            bottom=Contact("zero-field"), dz=0.25)
    for v, r in zip(V, res):
        e = solve_electrons(L, phi_B=1.0 - v, dz=0.25,
                            quantum_window=(1024.0, 1025.0),
                            classical_outside=True)
        assert r.sheet() == pytest.approx(e.sheet_classical_cm2, rel=1e-3)
        assert r.current_density == 0.0
    # the charge moves by q dn_s/dV, and C sits below the barrier's eps/d
    dns = (res[1].sheet() - res[0].sheet()) / 2.0
    assert (Q[1] - Q[0]) / 2.0 == pytest.approx(_Q * dns, rel=2e-2)
    eps_d = (0.25 * 10.7 + 0.75 * 10.4) * _EPS0 / 25e-9 * 1e-4
    assert np.all(C < eps_d) and np.all(C > 0.85 * eps_d)
    with pytest.raises(ValueError, match="acceptors"):
        solve_quasi_equilibrium([StackLayer(algan(0.0), 100.0, acceptors=1e-4)],
                                Contact("schottky", barrier=1.0))
    with pytest.raises(ValueError, match="zero-field"):
        solve_drift_diffusion(L, Transport(1.0, 1.0, 1.0, reference="t"),
                              top=Contact("ohmic"), bottom=Contact("zero-field"))


def test_dd_agrees_with_the_classical_electron_solver():
    L = [StackLayer(algan(0.0), 600.0, donors=1e-5), StackLayer(algan(0.25), 25.0)]
    tr = Transport(1500.0, 20.0, 1.0, reference="test values")
    r = solve_drift_diffusion(L, tr, top=Contact("schottky", barrier=1.0), dz=0.25)
    e = solve_electrons(L, phi_B=1.0, dz=0.25, quantum_window=(624.0, 625.0),
                        classical_outside=True)
    top = r.z > 500
    assert np.max(np.abs(r.ec[top] - e.ec[e.z > 500])) < 1e-3
    m = (e.z >= 580 - 1e-9) & (e.z <= 625 + 1e-9)
    ns_sp = np.sum(0.5 * (e.n[m][1:] + e.n[m][:-1]) * np.diff(e.z[m])) * 1e14
    assert r.sheet(580, 625) == pytest.approx(ns_sp, rel=1e-3)
    assert r.sheet(580, 625) > 1e13


def test_dd_generation_and_srh_match_the_diffusion_profile():
    tau, NA, G, Lb = 1e-8, 1e17, 1e20, 100000.0
    tr = Transport(1000.0, 400.0, 0.81, reference="test values",
                   tau_n=tau, tau_p=tau)
    L = [StackLayer(_TM, Lb, acceptors=NA * 1e-21)]
    kw = dict(top=Contact("ohmic"), dz=100.0, statistics="boltzmann")
    r0 = solve_drift_diffusion(L, tr, **kw)
    r = solve_drift_diffusion(L, tr, generation=G, **kw)
    Ln = math.sqrt(1000 * KB * 300 * tau) * 1e7
    ana = G * tau * (1 - np.cosh((r.z - Lb / 2) / Ln) / np.cosh(Lb / (2 * Ln)))
    i = slice(50, -50)
    assert np.max(np.abs((r.n - r0.n)[i] / ana[i] - 1)) < 1e-4
    assert abs(r.current_density) < 1e-12


def test_dd_iv_curve_and_numpy_fallback():
    L = [StackLayer(_TM, 400.0, donors=1e-4), StackLayer(_TM, 400.0, acceptors=1e-4)]
    V, J, res = iv_curve(L, _TT, [0.3, 0.35], top=Contact("ohmic"),
                         statistics="boltzmann", dz=8.0)
    assert J[1] < J[0] < 0
    r = solve_drift_diffusion(L, _TT, top=Contact("ohmic", voltage=0.3),
                              statistics="boltzmann", dz=8.0, use_scipy=False)
    assert r.current_density == pytest.approx(J[0], rel=1e-8)
    assert res[1].top.voltage == 0.35


def test_dd_refusals():
    with pytest.raises(ValueError, match="reference"):
        Transport(1000.0, 30.0, 1.0, reference="")
    with pytest.raises(ValueError, match="both SRH"):
        Transport(1000.0, 30.0, 1.0, reference="t", tau_n=1e-9)
    with pytest.raises(ValueError):
        Contact("schottky")
    with pytest.raises(ValueError):
        Contact("ohmic", barrier=1.0)
    L = [StackLayer(algan(0.0), 100.0, acceptors=1e-5)]
    with pytest.raises(ValueError, match="no holes"):
        solve_electrons(L, phi_B=1.0)
    with pytest.raises(ValueError, match="no conduction-band edge"):
        solve_drift_diffusion([StackLayer(gan("ambacher2021"), 50.0),
                               StackLayer(scaln(0.1), 10.0)],
                              _TT, top=Contact("schottky", barrier=1.0))


# ------------------------------------------------------------------- app
from kpenvelope.gui import api as gui_api  # noqa: E402

_HEMT = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
         "layers": [{"material": "GaN", "thickness": 1000},
                    {"material": "AlGaN", "x": 0.25, "thickness": 25}],
         "surface_barrier": 1.0}


def test_app_charges_and_electrons_equal_the_library():
    r = gui_api.handle("charges", _HEMT)
    L = [StackLayer(algan(0.0), 1000.0), StackLayer(algan(0.25), 25.0)]
    assert r["interfaces"][0]["charge_cm2"] == pytest.approx(
        interface_charges(L)[0].sigma_cm2, rel=1e-12)
    assert r["interfaces"][0]["meaning"] == "attracts electrons"
    e = gui_api.handle("electrons", _HEMT)
    ref = solve_electrons(L, phi_B=1.0, dz=0.2, quantum_window=(905.0, 1025.0),
                          classical_outside=True)
    assert e["sheet_cm2"] == pytest.approx(ref.sheet_quantum_cm2, rel=1e-9)


def test_app_script_reproduces_the_app(tmp_path):
    spec = dict(_HEMT, layers=[{"material": "GaN", "thickness": 200},
                               {"material": "AlGaN", "x": 0.25, "thickness": 25}])
    code = gui_api.handle("code", spec)["code"]
    ns = {}
    exec(compile(code.replace("print(", "(lambda *a: None)("), "app", "exec"), ns)
    assert ns["r"].sheet_quantum_cm2 == pytest.approx(
        gui_api.handle("electrons", spec)["sheet_cm2"], rel=1e-12)


def test_app_errors_are_plain_sentences():
    bad = gui_api.handle("charges", {"layers": [{"material": "InGaN", "x": 0.1,
                                                 "thickness": 5}]})
    assert "has no InGaN" in bad["error"]
    sc = {"parameter_set": "ambacher2021", "surface_barrier": 1.0,
          "layers": [{"material": "GaN", "thickness": 500},
                     {"material": "ScAlN", "x": 0.18, "thickness": 10}]}
    assert "ScAlN has no band gap" in gui_api.handle("electrons", sc)["error"]
    no_src = dict(sc, band_numbers={"ScAlN": {"eg": 5, "ec": 1, "me": 0.3}})
    assert "come from" in gui_api.handle("electrons", no_src)["error"]
    relaxed = {"layers": [{"material": "GaN", "thickness": 100},
                          {"material": "AlGaN", "x": 0.3, "thickness": 30,
                           "relaxation": 40}],
               "line_charge": {"mode": "custom", "value": ""}}
    assert "line charge" in gui_api.handle("charges", relaxed)["error"]
    assert "thickness" in gui_api.handle(
        "charges", {"layers": [{"material": "GaN", "thickness": "x"}]})["error"]
    assert gui_api.handle("device", dict(_HEMT, device={
        "mode": "iv", "v_start": 0, "v_stop": 1, "v_step": 0.5,
        "barrier": 1}))["error"].startswith("Please say where")


def test_app_cv_and_line_charge_and_plan():
    r = gui_api.handle("device", dict(_HEMT, device={
        "mode": "cv", "barrier": 1.0, "v_start": -2, "v_stop": 0,
        "v_step": 1, "bottom": "thick"}))
    assert len(r["c_nF_cm2"]) == 3 and all(c > 0 for c in r["c_nF_cm2"])
    lc = gui_api.handle("line_charge", {"constraints": ["shen-density-pbesol",
                                                        "shen-mobility"]})
    post = combine_line_charge(["shen-density-pbesol", "shen-mobility"])
    assert lc["mean"] == post.mean and lc["sd"] == post.sd
    pl = gui_api.handle("plan", {"plan": {"material": "AlGaN", "x": 0.25,
                                          "relaxation": 50,
                                          "charge_error": 1e-3}})
    g, f = algan(0.0), algan(0.25)
    eps = (g.a - f.a) / f.a
    assert pl["holography_sigma"] == pytest.approx(
        plan_holography(f, eps, 0.5, 1e-3), rel=1e-12)


def test_app_uncertainty_linear_budget_sums_to_one():
    r = gui_api.handle("uncertainty", dict(_HEMT, uncertainty={
        "barrier": 0.1, "layers": [{}, {"x": 0.01, "thickness": 1}]}))
    assert r["sigma_cm2"] > 0
    assert sum(b["share"] for b in r["budget"]) == pytest.approx(1.0)


def test_app_server_needs_its_token():
    import json
    import threading
    import urllib.error
    import urllib.request
    from kpenvelope.gui import make_server

    srv, tok = make_server(0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert tok in page and "__KPE_TOKEN__" not in page
        body = json.dumps(_HEMT).encode()
        req = urllib.request.Request(base + "/api/charges", data=body,
                                     headers={"X-Token": tok,
                                              "Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req).read())
        assert out["interfaces"][0]["charge_cm2"] > 1e13
        bad = urllib.request.Request(base + "/api/charges", data=body,
                                     headers={"X-Token": "wrong"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------- phonons
from kpenvelope import (CM1_TO_MEV, GAN_BARKER1973, PolarPhonons,  # noqa: E402
                        frohlich_constant, single_interface_modes, slab_modes)


def test_gan_phonons_are_the_listed_numbers_and_pass_lst():
    g = GAN_BARKER1973
    assert CM1_TO_MEV == pytest.approx(0.12398419843320026, rel=1e-15)
    assert g.w_LO_perp / CM1_TO_MEV == pytest.approx(746.0)
    assert g.w_LO_par / CM1_TO_MEV == pytest.approx(744.0)
    assert g.w_TO_perp / CM1_TO_MEV == pytest.approx(559.0)
    assert g.w_TO_par / CM1_TO_MEV == pytest.approx(533.0)
    assert g.eps_inf_perp == 5.35
    assert g.eps0_par == pytest.approx(10.4, rel=1e-14)
    # an independent check of the copied numbers: LST with the E1 set
    assert g.eps0_perp == pytest.approx(9.5, rel=5e-3)
    assert g.eps_perp(0.0) == pytest.approx(g.eps0_perp)


def test_slab_modes_are_the_fuchs_kliewer_closed_form():
    iso = PolarPhonons("iso", 70.0, 90.0, 70.0, 90.0, 5.0, 5.0, reference="test")
    d = 2.0
    qs = np.array([0.01, 0.3, 1.0, 50.0])
    r = slab_modes(iso, 1.0, d, qs)
    assert len(r["even"]) == 1 and len(r["odd"]) == 1
    for k, fn in (("even", lambda x: 1 / np.tanh(x)), ("odd", np.tanh)):
        c = fn(qs * d / 2)
        exact = np.sqrt((5 * 90.0 ** 2 + c * 70.0 ** 2) / (5 + c))
        assert np.allclose(r[k][0], exact, rtol=1e-9)
    # thin slab: even -> TO, odd -> LO; thick slab: both -> eps = -1
    assert r["even"][0][0] == pytest.approx(70.0, rel=2e-2)
    assert r["odd"][0][0] == pytest.approx(90.0, rel=1e-3)
    ws = math.sqrt((5 * 90.0 ** 2 + 70.0 ** 2) / 6)
    assert r["even"][0][-1] == pytest.approx(ws, rel=1e-9)
    assert r["odd"][0][-1] == pytest.approx(ws, rel=1e-9)
    assert single_interface_modes(iso, 1.0)[0] == pytest.approx(ws, rel=1e-9)


def test_isotropic_interface_is_the_quadratic():
    a = PolarPhonons("a", 60.0, 80.0, 60.0, 80.0, 5.0, 5.0, reference="test")
    b = PolarPhonons("b", 75.0, 100.0, 75.0, 100.0, 4.0, 4.0, reference="test")
    # 5 (x - 80^2)(x - 75^2) + 4 (x - 100^2)(x - 60^2) = 0
    c = [9.0, -(5 * (80 ** 2 + 75 ** 2) + 4 * (100 ** 2 + 60 ** 2)),
         5 * 80 ** 2 * 75 ** 2 + 4 * 100 ** 2 * 60 ** 2]
    exact = sorted(math.sqrt(x) for x in np.roots(c).real if x > 0)
    got = sorted(single_interface_modes(a, b))
    keep = [w for w in exact if a.eps_perp(w) * b.eps_perp(w) < 0]
    assert np.allclose(got, keep, rtol=1e-9)


def test_wurtzite_gan_surface_mode_solves_eps_perp_eps_par_equal_one():
    g = GAN_BARKER1973
    w = single_interface_modes(g, 1.0)
    assert len(w) == 1
    A = g.eps_inf_perp * g.eps_inf_par
    L1, L2 = g.w_LO_perp ** 2, g.w_LO_par ** 2
    T1, T2 = g.w_TO_perp ** 2, g.w_TO_par ** 2
    xs = np.roots([A - 1, -A * (L1 + L2) + (T1 + T2), A * L1 * L2 - T1 * T2])
    ok = [math.sqrt(x) for x in xs.real if x > 0
          and g.eps_par(math.sqrt(x)) < 0 and g.eps_perp(math.sqrt(x)) < 0]
    assert w[0] == pytest.approx(ok[0], rel=1e-10)


def test_frohlich_constant_two_forms():
    m, w, ei, e0 = 0.2, 92.0, 5.35, 9.5
    e, hbar, eps0 = 1.602176634e-19, 1.054571817e-34, 8.8541878128e-12
    om = w * 1e-3 * e / hbar
    alt = (e * e / (8 * math.pi * eps0 * hbar * om)
           * math.sqrt(2 * m * 9.1093837015e-31 * om / hbar) * (1 / ei - 1 / e0))
    assert frohlich_constant(m, w, ei, e0) == pytest.approx(alt, rel=1e-12)
    with pytest.raises(ValueError):
        frohlich_constant(0.2, 92.0, 9.5, 5.35)
    with pytest.raises(ValueError):
        PolarPhonons("x", 90.0, 70.0, 70.0, 90.0, 5.0, 5.0, reference="t")


def test_quantum_cv_charge_and_sheets():
    from kpenvelope import cv_curve_quantum
    L = [StackLayer(algan(0.0), 300.0), StackLayer(algan(0.25), 25.0)]
    V, C, Q, res = cv_curve_quantum(L, [-3.0, -1.0], barrier=1.0)
    for v, r in zip(V, res):
        e = solve_electrons(L, phi_B=1.0 - v, dz=0.2,
                            quantum_window=(205.0, 325.0), classical_outside=True)
        assert r.sheet_quantum_cm2 == pytest.approx(e.sheet_quantum_cm2, rel=1e-12)
    # the metal's charge moves by q times the change of all the electrons
    ns = [r.sheet_quantum_cm2 + r.sheet_classical_cm2 for r in res]
    assert (Q[1] - Q[0]) == pytest.approx(_Q * (ns[1] - ns[0]), rel=2e-3)
    eps_d = (0.25 * 10.7 + 0.75 * 10.4) * _EPS0 / 25e-9 * 1e-4
    assert np.all(C > 0.85 * eps_d) and np.all(C < eps_d)
    # within a few percent of the classical quasi-equilibrium C-V
    Vc, Cc, Qc, _ = cv_curve(L, [-3.0, -1.0], top=Contact("schottky", barrier=1.0),
                             bottom=Contact("zero-field"), dz=0.2)
    assert np.allclose(C, Cc, rtol=0.03)
    r = gui_api.handle("device", dict(_HEMT, layers=[
        {"material": "GaN", "thickness": 300}, {"material": "AlGaN", "x": 0.25,
                                                "thickness": 25}],
        device={"mode": "cv", "barrier": 1.0, "v_start": -1, "v_stop": -1,
                "v_step": 1, "bottom": "thick", "quantum": True}))
    assert r["quantum"] and r["c_nF_cm2"][0] == pytest.approx(C[1] * 1e9, rel=1e-9)
