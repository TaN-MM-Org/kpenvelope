"""Checks of the features new in 0.33.0: viscoelastic oxide and elastic
masks, negative ions in the plasma model."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])


# ------------------------------------------------- viscoelastic flow
def test_maxwell_flow_against_exact_results():
    from kpenvelope.process import Structure
    # a uniform swelling layer across the whole width: no sideways motion,
    # tau_zz builds up as 2 eta q (1 - chi^n) and sigma_xx = -tau_zz
    st = Structure(20.0, 0.0, 60.0, 2.0, "O", 40.0)
    ox = st.mat == "O"
    q = np.where(ox, 1e-3, 0.0)
    eta, G, dt = 1e12, 1e10, 50.0
    chi = eta / (eta + G * dt)
    hist = None
    for n in range(1, 6):
        r = st.stokes({"O": eta}, source=q, shear_modulus={"O": G}, dt=dt, history=hist)
        hist = r["tau"]
        want = -2 * eta * 1e-3 * (1 - chi ** n)
        assert np.max(np.abs(r["sxx"][ox] / want - 1)) < 1e-10
    # an elastic solid (no viscosity): the stress grows as 2 G q t
    hist = None
    for n in range(1, 4):
        r = st.stokes({}, source=q, shear_modulus={"O": G}, dt=dt, history=hist)
        hist = r["tau"]
        assert np.max(np.abs(r["sxx"][ox] / (-2 * G * 1e-3 * dt * n) - 1)) < 1e-10
    # a step of oxide with a steady source: the Maxwell stress settles on the
    # viscous one (it checks the shear memory at the corners and edges)
    st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
    st.etch({"Si": 1.0}, 40.0, mode="directional", window=(40, 80))
    st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d")
    ox = st.mat == "O"
    q = np.where(ox, 1e-3 * (1 + np.random.default_rng(0).random(ox.shape)), 0.0)
    v = st.stokes({"O": eta}, source=q)
    hist = None
    for _ in range(60):
        r = st.stokes({"O": eta}, source=q, shear_modulus={"O": G}, dt=100.0, history=hist)
        hist = r["tau"]
    for k in ("sxx", "szz", "sxz"):
        assert np.nanmax(np.abs(r[k][ox] - v[k][ox])) < 1e-10 * np.nanmax(np.abs(v[k][ox]))
    with pytest.raises(ValueError, match="dt"):
        st.stokes({"O": eta}, source=q, shear_modulus={"O": G})


def test_viscoelastic_oxidation():
    from kpenvelope.process import Structure

    def run(**kw):
        st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
        st.etch({"Si": 1.0}, 40.0, mode="directional", window=(40, 80))
        st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d",
                   viscosity={"O": 1e12}, stress_volume_nm3=0.025, temperature_K=1273.0, **kw)
        ox = st.mat == "O"
        S = st.oxide_stress
        return st.area("O"), min(np.nanmin(S["sxx"][ox]), np.nanmin(S["szz"][ox]))
    a, sa = run()
    b, sb = run(shear_modulus={"O": 1e30})          # no elasticity left
    assert a == b and sb == pytest.approx(sa, rel=1e-9)
    c, sc = run(shear_modulus={"O": 3e10})
    assert abs(sc) < abs(sa)                        # the stress has to build up
    with pytest.raises(ValueError, match="viscosity"):
        Structure(40.0, 0.0, 40.0, 2.0, "Si", 20.0).oxidize(
            "Si", "O", 0.5, 0.05, 10.0, 0.44, "t", mode="2d", shear_modulus={"O": 1e10})


def test_app_elastic_mask():
    from kpenvelope.gui import api
    from kpenvelope.process import Structure
    rec = ("deposit material=SiO2 thickness=4 mode=directional\n"
           "deposit material=Si3N4 thickness=10 mode=directional from=0 to=80\n"
           "oxidize material=Si oxide=SiO2 B=0.5 BA=0.05 time=1500 consumed=0.44 mode=2d "
           "viscosity=1e12 modulus=3e10 modulus_Si3N4=1e11")
    spec = dict(width=160, height=160, cell=2, substrate="Si", substrate_top=100,
                source="t", recipe=rec)
    r = api.run_process({"process": spec})
    st = Structure(160.0, 0.0, 160.0, 2.0, "Si", 100.0)
    st.deposit("SiO2", 4.0, mode="directional")
    st.deposit("Si3N4", 10.0, mode="directional", window=(0, 80))
    st.oxidize("Si", "SiO2", 0.5, 0.05, 1500.0, 0.44, "t", mode="2d",
               viscosity={"SiO2": 1e12}, shear_modulus={"SiO2": 3e10, "Si3N4": 1e11})
    s = st.oxide_stress["sxx"][st.mat == "Si3N4"]
    assert r["solid_stress_GPa"]["Si3N4"] == pytest.approx([np.nanmin(s) / 1e9,
                                                            np.nanmax(s) / 1e9], rel=1e-12)
    assert np.nanmin(s) < 0 < np.nanmax(s)          # the mask bends
    spec["recipe"] = "oxidize material=Si oxide=SiO2 B=0.5 BA=0.05 time=10 modulus=3e10"
    assert "viscosity" in api.handle("process", {"process": spec})["error"]


# ---------------------------------------------- negative ions in a plasma
def test_plasma_with_negative_ions():
    from scipy.optimize import fsolve
    from kpenvelope.plasma import AMU, ME, QE, SCCM, GlobalPlasma, Reaction, Species
    ill = "illustrative"
    A = Species("A", 40.0)
    Ap = Species("A+", 40.0, 1, wall_products={"A": 1})
    Am = Species("A-", 40.0, -1)
    rx = [Reaction(("e", "A"), ("e", "e", "A+"), 2.3e-14, ill, b=0.59, C=17.44, energy_eV=15.76),
          Reaction(("e", "A"), ("A-",), 1e-16, ill),
          Reaction(("A-", "A+"), ("A", "A"), 1e-13, ill),
          Reaction(("e", "A"), ("e", "A"), 2.3e-14, ill, b=1.6, elastic=True)]
    Q, S, P, h = 20 * SCCM, 0.05, 500.0, 0.3
    pl = GlobalPlasma([A, Ap, Am], rx, 0.15, 0.3, P, {"A": Q}, S, h, ill)
    r = pl.solve()
    V, Aw, M = pl.V, pl.A, 40 * AMU

    # the balances written out by hand: the negative ions never reach the
    # wall, the electrons are the positive ions less the negative ones
    def eqs(y):
        ng, npl, nm, T = np.exp(y)
        ne = npl - nm
        uB = math.sqrt(QE * T / M)
        Riz = 2.3e-14 * T ** 0.59 * math.exp(-17.44 / T) * ne * ng
        Ra, Rr = 1e-16 * ne * ng, 1e-13 * nm * npl
        Rel = 2.3e-14 * T ** 1.6 * ne * ng
        wall = h * uB * npl * Aw / V
        sc = Q / V
        return [(Riz - Rr - npl * S / V - wall) / sc, (Ra - Rr - nm * S / V) / sc,
                (Q / V - ng * S / V - Riz - Ra + 2 * Rr + wall) / sc,
                (P / (QE * V) - Riz * 15.76 - Rel * 3 * ME / M * T
                 - wall * (2.5 * T + 0.5 * T * math.log(M / (2 * math.pi * ME))))
                / (P / (QE * V))]
    d = r["densities"]
    got = np.array([d["A"], d["A+"], d["A-"], r["Te_eV"]])
    hand = np.exp(fsolve(eqs, np.log(got) + 0.05, xtol=1e-14))
    assert np.max(np.abs(got / hand - 1)) < 1e-9
    assert r["ne"] == pytest.approx(d["A+"] - d["A-"], rel=1e-12)
    assert "A-" not in r["ion_flux"] and r["wall_loss"]["A-"] == 0.0
    # strong attachment, reached by starting each solve from the last
    prev = r
    for k_a in (3e-16, 1e-15):
        rx[1] = Reaction(("e", "A"), ("A-",), k_a, ill)
        prev = GlobalPlasma([A, Ap, Am], rx, 0.15, 0.3, P, {"A": Q}, S, h, ill).solve(
            guess=prev)
        assert prev["residual"] < 1e-9 and prev["ne"] > 0
    assert prev["densities"]["A-"] / prev["ne"] > 10       # mostly negative ions
