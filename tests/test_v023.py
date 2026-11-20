"""Tests for v0.23.0: Peltier and Thomson heat, band-to-band tunneling
together with heating, hot electrons and a set lattice temperature,
nonlocal band-to-band tunneling on meshes, cited gap shifts for alloys,
and the app's new actions."""
import dataclasses
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan, inaln, ingan  # noqa: E402
from kpenvelope.device import (BandToBand, Device, Electrode, Mobility, Surface,  # noqa: E402
                               Varshni)
from kpenvelope.electrons import KB  # noqa: E402
from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_grid, mesh_from_points  # noqa: E402
from kpenvelope.nitride import gap_shift  # noqa: E402
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_v022 import _esaki, _no_pol  # noqa: E402

Q = 1.602176634e-19


def _bar(heat_model, temperature_model="mobility", L=2000.0, nd=1e18):
    z = np.linspace(0, L, 201)
    return Device([StackLayer(algan(0.0), L, donors=nd * 1e-21)], 20.0,
                  [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")],
                  Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0., 10., 20.]), z=z), statistics="boltzmann",
                  heat_model=heat_model, temperature_model=temperature_model)


# ------------------------------------------------ Peltier and Thomson heat
def test_peltier_heat_at_the_contacts_and_in_total():
    a, b = _bar("joule"), _bar("thermoelectric")
    sa, sb = a.solve({"bot": 0.5}), b.solve({"bot": 0.5})
    # it only moves heat: the total is the electrical power in both models
    assert sa.heat.sum() == pytest.approx(sa.power, rel=1e-12)
    assert sb.heat.sum() == pytest.approx(sb.power, rel=1e-12)
    assert sb.current["bot"] == sa.current["bot"]
    d = sb.heat - sa.heat
    # uniform bar at one temperature: no Thomson heat inside; the Peltier heat
    # Pi I / q with Pi = kT (5/2 + ln(N_c / n)) is released where the electrons
    # leave (the anode, bottom here) and taken where they enter
    Pi = KB * 300 * (2.5 + np.log(b.Nc[0] * 1e21 / 1e18))
    I = abs(sa.current["bot"])
    assert d[:, :3].sum() == pytest.approx(Pi * I, rel=1e-9)
    assert d[:, -3:].sum() == pytest.approx(-Pi * I, rel=1e-9)
    assert np.abs(d[:, 5:-5]).max() < 1e-12 * Pi * I


def test_thomson_heat_along_a_temperature_gradient():
    # with T linear along the bar and n = N_d: -Gamma . grad Pi per node, with
    # d Pi / dz = k (dT/dz) (5/2 + ln(N_c(T) / N_d) + 3/2)
    L, nd = 2000.0, 1e18
    c = _bar("thermoelectric", "full")
    j = _bar("joule", "full")
    z = c.axes[-1]
    T = np.tile(300 + 100 * z / L, 3)
    sc = c.solve({"bot": 0.5}, lattice_temperature=T)
    sj = j.solve({"bot": 0.5}, lattice_temperature=T)
    d = (sc.heat - sj.heat)[1]
    I = abs(sc.current["bot"])
    for k in (50, 100, 150):
        Tk = 300 + 100 * z[k] / L
        Nck = c._lat["Nc"][0] * (Tk / 300) ** 1.5 * 1e21
        dPi = KB * (100 / L) * (2.5 + np.log(Nck / nd) + 1.5)
        assert d[k] == pytest.approx(I / 20.0 * dPi * 10.0 * 10.0, rel=1e-6)
    assert sc.heat.sum() == pytest.approx(sc.power, rel=1e-12)


# --------------------------------- band-to-band tunneling with heating
def _tunnel_diode(T=300.0, full=False, shift=False):
    V = Varshni(4.73e-4, 636.0, 0.5, "t")
    m = dataclasses.replace(algan(0.0), name="p", eg=0.66, me=0.04, eps_r=16.0)
    if shift:
        d = float(V.shift(T, 300.0))
        m = dataclasses.replace(m, eg=m.eg + d, ec=m.ec + 0.5 * d)
    m2 = dataclasses.replace(m, name="n")
    layers = [StackLayer(m, 100.0, acceptors=3e-2), StackLayer(m2, 100.0, donors=1e-2)]
    kw = dict(temperature_model="full", band_gap_temperature=V) if full else {}
    return Device(layers, 20.0, [Electrode("top", "ohmic", 0, 20),
                                 Electrode("bot", "ohmic", 0, 20, face="bottom")],
                  Mobility(1000.0, "t", temperature_exponent=1.5), Surface("pinned", barrier=0.5),
                  grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 200, 401)),
                  extra_sheet=_no_pol(layers), hole_mass=0.3, hole_mobility=Mobility(100.0, "t"),
                  band_to_band=BandToBand("t", reduced_mass_m0=0.02, model="nonlocal"),
                  temperature_K=T, **kw)


def test_band_to_band_at_a_set_lattice_temperature():
    full, iso = _tunnel_diode(full=True), _tunnel_diode(360.0, shift=True)
    for V in (-0.1, 0.1):
        a = iso.solve({"bot": V})
        b = full.solve({"bot": V}, lattice_temperature=360.0)
        assert b.current["bot"] == pytest.approx(a.current["bot"], rel=1e-8)


def test_band_to_band_with_self_heating_and_hot_electrons():
    d = _tunnel_diode(full=True)
    th = ThermalModel(ThermalMaterial("d", 10.0, 0.0, "t"),
                      substrate=(ThermalLayer("s", 1e4, ThermalMaterial("s", 50.0, 0.0, "t")),))
    st = d.solve({"bot": -0.15}, thermal=th)
    assert st.temperature.max() > 320
    assert st.heat.sum() == pytest.approx(st.power, rel=1e-10)
    # the result is a fixed point: solving at its temperature gives its current,
    # and its heat gives its temperature
    chk = d.solve({"bot": -0.15}, lattice_temperature=st.temperature.ravel())
    assert chk.current["bot"] == pytest.approx(st.current["bot"], rel=1e-6)
    Tn = d._temperature(d._thermal_setup(th), st.heat, st.temperature.ravel())
    assert np.max(np.abs(Tn - st.temperature.ravel())) < 0.01
    # hot electrons that relax at once change nothing
    e = _tunnel_diode()
    a = e.solve({"bot": 0.1})
    h = e.solve({"bot": 0.1}, energy_relaxation=1e-16)
    assert h.current["bot"] == pytest.approx(a.current["bot"], rel=1e-5)


# ------------------------------------- nonlocal tunneling on meshes
def test_nonlocal_tunneling_on_a_mesh_equals_the_tensor_grid():
    T = _esaki()
    x, z = T.axes
    m = dataclasses.replace(algan(0.0), name="p", eg=0.66, me=0.04, eps_r=16.0)
    m2 = dataclasses.replace(m, name="n")
    mesh = mesh_from_grid(x, z, lambda xx, zz: "p" if zz < 100 else "n")
    P = mesh.points
    con = {"top": np.flatnonzero(np.isclose(P[:, 1], 200)),
           "bot": np.flatnonzero(np.isclose(P[:, 1], 0))}
    R = {"p": MeshRegion(material=m, mobility=Mobility(1000.0, "t"), acceptors=3e-2,
                         hole_mobility=Mobility(100.0, "t")),
         "n": MeshRegion(material=m2, mobility=Mobility(1000.0, "t"), donors=1e-2,
                         hole_mobility=Mobility(100.0, "t"))}
    bt = BandToBand("test", reduced_mass_m0=0.02, model="nonlocal", path_spacing=10.0)
    M = MeshDevice(mesh, R, [Electrode("top", "ohmic", 0, 1), Electrode("bot", "ohmic", 0, 1)],
                   con, hole_mass=0.3, band_to_band=bt)
    assert np.allclose(M._bt["area"], T._bt["area"])
    for V in (-0.1, 0.1):
        a, c = T.solve({"bot": V}), M.solve({"bot": V})
        assert c.current["bot"] == pytest.approx(a.current["bot"], rel=1e-9)


def test_nonlocal_tunneling_on_an_unstructured_mesh_in_any_direction():
    # a uniform field along the path direction, full supply: Kane's rate
    # integrated over each line's tunneling region, whatever the mesh
    m = dataclasses.replace(algan(0.0), name="nar", eg=0.66, me=0.04, eps_r=16.0)
    W, H = 300.0, 200.0
    rng = np.random.default_rng(3)
    xb, zb = np.linspace(0, W, 61), np.linspace(0, H, 41)
    bnd = np.unique(np.r_[np.c_[xb, 0 * xb], np.c_[xb, H + 0 * xb], np.c_[0 * zb, zb],
                          np.c_[W + 0 * zb, zb]], axis=0)
    X, Z = np.meshgrid(xb[1:-1], zb[1:-1], indexing="ij")
    inner = np.c_[X.ravel(), Z.ravel()] + rng.uniform(-0.3, 0.3, (X.size, 2)) * 5.0
    mesh = mesh_from_points(np.r_[bnd, inner], lambda x, z: "g")
    P = mesh.points
    con = {"a": np.flatnonzero(np.isclose(P[:, 1], H)), "b": np.flatnonzero(np.isclose(P[:, 1], 0))}
    F = 0.01
    G0 = BandToBand.rate(F * 1e7, 0.66, 0.02)[0] * 1e-21
    for d in ((0.0, 1.0), (1.0, 0.0), (0.6, 0.8)):
        bt = BandToBand("t", reduced_mass_m0=0.02, model="nonlocal", direction=d, path_spacing=4.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M = MeshDevice(mesh, {"g": MeshRegion(material=m, mobility=Mobility(1000.0, "t"),
                                                  donors=1e-3, hole_mobility=Mobility(100.0, "t"))},
                           [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)], con,
                           hole_mass=0.3, band_to_band=bt)
        dd = np.array(d) / np.hypot(*d)
        Xs = np.zeros((M.N, M.nv))
        Xs[:, 0] = F * (P @ dd)
        Xs[:, 1], Xs[:, 2] = -10.0, 10.0
        Gn, Gp = M._nonlocal_rates(Xs.ravel())
        exp = sum(L_["area"] * G0 * max(F * np.ptp(L_["s"][L_["ok_s"]]) - 0.66, 0.0) / F
                  for L_ in M._bt["lines"])
        assert float(np.sum(Gn * M.vol)) == pytest.approx(exp, rel=1e-7)
        assert float(np.sum(Gp * M.vol)) == pytest.approx(exp, rel=1e-7)


# ------------------------------------------------ cited gap shifts
def test_gap_shift_is_what_the_alloy_records_give():
    for name, f, x in (("AlGaN", algan, 0.25), ("InGaN", ingan, 0.15), ("InAlN", inaln, 0.18)):
        gs = gap_shift(name, x)
        for T in (250.0, 400.0, 600.0):
            assert gs.shift(T, 300.0) == pytest.approx(f(x, T).eg - f(x, 300.0).eg, abs=1e-12)
    with pytest.raises(ValueError, match="ScAlN"):
        gap_shift("ScAlN", 0.1)


# ------------------------------------------------------------ the app
def _app_spec():
    return {"parameter_set": "pbesol-mahim2026", "temperature": 300, "surface_barrier": 1.0,
            "layers": [{"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                        "doping": "none"},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0,
                        "doping": "none"}],
            "device": {"barrier": 1.0},
            "transistor": {"gate_length": 100, "gap_sg": 200, "gap_gd": 300, "contact": 100,
                           "heating": True, "substrate": "SiC", "substrate_um": 100,
                           "pitch_um": 50, "sweep": "drain", "fixed": 0, "from": 2, "to": 4,
                           "step": 2}}


def test_app_transistor_options_and_runaway():
    from kpenvelope.gui.api import handle
    s = _app_spec()
    s["transistor"].update(full_temp=True, peltier=True, thermionic=True, tau_e=0.47,
                           hot_coupled=True)
    r = handle("transistor", s)
    assert "error" not in r, r.get("error")
    assert r["drain"][-1] > 0 and r["tmax"][-1] > 300 and r["hot"]["coupled"]
    assert any("Varshni" in x for x in r["sources"]) and any("Wachutka" in x for x in r["sources"])
    s2 = _app_spec()
    s2["runaway"] = {"gate": 0.0, "to": 4, "step": 2}
    r = handle("runaway", s2)
    assert "error" not in r, r.get("error")
    assert r["runaway"] is None and r["stable"] == 4.0
    s3 = _app_spec()
    s3["layers"][1] = {"material": "ScAlN", "x": 0.1, "thickness": 10, "relaxation": 0,
                       "doping": "none"}
    s3["parameter_set"] = "ambacher2021"
    s3["transistor"]["full_temp"] = True
    assert "error" in handle("transistor", s3)


def test_app_tunnel_junction_and_vertical_current():
    from kpenvelope.driftdiffusion import Contact
    from kpenvelope.gui.api import _charges, _vertical_device, build_layers, handle
    spec = {"parameter_set": "hse-dreyer2016", "temperature": 300,
            "layers": [{"material": "InGaN", "x": 1.0, "thickness": 100, "relaxation": 0,
                        "doping": "p", "doping_cm3": 3e19},
                       {"material": "InGaN", "x": 1.0, "thickness": 100, "relaxation": 0,
                        "doping": "n", "doping_cm3": 1e19}],
            "transport": {"mobility_n": 1000, "mobility_p": 50, "hole_mass": 0.5,
                          "source": "illustrative"},
            "device": {"mode": "iv", "contact": "ohmic", "v_start": 0.0, "v_stop": -0.4,
                       "v_step": 0.1, "dz": 0.5, "btbt": True}}
    r = handle("device", spec)
    assert "error" not in r, r.get("error")
    J = dict(zip([round(v, 2) for v in r["v"]], r["j_A_cm2"]))
    assert J[-0.1] > 10 * J[-0.3] > 0                # the tunnel diode's dip
    # the vertical device without tunneling is the 1D solver's diode
    spec2 = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
             "layers": [{"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                         "doping": "p", "doping_cm3": 1e18},
                        {"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                         "doping": "n", "doping_cm3": 1e18}],
             "transport": {"mobility_n": 300, "mobility_p": 10, "hole_mass": 1.0,
                           "source": "illustrative", "lifetime": 1e-9},
             "device": {"mode": "iv", "contact": "ohmic", "v_start": -3.0, "v_stop": -3.4,
                        "v_step": 0.2, "dz": 1.0}}
    one = handle("device", spec2)
    layers = build_layers(spec2)
    ic, _ = _charges(layers, spec2)
    d = _vertical_device(spec2, layers, ic, 300.0, 1.0, Contact("ohmic"), False)
    prev = None
    for v, j in zip(one["v"], one["j_A_cm2"]):
        prev = d.solve({"top": v}, guess=prev)
        assert -prev.current["top"] / (20e-7 * 0.1) == pytest.approx(j, rel=1e-4)


def test_app_process_steps():
    from kpenvelope.gui.api import handle
    p = {"source": "illustrative", "width": 1000, "height": 400, "cell": 5, "substrate": "GaN",
         "substrate_top": 300,
         "recipe": "deposit material=SiN thickness=50 mode=conformal\n"
                   "etch SiN=1 time=60 mode=directional from=100 to=900  # open a window\n"
                   "implant species=donor dose=1e13 range=30 straggle=15 from=100 to=900\n"
                   "anneal time=60 temperature=1300 D0=1e-4 Ea=3"}
    r = handle("process", {"process": p})
    assert "error" not in r, r.get("error")
    assert r["materials"] == ["GaN", "SiN"] and r["areas"]["SiN"] == pytest.approx(2 * 100 * 50)
    assert 0.9 * 0.8e13 < r["donor_dose"] < 0.8e13      # the window's dose, less what the mask took
    o = handle("process", {"process": dict(p, substrate="Si",
                                           recipe="oxidize material=Si oxide=SiO2 B=2.89 "
                                                  "BA=0.0209 time=7200 consumed=0.44")})
    A = 2.89 / 0.0209
    x = 0.5 * A * (np.sqrt(1 + 4 * 2.89 * 7200 / A ** 2) - 1)
    assert o["oxide_exact"] == pytest.approx(x, rel=1e-12)
    for bad in ("deposit SiN 50", "grow material=SiN thickness=5", "etch time=5"):
        e = handle("process", {"process": dict(p, recipe=bad)})
        assert "error" in e and "Step 1" in e["error"]
    assert "error" in handle("process", {"process": dict(p, source="")})


def test_thermal_runaway_stops_once_too_hot():
    b = _bar("joule", L=500.0)
    th = ThermalModel(ThermalMaterial("d", 10.0, 0.0, "t"),
                      substrate=(ThermalLayer("s", 1e4, ThermalMaterial("s", 200.0, 0.0, "t")),))
    full = b.thermal_runaway("bot", [0.05, 0.1, 0.15, 0.2], th)
    T = full["hottest"]
    assert full["too_hot"] is None and np.all(np.diff(T) > 0)
    stop = 0.5 * (T[1] + T[2])
    r = b.thermal_runaway("bot", [0.05, 0.1, 0.15, 0.2], th, stop_above=stop)
    assert r["too_hot"] == 0.15 and r["runaway"] is None
    assert np.allclose(r["hottest"], T[:3], rtol=1e-9)
    with pytest.raises(ValueError, match="stop_above"):
        b.thermal_runaway("bot", [0.1], th, stop_above=-1.0)
