"""Checks of the features new in 0.35.0: phonons of air, alloys and
boundaries anywhere; masks that ride on the flowing oxide, and the
oxide's exact balance; the plasma's chemistry in the app."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

_GAN = (0.0916, 9.5, 5.35)
_BAR = (0.112, 8.5, 4.6)
_ALLOY = {"eps_high": 4.8, "lo_eV": [0.088, 0.110], "to_eV": [0.070, 0.095]}


def _profiles():
    z = np.linspace(-20, 20, 401)
    w = np.full(z.size, z[1] - z[0])
    w[[0, -1]] *= 0.5
    A = np.stack([np.exp(-((z + 5) / 3) ** 2), np.exp(-((z - 2) / 2) ** 2) * (z - 2)], 1)
    return z, w, A


# ------------------------------------------- phonons: air, alloys, anywhere
def test_layer_modes_of_air_alloys_and_boundaries_anywhere():
    from kpenvelope.scattering import (_eps_static, _layer_solve, _mat, _pair,
                                       _sheet_transforms, interface_modes, layer_modes)
    z, w, A = _profiles()
    q = 0.3
    aw = A * w[:, None]

    def H(a):
        return np.array([_pair(z, np.array([q]), a[:, k], a[:, k])[0] for k in range(2)])

    def half(zi, side, weight):
        a = aw * side[:, None]
        Fs = _sheet_transforms(z, np.array([q]), a, np.array([zi]))[0, :, 0]
        return (H(a) - Fs ** 2) * weight

    scale = float(np.max(np.abs(H(aw))))

    def rel(G, want):
        # each pair's weight against its closed form (where the closed form
        # is zero, as for a material without charge on its side, against
        # the bulk weight)
        nz = want != 0
        e = np.abs(G[nz] / want[nz] - 1).max(initial=0.0)
        return float(max(e, np.abs(G[~nz]).max(initial=0.0) / scale))
    # GaN under air: the surface phonon eps(w) = -1 in closed form, energy
    # and weight 4 F^2 / (w deps/dw), and GaN's half-space LO; the surface
    # between grid points and outside the grid on either side
    L, es, ei = _GAN
    T = L * math.sqrt(ei / es)
    ws = math.sqrt((ei * L * L + T * T) / (ei + 1))
    for zi in (0.05, 25.0, -23.3):
        modes = layer_modes(z, w, A, [zi], [_GAN, 1.0], q)
        assert len(modes) == 2
        F = (aw * np.exp(-q * np.abs(z - zi))[:, None]).sum(0)
        for p, G in modes:
            if abs(p - L) < 1e-12:
                assert rel(G, half(zi, z < zi, 1 / ei - 1 / es)) < 1e-6      # 2e-8
            else:
                assert p == pytest.approx(ws, rel=1e-12)
                de = 2 * ws * ei * (L * L - T * T) / (ws * ws - T * T) ** 2
                assert rel(G, F ** 2 * 4 / (ws * de)) < 1e-6   # 2e-8, 1.1e-7 with old NumPy
    # two polar materials with the boundary between grid points and outside
    for zi in (0.05, 25.0, -23.3):
        F = (aw * np.exp(-q * np.abs(z - zi))[:, None]).sum(0)
        for p, G in layer_modes(z, w, A, [zi], [_GAN, _BAR], q):
            if min(abs(p - _GAN[0]), abs(p - _BAR[0])) < 1e-12:
                m_ = _GAN if abs(p - _GAN[0]) < 1e-12 else _BAR
                want = half(zi, (z < zi) if m_ is _GAN else (z > zi), 1 / m_[2] - 1 / m_[1])
            else:
                want = F ** 2 * [c for w_, c in interface_modes(_GAN, _BAR)
                                 if abs(w_ - p) < 1e-9][0]
            assert rel(G, want) < 1e-6                                      # 1.3e-7
    # an alloy with two oscillators in bulk: a mode at each LO with the
    # weight H (2 / LO_k) Res(1 / eps) in closed form
    Ls, Ts = _ALLOY["lo_eV"], _ALLOY["to_eV"]
    modes = layer_modes(z, w, A, [0.0], [_ALLOY, _ALLOY], q)
    assert len(modes) == 2
    for p, G in modes:
        k = [i for i, l_ in enumerate(Ls) if abs(l_ - p) < 1e-9][0]
        res = (np.prod([Ls[k] ** 2 - t * t for t in Ts]) / _ALLOY["eps_high"]
               / (2 * Ls[k] * np.prod([Ls[k] ** 2 - l_ * l_ for i, l_ in enumerate(Ls) if i != k])))
        assert rel(G, H(aw) * (2 / Ls[k]) * res) < 1e-10                    # 5e-14
    # the sum rule with alloys, air and boundaries between grid points
    for zi, mats in (([0.0], [_ALLOY, _GAN]), ([-2.0, 2.0], [_GAN, _ALLOY, _BAR]),
                     ([-1.0, 1.5, 6.0], [_GAN, _ALLOY, _GAN, 1.0]),
                     ([0.07, 3.0], [_ALLOY, _BAR, 1.0])):
        zg = np.sort(np.concatenate([z, [v for v in zi if np.min(np.abs(z - v)) > 1e-9]]))
        r = np.zeros((zg.size, 2), complex)
        r[np.searchsorted(zg, z)] = aw
        idx = np.searchsorted(np.asarray(zi), 0.5 * (zg[1:] + zg[:-1]))
        for qq in (0.01, 0.3, 3.0):
            tot = sum(G / (2 * qq) for _, G in layer_modes(z, w, A, zi, mats, qq))
            F = [np.real((r * _layer_solve(zg, None, idx, [f(m) for m in mats], qq, 0.5, r)).sum(0))
                 for f in (_eps_static, lambda m: _mat(m)[1])]
            assert np.max(np.abs(tot / (F[1] - F[0]) - 1)) < 1e-6           # 1.6e-7
    with pytest.raises(ValueError, match="interlace"):
        _mat({"eps_high": 5.0, "lo_eV": [0.09, 0.11], "to_eV": [0.095, 0.10]})


def test_layered_phonons_with_air_and_alloys_in_the_mobility():
    from test_v029 import _three
    sb = _three(300.0, 5e12)
    bulk = sb.mobility(sb.polar_optical(*_GAN))["mobility_cm2Vs"]
    # air 500 nm beyond the grid: bulk
    far = sb.mobility(sb.layered_phonons([540.0], [_GAN, 1.0]))["mobility_cm2Vs"]
    assert far == pytest.approx(bulk, rel=1e-5)
    # a second oscillator with no strength: the one-oscillator result
    L, es, ei = _BAR
    T = L * math.sqrt(ei / es)
    weak = {"eps_high": ei, "lo_eV": [L, 0.2 + 1e-9], "to_eV": [T, 0.2]}
    one = sb.mobility(*sb.interface_phonons(30.0, _GAN, _BAR))["mobility_cm2Vs"]
    two = sb.mobility(sb.layered_phonons([30.0], [_GAN, weak]))["mobility_cm2Vs"]
    assert two == pytest.approx(one, rel=1e-5)
    # air on a 10 nm barrier: at q = 1 / nm its surface mode is the
    # barrier's eps(w) = -1 (the boundaries 10 nm apart hardly feel each
    # other), and the electrons, 10 nm below, hardly feel it
    from kpenvelope.scattering import _layer_roots
    ws = math.sqrt((ei * L * L + T * T) / (ei + 1))
    roots = _layer_roots(1.0, [30.0, 40.0], [_GAN, _BAR, 1.0])
    assert min(abs(r_ / ws - 1) for r_ in roots) < 1e-7
    air = sb.mobility(sb.layered_phonons([30.0, 40.0], [_GAN, _BAR, 1.0]))["mobility_cm2Vs"]
    assert air == pytest.approx(one, rel=1e-4)                              # 1.5e-6
    with pytest.raises(ValueError, match="no layer has phonons"):
        sb.layered_phonons([30.0], [1.0, 4.0])


# ----------------------------------------- masks that ride on the oxide
def _locos(shape, **kw):
    from kpenvelope.process import Structure
    st = Structure(160.0, 0.0, 160.0, 2.0, "Si", 100.0)
    st.deposit("SiO2", 4.0, mode="directional")
    st.deposit("Si3N4", 10.0, mode="directional", window=(40, 120))
    st.oxidize("Si", "SiO2", 0.5, 0.05, 1500.0, 0.44, "t", mode="2d", shape=shape, **kw)
    return st


def test_masks_ride_on_the_flowing_oxide_and_the_oxide_is_kept():
    def bottom(st):
        nit = st.mat == "Si3N4"
        return np.array([np.min(np.nonzero(nit[i])[0]) if nit[i].any() else -1
                         for i in range(nit.shape[0])])
    runs = {"volume": _locos("volume", viscosity={"SiO2": 1e12}),
            "rigid": _locos("flow", viscosity={"SiO2": 1e12}),
            "flows": _locos("flow", viscosity={"SiO2": 1e12, "Si3N4": 1e13}),
            "springy": _locos("flow", viscosity={"SiO2": 1e12},
                              shear_modulus={"SiO2": 3e10, "Si3N4": 1e11})}
    for st in runs.values():
        b = st.oxidation_balance
        # every bit of oxide made is in whole cells or still in parts of
        # cells; the end rounding is less than a cell per surface cell
        assert b["whole_cells_nm2"] + b["parts_nm2"] == pytest.approx(b["made_nm2"], rel=1e-9)
        assert abs(b["rounding_nm2"]) < b["parts_nm2"]
    start = 52                                       # the nitride's bottom cell at first
    assert np.all(bottom(runs["rigid"])[20:60] == start)            # a rigid mask stays
    for key in ("flows", "springy"):
        b = bottom(runs[key])[20:60]
        assert b[0] > start and b[-1] > start        # lifted at both edges,
        assert np.all(b[12:28] == start)             # not in the middle,
        assert np.array_equal(b, b[::-1])            # and the same on both sides
    with pytest.raises(ValueError, match="viscosity"):
        _locos("flow")


def test_app_masks_that_flow():
    from kpenvelope.gui import api
    rec = ("deposit material=SiO2 thickness=4 mode=directional\n"
           "deposit material=Si3N4 thickness=10 mode=directional from=40 to=120\n"
           "oxidize material=Si oxide=SiO2 B=0.5 BA=0.05 time=1500 consumed=0.44 mode=2d "
           "viscosity=1e12 viscosity_Si3N4=1e13 shape=flow")
    spec = dict(width=160, height=160, cell=2, substrate="Si", substrate_top=100,
                source="t", recipe=rec)
    r = api.run_process({"process": spec})
    st = _locos("flow", viscosity={"SiO2": 1e12, "Si3N4": 1e13})
    assert r["areas"] == {m: st.area(m) for m in r["areas"]}
    spec["recipe"] = rec.replace("viscosity_Si3N4", "viscosity_SiO2")
    assert "other solids" in api.handle("process", {"process": spec})["error"]
    spec["recipe"] = ("oxidize material=Si oxide=SiO2 B=0.5 BA=0.05 time=10 "
                      "viscosity_Si3N4=1e13")
    assert "need viscosity" in api.handle("process", {"process": spec})["error"]


# ------------------------------------------------- the app: phonons
def test_app_alloy_barrier_and_the_air_above():
    from kpenvelope.gui.api import handle
    gan = {"material": "GaN", "x": 0, "relaxation": 0}
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [dict(gan, thickness=1000, doping="n", doping_cm3=1e16),
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0}],
            "surface_barrier": 1.0}
    bar = {"lo_meV": 105, "eps_static": 9.0, "eps_high": 5.0, "source": "test"}

    def mu(**kw):
        return handle("mobility", dict(spec, scattering={"interface_phonons": dict(bar, **kw)}))
    one = mu()["mobility_cm2Vs"]
    # the air 25 nm above the electrons hardly matters
    assert mu(all_layers=True)["mobility_cm2Vs"] == pytest.approx(one, rel=1e-4)   # 1e-5
    # a second vibration with no strength changes nothing
    assert mu(lo2_meV=110.001, to2_meV=110.0)["mobility_cm2Vs"] == pytest.approx(one, rel=1e-5)
    two = mu(lo_meV=95, lo2_meV=110, to2_meV=100)
    assert "mobility_cm2Vs" in two and len(two["each"]) == 1
    assert "do not fit together" in mu(lo2_meV=110, to2_meV=60)["error"]


# --------------------------------------------- the app: plasma chemistry
def test_app_plasma_chemistry_feeds_the_etch():
    from kpenvelope.gui import api
    from kpenvelope.plasma import SCCM, GlobalPlasma, Reaction, Species, fluxes_cm2
    from kpenvelope.process import EtchKinetics, Structure
    chem = ("species A2 mass=70.9\n"
            "species A mass=35.45 wall=0.1 returns=A2:0.5\n"
            "species A2+ mass=70.9 charge=1 returns=A2\n"
            "reaction e + A2 -> e + e + A2+ A=1e-14 b=0.5 C=12 energy=11.5 source=ill\n"
            "reaction e + A2 -> e + A + A A=5e-14 C=3 energy=2.5 source=ill\n"
            "reaction e + A2 -> e + A2 A=3e-14 b=1 elastic source=ill\n"
            "chamber radius=0.15 length=0.3 power=800 pump=0.05 h=0.3 h_source=ill h\n"
            "feed A2=30\n")
    rec = ("deposit material=M thickness=20 mode=directional from=0 to=30\n"
           "deposit material=M thickness=20 mode=directional from=50 to=80\n"
           "plasmaetch fluxes=plasma neutral=A ion=A2+ stick_Si=0.2 yield_Si=5 "
           "atoms_Si=5e22 time=5")
    spec = dict(width=80, height=140, cell=2, substrate="Si", substrate_top=100,
                source="t", recipe=rec, plasma=chem)
    r = api.handle("process", {"process": spec})
    sp = [Species("A2", 70.9), Species("A", 35.45, wall=0.1, wall_products={"A2": 0.5}),
          Species("A2+", 70.9, 1, wall_products={"A2": 1})]
    rx = [Reaction(("e", "A2"), ("e", "e", "A2+"), 1e-14, "ill", b=0.5, C=12.0, energy_eV=11.5),
          Reaction(("e", "A2"), ("e", "A", "A"), 5e-14, "ill", C=3.0, energy_eV=2.5),
          Reaction(("e", "A2"), ("e", "A2"), 3e-14, "ill", b=1.0, elastic=True)]
    res = GlobalPlasma(sp, rx, 0.15, 0.3, 800.0, {"A2": 30 * SCCM}, 0.05, 0.3, "ill h").solve()
    nf, jf = fluxes_cm2(res, "A", "A2+")
    st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
    st.deposit("M", 20.0, mode="directional", window=(0, 30))
    st.deposit("M", 20.0, mode="directional", window=(50, 80))
    st.plasma_etch(EtchKinetics(nf, jf, 1e15, {"Si": 0.2}, {"Si": 5.0}, {"Si": 5e22}, "t"), 5.0)
    assert r["plasma"]["Te_eV"] == res["Te_eV"]
    assert r["plasma"]["neutral_flux_cm2"]["A"] == nf
    assert r["areas"]["Si"] == st.area("Si")
    assert "ill h" in r["sources"]
    bad = chem.replace("source=ill\nreaction e + A2 -> e + A + A", "\nreaction e + A2 -> e + A + A")
    assert "where the rate comes from" in api.handle("process", {"process": dict(spec, plasma=bad)})["error"]
    assert "needs the plasma" in api.handle("process", {"process": dict(spec, plasma="")})["error"]
    assert "becomes at the wall" in api.handle(
        "process", {"process": dict(spec, plasma=chem.replace("charge=1 returns=A2", "charge=1"))})["error"]
