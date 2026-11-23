"""Checks of the features new in 0.34.0: polar phonons of many layers, and
oxide whose shape follows its flow."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

_GAN = (0.0916, 9.5, 5.35)
_BAR = (0.112, 8.5, 4.6)


def _profiles():
    z = np.linspace(-20, 20, 401)
    w = np.full(z.size, z[1] - z[0])
    w[[0, -1]] *= 0.5
    A = np.stack([np.exp(-((z + 5) / 3) ** 2), np.exp(-((z - 2) / 2) ** 2) * (z - 2)], 1)
    return z, w, A


# ------------------------------------------------- phonons of many layers
def test_layer_modes_against_closed_forms_and_the_sum_rule():
    from kpenvelope.phonons import PolarPhonons, slab_modes
    from kpenvelope.scattering import (_layer_roots, _pair, _sheet_transforms,
                                       interface_modes, layer_modes)
    z, w, A = _profiles()
    q = 0.3
    H = np.array([_pair(z, np.array([q]), A[:, k] * w, A[:, k] * w)[0] for k in range(2)])
    # one material: the bulk LO phonon with H (1/eps_high - 1/eps_static)
    (p, G), = [m for m in layer_modes(z, w, A, [0.0], [_GAN, _GAN], q)]
    assert p == pytest.approx(_GAN[0], rel=1e-12)
    assert np.max(np.abs(G / (H * (1 / _GAN[2] - 1 / _GAN[1])) - 1)) < 1e-9
    # one interface: the half-space and interface modes in closed form
    modes = layer_modes(z, w, A, [0.0], [_GAN, _BAR], q)
    F = (A * w[:, None] * np.exp(-q * np.abs(z))[:, None]).sum(0)
    for p, G in modes:
        if min(abs(p - _GAN[0]), abs(p - _BAR[0])) < 1e-12:
            m_ = _GAN if abs(p - _GAN[0]) < 1e-12 else _BAR
            side = (z < 0) if m_ is _GAN else (z > 0)
            a = A * (w * side)[:, None]
            Fs = _sheet_transforms(z, np.array([q]), a, np.array([0.0]))[0, :, 0]
            want = (np.array([_pair(z, np.array([q]), a[:, k], a[:, k])[0] for k in range(2)])
                    - Fs ** 2) * (1 / m_[2] - 1 / m_[1])
        else:
            C = [c for w_, c in interface_modes(_GAN, _BAR) if abs(w_ - p) < 1e-9][0]
            want = F ** 2 * C
        assert np.max(np.abs(G / want - 1)) < 1e-7
    # a slab of the barrier in GaN: its interface modes are those of the
    # phonons module (isotropic records), at every q
    def rec(m):
        L, es, ei = m
        T = L * math.sqrt(ei / es)
        return PolarPhonons("t", T * 1e3, L * 1e3, T * 1e3, L * 1e3, ei, ei, "t")
    for qq in (0.05, 0.5, 2.0):
        sm = slab_modes(rec(_BAR), rec(_GAN), 3.0, qq)
        ref = sorted(float(b[0]) / 1e3 for k in ("even", "odd") for b in sm[k]
                     if np.isfinite(b[0]))
        got = _layer_roots(qq, [-1.5, 1.5], [_GAN, _BAR, _GAN])
        assert len(got) == len(ref)
        assert np.max(np.abs(np.array(got) - np.array(ref))) < 1e-9
    # the sum rule: sum_p (2 / w_p) Res_p F = F(eps_high) - F(eps_static)
    # (every mode found, with its weight), for three and four layers
    from kpenvelope.scattering import _layer_solve
    for zi, mats in (([-2.0, 2.0], [_GAN, _BAR, _GAN]), ([-1.0, 1.5, 6.0], [_GAN, _BAR, _GAN, _BAR]),
                     ([0.0, 0.5], [_GAN, _BAR, _GAN])):
        idx = np.searchsorted(np.asarray(zi), 0.5 * (z[1:] + z[:-1]))
        rhs = (A * w[:, None]).astype(complex)
        for qq in (0.03, 0.3, 2.0):
            tot = sum(G / (2 * qq) for _, G in layer_modes(z, w, A, zi, mats, qq))

            def F_static(k):
                st = [m[1 + k] for m in mats]     # constant eps
                phi = _layer_solve(z, w, idx, st, qq, 0.5, rhs)
                return np.real((rhs * phi).sum(0))
            assert np.max(np.abs(tot / (F_static(1) - F_static(0)) - 1)) < 1e-7  # 5e-9


def test_layered_phonons_in_the_mobility():
    from test_v029 import _three
    sb = _three(300.0, 5e12)
    bulk = sb.mobility(sb.polar_optical(*_GAN))["mobility_cm2Vs"]
    one = sb.mobility(sb.layered_phonons([30.0], [_GAN, _GAN]))["mobility_cm2Vs"]
    assert one == pytest.approx(bulk, rel=1e-5)             # measured 9e-7
    iface = sb.mobility(*sb.interface_phonons(30.0, _GAN, _BAR))["mobility_cm2Vs"]
    two = sb.mobility(sb.layered_phonons([30.0], [_GAN, _BAR]))["mobility_cm2Vs"]
    assert two == pytest.approx(iface, rel=1e-5)            # measured 1e-6
    # a barrier 0.5 nm thin in GaN hardly changes anything
    thin = sb.mobility(sb.layered_phonons([30.0, 30.5], [_GAN, _BAR, _GAN]))
    assert thin["mobility_cm2Vs"] == pytest.approx(bulk, rel=1e-3)
    with pytest.raises(ValueError, match="increasing"):
        sb.layered_phonons([30.0, 30.0], [_GAN, _BAR, _GAN])


# ---------------------------------------------- oxide shaped by its flow
def test_oxide_shape_follows_the_flow():
    from kpenvelope.process import Structure

    def flat(shape):
        st = Structure(20.0, 0.0, 100.0, 2.0, "Si", 60.0)
        st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d",
                   viscosity={"O": 1e12}, shape=shape)
        return st.area("O"), st.thickness("O", 10.0)
    assert flat("flow") == flat("volume")                    # a flat surface
    st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
    st.etch({"Si": 1.0}, 40.0, mode="directional", window=(20, 60))
    si0 = st.area("Si")
    st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d",
               viscosity={"O": 1e12}, shape="flow")
    ox = st.mat == "O"
    assert np.array_equal(ox, ox[::-1])                      # a symmetric trench
    # the new oxide's volume: what the silicon gave, over 0.44, to a few cells
    assert st.area("O") == pytest.approx((si0 - st.area("Si")) / 0.44, rel=0.02)
    with pytest.raises(ValueError, match="viscosity"):
        Structure(20.0, 0.0, 40.0, 2.0, "Si", 20.0).oxidize(
            "Si", "O", 0.5, 0.05, 10.0, 0.44, "t", mode="2d", shape="flow")


def test_app_flowing_oxide_shape():
    from kpenvelope.gui import api
    from kpenvelope.process import Structure
    rec = ("etch Si=1 time=40 mode=directional from=20 to=60\n"
           "oxidize material=Si oxide=O B=0.5 BA=0.05 time=2000 consumed=0.44 mode=2d "
           "viscosity=1e12 shape=flow")
    spec = dict(width=80, height=140, cell=2, substrate="Si", substrate_top=100,
                source="t", recipe=rec)
    r = api.run_process({"process": spec})
    st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
    st.etch({"Si": 1.0}, 40.0, mode="directional", window=(20, 60))
    st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d", viscosity={"O": 1e12},
               shape="flow")
    assert r["areas"]["O"] == st.area("O")
    spec["recipe"] = "oxidize material=Si oxide=O B=0.5 BA=0.05 time=10 shape=flow"
    assert "viscosity" in api.handle("process", {"process": spec})["error"]


# ------------------------------------- thick layers at large q, and the app
def test_layer_modes_of_thick_layers_keep_the_sum_rule():
    # a GaN channel, a 1 nm spacer, a 24 nm barrier and a GaN cap: at large
    # q the modes of separate boundaries come close in energy (an earlier
    # scan for the roots lost some and was 17 percent off the sum rule;
    # the cosh of a thick layer also overflowed)
    from kpenvelope.scattering import _layer_solve, layer_modes
    bar = (0.105, 9.0, 5.0)
    z = np.arange(-94.0, 27.0001, 0.2)
    w = np.full(z.size, 0.2)
    A = np.stack([np.exp(-((z + 3) / 2) ** 2), np.exp(-((z + 5) / 3) ** 2) * (z + 5)], 1)
    rhs = (A * w[:, None]).astype(complex)
    for zi, mats in (([0.0, 1.0, 25.0], [_GAN, _BAR, bar, _GAN]), ([0.0, 25.0], [_GAN, bar, _GAN])):
        idx = np.searchsorted(np.asarray(zi), 0.5 * (z[1:] + z[:-1]))
        for qq in np.exp(np.linspace(math.log(1e-4), math.log(30.0), 16)):
            tot = sum(G / (2 * qq) for _, G in layer_modes(z, w, A, zi, mats, qq))
            F = [np.real((rhs * _layer_solve(z, w, idx, [m[1 + k] for m in mats],
                                             qq, 0.5, rhs)).sum(0)) for k in (0, 1)]
            assert np.max(np.abs(tot / (F[1] - F[0]) - 1)) < 1e-6   # measured 2e-7


def test_app_every_layer_of_the_exact_region():
    from kpenvelope.gui.api import handle
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    gan = {"material": "GaN", "x": 0, "relaxation": 0}
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [dict(gan, thickness=1000, doping="n", doping_cm3=1e16),
                       {"material": "AlGaN", "x": 1.0, "thickness": 1, "relaxation": 0},
                       {"material": "AlGaN", "x": 0.25, "thickness": 24, "relaxation": 0},
                       dict(gan, thickness=2)],
            "surface_barrier": 1.0}
    bar = {"lo_meV": 105, "eps_static": 9.0, "eps_high": 5.0, "source": "test"}
    G = {"spacer_lo_meV": ph.w_LO_perp, "spacer_eps_static": ph.eps0_perp,
         "spacer_eps_high": ph.eps_inf_perp, "spacer_source": "GaN's"}

    def mu(**kw):
        return handle("mobility", dict(spec, scattering={"interface_phonons": dict(bar, **kw)}))
    bulk = handle("mobility", dict(spec, scattering={"polar": True}))["mobility_cm2Vs"]
    # GaN's numbers everywhere: bulk GaN
    same = mu(all_layers=True, lo_meV=ph.w_LO_perp, eps_static=ph.eps0_perp,
              eps_high=ph.eps_inf_perp)
    assert same["mobility_cm2Vs"] == pytest.approx(bulk, rel=1e-5)
    # spacer numbers equal to the barrier's: the same as without them
    a = mu(all_layers=True)
    b = mu(all_layers=True, spacer_lo_meV=105, spacer_eps_static=9.0, spacer_eps_high=5.0,
           spacer_source="test")
    assert a["mobility_cm2Vs"] == b["mobility_cm2Vs"] and len(a["each"]) == 1
    # the cap 25 nm away hardly matters: as the boundary next to the electrons alone
    one = mu()
    assert len(one["each"]) == 4
    assert a["mobility_cm2Vs"] == pytest.approx(one["mobility_cm2Vs"], rel=1e-5)  # 3e-7
    c = mu(all_layers=True, spacer_lo_meV=112, spacer_eps_static=8.5, spacer_eps_high=4.6,
           spacer_source="t2")
    assert c["mobility_cm2Vs"] != a["mobility_cm2Vs"] and "t2" in c["each"][0]["name"]
    assert "ticked" in mu(**G)["error"]
    assert "come from" in mu(all_layers=True, spacer_lo_meV=112)["error"]
