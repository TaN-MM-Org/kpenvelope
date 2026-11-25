"""Checks of the features new in 0.37.0: polar phonons of uniaxial
(wurtzite) layers, whose vibrations differ along and across the c axis,
found per bin of energy by contour integrals; in the mobility and the app."""

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


def test_bins_against_poles_closed_forms_and_the_sum_rule():
    from scipy.integrate import quad
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    from kpenvelope.scattering import _eps2, _layer_solve, layer_modes
    z, w, A = _profiles()
    aw = A * w[:, None]
    # isotropic stacks: the bins hold the weight of the exact poles, at
    # their mean energy
    for zi, mats in (([0.0], [_GAN, _BAR]), ([-2.0, 2.0], [_GAN, _BAR, _GAN]),
                     ([0.05, 3.0], [_GAN, _BAR, 1.0])):
        for q in (0.03, 0.3, 3.0):
            bb = layer_modes(z, w, A, zi, mats, q, bins=True)
            pp = layer_modes(z, w, A, zi, mats, q)
            Gb, Gp = sum(g for _, g in bb), sum(g for _, g in pp)
            assert np.max(np.abs(Gb / Gp - 1)) < 1e-5                       # 8e-7
            Mb, Mp = sum(e * g for e, g in bb), sum(e * g for e, g in pp)
            assert np.max(np.abs(Mb / Gb - Mp / Gp)) < 5e-4                 # 0.17 meV
    # bulk GaN, uniaxial: the weight of the exact Coulomb potential
    # exp(-kappa |z - z'|) / (2 eps_par kappa), kappa = q sqrt(eps_perp /
    # eps_par), in its band of TO-like modes (which only the anisotropy
    # couples) and, at small q, in its band of LO modes; and at every q the
    # sum rule with eps_perp and eps_par
    D = np.abs(z[:, None] - z[None, :])

    def dens(om, q, p):
        ep, ez = _eps2(ph, om + 1e-13j)
        kap = q * np.sqrt(ep / ez)
        K = np.exp(-kap * D) / (2 * ez * kap)
        return -np.imag(aw[:, p] @ K @ aw[:, p]) / om / math.pi * 4 * q

    def band(q, p, a_, b_):
        return quad(lambda x: dens(x, q, p) * math.sqrt(max((x - a_) * (b_ - x), 0.0)),
                    a_, b_, weight="alg", wvar=(-0.5, -0.5), limit=400, epsrel=1e-11)[0]
    TO = (ph.w_TO_par / 1000, ph.w_TO_perp / 1000)
    LO = (ph.w_LO_par / 1000, ph.w_LO_perp / 1000)
    zero = np.zeros(z.size - 1, int)
    for q in (0.05, 0.5, 3.0):
        bb = layer_modes(z, w, A, [0.0], [ph, ph], q)
        to_b = sum(g for e, g in bb if e < 0.08)
        to_c = np.array([band(q, p, *TO) for p in range(2)])
        assert np.max(np.abs(to_b / to_c - 1)) < 2e-3                       # 1e-3 at q = 3
        tot = sum(g for _, g in bb) / (2 * q)
        F = [np.real((aw * _layer_solve(z, w, zero, [{"perp": a, "par": b}], q, 0.5,
                                        aw.astype(complex))).sum(0))
             for a, b in ((ph.eps0_perp, ph.eps_inf_par * (LO[0] / TO[0]) ** 2),
                          (ph.eps_inf_perp, ph.eps_inf_par))]
        assert np.max(np.abs(tot / (F[1] - F[0]) - 1)) < 1e-6               # 7e-7
        if q < 0.1:
            lo_b = sum(g for e, g in bb if e > 0.08)
            lo_c = np.array([band(q, p, *LO) for p in range(2)])
            assert np.max(np.abs(lo_b / lo_c - 1)) < 1e-3                   # 2.4e-4


def test_uniaxial_gan_in_the_mobility():
    from test_v029 import _three
    import kpenvelope.scattering as S
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    sb = _three(300.0, 5e12)
    iso = (ph.w_LO_perp / 1000, ph.eps0_perp, ph.eps_inf_perp)
    bulk = sb.mobility(sb.polar_optical(*iso))["mobility_cm2Vs"]
    orig = S.layer_modes
    S.layer_modes = lambda *a, **k: orig(*a, bins=True, **k)
    try:
        by_bins = sb.mobility(sb.layered_phonons([30.0], [iso, iso]))["mobility_cm2Vs"]
    finally:
        S.layer_modes = orig
    assert by_bins == pytest.approx(bulk, rel=1e-5)                        # 1.9e-7
    uni = sb.mobility(sb.layered_phonons([30.0], [ph, ph]))["mobility_cm2Vs"]
    # GaN couples a little more strongly along c (1/eps_inf - 1/eps_0 is
    # 0.091 there, 0.082 across): the mobility is lower, by 3.6 percent here
    assert 0.94 < uni / bulk < 0.99
    from kpenvelope.scattering import _mat
    with pytest.raises(ValueError, match="one isotropic"):
        _mat({"perp": ph, "par": _GAN})


def test_app_uniaxial_gan():
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
    uni = mu(anisotropic=True)
    assert 0.95 < uni["mobility_cm2Vs"] / one < 0.995                       # 0.978
    assert "along and across" in uni["each"][0]["name"]
