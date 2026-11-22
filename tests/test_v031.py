"""Checks of the features new in 0.31.0."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

_GAN = (0.0916, 9.5, 5.35)          # hbar w_LO (eV), eps_static, eps_high
_BAR = (0.112, 8.5, 4.6)            # an illustrative barrier


def _eps(m, w, g=0.0):
    L, es, ei = m
    T2 = L * L * ei / es
    return ei * (w * w - L * L + 1j * g * w) / (w * w - T2 + 1j * g * w)


# ------------------------------------------------- interface phonon modes
def test_interface_modes_and_their_weights():
    from scipy.integrate import quad
    from kpenvelope.phonons import PolarPhonons, single_interface_modes
    from kpenvelope.scattering import interface_modes
    modes = interface_modes(_GAN, _BAR)

    def rec(m):
        L, es, ei = m
        T = L * math.sqrt(ei / es)
        return PolarPhonons("t", T * 1e3, L * 1e3, T * 1e3, L * 1e3, ei, ei, "t")
    other = single_interface_modes(rec(_GAN), rec(_BAR))
    assert len(modes) == len(other) == 2
    for (w, _), w2 in zip(modes, other):
        assert w == pytest.approx(w2 / 1e3, rel=1e-9)
    # each weight is the area of its peak in -Im 2 / (eps_1 + eps_2) with
    # a small damping, times 2 / (pi w)
    for w0, C in modes:
        val = quad(lambda w: -(2 / (_eps(_GAN, w, 1e-5) + _eps(_BAR, w, 1e-5))).imag
                   * 2 / (math.pi * w), w0 - 0.004, w0 + 0.004, points=[w0], limit=500,
                   epsabs=0)[0]
        assert val == pytest.approx(C, rel=1.5e-3)
    # the same material on both sides: one mode at w_LO with the bulk coupling
    (w, C), = interface_modes(_GAN, _GAN)
    assert w == pytest.approx(_GAN[0], rel=1e-12)
    assert C == pytest.approx(1 / _GAN[2] - 1 / _GAN[1], rel=1e-12)


def test_modes_add_up_to_the_layered_coulomb_potential():
    """The sum over the modes of coupling x kernel against the potential of
    a point charge solved on a grid with damped permittivities, -Im of it
    integrated over the frequency with weight 2 / (pi w)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from scipy.integrate import quad
    from kpenvelope.scattering import interface_modes
    modes = interface_modes(_GAN, _BAR)
    q, z1, z2, g, h, Lb = 0.2, -3.0, 2.0, 1e-4, 0.1, 150.0
    z = np.arange(-Lb, Lb + h / 2, h)
    k1, k2 = int(round((z1 + Lb) / h)), int(round((z2 + Lb) / h))

    def phi(w):
        ea, eb = _eps(_GAN, w, g), _eps(_BAR, w, g)
        e = np.where(np.abs(z) < h / 2, 0.5 * (ea + eb), np.where(z < 0, ea, eb))
        ef = np.where(0.5 * (z[:-1] + z[1:]) < 0, ea, eb)
        main = np.ones(z.size, complex)
        main[1:-1] = (ef[:-1] + ef[1:]) / h ** 2 + q * q * e[1:-1]
        lower = -ef / h ** 2
        upper = -ef / h ** 2
        lower[-1] = 0.0
        upper[0] = 0.0
        M = sp.diags([main, lower, upper], [0, -1, 1], format="csc")
        b = np.zeros(z.size, complex)
        b[k2] = 1 / h
        return spl.spsolve(M, b)[k1]
    poles = sorted([_GAN[0], _BAR[0]] + [w for w, _ in modes])
    edges = sorted([0.03, 0.2] + [p - 0.01 for p in poles] + [p + 0.01 for p in poles])
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        pts = [p for p in poles if a < p < b]
        tot += quad(lambda w: -phi(w).imag * 2 / (math.pi * w), a, b, points=pts or None,
                    limit=400, epsabs=1e-10, epsrel=1e-7)[0]
    # z1 and z2 are on opposite sides: only the interface modes couple them
    want = sum(C for _, C in modes) * math.exp(-q * (abs(z1) + abs(z2))) / (2 * q)
    assert tot == pytest.approx(want, rel=1e-3)


def _three(T=300.0, sheet=5e12):
    from test_v029 import _three as t3
    return t3(T, sheet)


def test_interface_phonons_in_the_mobility():
    sb = _three()
    bulk = sb.mobility(sb.polar_optical(*_GAN))["mobility_cm2Vs"]
    # the same material on both sides: the modes add up to the bulk phonon
    same = sb.mobility(*sb.interface_phonons(30.0, _GAN, _GAN))["mobility_cm2Vs"]
    assert same == pytest.approx(bulk, rel=1e-10)
    # an interface far away: only the bulk-like mode of the electrons' side
    far = sb.mobility(*sb.interface_phonons(-200.0, _BAR, _GAN))
    assert far["mobility_cm2Vs"] == pytest.approx(bulk, rel=1e-8)
    # a real heterointerface at the well's edge: detailed balance on the
    # grid, and finer grids agree
    modes = sb.interface_phonons(30.0, _GAN, _BAR)
    assert len(modes) == 4
    r = sb.mobility(*modes)
    M, Em = sb._last_system
    M = M.toarray()
    f = 0.5 * (1 - np.tanh(0.5 * (Em - sb.fermi) / sb.kT))
    S = M * (f * (1 - f))[None, :]
    assert np.max(np.abs(S - S.T)) < 1e-10 * np.max(np.abs(S))
    fine = sb.mobility(*modes, n_energy=4 * int(round(_GAN[0] / r["energy_step_eV"])))
    assert r["mobility_cm2Vs"] == pytest.approx(fine["mobility_cm2Vs"], rel=1e-3)
    assert r["mobility_cm2Vs"] < min(r["each"])
    with pytest.raises(ValueError, match="eps_static"):
        sb.interface_phonons(30.0, _GAN, (0.1, 4.0, 5.0))


# ------------------------------------------------------------ the app
def test_app_interface_vibrations_and_level_changes():
    from test_v020 import _spec
    from kpenvelope.gui import api
    from kpenvelope.gui.api import handle
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [{"material": "GaN", "x": 0, "thickness": 1000, "relaxation": 0,
                        "doping": "n", "doping_cm3": 1e16},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0}],
            "surface_barrier": 1.0}
    bulk = handle("mobility", dict(spec, scattering={"polar": True}))["mobility_cm2Vs"]
    # GaN's own numbers on the barrier side: the modes add up to bulk GaN
    same = handle("mobility", dict(spec, scattering={"interface_phonons": {
        "lo_meV": ph.w_LO_perp, "eps_static": ph.eps0_perp, "eps_high": ph.eps_inf_perp,
        "source": "test"}}))
    assert same["mobility_cm2Vs"] == pytest.approx(bulk, rel=1e-10)
    assert same["interface"] and len(same["each"]) == 3
    assert "come from" in handle("mobility", dict(spec, scattering={"interface_phonons": {
        "lo_meV": 112, "eps_static": 8.5, "eps_high": 4.6}}))["error"]
    # the wave current with electrons changing level: the library's coupled
    # mode space with four modes
    S = dict(contact="50", gap_sg="35", gate_length="30", gap_gd="35", mobility="1500")
    r = api.run_quantum(_spec(transistor=dict(S), quantum_transport={
        "gate": -1, "drain": 0.05, "mixing": True}))
    dev, _ = api._transistor_device(_spec(transistor=dict(S, quantum=True)))
    b = {"gate": -1.0, "drain": 0.05}
    st = dev.negf(b, guess=dev.solve(b), dx=0.5, coupled_modes=4)
    assert r["negf"] == pytest.approx(st.negf["current"], rel=1e-12)
    assert r["mixing"] and r["subbands"] == 4
    with pytest.raises(api.SpecError, match="without bumping"):
        api.run_quantum(_spec(transistor=dict(S), quantum_transport={
            "gate": -1, "drain": 0.05, "mixing": True, "scattering": True}))


# ------------------------------------------------------- plasma chemistry
def test_global_plasma_against_its_own_balances():
    from scipy.optimize import brentq
    from kpenvelope.plasma import (AMU, KB, ME, QE, SCCM, GlobalPlasma, Reaction, Species,
                                   fluxes_cm2)
    ill = "illustrative"
    X = Species("X", 40.0)
    Xp = Species("X+", 40.0, 1, wall_products={"X": 1})
    rx = [Reaction(("e", "X"), ("e", "e", "X+"), 2.3e-14, ill, b=0.59, C=17.44, energy_eV=15.76),
          Reaction(("e", "X"), ("e", "X"), 2.5e-14, ill, b=0.33, C=12.78, energy_eV=12.14),
          Reaction(("e", "X"), ("e", "X"), 2.3e-14, ill, b=1.6, elastic=True)]
    Q, S, P, h = 20 * SCCM, 0.05, 500.0, 0.3
    pl = GlobalPlasma([X, Xp], rx, 0.15, 0.3, P, {"X": Q}, S, h, ill)
    r = pl.solve()
    # the same balances written out by hand: ions k_iz n_g = S/V + h u_B A/V,
    # atoms n_g + n_i = Q / S, and the power balance for n_i
    V, A, M = pl.V, pl.A, 40 * AMU

    def kiz(T):
        return 2.3e-14 * T ** 0.59 * math.exp(-17.44 / T)

    def uB(T):
        return math.sqrt(QE * T / M)
    ni = 1e15
    for _ in range(300):
        T = brentq(lambda T: kiz(T) * (Q / S - ni) - S / V - h * uB(T) * A / V, 0.5, 50)
        ng = Q / S - ni
        per = (ng * (kiz(T) * 15.76 + 2.5e-14 * T ** 0.33 * math.exp(-12.78 / T) * 12.14
                     + 2.3e-14 * T ** 1.6 * 3 * ME / M * T)
               + h * uB(T) * A / V * (2.5 * T + 0.5 * T * math.log(M / (2 * math.pi * ME))))
        new = P / (QE * V) / per
        if abs(new / ni - 1) < 1e-14:
            break
        ni = new
    assert r["Te_eV"] == pytest.approx(T, rel=1e-10)
    assert r["ne"] == pytest.approx(ni, rel=1e-10)
    assert r["pressure_Pa"] == pytest.approx(ng * KB * 300.0, rel=1e-10)
    # a molecule that breaks up: the atoms' balance in closed form, and
    # every atom fed is pumped out
    A2 = Species("A2", 70.9)
    At = Species("A", 35.45, wall=0.1, wall_products={"A2": 0.5})
    A2p = Species("A2+", 70.9, 1, wall_products={"A2": 1})
    rx2 = [Reaction(("e", "A2"), ("e", "e", "A2+"), 1e-14, ill, b=0.5, C=12.0, energy_eV=11.5),
           Reaction(("e", "A2"), ("e", "A", "A"), 5e-14, ill, C=3.0, energy_eV=2.5),
           Reaction(("e", "A2"), ("e", "A2"), 3e-14, ill, b=1.0, elastic=True)]
    pl2 = GlobalPlasma([A2, At, A2p], rx2, 0.15, 0.3, 800.0, {"A2": 30 * SCCM}, 0.05, 0.3, ill)
    r2 = pl2.solve()
    d, ne, T = r2["densities"], r2["ne"], r2["Te_eV"]
    vA = math.sqrt(8 * KB * 300.0 / (math.pi * 35.45 * AMU))
    nA = 2 * 5e-14 * math.exp(-3.0 / T) * ne * d["A2"] / (0.05 / pl2.V + 0.1 * vA / 4 * pl2.A / pl2.V)
    assert d["A"] == pytest.approx(nA, rel=1e-10)
    assert (2 * d["A2"] + d["A"] + 2 * d["A2+"]) * 0.05 == pytest.approx(2 * 30 * SCCM, rel=1e-10)
    nf, inf_ = fluxes_cm2(r2, "A", "A2+")
    assert nf == pytest.approx(d["A"] * vA / 4 * 1e-4, rel=1e-12)
    assert inf_ == pytest.approx(0.3 * d["A2+"] * math.sqrt(QE * T / (70.9 * AMU)) * 1e-4,
                                 rel=1e-12)
    with pytest.raises(ValueError, match="charge"):
        GlobalPlasma([X, Xp], [Reaction(("e", "X"), ("e", "X+"), 1e-14, ill)], 0.1, 0.1,
                     100.0, {"X": Q}, S, h, ill)
    with pytest.raises(ValueError, match="source of h"):
        GlobalPlasma([X, Xp], rx, 0.1, 0.1, 100.0, {"X": Q}, S, h, "")
