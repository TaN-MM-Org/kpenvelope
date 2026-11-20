"""Tests for v0.25.0: quantum holes, the quantum channel with band-to-band
tunneling, the ballistic limit of the current, and re-emission of
particles in the flux process steps."""
import dataclasses
import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan  # noqa: E402
from kpenvelope.device import (BandToBand, Device, Electrode, Mobility,  # noqa: E402
                               QuantumChannel, Surface)
from kpenvelope.process import Structure  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])


# ------------------------------------------------------- quantum holes
def test_quantum_holes_mirror_quantum_electrons():
    # a well for electrons and its mirror image for holes (same mass, band
    # profile turned upside down, donors -> acceptors): the confined holes
    # must be the confined electrons exactly
    base = dataclasses.replace(algan(0.0), eg=3.4, me=0.2, eps_r=9.0)
    z = np.linspace(0, 110, 1101)
    out = {}
    for holes in (False, True):
        def mk(c, name):
            return dataclasses.replace(base, name=name, ec=(-c + 3.4) if holes else c)
        dop = dict(acceptors=1e-3) if holes else dict(donors=1e-3)
        layers = [StackLayer(mk(0.0, "b1"), 50.0, **dop), StackLayer(mk(-0.3, "w"), 10.0),
                  StackLayer(mk(0.0, "b2"), 50.0, **dop)]
        el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
        kw = dict(hole_mass=0.2, hole_mobility=Mobility(100.0, "t")) if holes else {}
        q = (QuantumChannel((40.0, 70.0), x_range=(0, 20), electrons=False,
                            holes=(0.2, 0.2, "test"), tol=1e-9) if holes
             else QuantumChannel((40.0, 70.0), x_range=(0, 20), tol=1e-9))
        d = Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                   grid=dict(x=np.array([0., 10., 20.]), z=z), quantum=q, **kw)
        eq = d.solve({})
        on = d.solve({"bot": -0.2 if holes else 0.2}, guess=eq)
        out[holes] = (eq, on)
    a, b = out[False][0], out[True][0]
    assert b.quantum["hole_sheet_cm2"][1] == pytest.approx(a.quantum["sheet_cm2"][1], rel=1e-9)
    assert np.allclose(b.quantum["hole_subbands"][1][1][:3], -a.quantum["subbands"][1][1][:3],
                       atol=1e-9)
    assert np.max(np.abs(a.n[1] - b.p[1])) < 1e-9 * a.n.max()
    # the hole current is the electron one scaled by the mobility (100 / 1000)
    assert -out[True][1].current["bot"] == pytest.approx(0.1 * out[False][1].current["bot"],
                                                         rel=1e-8)
    with pytest.raises(ValueError, match="holes"):
        QuantumChannel((1.0, 2.0), holes=(0.2, 0.2))


def test_quantum_channel_with_band_to_band_tunneling():
    from test_v022 import _no_pol
    m = dataclasses.replace(algan(0.0), name="p", eg=0.66, me=0.04, eps_r=16.0)
    m2 = dataclasses.replace(m, name="n")
    layers = [StackLayer(m, 100.0, acceptors=3e-2), StackLayer(m2, 100.0, donors=1e-2)]

    def mk(q):
        return Device(layers, 20.0, [Electrode("top", "ohmic", 0, 20),
                                     Electrode("bot", "ohmic", 0, 20, face="bottom")],
                      Mobility(1000.0, "t"), Surface("pinned", barrier=0.5),
                      grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 200, 401)),
                      extra_sheet=_no_pol(layers), hole_mass=0.3,
                      hole_mobility=Mobility(100.0, "t"),
                      band_to_band=BandToBand("t", reduced_mass_m0=0.02, model="nonlocal"),
                      quantum=q)
    dq = mk(QuantumChannel((90.0, 190.0), x_range=(0, 20), holes=(0.3, 0.3, "t")))
    dc = mk(None)
    eq = dq.solve({})
    assert all(abs(v) < 1e-12 for v in eq.current.values())
    st = dq.solve({"bot": -0.1}, guess=eq)
    assert abs(sum(st.current.values())) < 1e-9 * abs(st.current["bot"])
    assert np.max(np.abs(st.quantum_potential)) > 0.1
    # the tunneling sees the real bands: the same rates as a device without
    # the quantum channel, given the same potentials
    dq._use_q(st)
    Gq, Gc = dq._nonlocal_rates(st._x), dc._nonlocal_rates(st._x)
    assert np.max(np.abs(Gq[0] - Gc[0])) < 1e-8 * np.max(np.abs(Gc[0]))


# ---------------------------------------------------------- ballistic
def test_ballistic_limit():
    from scipy.integrate import quad
    from kpenvelope.electrons import HBAR, KB, M0, fermi_dirac_half
    from kpenvelope.nitride import QE
    # the formula against a direct integral over the 2D Fermi gas moving +x
    m, kT = 0.2 * M0, KB * 300.0
    pre = QE * math.sqrt(2 * M0) * (kT * QE) ** 1.5 / (2 * math.pi ** 1.5 * HBAR ** 2)
    for eta in (-3.0, 0.0, 5.0):
        f = (lambda k: 1.0 / (1.0 + math.exp(min(HBAR ** 2 * k * k / (2 * m) / (kT * QE)
                                                   - eta, 700.0))))
        direct = QE * HBAR / (math.pi ** 2 * m) * quad(lambda k: k * k * f(k), 0, 5e9,
                                                      limit=400)[0]
        assert pre * math.sqrt(0.2) * float(fermi_dirac_half(eta)[0]) == pytest.approx(
            direct, rel=1e-9)
    from test_v024 import _small_hemt
    q = _small_hemt(True)
    zero = q.solve({"gate": -1.0})
    I0, _ = q.ballistic_current(zero)
    assert abs(I0) < 1e-12
    st = q.solve({"drain": 1.0, "gate": -1.0}, guess=zero)
    I, per = q.ballistic_current(st)
    assert I > st.current["drain"] > 0 and sum(j for _, j in per) == pytest.approx(I)
    assert per[0][0] < per[1][0]                         # levels in order
    with pytest.raises(ValueError, match="QuantumChannel"):
        _small_hemt(False).ballistic_current(st)


# ------------------------------------------------ re-emission (sticking)
def _mc_trench(w, d, s, N, seed=0):
    """Independent Monte Carlo: particles enter a w x d trench through its
    opening by the 2D cosine law, stick with probability s where they land
    and are re-emitted by the cosine law otherwise; returns the landing
    points on the bottom and the number launched."""
    rng = np.random.default_rng(seed)
    a, b = -w / 2, w / 2
    x = rng.uniform(a, b, N)
    y = np.zeros(N)
    ph = np.arcsin(2 * rng.random(N) - 1)
    ux, uy = np.sin(ph), -np.cos(ph)
    alive = np.ones(N, bool)
    bottom = []
    for _ in range(300):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        X, Y, UX, UY = x[idx], y[idx], ux[idx], uy[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.stack([np.where(UY < 0, (-d - Y) / UY, np.inf),
                          np.where(UX < 0, (a - X) / UX, np.inf),
                          np.where(UX > 0, (b - X) / UX, np.inf),
                          np.where(UY > 0, -Y / UY, np.inf)], 1)
        T[T <= 1e-12] = np.inf
        k = np.argmin(T, 1)
        t = T[np.arange(idx.size), k]
        X2, Y2 = X + t * UX, Y + t * UY
        alive[idx[k == 3]] = False
        bottom.append(X2[k == 0])
        stick = rng.random(idx.size) < s
        alive[idx[(k < 3) & stick]] = False
        go = (k < 3) & ~stick
        al = np.arcsin(2 * rng.random(idx.size) - 1)
        nx = np.where(k == 1, 1.0, np.where(k == 2, -1.0, 0.0))
        ny = np.where(k == 0, 1.0, 0.0)
        x[idx[go]], y[idx[go]] = X2[go], Y2[go]
        ux[idx[go]] = (np.cos(al) * nx + np.sin(al) * ny)[go]
        uy[idx[go]] = (np.cos(al) * ny - np.sin(al) * nx)[go]
    return np.concatenate(bottom), N


def test_view_factors_and_re_emission():
    w, d = 40.0, 60.0
    st = Structure(200.0, 0.0, 200.0, 2.0, "Si", 150.0)
    st.etch({"Si": 1.0}, d, mode="directional", window=(100 - w / 2, 100 + w / 2))
    faces, F = st.view_factors()
    # what a face does not send to other faces goes to the sky: the direct
    # flux it receives from the sky (reciprocity with the sky)
    direct = {(i, k, sd): f for (i, k, sd, f) in st.fluxes(1.0)}
    err = [abs(1 - F[q].sum() - direct[(f[0], f[1], f[2])]) for q, f in enumerate(faces)
           if (f[0], f[1], f[2]) in direct]
    assert max(err) < 1e-3
    assert np.abs(F - F.T).max() < 2e-3                 # reciprocity (equal faces)
    kb = int(np.argmin(np.abs(st.z - (150 - d)))) - 1
    for s_ in (0.3, 0.05):
        hb, N = _mc_trench(w, d, s_, 200000, seed=1)
        sel = np.abs(hb) < 4.0
        mc = sel.sum() / N * w / 8.0
        sig = math.sqrt(sel.sum()) / N * w / 8.0
        ff = st._face_fluxes(1.0, s_)
        rad = np.mean([f for (i, k, sd, f) in ff if sd == 0 and k == kb
                       and abs(st.xc[i] - 100) < 4.0])
        assert abs(rad - mc) < 4 * sig + 0.01 * mc


def test_deposition_with_low_sticking_is_conformal():
    out = {}
    for s_ in (1.0, 0.01):
        st = Structure(200.0, 0.0, 250.0, 2.0, "Si", 150.0)
        st.etch({"Si": 1.0}, 100, mode="directional", window=(80, 120))
        st.deposit("O", 10.0, mode="flux", sticking=s_)
        out[s_] = (st.thickness("O", 20), st.thickness("O", 100))
    assert out[1.0][0] == pytest.approx(10.0) and out[1.0][1] < 5.0
    assert out[0.01][0] == pytest.approx(10.0) and out[0.01][1] == pytest.approx(10.0, abs=2.0)
    with pytest.raises(ValueError, match="sticking"):
        Structure(40.0, 0.0, 40.0, 2.0, "Si", 20.0).deposit("O", 2.0, mode="flux",
                                                             sticking=0.0)
