"""Checks of the features new in 0.36.0: NEGF with phonons that take and
give energy (the self-consistent Born approximation for a local coupling,
with the motion across the width on a grid), in the library and the app."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])

_HW = 0.0925
_KT = 0.02585


def _barrier_chain():
    from kpenvelope.negf import Chain
    U = 0.15 * np.exp(-((np.arange(81) - 40) / 15.0) ** 2)
    return Chain(U, 0.2, 0.5)


def test_phonon_transport_exact_checks():
    from kpenvelope.negf import chain_transport, phonon_transport
    ch = _barrier_chain()
    # without coupling: the closed-form ballistic current and density
    r0 = phonon_transport(ch, 0.05, 0.0, _KT, _HW, 0.0)
    rc = chain_transport(ch, 0.05, 0.0, _KT)
    assert r0["I"] == pytest.approx(rc["I"], rel=1e-5)                 # 3e-7
    assert np.max(np.abs(r0["n"] / rc["n"] - 1)) < 1e-2               # 3e-3
    # at equilibrium every state is filled as Fermi-Dirac says, at every
    # site and energy (detailed balance), and no current flows
    eq = phonon_transport(ch, 0.03, 0.03, _KT, _HW, 0.05, tol=1e-11)
    f = 1.0 / (1.0 + np.exp((eq["E"] - 0.03) / _KT))
    assert np.max(np.abs(eq["G_n"] - f[:, None] * eq["A"])) < 1e-12 * np.max(eq["A"])
    assert abs(eq["I"]) < 1e-12 * abs(rc["I"])
    # with a bias: the current leaving equals the current entering, and the
    # power the contacts lose is hbar w times the net emission
    r = phonon_transport(ch, 0.05, 0.0, _KT, _HW, 0.05, tol=1e-11, max_iter=1000)
    assert r["I"] == pytest.approx(r["I_drain"], rel=1e-8)            # 9e-10
    assert r["power"] == pytest.approx(r["power_phonons"], rel=1e-6)  # 4e-9
    assert 0 < r["power"] < r["I"] * 0.05
    assert 0 < r["I"] < r0["I"]
    with pytest.raises(ValueError, match="probes"):
        from kpenvelope.negf import Chain
        phonon_transport(Chain(ch.U, 0.2, 0.5, eta=0.01), 0.05, 0.0, _KT, _HW, 0.05)


def test_phonon_transport_against_the_sum_of_1d_results():
    # a uniform channel 75 nm long: each part of the width (energy along x)
    # is a 1D wire, whose exact conductance with momentum-randomizing
    # scattering is lambda / (lambda + L) of the ballistic one, lambda =
    # 2 hbar v_x / Gamma (Gamma from the run); summed over the width
    from kpenvelope.negf import _G0, _H2M, Chain, phonon_transport
    m, a, L, F, V = 0.2, 0.5, 75.0, 0.03, 0.001
    t = _H2M / (m * a * a)
    r = phonon_transport(Chain(np.zeros(int(L / a) + 1), m, a), F + V / 2, F - V / 2, _KT,
                         _HW, 0.01, energy_step=_KT / 10, tol=1e-9, max_iter=800)
    E = r["E"]
    Gam = (r["sigma"][0] + r["sigma"][1])[:, r["n"].size // 2]
    mf = 1.0 / (4.0 * _KT * np.cosh((E - F) / (2.0 * _KT)) ** 2)
    u = (np.arange(4000) + 0.5) / 4000
    tot = 0.0
    for e, g, wf in zip(E, Gam, mf):
        if e <= 0:
            continue
        Ex = e * 0.5 * (1 - np.cos(math.pi * u))
        wE = e * 0.5 * math.pi * np.sin(math.pi * u) / u.size
        g1 = math.sqrt(m / _H2M) / (2 * math.pi * np.sqrt(e - Ex))
        hv = 2 * t * a * np.sqrt(np.clip(1 - (1 - Ex / (2 * t)) ** 2, 0, None))
        lam = 2 * hv / g
        tot += wf * np.sum(wE * g1 * lam / (lam + L)) * (E[1] - E[0])
    assert r["I"] / V == pytest.approx(_G0 * tot, rel=0.03)           # 1.1 percent


def test_phonon_coupling_for_mobility():
    from kpenvelope.electrons import HBAR, M0
    from kpenvelope.negf import _H2M, phonon_coupling_for_mobility
    from kpenvelope.nitride import QE
    T, mu, m = 300.0, 1500.0, 0.2
    C = phonon_coupling_for_mobility(mu, m, _HW, T)
    kT = 8.617333262e-5 * T
    g = m / (2 * math.pi * 2 * _H2M)
    Nq = 1 / math.expm1(_HW / kT)
    E = np.linspace(1e-7, 2.0, 2000001)
    rate = 2 * math.pi * C * g * (Nq + (Nq + 1) * (E > _HW))
    tau = HBAR / (rate * QE)
    y = tau * E * np.exp(-E / kT)
    mu_num = QE / (m * M0) * float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(E))) / kT ** 2 * 1e4
    assert mu_num == pytest.approx(mu, rel=1e-5)                       # 2e-6
    # with a Fermi level: the same formula with Fermi-Dirac filling, the
    # thin-gas limit and an independent integration
    from scipy.integrate import quad
    assert phonon_coupling_for_mobility(mu, m, _HW, T, fermi_eV=-0.4) == pytest.approx(
        C, rel=1e-6)                                                     # 7e-8
    F = 0.05

    def fd(e):
        return 1 / (1 + math.exp((e - F) / kT))

    def gam(e):
        out = Nq * (1 - fd(e + _HW)) + (Nq + 1) * fd(e + _HW)
        if e > _HW:
            out += (Nq + 1) * (1 - fd(e - _HW)) + Nq * fd(e - _HW)
        return 2 * math.pi * g * out

    def mf(e):
        return fd(e) * (1 - fd(e)) / kT
    num = sum(quad(lambda e: HBAR / QE / gam(e) * e * mf(e), a_, b_, limit=200)[0]
              for a_, b_ in ((0, _HW), (_HW, 2.0)))
    tau1 = num / quad(fd, 0, 2.0, limit=200)[0]
    want = tau1 / (mu * 1e-4 * m * M0 / QE)
    assert phonon_coupling_for_mobility(mu, m, _HW, T, fermi_eV=F) == pytest.approx(
        want, rel=1e-6)
    assert want < C                    # a dense gas emits more
    with pytest.raises(ValueError):
        phonon_coupling_for_mobility(-1.0, m, _HW, T)


def test_device_negf_with_phonons_holes_mirror_electrons():
    from test_v027 import _mirror_fet
    out = {}
    for h in (False, True):
        d = _mirror_fet(h)
        s = -1.0 if h else 1.0
        b = {"gate": -0.3 * s, "drain": 0.02 * s}
        g = d.solve(b)
        car = "holes" if h else "electrons"
        out[h] = (d.negf(b, guess=g, dx=1.0, carrier=car),
                  d.negf(b, guess=g, dx=1.0, carrier=car, phonon_energy=_HW,
                         phonon_coupling=0.03))
    (a0, a), (b0, b_) = out[False], out[True]
    assert 0 < a.current["drain"] < a0.current["drain"]
    assert -b_.current["drain"] == pytest.approx(a.current["drain"], rel=1e-8)   # 1e-13
    assert b_.negf["phonon_power"] == pytest.approx(a.negf["phonon_power"], rel=1e-8)
    assert 0 < a.negf["phonon_power"] < a.current["drain"] * 0.02
    d = _mirror_fet(False)
    with pytest.raises(ValueError, match="both"):
        d.negf({}, phonon_energy=_HW)
    with pytest.raises(ValueError, match="not combined"):
        d.negf({}, phonon_energy=_HW, phonon_coupling=0.01, scattering_time=1e-13)


def test_app_wave_current_with_phonons():
    from test_v020 import _spec
    from kpenvelope import device
    from kpenvelope.gui import api
    from kpenvelope.negf import phonon_coupling_for_mobility
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    seen = {}

    class Stop(Exception):
        pass

    def fake(self, bias, **kw):
        seen.update(kw, T=self.T)
        raise Stop
    S = dict(contact="50", gap_sg="35", gate_length="30", gap_gd="35", mobility="1500")
    spec = _spec(transistor=dict(S), quantum_transport={"gate": -1, "drain": 0.05,
                                                        "phonons": True})
    orig = device.Device.negf
    device.Device.negf = fake
    try:
        with pytest.raises(Stop):
            api.run_quantum(spec)
    finally:
        device.Device.negf = orig
    dev, info = api._transistor_device(_spec(transistor=dict(S, quantum=True)))
    m = info["layers"][0].material.me
    assert seen["phonon_energy"] == ph.w_LO_perp / 1000.0
    dd = dev.solve({"gate": -1.0, "drain": 0.05})
    F = api._gate_fermi(dev, dd, m)
    assert seen["phonon_coupling"] == phonon_coupling_for_mobility(
        1500.0, m, ph.w_LO_perp / 1000.0, seen["T"], fermi_eV=F)
    spec["quantum_transport"]["scattering"] = True
    assert "on their own" in api.handle("quantum", spec)["error"]
