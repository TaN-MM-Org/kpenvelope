"""Tests for v0.24.0: confined electrons in the device solver
(QuantumChannel), and process steps with line-of-sight fluxes and
two-dimensional oxidation."""
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan, solve_electrons  # noqa: E402
from kpenvelope.device import (BandToBand, Device, Electrode, Mobility,  # noqa: E402
                               QuantumChannel, Surface, Trap, column_states, _H2M)
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel  # noqa: E402


# ------------------------------------------------ the column eigenproblem
def test_column_states_square_well_and_airy():
    from scipy.special import ai_zeros
    m, L = 0.2, 10.0
    errs = []
    for n in (100, 200):
        zz = np.linspace(0, L, n + 1)
        zz = np.sort(np.r_[zz, 0.5 * (zz[:-1] + zz[1:])[::3]])        # uneven spacing
        E, p2, w = column_states(zz[1:-1], np.zeros(zz.size - 2), np.full(zz.size - 2, m),
                                 3, 0.0, L)
        exact = _H2M * np.pi ** 2 * np.arange(1, 4) ** 2 / (m * L ** 2)
        errs.append(np.abs(E / exact - 1))
        assert np.allclose((w[:, None] * p2).sum(0), 1.0, rtol=1e-12)
    assert np.all(errs[1] < 1e-3) and np.all(errs[0] / errs[1] > 3.8)       # second order
    a = -ai_zeros(3)[0]
    F = 0.01                                                                  # V/nm
    errs = []
    for n in (400, 800):
        z = np.linspace(0, 60, n + 1)[1:]
        E, _, _ = column_states(z, F * z, np.full(z.size, m), 3, 0.0, 60 + 60 / n)
        errs.append(np.abs(E / ((_H2M / m) ** (1 / 3) * F ** (2 / 3) * a) - 1))
    assert np.all(errs[1] < 1e-4) and np.all(errs[0] / errs[1] > 3.8)


# ---------------------------------- the device against Schroedinger-Poisson
def _hemt_1d(dz, quantum=True):
    layers = [StackLayer(algan(0.0), 300.0, donors=1e-4), StackLayer(algan(0.25), 25.0)]
    win = (280.0, 324.0)
    r = solve_electrons(layers, 1.0, dz=dz, quantum_window=win, classical_outside=True,
                        n_states=40, tol=1e-10)
    dev = Device(layers, 20.0, [Electrode("bot", "ohmic", 0, 20, face="bottom")],
                 Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                 grid=dict(x=np.array([0., 10., 20.]), z=r.z),
                 quantum=QuantumChannel(win, tol=1e-9) if quantum else None)
    return r, dev, dev.equilibrium()


def test_quantum_device_equals_schroedinger_poisson():
    out = []
    for dz in (0.2, 0.1):
        r, dev, st = _hemt_1d(dz)
        dE = np.abs(st.quantum["subbands"][1][1][:3] - r.subband_energies[:3])
        dEc = float(np.nanmax(np.abs(st.ec[1] - r.ec)))
        dns = abs(st.quantum["sheet_cm2"][1] / r.sheet_quantum_cm2 - 1)
        out.append((dEc, dE.max(), dns))
    # the two codes differ only in how the material step is put on the grid:
    # the difference is first order in the step and halves with it
    assert out[1][0] < 6e-5 and out[1][1] < 3e-6 and out[1][2] < 1e-5
    assert 1.7 < out[0][0] / out[1][0] < 2.3
    # the effect being checked is far larger than that difference
    rc, devc, stc = _hemt_1d(0.1, quantum=False)
    assert np.nanmax(np.abs(stc.ec[1] - st.ec[1])) > 100 * out[1][0]
    z = r.z
    m = (z > 280) & (z < 300)
    cq = (st.n[1][m] * (300 - z[m])).sum() / st.n[1][m].sum()
    cc = (stc.n[1][m] * (300 - z[m])).sum() / stc.n[1][m].sum()
    c1 = (r.n[m] * (300 - z[m])).sum() / r.n[m].sum()
    assert cq == pytest.approx(c1, abs=2e-3) and cq > cc + 0.15     # the gas moves away


def test_quantum_channel_at_a_set_lattice_temperature():
    # the full temperature model (gap held fixed here) at 360 K against the
    # 1D Schroedinger-Poisson solver at 360 K
    from kpenvelope.device import Varshni
    layers = [StackLayer(algan(0.0), 300.0, donors=1e-4), StackLayer(algan(0.25), 25.0)]
    win = (280.0, 324.0)
    r = solve_electrons(layers, 1.0, temperature_K=360.0, dz=0.2, quantum_window=win,
                        classical_outside=True, n_states=40, tol=1e-10)
    d = Device(layers, 20.0, [Electrode("bot", "ohmic", 0, 20, face="bottom")],
               Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
               grid=dict(x=np.array([0., 10., 20.]), z=r.z),
               quantum=QuantumChannel(win, tol=1e-9), temperature_model="full",
               band_gap_temperature=Varshni(1e-12, 1.0, 0.5, "no gap change, for the check"))
    st = d.solve({}, lattice_temperature=360.0)
    assert np.nanmax(np.abs(st.ec[1] - r.ec)) < 1.2e-4
    assert abs(st.quantum["subbands"][1][1][0] - r.subband_energies[0]) < 1e-5
    assert st.quantum["sheet_cm2"][1] == pytest.approx(r.sheet_quantum_cm2, rel=3e-5)


def test_quantum_channel_under_a_biased_gate():
    layers = [StackLayer(algan(0.0), 150.0, donors=1e-4), StackLayer(algan(0.25), 20.0)]
    win = (130.0, 169.5)
    r = solve_electrons(layers, 2.0, dz=0.25, quantum_window=win, classical_outside=True,
                        n_states=40, tol=1e-10)
    x = np.linspace(0, 2000, 9)
    el = [Electrode("s", "ohmic", 0, 100, face="bottom"),
          Electrode("gate", "schottky", 0, 2000, barrier=1.0)]
    dev = Device(layers, 2000.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                 grid=dict(x=x, z=r.z), quantum=QuantumChannel(win))
    st = dev.solve({"gate": -1.0})
    assert np.nanmax(np.abs(st.ec[4] - r.ec)) < 2e-4
    assert st.quantum["sheet_cm2"][4] == pytest.approx(r.sheet_quantum_cm2, rel=5e-5)


def _small_hemt(quantum=True, traps=(), holes=False):
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    mob = Mobility(1500.0, "t", saturation_velocity=1.0e7, temperature_exponent=1.5)
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 300, 400, barrier=1.0),
          Electrode("drain", "ohmic", 600, 700, implant_depth=30, implant_donors=1e20)]
    kw = dict(hole_mass=1.0, hole_mobility=Mobility(10.0, "t")) if holes else {}
    return Device(layers, 700.0, el, mob, Surface("pinned", barrier=1.0), traps=traps,
                  dx_fine=10, dx_max=50, dz_fine=0.5, dz_max=50,
                  quantum=QuantumChannel((270.0, 319.5), x_range=(120, 580)) if quantum
                  else None, **kw)


def test_quantum_transistor_balances():
    tr = [Trap("acceptor", 0.7, "ec", 1e17, 1e-8, 1e-8, "t", "buffer")]
    dev = _small_hemt(traps=tr, holes=True)
    eq = dev.equilibrium()
    assert all(abs(v) < 1e-12 for v in eq.current.values())     # detailed balance
    st = dev.solve({"drain": 2.0}, guess=eq)
    assert abs(sum(st.current.values())) < 1e-9 * abs(st.current["drain"])
    assert st.quantum["last_change"] < 1e-5 and st.quantum_potential.shape == st.n.shape
    # ec is the real band edge: Lambda sits only inside the window
    z = dev.axes[-1]
    assert np.all(st.quantum_potential[:, (z < 269) | (z > 320)] == 0)
    th = ThermalModel(ThermalMaterial("d", 130.0, 0.0, "t"),
                      substrate=(ThermalLayer("s", 1e3, ThermalMaterial("s", 300.0, 0.0, "t")),))
    hot = dev.solve({"drain": 2.0}, guess=st, thermal=th)
    assert hot.temperature.max() > 310
    assert hot.heat.sum() == pytest.approx(hot.power, rel=1e-9)
    ports, Y = dev.small_signal(st, [1e9])
    assert np.all(np.isfinite(Y))


def test_quantum_channel_near_pinch_off_and_what_it_refuses():
    c, q = _small_hemt(False), _small_hemt(True)
    a = c.solve({"drain": 1.0, "gate": -3.0})
    b = q.solve({"drain": 1.0, "gate": -3.0})
    # near pinch-off the confinement energy shifts the threshold: less
    # current at the same voltages (about 1 percent here)
    assert 0.95 * a.current["drain"] < b.current["drain"] < 0.995 * a.current["drain"]
    # the same state reached through a sweep
    prev = None
    for vg in (-1.5, -3.0):
        prev = q.solve({"drain": 1.0, "gate": vg}, guess=prev)
    assert prev.current["drain"] == pytest.approx(b.current["drain"], rel=1e-5)
    # hot electrons that relax at once (constant mobility): the quantum channel
    # changes the hydrodynamic current as it changes drift-diffusion
    layers = [StackLayer(algan(0.0), 300.0), StackLayer(algan(0.25), 20.0)]
    ratio = []
    for qc in (None, QuantumChannel((270.0, 319.5))):
        d = Device(layers, 700.0, q.electrodes, Mobility(1500.0, "t"),
                   Surface("pinned", barrier=1.0), dx_fine=10, dx_max=50, dz_fine=0.5,
                   dz_max=50, quantum=qc)
        dd = d.solve({"drain": 1.0, "gate": -3.0})
        hy = d.solve({"drain": 1.0, "gate": -3.0}, guess=dd, energy_relaxation=1e-16)
        ratio.append(hy.current["drain"] / dd.current["drain"])
    assert abs(ratio[0] - 1) < 1e-3 and ratio[1] == pytest.approx(ratio[0], abs=1e-4)
    layers = [StackLayer(algan(0.0), 300.0, donors=1e-4), StackLayer(algan(0.25), 20.0)]
    el = [Electrode("s", "ohmic", 0, 20, face="bottom")]
    base = dict(grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 320, 641)))
    with pytest.raises(ValueError, match="band_to_band"):
        Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
               hole_mass=1.0, hole_mobility=Mobility(10.0, "t"),
               band_to_band=BandToBand("t", reduced_mass_m0=0.1),
               quantum=QuantumChannel((280.0, 319.0)), **base)
    with pytest.raises(ValueError, match="thermionic"):
        Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
               heterojunctions="thermionic", quantum=QuantumChannel((280.0, 319.0)), **base)
    with pytest.raises(ValueError, match="no semiconductor column"):
        Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
               quantum=QuantumChannel((400.0, 500.0)), **base)
    with pytest.raises(ValueError, match="window"):
        QuantumChannel((5.0, 1.0))


# ------------------------------------------------ line-of-sight fluxes
def _trench(w, d, h=2.0):
    from kpenvelope.process import Structure
    s = Structure(400.0, 0.0, 300.0, h, "Si", 200.0)
    s.etch({"Si": 1.0}, d, mode="directional", window=(200 - w / 2, 200 + w / 2))
    return s


def test_fluxes_are_the_exact_view_factors():
    import math
    from scipy.integrate import quad
    for (w, d) in ((40, 40), (20, 80), (100, 20)):
        s = _trench(w, d)
        for n in (1.0, 5.0, 50.0):
            F = s.fluxes(n)
            ib = int(np.argmin(np.abs(s.xc - 201)))
            kb = int(np.argmin(np.abs(s.z - (200 - d)))) - 1
            f = [f for (i, k, side, f) in F if i == ib and k == kb and side == 0][0]
            a = math.atan((w / 2 + s.xc[ib] - 200) / d)
            b = math.atan((w / 2 - s.xc[ib] + 200) / d)
            Cn = math.sqrt(math.pi) * math.gamma((n + 1) / 2) / math.gamma(n / 2 + 1)
            exact = quad(lambda p: math.cos(p) ** n / Cn, -a, b)[0]
            assert f == pytest.approx(exact, abs=2e-6)
            # every particle that enters the opening lands inside the trench
            tot = sum(f_ * s.h for (i, k, side, f_) in F
                      if (side == 0 and 200 - w / 2 < s.xc[i] < 200 + w / 2)
                      or (side == 1 and abs(s.x[i + 1] - (200 - w / 2)) < 1e-9)
                      or (side == 2 and abs(s.x[i] - (200 + w / 2)) < 1e-9))
            assert tot == pytest.approx(w, rel=1e-3)
        # the walls of a gas at rest: 1/2 (1 - cos(atan(w / depth)))
        if n == 50.0:
            for (i, k, side, f_) in s.fluxes(1.0):
                if side == 1 and abs(s.x[i + 1] - (200 - w / 2)) < 1e-9:
                    assert f_ == pytest.approx(
                        0.5 * (1 - math.cos(math.atan(w / (200 - s.zc[k])))), abs=1e-6)


def test_flux_etching_and_deposition():
    from kpenvelope.process import Structure
    flat = Structure(200.0, 0.0, 300.0, 2.0, "Si", 200.0)
    flat.etch({"Si": 1.0}, 50.0, mode="flux")
    assert flat.thickness("Si", 100) == pytest.approx(150.0)
    # aspect-ratio dependent etching: the narrow trench etches more slowly
    narrow, wide = _trench(20, 40), _trench(100, 40)
    for s in (narrow, wide):
        s.etch({"Si": 1.0}, 40.0, mode="flux")
    dn, dw = 200 - narrow.top_surface()[100], 200 - wide.top_surface()[100]
    assert dn < dw - 10
    # a narrow ion beam etches straight down through a mask opening
    m = Structure(400.0, 0.0, 300.0, 2.0, "Si", 200.0)
    m.deposit("M", 10.0, mode="directional")
    m.etch({"M": 1.0}, 10.0, mode="directional", window=(150, 250))
    m.etch({"Si": 1.0}, 60.0, mode="flux", exponent=2000.0)
    assert m.thickness("Si", 200) == pytest.approx(140.0, abs=2.0)
    assert m.thickness("Si", 140) == pytest.approx(200.0)       # under the mask
    # deposition: the flat thickness, and the volume is what fell in
    s = _trench(100, 100)
    before = s.area("Si")
    s.deposit("O", 20.0, mode="flux")
    assert s.thickness("O", 20) == pytest.approx(20.0)
    assert s.area("O") == pytest.approx(400.0 * 20.0, rel=0.01)
    assert s.area("Si") == before
    with pytest.raises(ValueError, match="exponent"):
        s.fluxes(0.5)


def test_two_dimensional_oxidation():
    import math
    from kpenvelope.process import Structure
    kT = 8.617333262e-5 * (1000 + 273.15)
    B = 7.72e2 * math.exp(-1.23 / kT) * 1e6 / 3600        # nm^2/s (as in example 46)
    BA = 6.23e6 * math.exp(-2.0 / kT) * 1e3 / 3600
    A = B / BA
    # the oxidant solver: in one dimension c = A / (A + 2 x) at the interface
    s = Structure(40.0, 0.0, 300.0, 2.0, "Si", 200.0)
    s.mat[:, 100:110] = "SiO2"                              # 20 nm of oxide
    li, lk, c = s._oxidant("Si", "SiO2", A)
    assert np.allclose(c, A / (A + 2 * 20.0), rtol=1e-12)
    # far from any edge: Deal-Grove within one cell, also from a pad oxide
    for x0, T in ((0.0, 7200.0), (10.0, 14400.0)):
        w = Structure(40.0, 0.0, 700.0, 2.0, "Si", 400.0)
        if x0:
            w.deposit("SiO2", x0, mode="directional")
        si0, ox0 = w.area("Si"), w.area("SiO2")
        w.oxidize("Si", "SiO2", B, BA, T, 0.44, "test", mode="2d")
        tau = (x0 * x0 + A * x0) / B
        x = 0.5 * A * (math.sqrt(1 + 4 * B * (T + tau) / A ** 2) - 1)
        assert w.thickness("SiO2", 20) == pytest.approx(x, abs=2.0)
        assert w.thickness("Si", 20) == pytest.approx(400 - 0.44 * (x - x0), abs=2.0)
        # the oxide's volume is the consumed silicon over 0.44
        assert w.area("SiO2") - ox0 == pytest.approx((si0 - w.area("Si")) / 0.44, rel=0.02)
    # LOCOS: under a nitride mask on a pad oxide the oxide reaches in (bird's
    # beak) and lifts the mask; without the pad oxide it reaches in less
    beak = {}
    for pad in (10.0, 0.0):
        w = Structure(600.0, 0.0, 700.0, 2.0, "Si", 400.0)
        if pad:
            w.deposit("SiO2", pad, mode="directional")
        w.deposit("Si3N4", 40.0, mode="directional", window=(0, 300))
        w.oxidize("Si", "SiO2", B, BA, 14400.0, 0.44, "test", mode="2d")
        beak[pad] = [w.thickness("SiO2", x) - pad for x in (200.0, 250.0)]
        if pad:
            assert w.thickness("SiO2", 50) == pytest.approx(pad)          # far under the mask
            assert w.top_surface()[100] > 450.0 + 4                         # the mask lifted
    assert beak[10.0][0] > beak[0.0][0] + 10 and beak[10.0][1] > beak[0.0][1] + 10
    # the column mode is unchanged and refuses a wrong mode
    with pytest.raises(ValueError, match="mode"):
        Structure(40.0, 0.0, 300.0, 2.0, "Si", 200.0).oxidize("Si", "SiO2", B, BA, 10.0,
                                                              0.44, "t", mode="3d")


# ------------------------------------------------------------ the app
def test_app_quantum_transistor_and_new_process_modes():
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_v023 import _app_spec
    from kpenvelope.gui.api import handle
    s = _app_spec()
    s["transistor"].update(heating=False, quantum=True, sweep="gate", fixed=5,
                           **{"from": -2, "to": 0, "step": 2})
    r = handle("transistor", s)
    assert "error" not in r, r.get("error")
    c = _app_spec()
    c["transistor"].update(heating=False, sweep="gate", fixed=5, **{"from": -2, "to": 0, "step": 2})
    rc = handle("transistor", c)
    assert any("Schroedinger" in x for x in r["sources"])
    assert np.allclose(r["drain"], rc["drain"], rtol=0.05)
    assert not np.allclose(r["drain"], rc["drain"], rtol=1e-4)
    s["transistor"]["thermionic"] = True
    assert "thermionic" in handle("transistor", s)["error"]
    p = {"source": "illustrative", "width": 400, "height": 300, "cell": 2, "substrate": "Si",
         "substrate_top": 200,
         "recipe": "etch Si=1 time=80 mode=directional from=180 to=220\n"
                   "deposit material=SiO2 thickness=20 mode=flux exponent=1\n"
                   "oxidize material=Si oxide=SiO2 B=2.89 BA=0.0209 time=600 consumed=0.44 "
                   "mode=2d"}
    r = handle("process", {"process": p})
    assert "error" not in r, r.get("error")
    assert r["oxide_exact"] is None and r["materials"] == ["Si", "SiO2"]
    p["recipe"] = "etch Si=1 time=30 mode=flux from=1 to=2"
    assert "no window" in handle("process", {"process": p})["error"]
