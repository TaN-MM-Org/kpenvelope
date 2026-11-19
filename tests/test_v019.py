"""Tests for v0.19.0: 2D/3D device simulation with current, traps in time,
field- and temperature-dependent mobility, and self-heating."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan, gan, threshold_voltage  # noqa: E402
from kpenvelope.device import (Device, Electrode, Mobility, Surface, Trap,  # noqa: E402
                               capture_coefficient)
from kpenvelope.driftdiffusion import Contact, Transport, solve_drift_diffusion  # noqa: E402
from kpenvelope.electrons import KB, M0  # noqa: E402
from kpenvelope.nitride import QE  # noqa: E402
from kpenvelope.thermal import (GAN_BULK, SIC_4H_ROOM, ThermalLayer,  # noqa: E402
                                ThermalMaterial, ThermalModel, boundary_resistance,
                                solve_heat)


def _hemt(width=700.0, buffer=200.0, **kw):
    layers = [StackLayer(algan(0.0), buffer, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
          Electrode("gate", "schottky", 300, 400, barrier=1.0),
          Electrode("drain", "ohmic", 600, width, implant_depth=40, implant_donors=1e20)]
    mob = kw.pop("mobility", Mobility(1800.0, "test", saturation_velocity=1e7))
    grid = dict(dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40, fine_depth=40)
    grid.update(kw)
    return layers, Device(layers, width, el, mob, Surface("pinned", barrier=1.0), **grid)


def _vertical(layers, V, holes=True, **kw):
    z = np.linspace(0, sum(L.thickness for L in layers), kw.pop("nz", 201))
    x = np.array([0.0, 10.0, 20.0])
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    mob = kw.pop("mobility", Mobility(300.0, "test"))
    extra = dict(hole_mass=1.0, hole_mobility=Mobility(20.0, "test")) if holes else {}
    extra.update(kw)
    dev = Device(layers, 20.0, el, mob,
                 Surface("pinned", barrier=1.0), grid=dict(x=x, z=z), **extra)
    return dev


# ------------------------------------------------------------ inputs
def test_inputs_refused():
    with pytest.raises(ValueError):
        Mobility(1000.0, "")
    with pytest.raises(ValueError):
        Mobility(1000.0, "r", beta=0.5)
    with pytest.raises(ValueError):
        Electrode("g", "schottky", 0, 1)
    with pytest.raises(ValueError):
        Electrode("s", "ohmic", 0, 1, implant_depth=10)
    with pytest.raises(ValueError):
        Trap("acceptor", 0.5, "ec", 1e17, 1e-8, 0.0, "", where="buffer")
    with pytest.raises(ValueError):
        Surface("pinned")
    layers = [StackLayer(algan(0.0), 100.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    with pytest.raises(ValueError, match="no charge-neutral"):
        Device(layers, 300.0, [Electrode("s", "ohmic", 0, 100)],
               Mobility(1000.0, "r"), Surface("pinned", barrier=1.0))
    with pytest.raises(ValueError, match="surface traps"):
        Device(layers, 300.0, [Electrode("s", "ohmic", 0, 100, implant_depth=30,
                                         implant_donors=1e20)],
               Mobility(1000.0, "r"), Surface("pinned", barrier=1.0),
               traps=[Trap("donor", 1.0, "ec", 1e13, 1e-8, 0.0, "r", where="surface")])


def test_mobility_formula():
    m = Mobility(1000.0, "r", saturation_velocity=1e7, beta=2.0, temperature_exponent=1.5)
    E = 3e4
    mu0 = 1000.0 * (400 / 300.0) ** -1.5
    E0 = 1e-3 * 1e7 / mu0
    Ee = math.sqrt(E * E + E0 * E0) - E0
    assert m.value(E, 400.0) == pytest.approx(mu0 / math.sqrt(1 + (mu0 * Ee / 1e7) ** 2), rel=1e-14)
    # the rounding moves mu by less than 0.1 percent of the saturation term
    assert m.value(E, 400.0) == pytest.approx(mu0 / math.sqrt(1 + (mu0 * E / 1e7) ** 2), rel=1e-3)
    assert m.value(0.0) == pytest.approx(1000.0, rel=1e-15)
    assert Mobility(500.0, "r").value(1e6) == pytest.approx(500.0)


def test_capture_coefficient():
    c = capture_coefficient(1e-15, 0.2, 300.0)
    v = math.sqrt(8 * KB * QE * 300.0 / (math.pi * 0.2 * M0)) * 100
    assert c == pytest.approx(1e-15 * v, rel=1e-14)
    assert 2.3e-8 < c < 2.5e-8


# ------------------------------------------------------------ against 1D
@pytest.mark.parametrize("V", [0.0, 2.5, 2.9])
def test_2d_equals_1d_diode(V):
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 100.0, acceptors=1e-4, name="p"),
              StackLayer(g, 100.0, donors=1e-4, name="n")]
    tr = Transport(300.0, 20.0, 1.0, reference="test")
    r1 = solve_drift_diffusion(layers, [tr, tr], Contact("ohmic"),
                               Contact("ohmic", voltage=V), dz=1.0)
    dev = _vertical(layers, V)
    st = dev.solve({"bot": V})
    J2 = st.current["bot"] / (20e-7 * 0.1)          # A/mm over 20 nm -> A/cm^2
    assert np.max(np.abs(st.psi[1] - r1.psi)) < 1e-9
    if V == 0:
        assert abs(J2) < 1e-20
    else:
        assert J2 == pytest.approx(r1.current_density, rel=1e-8)
    assert abs(st.current["top"] + st.current["bot"]) <= 1e-9 * abs(st.current["bot"]) + 1e-15


@pytest.mark.parametrize("beta,force", [(1.0, "field"), (2.0, "field"),
                                        (1.0, "quasi-fermi")])
def test_velocity_saturation_uniform_bar(beta, force):
    # a uniformly doped bar: n = N_D and a uniform field, so the discrete
    # current must equal q n mu(E) E exactly
    g = algan(0.0)
    layers = [StackLayer(g, 500.0, donors=1e-3)]
    mob = Mobility(1000.0, "test", saturation_velocity=1.5e7, beta=beta)
    dev = _vertical(layers, 0.0, holes=False, mobility=mob, nz=101,
                    driving_force=force, statistics="boltzmann")
    for V in (0.01, 1.0, 20.0):
        st = dev.solve({"bot": V})
        E = V / 500e-7
        J = QE * 1e18 * mob.value(E) * E
        J2 = st.current["bot"] / (20e-7 * 0.1)
        assert J2 == pytest.approx(J, rel=1e-6)


def test_3d_equals_2d():
    layers, d2 = _hemt()
    s2 = d2.solve({"drain": 2.0, "gate": -1.0})
    x, z = d2.axes
    el = d2.electrodes
    d3 = Device(layers, 700.0, el, Mobility(1800.0, "test", saturation_velocity=1e7),
                Surface("pinned", barrier=1.0), depth=40.0,
                grid=dict(x=x, y=np.array([0.0, 20.0, 40.0]), z=z))
    s3 = d3.solve({"drain": 2.0, "gate": -1.0})
    assert s3.current["drain"] / 4e-5 == pytest.approx(s2.current["drain"], rel=1e-9)
    assert np.max(np.abs(s3.psi[:, 1, :] - s2.psi)) < 1e-9


# ------------------------------------------------------------ transistor
def test_hemt_kirchhoff_and_threshold():
    layers, dev = _hemt()
    V, I, S = dev.sweep("drain", [0.5, 2.0, 5.0], bias={"gate": 0.0})
    for s in S:
        tot = sum(s.current.values())
        assert abs(tot) < 1e-9 * abs(s.current["drain"])
        assert s.heat.sum() == pytest.approx(s.power, rel=1e-8)
    assert np.all(np.diff(I["drain"]) > 0)
    # the drain current at a small drain voltage vanishes near the 1D
    # threshold of the layers under the gate
    vt = threshold_voltage(layers, 1.0)
    on = dev.solve({"drain": 0.1, "gate": vt + 1.0})
    off = dev.solve({"drain": 0.1, "gate": vt - 0.5}, guess=on)
    assert off.current["drain"] < 0.05 * on.current["drain"]


# ------------------------------------------------------------ traps
def test_trap_equilibrium_is_fermi_function():
    g = algan(0.0)
    layers = [StackLayer(g, 300.0, donors=1e-3, name="n")]
    t = Trap("acceptor", 0.3, "ec", 1e16, 1e-8, 0.0, "test", where="n")
    dev = _vertical(layers, 0.0, holes=False, traps=[t], statistics="boltzmann", nz=61)
    st = dev.equilibrium()
    Et = st.ec - 0.3
    f = 1.0 / (1.0 + np.exp(Et / (KB * 300.0)))
    assert np.nanmax(np.abs(st.occupancy[0] - f)) < 1e-12


def test_trap_emission_backward_euler_exact():
    # few traps in a doped layer: n stays fixed, so each step of backward
    # Euler multiplies the distance to the final occupancy by 1/(1 + lam h)
    g = algan(0.0)
    layers = [StackLayer(g, 300.0, donors=1e-3, name="n")]
    cn = 1e-8
    t = Trap("acceptor", 0.6, "ec", 1e8, cn, 0.0, "test", where="n")
    dev = _vertical(layers, 0.0, holes=False, traps=[t], statistics="boltzmann", nz=31)
    st = dev.equilibrium()
    tr = dev._traps[0]
    n = 1e-3 * 1e21 * 1e-21                       # nm^-3
    lam = tr["cn"] * n + tr["en"][0]
    finf = tr["cn"] * n / lam
    times = np.logspace(-12, -6, 7)
    r = dev.transient(st, {}, times, rise_time=0.0, initial_occupancy=[0.0])
    f = 0.0
    cum, k = 0.0, 0
    expected = []
    for h in r.steps:
        f = finf + (f - finf) / (1.0 + lam * h)
        cum += h
        if k < times.size and cum >= times[k] * (1 - 1e-12):
            expected.append(f)
            k += 1
    assert np.allclose(r.mean_occupancy[0], expected, rtol=1e-6)
    assert r.mean_occupancy[0][-1] == pytest.approx(finf, rel=1e-3)


def test_drain_lag_recovers_to_steady_state():
    layers = [StackLayer(algan(0.0), 200.0, name="buffer", donors=1e-6),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    cn = capture_coefficient(1e-15, 0.2)
    fe = Trap("acceptor", 0.5, "ec", 1e17, cn, 0.0, "test", where="buffer",
              z_range=(0.0, 150.0))
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
          Electrode("gate", "schottky", 300, 400, barrier=1.0),
          Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20)]
    dev = Device(layers, 700.0, el, Mobility(1800.0, "test", saturation_velocity=1e7),
                 Surface("pinned", barrier=1.0), traps=[fe], dx_fine=10, dx_max=50,
                 dz_fine=2.0, dz_max=40, fine_depth=40)
    stress = dev.solve({"drain": 10.0})
    final = dev.solve({"drain": 2.0}, guess=stress)
    r = dev.transient(stress, {"drain": 2.0}, np.logspace(-9, 0, 10))
    # the current starts low (traps filled during the stress hold
    # electrons) and recovers to the steady state of the new bias
    assert r.current["drain"][-1] == pytest.approx(final.current["drain"], rel=1e-6)
    assert r.current["drain"][0] < 0.99 * final.current["drain"]
    assert r.mean_occupancy[0][0] > r.mean_occupancy[0][-1]


def test_surface_states_close_to_pinned():
    # many surface donors at E_c - 1.0 eV pin the surface near 1.0 eV, so
    # the electron layer matches the pinned-surface model
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20)]
    kw = dict(dx_fine=10, dx_max=50, dz_fine=1.0, dz_max=40, fine_depth=40)
    pinned = Device(layers, 600.0, el, Mobility(1000.0, "t"),
                    Surface("pinned", barrier=1.0), **kw).equilibrium()
    don = Trap("donor", 1.0, "ec", 1e15, 1e-8, 0.0, "test", where="surface")
    st = Device(layers, 600.0, el, Mobility(1000.0, "t"), Surface("states"),
                traps=[don], **kw).equilibrium()
    ns_p = pinned.sheet()[-1]
    ns_s = st.sheet()[-1]
    assert ns_s == pytest.approx(ns_p, rel=0.03)
    surf = st.ec[-1, -1]
    assert 0.8 < surf < 1.05


# ------------------------------------------------------------ heat
def test_heat_kirchhoff_slab():
    z = np.linspace(0, 100e3, 401)
    x = np.array([0.0, 500.0, 1000.0])
    q = 2e6
    H = np.zeros((3, z.size))
    H[:, -1] = q * np.array([250, 500, 250]) * 1e-9
    T, _ = solve_heat([x], z, [GAN_BULK] * (z.size - 1), H, 300.0)
    a = GAN_BULK.exponent
    exact = 300 * (1 + q * 100e3 * 1e-9 * (1 - a) / (GAN_BULK.k300 * 300)) ** (1 / (1 - a))
    assert T[1, -1] == pytest.approx(exact, abs=1e-6)


def test_heat_strip_fourier():
    k = 150.0
    mk = ThermalMaterial("c", k, 0.0, "test")
    W, L, a, q0 = 20e3, 50e3, 2e3, 1e7
    x = np.linspace(0, W, 801)
    z = np.linspace(0, L, 501)
    h = np.diff(x)
    lo = np.r_[0, h / 2]
    hi = np.r_[h / 2, 0]
    qq = np.array([q0 * max(0.0, min(xi + r, a) - max(xi - l_, 0.0))
                   for xi, l_, r in zip(x, lo, hi)]) * 1e-9
    H = np.zeros((x.size, z.size))
    H[:, -1] = qq
    T, _ = solve_heat([x], z, [mk] * (z.size - 1), H, 300.0)
    Wm, Lm, am = W * 1e-9, L * 1e-9, a * 1e-9

    def exact(xx):
        s = q0 * am * Lm / (k * Wm)
        for m in range(1, 6000):
            b = m * math.pi / Wm
            s += 2 * q0 * math.sin(b * am) / (k * Wm * b * b) * math.cos(b * xx) * math.tanh(b * Lm)
        return 300 + s

    for xi in (0.0, 1000.0, 5000.0, 15000.0):
        i = int(np.argmin(np.abs(x - xi)))
        assert T[i, -1] - 300 == pytest.approx(exact(xi * 1e-9) - 300, rel=3e-4)


def test_boundary_resistance_layer():
    L = boundary_resistance(1e-8, "test", thickness_nm=10.0)
    assert L.material.k300 * 1e-8 == pytest.approx(10e-9)
    # 1D: a flux q through the layer raises the temperature by q R
    z = np.linspace(0, 10.0, 11)
    H = np.zeros((2, z.size))
    q = 1e8
    H[:, -1] = q * np.array([5, 5]) * 1e-9
    T, _ = solve_heat([np.array([0.0, 10.0])], z, [L.material] * 10, H, 300.0)
    assert T[0, -1] - 300 == pytest.approx(q * 1e-8, rel=1e-9)


def test_self_heating_energy_and_current_drop():
    layers, dev = _hemt(mobility=Mobility(1800.0, "test", saturation_velocity=1e7,
                                          temperature_exponent=2.35))
    th = ThermalModel(GAN_BULK, substrate=(ThermalLayer("SiC", 50e3, SIC_4H_ROOM),),
                      lateral_extension=5e3)
    iso = dev.solve({"drain": 8.0})
    hot = dev.solve({"drain": 8.0}, guess=iso, thermal=th)
    assert hot.heat.sum() == pytest.approx(hot.power, rel=1e-8)
    assert hot.temperature.max() > 310
    assert hot.current["drain"] < iso.current["drain"]
    # the temperature is the heat equation's answer for the final heat
    T2 = dev._temperature(dev._thermal_setup(th), hot.heat, hot.temperature.ravel())
    assert np.max(np.abs(T2 - hot.temperature.ravel())) < 0.05


# ------------------------------------------------------------ app
def _app_spec(**extra):
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [{"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                        "doping": "none"},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0,
                        "doping": "none"}],
            "surface_barrier": 1.0, "device": {"barrier": 1.0},
            "line_charge": {"mode": "measured"}}
    spec.update(extra)
    return spec


def test_app_transistor_curves_and_heating():
    from kpenvelope.gui.api import _bajaj_vsat, handle
    t = {"gate_length": 200, "gap_sg": 300, "gap_gd": 400, "contact": 200,
         "sweep": "drain", "from": 0, "to": 4, "step": 2, "fixed": 0,
         "heating": True, "substrate": "SiC", "substrate_um": 50, "pitch_um": 20}
    r = handle("transistor", _app_spec(transistor=t))
    assert "error" not in r, r.get("error")
    for k in range(3):
        assert abs(r["drain"][k] + r["gate"][k] + r["source"][k]) < 1e-8
    assert r["drain"][0] == pytest.approx(0.0, abs=1e-12)
    assert r["drain"][2] > r["drain"][1] > 0
    assert r["tmax"][2] > r["tmax"][1] > 300.0
    assert r["vsat"] == pytest.approx(_bajaj_vsat(r["ns"]), rel=1e-12)
    # Bajaj et al., Eq. 1: 1e7 / (0.38 + (n_s / 1.8e13)^0.45)
    assert _bajaj_vsat(1.8e13) == pytest.approx(1e7 / 1.38, rel=1e-14)
    bad = handle("transistor", _app_spec(transistor=dict(t, step=0.01, to=40)))
    assert "error" in bad


def test_app_drain_lag():
    from kpenvelope.gui.api import handle
    t = {"gate_length": 200, "gap_sg": 300, "gap_gd": 400, "contact": 200}
    r = handle("lag", _app_spec(transistor=t, lag={"stress_drain": 10, "drain": 3,
                                                    "density": 1e17, "clean": 80}))
    assert "error" not in r, r.get("error")
    assert r["drain"][-1] == pytest.approx(r["steady"], rel=1e-6)
    assert r["dip"] > 0.0
    assert 1e-4 < r["emission_time"] < 1e-1


@pytest.mark.parametrize("force", ["field", "quasi-fermi"])
def test_analytic_jacobian_matches_central_differences(force):
    layers = [StackLayer(algan(0.0), 60.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 40, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 80, 120, barrier=1.0),
          Electrode("drain", "ohmic", 160, 200, implant_depth=30, implant_donors=1e20)]
    dev = Device(layers, 200.0, el, Mobility(1500.0, "test", saturation_velocity=1e7),
                 Surface("pinned", barrier=1.0), dx_fine=20, dx_max=40, dz_fine=5.0,
                 dz_max=20, fine_depth=20)
    st = dev.solve({"drain": 1.0, "gate": -0.5})
    dev.force = force                  # the derivatives at a biased state
    x, b, T = st._x, st.bias, np.full(dev.N, 300.0)
    F0 = dev._residual(x, b, T)
    J = dev._jacobian(x, F0, b, T, None, None).toarray()
    scale = np.abs(J).max(axis=1) + 1e-300
    worst = 0.0
    for c in range(x.size):
        h = 1e-7 * max(1.0, abs(x[c]))
        xp, xm = x.copy(), x.copy()
        xp[c] += h
        xm[c] -= h
        cd = (dev._residual(xp, b, T) - dev._residual(xm, b, T)) / (2 * h)
        worst = max(worst, float(np.max(np.abs(J[:, c] - cd) / scale)))
    assert worst < 1e-5


def test_traps_with_holes_equal_srh_lifetimes():
    # a trap with N_t c_n = 1/tau_n and N_t c_p = 1/tau_p recombines like the
    # 1D solver's Shockley-Read-Hall lifetimes (its charge is negligible here)
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 100.0, acceptors=1e-4, name="p"),
              StackLayer(g, 100.0, donors=1e-4, name="n")]
    tau = 1e-9
    tr = Transport(300.0, 20.0, 1.0, reference="test", tau_n=tau, tau_p=tau,
                   trap_level=0.0)
    V = 2.6
    r1 = solve_drift_diffusion(layers, [tr, tr], Contact("ohmic"),
                               Contact("ohmic", voltage=V), dz=1.0, statistics="boltzmann")
    Nt = 1e8
    c = 1.0 / (Nt * tau)
    traps = [Trap("acceptor", g.eg / 2, "ec", Nt, c, c, "test", where=w) for w in ("p", "n")]
    dev = _vertical(layers, V, traps=traps, statistics="boltzmann")
    st = dev.solve({"bot": V})
    J2 = st.current["bot"] / (20e-7 * 0.1)
    assert J2 == pytest.approx(r1.current_density, rel=1e-6)
    r0 = solve_drift_diffusion(layers, Transport(300.0, 20.0, 1.0, reference="test"),
                               Contact("ohmic"), Contact("ohmic", voltage=V), dz=1.0,
                               statistics="boltzmann")
    assert r1.current_density > 2 * r0.current_density      # recombination matters


# ------------------------------------------------------------ 0.20.0
def test_small_signal_matches_dc_and_kirchhoff():
    layers, dev = _hemt()
    st = dev.solve({"drain": 3.0, "gate": -1.0})
    ports, Y = dev.small_signal(st, [0.0, 1e3, 1e10])
    assert ports == ["source", "gate", "drain", "surface"]
    big = np.abs(Y).max()
    assert np.abs(Y.sum(axis=1)).max() < 1e-8 * big       # currents conserved
    assert np.abs(Y.sum(axis=2)).max() < 1e-8 * big       # a common shift does nothing
    names = ["source", "gate", "drain"]
    d = 1e-4
    for k, nm in enumerate(names):
        bp, bm = dict(st.bias), dict(st.bias)
        bp[nm] += d
        bm[nm] -= d
        sp_ = dev.solve(bp, guess=st, tol=1e-11)
        sm = dev.solve(bm, guess=st, tol=1e-11)
        for c, nc in enumerate(names):
            g = (sp_.current[nc] - sm.current[nc]) / (2 * d)
            assert Y[0, ports.index(nc), k].real == pytest.approx(g, rel=1e-5, abs=1e-9)
        cg = (sp_.charge["gate"] - sm.charge["gate"]) / (2 * d)
        w = 2 * math.pi * 1e3
        assert Y[1, 1, k].imag / w == pytest.approx(cg, rel=1e-6)


def test_series_resistance_bar_and_two_port():
    from kpenvelope.device import with_series
    g = algan(0.0)
    layers = [StackLayer(g, 500.0, donors=1e-3)]
    z = np.linspace(0, 500.0, 101)
    x = np.array([0.0, 10.0, 20.0])
    mob = Mobility(1000.0, "test")

    def bar(r_top, r_bot):
        el = [Electrode("top", "ohmic", 0, 20, resistance=r_top),
              Electrode("bot", "ohmic", 0, 20, face="bottom", resistance=r_bot)]
        return Device(layers, 20.0, el, mob, Surface("pinned", barrier=1.0),
                      grid=dict(x=x, z=z), statistics="boltzmann")

    I0 = bar(0.0, 0.0).solve({"bot": 0.5}).current["bot"]
    Rbar = 0.5 / I0                                   # ohm mm, exactly linear here
    dev = bar(0.3 * Rbar, 0.2 * Rbar)
    st = dev.solve({"bot": 0.5})
    assert st.current["bot"] == pytest.approx(0.5 / (1.5 * Rbar), rel=1e-9)
    assert st.internal["bot"] - st.internal["top"] == pytest.approx(0.5 / 1.5, rel=1e-9)
    # the two-port through the resistors equals the DC conductance with them
    ports, Y = bar(0.0, 0.0).small_signal(bar(0.0, 0.0).solve({"bot": 0.5}), [0.0])
    Ye = with_series(ports, Y, {"top": 0.3 * Rbar, "bot": 0.2 * Rbar})
    assert Ye[0, ports.index("bot"), ports.index("bot")].real == pytest.approx(
        1 / (1.5 * Rbar), rel=1e-7)


def test_rf_figures_quasi_static_limit():
    from kpenvelope.device import rf_figures, two_port
    layers, dev = _hemt()
    st = dev.solve({"drain": 5.0, "gate": -1.0})
    f = np.array([1e6, 1e8, 1e9, 1e10, 1e11, 1e12])
    ports, Y = dev.small_signal(st, f)
    Y2 = two_port(ports, Y)
    r = rf_figures(f, Y2)
    gm = Y2[0, 1, 0].real
    cgg = Y2[0, 0, 0].imag / (2 * math.pi * 1e6)
    assert r["fT_extrapolated"] == pytest.approx(gm / (2 * math.pi * cgg), rel=1e-6)
    assert 0 < r["fT"] <= 1.05 * r["fT_extrapolated"]


def test_heat_in_time_semi_infinite_and_energy():
    from kpenvelope.thermal import _HeatGrid, solve_heat_transient
    m = ThermalMaterial("c", 150.0, 0.0, "test", density=6.15, cp_a=0.49)
    L = 400e3
    z = np.unique(np.r_[np.linspace(0, L - 20e3, 200), np.linspace(L - 20e3, L, 801)])
    x = np.array([0.0, 1000.0])
    q = 1e9
    H = np.zeros((2, z.size))
    H[:, -1] = q * np.array([500, 500]) * 1e-9
    times = np.logspace(-9, -4, 400)
    t, T = solve_heat_transient([x], z, [m] * (z.size - 1), H, 300.0, times)
    alpha = 150.0 / (6.15e6 * 0.49)
    for tt in (1e-7, 1e-5):
        k = int(np.argmin(np.abs(times - tt)))
        exact = 2 * q / 150.0 * math.sqrt(alpha * times[k] / math.pi)
        assert T[k][0, -1] - 300 == pytest.approx(exact, rel=5e-3)
    g = _HeatGrid([x], z, [m] * (z.size - 1))
    E = np.sum(g.capacity(T[-1]) * (T[-1].ravel() - 300))
    assert E == pytest.approx(q * 1e-6 * times[-1], rel=1e-4)


def test_heat_capacities_from_sources():
    # GaN: c_p = (38.1 + 8.96e-3 T) / M J/(g K); 0.49 J/(g K) is listed too
    assert GAN_BULK.cp_a + GAN_BULK.cp_b * 300.0 == pytest.approx(0.49, abs=0.005)
    assert GAN_BULK.volumetric_heat(300.0) == pytest.approx(
        6.15e6 * (38.1 + 8.96e-3 * 300) / (69.723 + 14.007), rel=1e-12)
    assert SIC_4H_ROOM.volumetric_heat(400.0) == pytest.approx(3.211e6 * 0.69, rel=1e-12)
    with pytest.raises(ValueError):
        ThermalMaterial("x", 1.0, 0.0, "r", density=1.0)


def test_pulse_reaches_the_steady_self_heated_state():
    layers, dev = _hemt(mobility=Mobility(1800.0, "test", saturation_velocity=1e7,
                                          temperature_exponent=2.35))
    th = ThermalModel(GAN_BULK, substrate=(ThermalLayer("SiC", 50e3, SIC_4H_ROOM),),
                      lateral_extension=5e3)
    hot = dev.solve({"drain": 8.0}, thermal=th)
    t, I, Tm, _ = dev.pulse({"drain": 8.0}, np.logspace(-9, -2, 15), th)
    assert np.all(np.diff(I["drain"]) <= 1e-6)          # it only heats up
    assert I["drain"][-1] == pytest.approx(hot.current["drain"], rel=1e-4)
    assert Tm[-1] == pytest.approx(hot.temperature.max(), abs=0.05)
    assert Tm[0] < Tm[-1]


def test_avalanche_gain_matches_mcintyre():
    from kpenvelope.device import IMPACT_GAN
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 50.0, acceptors=1e-2, name="p"), StackLayer(g, 300.0, name="i"),
              StackLayer(g, 50.0, donors=1e-2, name="n")]
    gen = (lambda xx, zz: 1e24 if 52 <= zz <= 54 else 0.0)
    kw = dict(holes=True, statistics="boltzmann", generation=gen, nz=401)
    d0 = _vertical(layers, 0.0, **kw)
    d1 = _vertical(layers, 0.0, ionization=IMPACT_GAN, **kw)
    g0 = g1 = None
    for V in (80.0, 95.0):
        s0 = d0.solve({"top": V}, guess=g0)
        s1 = d1.solve({"top": V}, guess=g1)
        g0, g1 = s0, s1
        M = s1.current["top"] / s0.current["top"]
        z = s1.axes[-1]
        E = -np.gradient(s1.psi[1], z) * 1e7
        an, ap = IMPACT_GAN.alpha(E)
        m = z >= 53
        zs, a, b = z[m] * 1e-7, an[m], ap[m]
        inner = np.concatenate([[0], np.cumsum(0.5 * ((a - b)[1:] + (a - b)[:-1]) * np.diff(zs))])
        f = a * np.exp(-inner)
        I = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zs))
        Mc = 1 / (1 - I)
        assert M - 1 == pytest.approx(Mc - 1, rel=0.05)      # McIntyre, electron injection
        assert M > 1.0
    assert M > 1.5


def test_ionization_integral_and_breakdown():
    from kpenvelope.device import IMPACT_GAN
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 50.0, acceptors=1e-2, name="p"), StackLayer(g, 300.0, name="i"),
              StackLayer(g, 50.0, donors=1e-2, name="n")]
    dev = _vertical(layers, 0.0, statistics="boltzmann", nz=401)
    st = dev.solve({"top": 95.0})
    r = dev.ionization_integral(st, IMPACT_GAN)
    z = st.axes[-1]
    E = np.abs(np.gradient(st.psi[1], z)) * 1e7
    an, ap = IMPACT_GAN.alpha(E)
    zc = z * 1e-7
    d = an - ap
    cum = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(zc))])
    f = an * np.exp(-cum)
    In = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zc))
    assert r["I_n"] == pytest.approx(In, rel=1e-3)
    vbr, V, I, _ = dev.breakdown_voltage("top", [95.0, 100.0, 105.0, 110.0, 115.0],
                                         guess=st, ionization=IMPACT_GAN)
    assert I[-1] >= 1.0 > I[-2]
    assert V[-2] < vbr < V[-1]
    # breakdown near 3.3 to 3.7 MV/cm over the 300 nm undoped layer
    assert 95.0 < vbr < 115.0


def test_electron_temperature_uniform_field():
    from kpenvelope.electrons import KB as kB
    g = algan(0.0)
    layers = [StackLayer(g, 3000.0, donors=1e-3)]
    mob = Mobility(1000.0, "test", saturation_velocity=1.5e7)
    dev = _vertical(layers, 0.0, holes=False, mobility=mob, nz=1201, statistics="boltzmann")
    tau = 0.47e-12
    for V in (1.5, 15.0):
        st = dev.solve({"bot": V})
        Tn = dev.electron_temperature(st, tau)
        E = V / 3000e-7
        exact = 300.0 + 2.0 / 3.0 * tau * mob.value(E) * E * E / kB
        assert Tn[1, 400] == pytest.approx(exact, rel=1e-4)    # 2 um downstream of the entry
        assert Tn[1, 0] == pytest.approx(300.0) and Tn[1, -1] == pytest.approx(300.0)


def test_gate_tunneling_matches_1d_under_a_wide_gate():
    from kpenvelope import gate_leakage
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 10.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 400, 2400, barrier=1.0),
          Electrode("drain", "ohmic", 2700, 2800, implant_depth=30, implant_donors=1e20)]
    dev = Device(layers, 2800.0, el, Mobility(1800.0, "test"), Surface("pinned", barrier=1.0),
                 dx_fine=10, dx_max=100, dz_fine=0.5, dz_max=40, fine_depth=40)
    st = dev.solve({"gate": -3.0})
    I, Jc = dev.gate_tunneling(st)
    nz = st.axes[-1].size
    xs = st.axes[0][dev.enodes["gate"] // nz]
    k = int(np.argmin(np.abs(xs - 1400.0)))
    _, J1, _ = gate_leakage(layers, 1.0, [-3.0], dz=0.5)
    assert Jc[k] == pytest.approx(J1[0], rel=5e-3)
    assert I < 0                     # electrons enter from the metal at reverse bias
