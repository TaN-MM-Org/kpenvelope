"""Tests for v0.22.0: the lattice temperature in the carrier statistics
(leakage that grows with temperature, thermoelectric currents, thermal
runaway from leakage), hot electrons together with self-heating,
self-heating and tunnel barriers on meshes, nonlocal band-to-band
tunneling (recombination, Esaki diodes), and process topography
(deposition, etching, oxidation)."""
import dataclasses
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan  # noqa: E402
from kpenvelope.device import (BandToBand, Device, Dielectric, Electrode,  # noqa: E402
                               Mobility, Surface, Trap, TunnelBarrier, Varshni)
from kpenvelope.electrons import HBAR, KB, M0  # noqa: E402
from kpenvelope.interfaces import interface_charges  # noqa: E402
from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_grid, mesh_from_points  # noqa: E402
from kpenvelope.process import Anneal, Implant, Structure  # noqa: E402
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel  # noqa: E402

Q = 1.602176634e-19
VARSHNI = Varshni(4.73e-4, 636.0, 0.5, "test numbers")


def _no_pol(layers):
    return {c.index: -c.sigma_nm2 for c in interface_charges(layers)}


def _narrow(T=300.0, shift=False, eg=1.12):
    """A made-up narrow-gap test material (numbers for the tests only)."""
    m = dataclasses.replace(algan(0.0), name="narrow", eg=eg, me=0.26, eps_r=11.7)
    if shift:
        d = float(VARSHNI.shift(T, 300.0))
        m = dataclasses.replace(m, eg=m.eg + d, ec=m.ec + 0.5 * d)
    return m


def _leaky_diode(T=300.0, full=True, shift=False, eg=1.12, nz=201, trap=1e15, dop=1e-4):
    m = _narrow(T, shift, eg)
    m1 = dataclasses.replace(m, name="narrow2")
    layers = [StackLayer(m, 2000.0, acceptors=dop), StackLayer(m1, 2000.0, donors=dop)]
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    tr = [Trap("acceptor", 0.5 * eg, "ec", trap, 1e-8, 1e-8, "test", ("narrow", "narrow2"))]
    kw = dict(temperature_model="full", band_gap_temperature=VARSHNI) if full else {}
    return Device(layers, 20.0, el, Mobility(1000.0, "t", temperature_exponent=1.5),
                  Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 4000, nz)),
                  hole_mass=0.81, hole_mobility=Mobility(400.0, "t"), temperature_K=T,
                  extra_sheet=_no_pol(layers), traps=tr, statistics="boltzmann", **kw)


# ------------------------------------------------- lattice temperature
def test_full_model_at_one_temperature_is_the_device_built_at_it():
    # N_c, N_v, the gap, trap emission, contacts and fluxes at T1 everywhere
    # must give exactly the device made for T1 (its gap shifted by hand)
    full = _leaky_diode()
    T1 = 380.0
    iso = _leaky_diode(T1, full=False, shift=True)
    for V in (-5.0, 0.4):
        a = iso.solve({"bot": V})
        b = full.solve({"bot": V}, lattice_temperature=T1)
        assert b.current["bot"] == pytest.approx(a.current["bot"], rel=1e-8)
        assert np.max(np.abs(a.psi - b.psi)) < 1e-9
    # and at the device temperature it is the old model to the last bit
    old = _leaky_diode(full=False)
    a, b = old.solve({"bot": -5.0}), full.solve({"bot": -5.0}, lattice_temperature=300.0)
    assert a.current["bot"] == b.current["bot"]
    # leakage grows with temperature
    assert abs(full.solve({"bot": -5.0}, lattice_temperature=T1).current["bot"]) > \
        30 * abs(a.current["bot"])


def test_thermoelectric_current_of_a_bar_in_a_temperature_gradient():
    # Gamma = -mu n [grad F_n + k (ln(N_c / n) + 5/2) grad T] (Boltzmann, n = N_d,
    # mu constant): with both ends grounded, Gamma L / (mu N_d) depends only on
    # the end temperatures, whatever the profile between them
    from scipy.integrate import quad
    L, nd = 4000.0, 1e18
    layers = [StackLayer(algan(0.0), L, donors=nd * 1e-21)]
    z = np.linspace(0, L, 201)
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    dev = Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                 grid=dict(x=np.array([0., 10., 20.]), z=z), statistics="boltzmann",
                 temperature_model="full")
    T = np.tile(300.0 + 100.0 * z / L + 60 * np.sin(np.pi * z / L) ** 2, 3)
    st = dev.solve({}, lattice_temperature=T)
    Nc0 = dev._lat["Nc"][0] * 1e21
    integ = quad(lambda t: np.log(Nc0 * (t / 300.0) ** 1.5 / nd) + 2.5, 300, 400)[0] * KB
    I = Q * 1000.0 * nd * integ / (L * 1e-7) * 20e-7 * 0.1
    assert abs(st.current["bot"]) == pytest.approx(I, rel=1e-6)
    assert st.current["bot"] > 0 and abs(sum(st.current.values())) < 1e-12


def test_jacobian_of_the_temperature_dependent_fluxes():
    layers = [StackLayer(algan(0.0), 300.0, acceptors=1e-2),
              StackLayer(algan(0.2), 30.0, donors=1e-3), StackLayer(algan(0.0), 300.0, donors=1e-2)]
    z = np.unique(np.r_[np.linspace(0, 630, 40), [300, 330]])
    el = [Electrode("top", "ohmic", 0, 40), Electrode("bot", "ohmic", 0, 40, face="bottom")]
    tr = [Trap("acceptor", 1.0, "ec", 1e16, 1e-8, 1e-8, "t", "GaN")]
    for force, stat in (("field", "fermi"), ("quasi-fermi", "boltzmann")):
        dev = Device(layers, 40.0, el, Mobility(300.0, "t", saturation_velocity=1e7,
                                                temperature_exponent=1.5),
                     Surface("pinned", barrier=1.0), grid=dict(x=np.linspace(0, 40, 4), z=z),
                     hole_mass=1.0, hole_mobility=Mobility(20.0, "t", saturation_velocity=1e7),
                     statistics=stat, temperature_model="full", traps=tr, driving_force=force,
                     band_gap_temperature=Varshni(0.909e-3, 830.0, 0.6, "t"))
        X, Z = dev.coords
        T = 300 + 150 * np.exp(-((Z - 320) / 150) ** 2) + 0.5 * X
        st = dev.solve({"bot": 2.0}, lattice_temperature=T)
        x = st._x + 1e-3 * np.random.default_rng(0).standard_normal(st._x.size)
        b = {"top": 0.0, "bot": 2.0}
        J = dev._jacobian(x, None, b, T, None, None).toarray()
        Jn = np.zeros_like(J)
        for k in range(x.size):
            e = np.zeros_like(x)
            e[k] = 1e-7
            Jn[:, k] = (dev._residual(x + e, b, T) - dev._residual(x - e, b, T)) / 2e-7
        sc = np.abs(Jn).max(axis=1, keepdims=True) + 1e-300
        assert np.max(np.abs(J - Jn) / sc) < 1e-6


def test_thermal_runaway_from_leakage_matches_the_lumped_model():
    # a leaky diode on a poor heat sink: the lumped model T = T0 + R_th V I(V, T)
    # (I from solves at a uniform lattice temperature) has no solution above
    # V* = max over T of the V that solves it
    from scipy.optimize import brentq, minimize_scalar
    dev = _leaky_diode(eg=0.66, nz=161, trap=1e16)
    th = ThermalModel(ThermalMaterial("d", 100.0, 0.0, "t"),
                      substrate=(ThermalLayer("sub", 1e5, ThermalMaterial("s", 3e-3, 0.0, "t")),))
    Rth = 1e-4 / (3e-3 * 20e-9) * 1e3                          # K per (W/mm)

    last = {}

    def Vof(T):
        def I(v):
            st = dev.solve({"bot": -v}, lattice_temperature=T, guess=last.get("st"))
            last["st"] = st
            return abs(st.current["bot"])
        return brentq(lambda v: Rth * v * I(v) - (T - 300.0), 0.5, 20.0, xtol=1e-7)
    # V(T) is flat at its maximum, so T needs no great precision there
    r = minimize_scalar(lambda T: -Vof(T), bounds=(303.0, 350.0), method="bounded",
                        options=dict(xatol=0.2))
    Vstar = -r.fun
    res = dev.thermal_runaway("bot", -np.linspace(0.6 * Vstar, 1.25 * Vstar, 7), th)
    assert abs(res["stable"]) <= Vstar * 1.003
    assert abs(res["runaway"]) >= Vstar * 0.997
    assert abs(res["runaway"] - res["stable"]) <= 2e-3 * Vstar


# --------------------------------------------- hot electrons + self-heating
def _uniform_bar():
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    mob = Mobility(1000.0, "test", saturation_velocity=1.5e7, temperature_exponent=1.5)
    return Device([StackLayer(algan(0.0), 3000.0, donors=1e-3, name="bar")], 20.0, el, mob,
                  Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 3000, 301)),
                  statistics="boltzmann")


def test_hot_electrons_hand_their_heat_to_the_lattice():
    dev = _uniform_bar()
    tau = 0.47e-12
    st = dev.solve({"bot": 3.0}, energy_relaxation=tau, lattice_temperature=300.0)
    # where the field is uniform, the energy lost by relaxation is the Joule heat
    assert st.lattice_heat[1, 150] == pytest.approx(st.heat[1, 150], rel=1e-4)
    # near the contact the electrons arrive hot, so the lattice there gets less
    # in the bulk and the contact gets the rest; in total it is the work
    # the field does on the electrons
    assert st.lattice_heat[1, 295] < 0.95 * st.heat[1, 295]
    assert np.sum(st.lattice_heat) == pytest.approx(st.power, rel=1e-3)
    th = ThermalModel({"bar": ThermalMaterial("b", 5000.0, 0.0, "t")},
                      substrate=(ThermalLayer("sub", 1e4, ThermalMaterial("s", 2000.0, 0.0, "t")),))
    a = dev.solve({"bot": 3.0}, thermal=th)
    b = dev.solve({"bot": 3.0}, thermal=th, energy_relaxation=tau)
    assert b.current["bot"] == pytest.approx(a.current["bot"], rel=1e-2)
    assert b.temperature.max() - 300 == pytest.approx(a.temperature.max() - 300, rel=1e-2)
    assert b.electron_temperature.max() > b.temperature.max() + 50


# ------------------------------------------------------------ meshes
def _two_layer(x, z):
    g = algan(0.0)
    mob = Mobility(500.0, "test", saturation_velocity=1e7, temperature_exponent=1.5)
    el = [Electrode("source", "ohmic", 0, 100),
          Electrode("gate", "schottky", 200, 300, barrier=0.8),
          Electrode("drain", "ohmic", 400, 500)]
    T = Device([StackLayer(g, 150.0, donors=1e-5, name="buf"),
                StackLayer(g, 50.0, donors=1e-3, name="ch")], 500.0, el, mob,
               Surface("states", fixed_charge=0.0), grid=dict(x=x, z=z), extra_sheet={0: 0.0})
    mesh = mesh_from_grid(x, z, lambda xx, zz: "buf" if zz < 150 else "ch")
    P = mesh.points
    top = np.isclose(P[:, 1], 200.0)
    con = {"source": np.flatnonzero(top & (P[:, 0] <= 100 + 1e-9)),
           "gate": np.flatnonzero(top & (P[:, 0] >= 200 - 1e-9) & (P[:, 0] <= 300 + 1e-9)),
           "drain": np.flatnonzero(top & (P[:, 0] >= 400 - 1e-9))}
    M = MeshDevice(mesh, {"buf": MeshRegion(material=g, mobility=mob, donors=1e-5),
                          "ch": MeshRegion(material=g, mobility=mob, donors=1e-3)}, el, con)
    return T, M


def test_self_heating_on_a_mesh_equals_the_tensor_grid():
    x = np.unique(np.r_[np.linspace(0, 100, 6), np.linspace(120, 380, 14),
                        np.linspace(400, 500, 6), np.linspace(150, 350, 11)])
    T, M = _two_layer(x, np.linspace(0, 200, 41))
    th = ThermalModel({"buf": ThermalMaterial("x", 3.0, 1.2, "t", density=6.15, cp_a=0.49,
                                              cp_b=1e-4),
                       "ch": ThermalMaterial("y", 1.0, 0.5, "t", density=6.0, cp_a=0.5)})
    b = {"drain": 3.0, "gate": 0.0}
    a, c = T.solve(b, thermal=th), M.solve(b, thermal=th)
    assert c.current["drain"] == pytest.approx(a.current["drain"], rel=1e-10)
    assert np.max(np.abs(a.temperature.ravel() - c.temperature)) < 1e-8
    assert a.temperature.max() > 400
    _, Ia, Ta, _ = T.pulse(b, [1e-11, 1e-10], th)
    _, Ic, Tc, _ = M.pulse(b, [1e-11, 1e-10], th)
    assert np.allclose(Ia["drain"], Ic["drain"], rtol=1e-10)
    assert np.allclose(Ta, Tc, rtol=1e-10)
    with pytest.raises(ValueError, match="substrate"):
        M.solve(b, thermal=ThermalModel(th.device_materials, substrate=(
            ThermalLayer("s", 100.0, ThermalMaterial("s", 1.0, 0.0, "t")),)))


def test_heat_on_an_unstructured_mesh_against_the_exact_solution():
    # uniform heating of a slab, sink at the bottom, k = k300 (T/300)^-a:
    # Kirchhoff's transform gives T exactly
    W, Hh, q, k3, al = 1000.0, 600.0, 2e15, 50.0, 1.3
    errs = []
    for nb in (11, 21, 41):
        rng = np.random.default_rng(1)
        h = Hh / (nb - 1)
        xb = np.linspace(0, W, int(round(W / h)) + 1)
        zb = np.linspace(0, Hh, nb)
        bnd = np.unique(np.r_[np.c_[xb, 0 * xb], np.c_[xb, Hh + 0 * xb],
                              np.c_[0 * zb, zb], np.c_[W + 0 * zb, zb]], axis=0)
        X, Z = np.meshgrid(xb[1:-1], zb[1:-1], indexing="ij")
        inner = np.c_[X.ravel(), Z.ravel()] + rng.uniform(-0.3, 0.3, (X.size, 2)) * h
        mesh = mesh_from_points(np.r_[bnd, inner], lambda x, z: "gan")
        P = mesh.points
        con = {"a": np.flatnonzero(np.isclose(P[:, 0], 0) & np.isclose(P[:, 1], Hh)),
               "b": np.flatnonzero(np.isclose(P[:, 0], W) & np.isclose(P[:, 1], Hh))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M = MeshDevice(mesh, {"gan": MeshRegion(material=algan(0.0),
                                                    mobility=Mobility(100.0, "t"), donors=1e-3)},
                           [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)], con)
        thd = M._thermal_setup(ThermalModel(ThermalMaterial("m", k3, al, "t")))
        T = M._temperature(thd, q * M.vol_all * 1e-18 / 1e3, None)
        zz = P[:, 1] * 1e-9
        theta = q * (Hh * 1e-9 * zz - zz ** 2 / 2) / k3
        Tex = 300.0 * (1 + (1 - al) * theta / 300.0) ** (1 / (1 - al))
        errs.append(np.max(np.abs(T - Tex)))
    assert errs[2] < 1e-3 and errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 8


def test_tunnel_barrier_on_a_mesh_equals_the_tensor_grid():
    thick = 3.0
    g, b = algan(0.0), algan(0.3)
    layers = [StackLayer(g, 60.0, donors=1e-2, name="bottom"), StackLayer(b, thick, name="barrier"),
              StackLayer(g, 60.0, donors=1e-2, name="top")]
    z = np.unique(np.round(np.r_[np.linspace(0, 59, 60), np.linspace(59, 67, 81),
                                 np.linspace(64, 123, 60)], 9))
    x = np.array([0, 10, 20.0])
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    T = Device(layers, 20.0, el, Mobility(300.0, "t"), Surface("states", fixed_charge=0.0),
               extra_sheet=_no_pol(layers), grid=dict(x=x, z=z),
               tunnel_barriers=[TunnelBarrier("barrier", n_energy=400)])
    mesh = mesh_from_grid(x, z, lambda xx, zz: "bottom" if zz < 60 else (
        "barrier" if zz < 60 + thick else "top"))
    P = mesh.points
    con = {"top": np.flatnonzero(np.isclose(P[:, 1], 123)),
           "bot": np.flatnonzero(np.isclose(P[:, 1], 0))}
    R = {"bottom": MeshRegion(material=g, mobility=Mobility(300.0, "t"), donors=1e-2),
         "barrier": MeshRegion(material=b, mobility=Mobility(300.0, "t")),
         "top": MeshRegion(material=g, mobility=Mobility(300.0, "t"), donors=1e-2)}
    M = MeshDevice(mesh, R, el, con, tunnel_barriers=[TunnelBarrier("barrier", n_energy=400)])
    assert abs(M.solve({"top": 0.0}).current["top"]) < 1e-12
    for V in (0.05, -0.1):
        a, c = T.solve({"top": V}), M.solve({"top": V})
        assert c.current["top"] == pytest.approx(a.current["top"], rel=1e-10)


# ------------------------------------------- nonlocal band-to-band
def _esaki(model="nonlocal"):
    m = dataclasses.replace(algan(0.0), name="p", eg=0.66, me=0.04, eps_r=16.0)
    m2 = dataclasses.replace(m, name="n")
    layers = [StackLayer(m, 100.0, acceptors=3e-2), StackLayer(m2, 100.0, donors=1e-2)]
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    bt = (BandToBand("test", reduced_mass_m0=0.02, model=model, n_energy=200)
          if model else None)
    return Device(layers, 20.0, el, Mobility(1000.0, "t"), Surface("pinned", barrier=0.5),
                  grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 200, 401)),
                  extra_sheet=_no_pol(layers), hole_mass=0.3,
                  hole_mobility=Mobility(100.0, "t"), band_to_band=bt)


def test_nonlocal_band_to_band_is_kane_in_a_uniform_field():
    m = dataclasses.replace(algan(0.0), name="nar", eg=0.66, me=0.04, eps_r=16.0)
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    dev = Device([StackLayer(m, 200.0, donors=1e-3)], 20.0, el, Mobility(1000.0, "t"),
                 Surface("pinned", barrier=0.5),
                 grid=dict(x=np.array([0., 10., 20.]), z=np.linspace(0, 200, 201)),
                 hole_mass=0.3, hole_mobility=Mobility(100.0, "t"),
                 band_to_band=BandToBand("test", reduced_mass_m0=0.02, model="nonlocal"))
    F = 0.01                                              # V/nm
    X = np.zeros((dev.N, dev.nv))
    X[:, 0] = F * dev.coords[1]
    X[:, 1], X[:, 2] = -10.0, 10.0                         # full supply
    Gn, Gp = dev._nonlocal_rates(X.ravel())
    tot_n = (Gn * dev.vol).reshape(3, -1).sum(axis=1) / dev._bt["area"]
    tot_p = (Gp * dev.vol).reshape(3, -1).sum(axis=1) / dev._bt["area"]
    Ec = dev.Ec0 - X[:, 0]
    G0, _ = BandToBand.rate(F * 1e7, 0.66, 0.02)
    expect = G0 * 1e-21 * ((Ec.max() - Ec.min()) - 0.66) / F
    assert np.allclose(tot_n, expect, rtol=1e-7) and np.allclose(tot_p, expect, rtol=1e-7)


def test_esaki_diode_with_nonlocal_tunneling():
    from scipy.integrate import quad
    dev, dd = _esaki(), _esaki(None)
    st = dev.solve({"bot": 0.0})
    assert abs(st.current["bot"]) < 1e-11 and st.band_to_band_iterations == 1
    I, prev = {}, None
    for V in (-0.1, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6):
        prev = dev.solve({"bot": V}, guess=prev)
        I[V] = prev.current["bot"]
        Gn, Gp = prev.band_to_band_rates
        pairs = float(np.sum(Gn * dev.vol.reshape(dev.shape))) * Q * 1e6
        assert pairs == pytest.approx(float(np.sum(Gp * dev.vol.reshape(dev.shape))) * Q * 1e6,
                                      rel=1e-9, abs=1e-30)
        if V == 0.15:
            st15 = prev
        if V in (-0.1, 0.1):
            # the tunneling current is the terminal current
            assert I[V] == pytest.approx(-pairs, rel=1e-4)
            assert abs(I[V]) > 1e6 * abs(dd.solve({"bot": V}).current["bot"])
    assert I[-0.1] < 0 < I[0.05]
    assert I[0.1] > 2 * I[0.3] > 0 and I[0.3] > 5 * I[0.45]        # negative resistance
    assert I[0.6] > I[0.45]                                         # then diffusion takes over
    # an independent quadrature of the same model on the solved bands
    zz = dev.axes[-1]
    Ec, Fn, Fp = st15.ec[1], st15.efn[1], st15.efp[1]
    Ev = Ec - 0.66
    lo, hi, nE = Ec.min(), Ev.max(), 200
    dE = (hi - lo) / nE
    pre = Q ** 2 / (18 * np.pi * HBAR ** 2) * np.sqrt(0.02 * M0 / (0.66 * Q))
    kT = KB * 300.0
    tot = 0.0
    for E in lo + (np.arange(nE) + 0.5) * dE:
        zv = np.interp(-E, -Ev, zz) if Ev[0] > Ev[-1] else np.interp(E, Ev, zz)
        zc = np.interp(-E, -Ec, zz) if Ec[0] > Ec[-1] else np.interp(E, Ec, zz)
        a, b = sorted((zv, zc))
        kap = lambda s: 2 * np.sqrt(0.02 * M0 * Q * max(  # noqa: E731
            (E - np.interp(s, zz, Ev)) * (np.interp(s, zz, Ec) - E), 0) / 0.66) / HBAR * 1e-9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph = quad(kap, a, b, points=list(zz[(zz > a) & (zz < b)])[:40], limit=400,
                      epsabs=0, epsrel=1e-11)[0]
        fv = 1 / (1 + np.exp((E - np.interp(zv, zz, Fp)) / kT))
        fc = 1 / (1 + np.exp((E - np.interp(zc, zz, Fn)) / kT))
        tot += pre * 0.66 / ((b - a) * 1e-9) * np.exp(-2 * ph) * (fv - fc) * dE * 1e-18
    Gn, _ = st15.band_to_band_rates
    mine = float(np.sum((Gn * dev.vol.reshape(dev.shape))[1])) / dev._bt["area"][1]
    assert mine == pytest.approx(tot, rel=1e-6)
    with pytest.raises(ValueError, match="nonlocal"):
        BandToBand("t", model="other")


# ------------------------------------------------- process topography
def test_etching_and_deposition_against_exact_shapes():
    W, w, R = 600.0, 100.0, 60.0
    errs = []
    for h in (2.0, 1.0):
        s = Structure(W, 0.0, 400.0, h, "Si", 300.0)
        s.deposit("M", 20.0, mode="directional")
        s.etch({"M": 1.0}, 25.0, mode="directional", window=(250, 350))
        a0 = s.area("Si")
        s.etch({"Si": 1.0}, R, mode="isotropic")        # the mask is not etched
        errs.append((a0 - s.area("Si")) / (w * R + math.pi * R * R / 2) - 1)
        assert s.area("M") == pytest.approx(500 * 20.0)
    assert abs(errs[1]) < 0.01 and abs(errs[1]) < 0.6 * abs(errs[0])
    # conformal film on a pillar: W t + 2 H t + (pi/2 - 2) t^2
    for h, tol in ((2.0, 3e-3), (1.0, 1e-3)):
        s = Structure(W, 0.0, 600.0, h, "Si", 200.0)
        s.deposit("P", 150.0, mode="directional", window=(250, 350))
        s.deposit("D", 40.0, mode="conformal")
        exact = W * 40 + 2 * 150 * 40 + (math.pi / 2 - 2) * 1600
        assert s.area("D") == pytest.approx(exact, rel=tol)
    # straight down through two materials at their rates
    s = Structure(100.0, 0.0, 200.0, 1.0, "Si", 100.0)
    s.deposit("N", 30.0, mode="directional")
    s.etch({"N": 2.0, "Si": 0.5}, 25.0, mode="directional")
    assert s.thickness("N", 50) == 0 and s.thickness("Si", 50) == 95.0


def test_deal_grove_oxidation():
    B, BA = 0.0104 * 1e6 / 3600, 0.0753 * 1e3 / 3600          # nm^2/s, nm/s
    A = B / BA
    s = Structure(200.0, 0.0, 400.0, 1.0, "Si", 300.0)
    s.deposit("N", 10.0, mode="directional", window=(0, 100))   # a nitride mask
    s.oxidize("Si", "SiO2", B, BA, 3600.0, 0.44, "test")
    x2 = s.oxidize("Si", "SiO2", B, BA, 3600.0, 0.44, "test")
    x = 0.5 * A * (math.sqrt(1 + 4 * B * 7200 / A ** 2) - 1)
    assert x2[150] == pytest.approx(x, rel=1e-12)              # two steps = one
    assert abs(s.thickness("SiO2", 150) - x) <= 1.0
    assert abs(s.thickness("Si", 150) - (300 - 0.44 * x)) <= 1.0
    assert s.thickness("SiO2", 50) == 0 and s.thickness("Si", 50) == 300
    with pytest.raises(ValueError, match="reference"):
        s.oxidize("Si", "SiO2", B, BA, 10.0, 0.44, "")


def test_process_flow_into_a_mesh_device():
    s = Structure(1000.0, 0.0, 400.0, 5.0, "GaN", 300.0)
    s.deposit("SiN", 50.0, mode="conformal")
    s.etch({"SiN": 1.0}, 60.0, mode="directional", window=(100, 900))
    s.implant(Implant("donor", 1e13, 30.0, 15.0, "test", 0.0))
    d0 = s.donors.sum() * (s.h * 1e-7) ** 2
    s.anneal(Anneal(60.0, 1300.0, 1e-4, 3.0, "test"))
    assert s.donors.sum() * (s.h * 1e-7) ** 2 == pytest.approx(d0, rel=1e-10)
    assert d0 == pytest.approx(1e13 * 800e-7, rel=2e-3)     # the window's dose, none in the mask
    mesh = s.mesh()
    reg = {"GaN": MeshRegion(material=algan(0.0), mobility=Mobility(300.0, "t")),
           "SiN": MeshRegion(dielectric=Dielectric("SiN", 1.0, 7.0, "t"))}
    con = {"a": s.surface_nodes(mesh, "GaN", 100, 200), "b": s.surface_nodes(mesh, "GaN", 800, 900)}
    assert con["a"].size == con["b"].size == 20     # the nodes under the mask edge are covered
    M = MeshDevice(mesh, reg, [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)],
                   con, doping=s.doping(), statistics="boltzmann")
    st = M.solve({"b": 0.01})
    assert st.current["b"] > 0 and abs(sum(st.current.values())) < 1e-12
    # epitaxial growth with in-situ doping
    s.deposit("GaN2", 20.0, mode="directional", donors=1e18)
    assert "GaN2" in s.semis and np.max(s.donors[s.mat == "GaN2"]) == 1e18


def test_new_options_are_checked():
    with pytest.raises(ValueError, match="temperature_model"):
        _leaky_diode(full=False).__class__([StackLayer(algan(0.0), 10.0, donors=1e-3)], 20.0,
                                          [Electrode("a", "ohmic", 0, 20)], Mobility(1.0, "t"),
                                          Surface("pinned", barrier=1.0),
                                          temperature_model="everything")
    with pytest.raises(ValueError, match="conduction_share"):
        Varshni(1e-4, 100.0, 1.5, "t")
    with pytest.raises(ValueError, match="lattice_temperature"):
        _leaky_diode().solve({"bot": 0.0}, lattice_temperature=-1.0)


def test_traps_keep_detailed_balance_with_fermi_statistics():
    # an InN p-n diode (degenerate n side) with traps: before 0.22.0 the
    # trap rates did not vanish in equilibrium and no equilibrium was found
    from kpenvelope import ingan
    inn = ingan(1.0)
    layers = [StackLayer(inn, 500.0, acceptors=1e-3, name="p"),
              StackLayer(dataclasses.replace(inn, name="InN n"), 500.0, donors=1e-3, name="n")]
    dev = Device(layers, 20.0, [Electrode("top", "ohmic", 0, 20),
                                Electrode("bottom", "ohmic", 0, 20, face="bottom")],
                 Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                 grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 1000, 201)),
                 hole_mass=0.5, hole_mobility=Mobility(50.0, "t"),
                 traps=[Trap("acceptor", 0.38, "ec", 1e17, 1e-8, 1e-8, "t", ("p", "n"))],
                 extra_sheet=_no_pol(layers))
    st = dev.equilibrium()
    assert st.current == {"top": 0.0, "bottom": 0.0}
    b = dev._balances(st._x, np.full(dev.N, 300.0))
    tr = dev._traps[0]
    gross = tr["Nt"] * tr["cn"] * b["n"]                # capture alone
    assert np.max(np.abs(b["Rn"])) < 1e-9 * np.max(gross)
    assert np.max(st.n) > 1.5 * np.max(dev.Nc) * 1e21     # the n side is degenerate (n > N_c)
