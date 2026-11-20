"""Tests for v0.21.0: hot electrons coupled to the current, thermionic
emission at band steps, tunneling through barriers inside the device,
band-to-band tunneling, breakdown by current and by thermal runaway,
implants and anneals, and devices on unstructured meshes."""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan, gan, gate_leakage  # noqa: E402
from kpenvelope.device import (BandToBand, Device, Dielectric, Electrode,  # noqa: E402
                               Mobility, Surface, TunnelBarrier)
from kpenvelope.electrons import KB  # noqa: E402
from kpenvelope.interfaces import interface_charges  # noqa: E402
from kpenvelope.leakage import tsu_esaki_current  # noqa: E402
from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_grid, mesh_from_points  # noqa: E402
from kpenvelope.process import Anneal, Implant, Process  # noqa: E402

Q = 1.602176634e-19


def _bar(layers, holes=False, nz=301, **kw):
    z = np.linspace(0, sum(L.thickness for L in layers), nz)
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    mob = kw.pop("mobility", Mobility(1000.0, "test", saturation_velocity=1.5e7))
    extra = dict(hole_mass=1.0, hole_mobility=Mobility(20.0, "test")) if holes else {}
    extra.update(kw)
    return Device(layers, 20.0, el, mob, Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0.0, 10.0, 20.0]), z=z), **extra)


def _no_polarization(layers):
    return {c.index: -c.sigma_nm2 for c in interface_charges(layers)}


# ------------------------------------------------------------ hydrodynamic
def test_hydrodynamic_flux_is_the_exact_edge_solution():
    # on an edge with a linear electron temperature the discrete flux makes
    # u = n T_n follow du/dx = -(dE/dx)/(k T_n) u - Gamma/(mu k) exactly
    from scipy.integrate import solve_ivp
    dev = _bar([StackLayer(algan(0.0), 3000.0, donors=1e-3)], statistics="boltzmann")
    st = dev.solve({"bot": 3.0})
    z = dev.axes[-1]
    Tn = np.tile(300 + 2000 * np.exp(-((z - 1500) / 400) ** 2), 3)
    dev._set_hydro(Tn, np.full(dev.N, 300.0), 0.47e-12)
    psi, fn, fp = dev._unpack_static(st._x)
    _, n, p, lnn, lnp = dev._carriers(psi, fn, fp)
    Gn, _ = dev._fluxes(psi, fn, fp, n, p, lnn, lnp, np.full(dev.N, 300.0))
    Ec = dev.Ec0 - psi
    mu = dev._hydro["mu"] * 1e14
    dev._hydro = None
    for e in np.flatnonzero(dev.eaxis == 1)[::23]:
        i, j, h = dev.ei[e], dev.ej[e], dev.eh[e]
        g, G, M, Ti, Tj = (Ec[j] - Ec[i]) / h, Gn[e], mu[e], Tn[i], Tn[j]
        sol = solve_ivp(lambda s, u: [-g / (KB * (Ti + (Tj - Ti) * s / h)) * u[0] - G / (M * KB)],
                        [0, h], [n[i] * Ti], rtol=1e-12, atol=1e-40)
        assert sol.y[0, -1] == pytest.approx(n[j] * Tj, rel=1e-9)


def test_hydrodynamic_uniform_bar_gives_the_drift_diffusion_velocity():
    # where the field is uniform the coupled model returns the user's
    # velocity-field curve and T_n = T_L + (2/3) tau mu E^2 / k
    mob = Mobility(1000.0, "test", saturation_velocity=1.5e7)
    dev = _bar([StackLayer(algan(0.0), 3000.0, donors=1e-3)], mobility=mob,
               statistics="boltzmann")
    tau = 0.47e-12
    st = dev.solve({"bot": 3.0}, energy_relaxation=tau)
    z, psi, n, Tn = st.axes[-1], st.psi[1], st.n[1], st.electron_temperature[1]
    k = 150
    E = abs(psi[k + 1] - psi[k - 1]) / (z[k + 1] - z[k - 1]) * 1e7
    v = abs(st.current["bot"]) / (20e-7 * 0.1) / (Q * n[k])
    assert v == pytest.approx(mob.value(E) * E, rel=1e-5)
    assert Tn[k] == pytest.approx(300 + 2 / 3 * tau * mob.value(E) * E * E / KB, rel=1e-5)
    assert Tn[0] == pytest.approx(300.0) and Tn[-1] == pytest.approx(300.0)


def test_velocity_overshoot_in_an_n_plus_n_n_plus_diode():
    mob = Mobility(1000.0, "test", saturation_velocity=1.5e7)
    g = algan(0.0)
    layers = [StackLayer(g, 200.0, donors=1e-2, name="a"), StackLayer(g, 200.0, donors=1e-4),
              StackLayer(g, 200.0, donors=1e-2, name="b")]
    peaks = []
    for nz in (301, 601):
        dev = _bar(layers, mobility=mob, nz=nz, statistics="boltzmann")
        dd = dev.solve({"bot": 2.0})
        hy = dev.solve({"bot": 2.0}, energy_relaxation=0.47e-12)
        z = dev.axes[-1]
        m = (z > 200) & (z < 400)

        def vel(s):
            return abs(s.current["bot"]) / (20e-7 * 0.1) / (Q * s.n[1])
        assert vel(dd)[m].max() <= 1.5e7 * (1 + 1e-6)          # drift-diffusion: never above v_sat
        peaks.append(vel(hy)[m].max())
        assert hy.current["bot"] != pytest.approx(dd.current["bot"], rel=0.05)
    assert min(peaks) > 2 * 1.5e7                                 # overshoot
    assert peaks[0] == pytest.approx(peaks[1], rel=0.03)          # resolved on the grid


# ------------------------------------------------- thermionic emission
def test_thermionic_heterojunction_against_an_independent_solution():
    # n GaN / n Al0.15GaN with the polarization charges removed: the device
    # with thermionic emission at the step against a boundary-value solution
    # of the same model (Poisson, drift-diffusion on each side, the emission
    # condition at the interface); the single-node interface makes the error
    # first order in the grid step, so two grids are extrapolated
    from scipy.integrate import solve_bvp
    g, a2 = algan(0.0), algan(0.15)
    L, ND, mu = 100.0, 1e-3, 300.0
    layers = [StackLayer(g, L, donors=ND, name="gan"), StackLayer(a2, L, donors=ND, name="algan")]
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    V = 0.02
    fluxes = []
    for h in (0.1, 0.04):
        z = np.linspace(0, 2 * L, int(round(2 * L / h)) + 1)
        dev = Device(layers, 20.0, el, Mobility(mu, "t"), Surface("pinned", barrier=1.0),
                     extra_sheet=_no_polarization(layers),
                     grid=dict(x=np.array([0, 10, 20.0]), z=z), heterojunctions="thermionic",
                     statistics="boltzmann")
        assert abs(dev.solve({"top": 0.0}).current["top"]) < 1e-10      # equilibrium
        fluxes.append(dev.solve({"top": V}).current["top"] / Q / (20.0 * 1e6))
    G_dev = fluxes[1] - (fluxes[0] - fluxes[1]) * 0.04 / (0.1 - 0.04)
    nz = z.size
    kT = KB * 300.0
    Ec1, Ec2, Nc1, Nc2 = dev.Ec0[0], dev.Ec0[nz - 1], dev.Nc[0], dev.Nc[nz - 1]
    e1, e2 = g.eps_r * 0.0552634936, a2.eps_r * 0.0552634936          # e / (V nm)
    vR, M = dev.vR[nz - 1], mu * 1e14
    psiL, FR = Ec1 + kT * math.log(ND / Nc1), -V
    psiR = Ec2 - FR + kT * math.log(ND / Nc2)

    def dens(ps, F, Ec, Nc):
        return Nc * np.exp((F - Ec + ps) / kT)

    def f(s, y, p):
        p1, E1, F1, p2, E2, F2 = y
        n1, n2 = dens(p1, F1, Ec1, Nc1), dens(p2, F2, Ec2, Nc2)
        return np.vstack([-E1 * L, (ND - n1) / e1 * L, -p[0] / (M * n1) * L,
                          -E2 * L, (ND - n2) / e2 * L, -p[0] / (M * n2) * L])

    def bc(ya, yb, p):
        n2i = dens(ya[3], ya[5], Ec2, Nc2)
        return np.array([ya[0] - psiL, ya[2], yb[3] - psiR, yb[5] - FR, yb[0] - ya[3],
                         e1 * yb[1] - e2 * ya[4],
                         (p[0] - vR * n2i * np.expm1((yb[2] - ya[5]) / kT)) / (vR * ND)])
    s = np.linspace(0, 1, 401)
    y0 = np.zeros((6, s.size))
    y0[0], y0[3], y0[5] = psiL, np.linspace(psiL, psiR, s.size), np.linspace(0, FR, s.size)
    sol = solve_bvp(f, bc, s, y0, p=[0.0], tol=1e-10, max_nodes=200000)
    assert sol.status == 0
    assert G_dev == pytest.approx(sol.p[0], rel=5e-3)


# ------------------------------------------------- tunneling in the device
def _barrier_stack(thick):
    g, b = algan(0.0), algan(0.3)
    layers = [StackLayer(g, 60.0, donors=1e-2, name="bottom"),
              StackLayer(b, thick, name="barrier"), StackLayer(g, 60.0, donors=1e-2, name="top")]
    z = np.unique(np.r_[np.linspace(0, 60, 121), np.linspace(60, 60 + thick, int(thick / 0.1) + 1),
                        np.linspace(60 + thick, 120 + thick, 121)])
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    return z, Device(layers, 20.0, el, Mobility(300.0, "t"), Surface("pinned", barrier=1.0),
                     extra_sheet=_no_polarization(layers),
                     grid=dict(x=np.array([0, 10, 20.0]), z=z),
                     tunnel_barriers=[TunnelBarrier("barrier", n_energy=800)])


def test_tunnel_barrier_current_is_the_tsu_esaki_current_of_the_bands():
    for thick in (2.0, 4.0):
        z, d = _barrier_stack(thick)
        assert abs(d.solve({"top": 0.0}).current["top"]) < 1e-10       # equilibrium
        for V in (0.05, 0.2):
            st = d.solve({"top": V})
            grp = d._tb["groups"][0]
            path = grp["path"][1]
            J = tsu_esaki_current(d.axes[-1][path % z.size], st.ec.ravel()[path],
                                  st.efn.ravel()[grp["a"][1]], st.efn.ravel()[grp["b"][1]],
                                  d.me_node[path], 300.0, n_energy=800,
                                  transport_mass=float(d.me_node[grp["a"][1]]))
            assert st.current["top"] / (20e-7 * 0.1) == pytest.approx(J, rel=1e-8)


def test_coupled_gate_tunneling_matches_the_1d_solver():
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 10.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 400, 1600, barrier=1.0),
          Electrode("drain", "ohmic", 1900, 2000, implant_depth=30, implant_donors=1e20)]
    dev = Device(layers, 2000.0, el, Mobility(1800.0, "test"), Surface("pinned", barrier=1.0),
                 tunnel_barriers=[TunnelBarrier("barrier")], dx_fine=10, dx_max=100,
                 dz_fine=0.5, dz_max=40, fine_depth=40)
    st = dev.solve({"gate": -3.0})
    psi, fn, fp = dev._unpack_static(st._x)
    grp = dev._tb["groups"][0]
    G = dev._link_current(grp, psi, fn)
    k = int(np.argmin(np.abs(dev.coords[0][grp["a"]] - 1000.0)))
    J_mid = -G[k] * Q * 1e14                         # A/cm^2, negative: into the stack
    _, J1, _ = gate_leakage(layers, 1.0, [-3.0], dz=0.5)
    assert J_mid == pytest.approx(-J1[0], rel=0.02)
    assert st.current["gate"] < 0
    assert abs(sum(st.current.values())) < 1e-9 * abs(st.current["gate"])


def test_band_to_band_tunneling():
    hbar, m0 = 1.054571817e-34, 9.1093837015e-31
    F, Eg, mr = 5e8, 3.4 * Q, 0.17 * m0
    hand = (Q ** 2 * F ** 2 / (18 * np.pi * hbar ** 2) * np.sqrt(mr / Eg)
            * np.exp(-np.pi * np.sqrt(mr) * Eg ** 1.5 / (2 * Q * hbar * F)) * 1e-6)
    G, dG = BandToBand.rate(5e6, 3.4, 0.17)
    assert G == pytest.approx(hand, rel=1e-12)
    assert dG == pytest.approx((BandToBand.rate(5e6 + 100, 3.4, 0.17)[0]
                                - BandToBand.rate(5e6 - 100, 3.4, 0.17)[0]) / 200, rel=1e-6)
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 40.0, acceptors=5e-2, name="p"), StackLayer(g, 40.0, donors=5e-2, name="n")]
    dev = _bar(layers, holes=True, nz=401, statistics="boltzmann",
               band_to_band=BandToBand("Kane, as in arXiv:2108.09328 Eq. 13"))
    # equilibrium: no generation, only the solver's rounding floor
    s0 = dev.solve({"top": 0.0})
    assert abs(s0.current["top"]) < 1e-10 and s0.band_to_band_pairs == 0.0
    for V in (5.0, 10.0):
        st = dev.solve({"top": V})
        psi = st._x.reshape(dev.N, dev.nv)[:, 0]
        I_gen = dev._band_to_band_nodes(psi, V).sum() * Q * 1e6
        assert abs(st.current["top"]) == pytest.approx(I_gen, rel=1e-4)
        assert I_gen > 1e-7


# ------------------------------------------------------------ breakdown
def test_breakdown_by_current_and_field_check():
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
          Electrode("gate", "schottky", 300, 400, barrier=1.0),
          Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20)]
    dev = Device(layers, 700.0, el, Mobility(1800.0, "t", saturation_velocity=1e7),
                 Surface("pinned", barrier=1.0), dx_fine=10, dx_max=50, dz_fine=2.0,
                 dz_max=40, fine_depth=40)
    r = dev.breakdown_by_current("drain", np.arange(1.0, 41.0, 1.0), bias={"gate": -8.0},
                                 current=1e-3)
    I = np.abs(r["current"]["drain"])
    assert r["voltage"] is not None and I[-1] >= 1e-3 > I[-2]
    assert r["voltages"][-2] <= r["voltage"] <= r["voltages"][-1]
    assert sum(r["share"].values()) == pytest.approx(1.0)
    # the field in a capacitor's dielectric is V/d
    sin = Dielectric("SiN", 50.0, 7.5, "test")
    d2 = Device([StackLayer(algan(0.0), 200.0, donors=1e-1, name="n+")], 1000.0,
                [Electrode("body", "ohmic", 0, 1000, face="bottom"),
                 Electrode("plate", "schottky", 0, 1000, barrier=1.0, z0=250.0, z1=250.0)],
                Mobility(300.0, "t"), Surface("pinned", barrier=1.0), cap=[sin],
                dx_fine=100, dx_max=200, dz_fine=1.0, dz_max=10, fine_depth=60)
    s0 = d2.solve({"plate": 0.0})
    s5 = d2.solve({"plate": 5.0}, guess=s0)
    f0 = d2.field_check(s0, {"SiN": (5.0, "test")})["SiN"]["peak"]
    f5 = d2.field_check(s5, {"SiN": (5.0, "test")})["SiN"]
    assert f5["peak"] - f0 == pytest.approx(5.0 / 50e-7 / 1e6, rel=2e-3)
    assert f5["ratio"] == pytest.approx(f5["peak"] / 5.0)


def test_thermal_runaway_matches_the_lumped_model():
    # a bar whose mobility rises as T^2, on a poor heat sink: runaway where
    # T = T0 + R_th G0 (T/T0)^2 V^2 stops having a solution
    from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel
    layers = [StackLayer(algan(0.0), 100.0, donors=1e-4, name="bar")]
    el = [Electrode("a", "ohmic", 0, 50, z0=50.0, z1=100.0),
          Electrode("b", "ohmic", 950, 1000, z0=50.0, z1=100.0)]
    d = Device(layers, 1000.0, el, Mobility(300.0, "test", temperature_exponent=-2.0),
               Surface("states", fixed_charge=0.0), dx_fine=25, dx_max=50, dz_fine=10,
               dz_max=10, fine_depth=100)
    th = ThermalModel({"bar": ThermalMaterial("bar", 5000.0, 0.0, "test")},
                      substrate=(ThermalLayer("sub", 1000.0, ThermalMaterial("s", 1.0, 0.0, "t")),))
    G0 = d.solve({"b": 0.01}).current["b"] / 0.01
    s1 = d.solve({"b": 0.01}, thermal=th)
    Rth = float(np.sum(s1.heat * (s1.temperature - 300.0)) / np.sum(s1.heat)) / s1.power
    Ts = 600.0
    Vc = math.sqrt((Ts - 300.0) / (Rth * G0 * (Ts / 300.0) ** 2))
    r = d.thermal_runaway("b", np.linspace(0.2 * Vc, 1.3 * Vc, 23), th)
    assert r["stable"] <= Vc * 1.003 and r["runaway"] >= Vc * 0.997
    assert r["runaway"] - r["stable"] < 2e-3 * Vc


# ------------------------------------------------------------ process
def test_implant_and_anneal_against_exact_solutions():
    an = Anneal(60000.0, 1300.0, 1e-4, 3.0, "test")
    D = an.diffusivity_nm2_s
    assert D == pytest.approx(1e-4 * math.exp(-3.0 / (8.617333262e-5 * 1300.0)) * 1e14)
    imp = Implant("donor", 1e14, 300.0, 30.0, "test", surface_z=1000.0)
    pr = Process([imp, an], width=100.0, z_bottom=0.0, z_top=1000.0, nx=5, nz=2001)
    s2 = math.sqrt(30.0 ** 2 + 2 * D * 60000.0)
    assert s2 > 55.0
    exact = 1e14 / (math.sqrt(2 * math.pi) * s2 * 1e-7) * np.exp(
        -0.5 * ((1000.0 - pr.z - 300.0) / s2) ** 2)
    assert np.max(np.abs(pr.donors[2] - exact)) < 1e-4 * exact.max()
    assert pr.sheet()[2] == pytest.approx(1e14, rel=1e-9)
    # an implant at the surface stays a half Gaussian (no flux out)
    imp2 = Implant("acceptor", 5e13, 0.0, 20.0, "test", surface_z=500.0)
    pr2 = Process([imp2, an], width=10.0, z_bottom=0.0, z_top=500.0, nx=3, nz=2001)
    s2b = math.sqrt(20.0 ** 2 + 2 * D * 60000.0)
    ex2 = 5e13 / (math.sqrt(2 * math.pi) * s2b * 1e-7) * np.exp(-0.5 * ((500.0 - pr2.z) / s2b) ** 2)
    assert np.max(np.abs(pr2.acceptors[1] - ex2)) < 1e-4 * ex2.max()
    assert pr2.sheet("acceptor")[1] == pytest.approx(2.5e13, rel=1e-9)
    # the mask edge is the Gaussian convolved with the window
    imp3 = Implant("donor", 1e13, 50.0, 10.0, "test", surface_z=200.0,
                   lateral_straggle_nm=15.0, window=(100.0, 300.0))
    xs = np.linspace(0, 400, 81)
    c = imp3.concentration(xs, np.full(xs.size, 150.0)) / (1e13 / (math.sqrt(2 * math.pi) * 1e-6))
    u = np.linspace(100, 300, 20001)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    conv = np.array([trap(np.exp(-0.5 * ((x - u) / 15.0) ** 2) / (math.sqrt(2 * math.pi) * 15.0), u)
                     for x in xs])
    assert np.max(np.abs(c - conv)) < 1e-7


def test_process_doping_in_a_device():
    imp = Implant("donor", 5e13, 30.0, 10.0, "test", surface_z=220.0, window=(0.0, 100.0))
    pr = Process([imp], width=700.0, z_bottom=0.0, z_top=220.0, nx=351, nz=441)
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100), Electrode("drain", "ohmic", 600, 700,
                                                          implant_depth=40, implant_donors=1e20)]
    dev = Device(layers, 700.0, el, Mobility(1800.0, "t"), Surface("pinned", barrier=1.0),
                 doping=pr.doping(), dx_fine=10, dx_max=50, dz_fine=1.0, dz_max=40,
                 fine_depth=60)
    # the device holds the implanted dose under the window (control volumes)
    x = dev.coords[0]
    x40 = dev.axes[0][np.argmin(np.abs(dev.axes[0] - 40.0))]
    col = np.isclose(x, x40)
    dose = float(np.sum(dev.ND[col] * dev.vol[col])) / float(np.sum(dev.lat_area[col][:1])) * 1e14
    assert dose == pytest.approx(pr.sheet()[np.argmin(np.abs(pr.x - x40))], rel=2e-2)
    st = dev.solve({"drain": 0.5})
    assert abs(sum(st.current.values())) < 1e-9 * abs(st.current["drain"])


# ------------------------------------------------------------ meshes
def test_mesh_of_right_triangles_equals_the_tensor_grid():
    g = algan(0.0)
    x = np.unique(np.r_[np.linspace(0, 100, 6), np.linspace(120, 380, 14),
                        np.linspace(400, 500, 6), np.linspace(150, 350, 11)])
    z = np.unique(np.r_[np.linspace(0, 150, 7), np.linspace(150, 200, 26)])
    mob = Mobility(500.0, "test", saturation_velocity=1e7)
    el = [Electrode("source", "ohmic", 0, 100),
          Electrode("gate", "schottky", 200, 300, barrier=0.8),
          Electrode("drain", "ohmic", 400, 500)]
    T = Device([StackLayer(g, 200.0, donors=1e-3)], 500.0, el, mob,
               Surface("states", fixed_charge=0.0), grid=dict(x=x, z=z))
    mesh = mesh_from_grid(x, z, lambda xx, zz: "gan")
    P = mesh.points
    top = np.isclose(P[:, 1], 200.0)
    con = {"source": np.flatnonzero(top & (P[:, 0] <= 100 + 1e-9)),
           "gate": np.flatnonzero(top & (P[:, 0] >= 200 - 1e-9) & (P[:, 0] <= 300 + 1e-9)),
           "drain": np.flatnonzero(top & (P[:, 0] >= 400 - 1e-9))}
    M = MeshDevice(mesh, {"gan": MeshRegion(material=g, mobility=mob, donors=1e-3)}, el, con)
    for b in ({"drain": 1.0, "gate": -1.0}, {"drain": 3.0, "gate": 0.0}):
        a, c = T.solve(b), M.solve(b)
        assert c.current["drain"] == pytest.approx(a.current["drain"], rel=1e-10)
        assert np.max(np.abs(a.psi.ravel() - c.psi)) < 1e-10


def test_curved_mesh_resistor_converges_to_the_exact_result():
    # a quarter annulus (radii a, b) with contacts on its two straight
    # edges: R = rho (pi/2) / ln(b/a) per unit depth
    g, a, b, mu, nd = algan(0.0), 200.0, 600.0, 500.0, 1e18
    R = 1 / (Q * nd * mu) * (np.pi / 2) / (np.log(b / a) * 0.1)
    errs = []
    for nr, nt in ((17, 25), (33, 49)):
        r, t = np.linspace(a, b, nr), np.linspace(0, np.pi / 2, nt)
        Rr, Tt = np.meshgrid(r, t, indexing="ij")
        pts = np.column_stack([(Rr * np.cos(Tt)).ravel(), (Rr * np.sin(Tt)).ravel()])
        mesh = mesh_from_points(pts, lambda x, z: "gan" if a - 1e-9 <= np.hypot(x, z) <= b + 1e-9
                                else None)
        P = mesh.points
        con = {"a": np.flatnonzero(np.isclose(P[:, 1], 0.0)),
               "b": np.flatnonzero(np.isclose(P[:, 0], 0.0))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            M = MeshDevice(mesh, {"gan": MeshRegion(material=g, mobility=Mobility(mu, "test"),
                                                    donors=nd * 1e-21)},
                           [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)], con,
                           statistics="boltzmann")
        errs.append(abs(M.solve({"b": 0.01}).current["b"]) / (0.01 / R) - 1)
    assert abs(errs[1]) < 5e-4
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)       # second order


# ------------------------------------------------------------ the app
def test_app_hot_electrons_coupled_and_breakdown_by_current():
    from kpenvelope.gui.api import handle
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [{"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                        "doping": "none"},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0,
                        "doping": "none"}],
            "surface_barrier": 1.0, "device": {"barrier": 1.0},
            "line_charge": {"mode": "measured"}}
    fet = {"gate_length": 200, "gap_sg": 300, "gap_gd": 500, "contact": 200,
           "sweep": "drain", "from": 1, "to": 3, "step": 2, "fixed": 0, "tau_e": 0.47}
    a = handle("transistor", dict(spec, transistor=dict(fet)))
    b = handle("transistor", dict(spec, transistor=dict(fet, hot_coupled=True)))
    assert "error" not in a and "error" not in b, (a.get("error"), b.get("error"))
    assert b["hot"]["coupled"] and not a["hot"]["coupled"]
    assert b["drain"][-1] != pytest.approx(a["drain"][-1], rel=1e-3)
    r = handle("breakdown", dict(spec, transistor=dict(fet),
                                 breakdown={"gate": -10, "to": 60, "step": 10,
                                            "criterion": "current", "current_mA": 1}))
    assert "error" not in r, r.get("error")
    assert r["criterion"] == "current"
    assert r["source_share"] + r["gate_share"] == pytest.approx(1.0)
