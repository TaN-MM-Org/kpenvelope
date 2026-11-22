"""Checks of the features new in 0.26.0: quantum transport by
non-equilibrium Green's functions (kpenvelope.negf, Device.negf)."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan  # noqa: E402
from kpenvelope.device import (Device, Electrode, Mobility, QuantumChannel,  # noqa: E402
                               Surface)
from kpenvelope.electrons import HBAR, M0, fermi_dirac_half  # noqa: E402
from kpenvelope.negf import _G0, _H2M, Chain, chain_transport  # noqa: E402
from kpenvelope.nitride import QE  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])

KT = 0.025852
_trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
G2D = 1.0 / (math.pi * 2.0 * _H2M)          # m0 / (pi hbar^2), per eV per nm^2


# ------------------------------------------------------------ the chain
def _t_barrier(E, V0, b, m):
    """Transmission of a rectangular barrier (height V0, width b) for a
    particle of mass m (the textbook continuum result)."""
    if E < V0:
        kap = math.sqrt((V0 - E) * m / _H2M)
        return 1.0 / (1.0 + V0 ** 2 * math.sinh(kap * b) ** 2 / (4 * E * (V0 - E)))
    q = math.sqrt((E - V0) * m / _H2M)
    return 1.0 / (1.0 + V0 ** 2 * math.sin(q * b) ** 2 / (4 * E * (E - V0)))


def test_chain_transmission_against_the_rectangular_barrier():
    m, V0, b = 0.2, 0.3, 2.0
    Es = np.array([0.05, 0.1, 0.2, 0.35, 0.5])
    exact = np.array([_t_barrier(E, V0, b, m) for E in Es])
    err = []
    for a in (0.1, 0.05):
        nb = int(round(b / a))
        U = np.zeros(nb + 41)
        U[20:20 + nb] = V0
        T = Chain(U, m, a).solve(Es)["T"]
        err.append(np.max(np.abs(T - exact)))
    # second order in the grid step
    assert err[1] < 1e-3 and err[0] / err[1] > 3.5


def test_chain_equilibrium_density_and_the_top_of_the_barrier():
    m = 0.2
    # a uniform chain at equilibrium holds the 2D gas density m kT / (pi
    # hbar^2) ln(1 + exp(F / kT)), to second order in the grid step
    for F in (-0.05, 0.1):
        n2 = G2D * m * KT * math.log1p(math.exp(F / KT))
        e = []
        for a in (0.5, 0.25):
            r = chain_transport(Chain(np.zeros(40), m, a), F, F, KT, energy_step=KT / 40)
            e.append(abs(r["n"][20] / n2 - 1))
            assert r["I"] == 0.0
        assert e[1] < 2e-3 and e[0] / e[1] > 3.5
    # a long, smooth barrier: the current tends to the top-of-the-barrier
    # formula q sqrt(2m)(kT)^1.5 / (2 pi^1.5 hbar^2) [F_1/2(eta_S) - F_1/2(eta_D)]
    FS, FD, Et = 0.0, -0.3, 0.15
    top = _G0 * math.sqrt(m * KT / (2 * math.pi * 2 * _H2M)) * KT * float(
        fermi_dirac_half((FS - Et) / KT)[0] - fermi_dirac_half((FD - Et) / KT)[0])
    dev = []
    for L in (20, 40, 80):
        x = np.arange(0, L + 40, 0.5)
        U = np.where((x > 20) & (x < 20 + L), Et * np.sin(np.pi * (x - 20) / L) ** 2, 0.0)
        dev.append(chain_transport(Chain(U, m, 0.5), FS, FD, KT)["I"] / top - 1)
    # tunneling through the top adds current; less as the barrier widens
    assert dev[0] > dev[1] > dev[2] > 0 and dev[2] < 0.015


def test_buttiker_probes_conserve_current_and_give_drude():
    m, tau = 0.2, 1e-13
    eta = HBAR / tau / QE
    for F in (0.0, 0.13):                      # nondegenerate and degenerate
        R, n = [], None
        for L in (127, 213):
            ch = Chain(np.zeros(int(L / 0.25)), m, 0.25, eta=eta)
            V = 1e-4
            r = chain_transport(ch, F + V / 2, F - V / 2, KT, energy_step=KT / 10,
                                emax_kT=18)
            assert abs(r["I"] / r["I_drain"] - 1) < 1e-8          # no current lost
            R.append(V / r["I"])
            n = r["n"][ch.N // 2]
        # the probes do not change the equilibrium density
        n2 = G2D * m * KT * math.log1p(math.exp(F / KT))
        assert n == pytest.approx(n2, rel=5e-3)
        # resistance grows linearly with length at the Drude rate
        sheet = (R[1] - R[0]) / (213 - 127)
        drude = 1.0 / (QE ** 2 * n2 * 1e18 * tau / (m * M0))
        assert sheet == pytest.approx(drude, rel=0.012)


# ------------------------------------------------------------ the device
def _short_hemt():
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 50, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 85, 115, barrier=1.0),
          Electrode("drain", "ohmic", 150, 200, implant_depth=30, implant_donors=1e20)]
    return Device(layers, 200.0, el, Mobility(1500.0, "t"), Surface("pinned", barrier=1.0),
                  dx_fine=5, dx_max=50, dz_fine=0.5, dz_max=50,
                  quantum=QuantumChannel((270.0, 319.5), x_range=(55, 145)))


def test_device_negf_equilibrium_and_ballistic_limit():
    dev = _short_hemt()
    qc = dev.solve({"gate": -1.0})
    eq = dev.negf({"gate": -1.0}, guess=qc, dx=0.5)
    assert eq.negf["current"] == 0.0 and eq.current["drain"] == 0.0
    # at equilibrium the channel holds what the QuantumChannel gives, away
    # from the ends (next to the leads the potential is not flat, and the
    # electrons' motion along x, which the columns of the QuantumChannel
    # leave out, shows: a few percent there)
    a, b = np.array(eq.quantum["sheet_cm2"]), np.array(qc.quantum["sheet_cm2"])
    k = [i for i, (x, _) in enumerate(qc.quantum["subbands"]) if 55 <= x <= 145]
    assert np.max(np.abs(a[2:-2] / b[k][2:-2] - 1)) < 2.5e-3
    # with a small drain voltage, the self-consistent ballistic current
    # against the top-of-the-barrier formula on the same potential
    for vg, vd in ((-1.0, 0.05), (-2.0, 0.02)):
        dd = dev.solve({"gate": vg, "drain": vd})
        st = dev.negf({"gate": vg, "drain": vd}, guess=dd, dx=0.5)
        top, _ = dev.ballistic_current(st)
        assert st.current["drain"] == -st.current["source"] == st.negf["current"]
        assert 0 < st.negf["current"] / top - 1 < 0.015       # plus a little tunneling
    with pytest.raises(ValueError, match="QuantumChannel"):
        Device(dev.layers, 200.0, dev.electrodes, Mobility(1500.0, "t"),
               Surface("pinned", barrier=1.0), dx_fine=5, dx_max=50, dz_fine=0.5,
               dz_max=50).negf({"drain": 0.1})
    with pytest.raises(ValueError, match="scattering_time"):
        dev.negf({"drain": 0.1}, scattering_time=0.0)


def _bar(L, mu):
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 50, implant_depth=30, implant_donors=1e20),
          Electrode("drain", "ohmic", 50 + L, 100 + L, implant_depth=30, implant_donors=1e20)]
    return Device(layers, 100.0 + L, el, Mobility(mu, "t"), Surface("pinned", barrier=1.0),
                  dx_fine=5, dx_max=10, dz_fine=0.5, dz_max=50,
                  quantum=QuantumChannel((285.0, 319.5), x_range=(55, 45 + L)))


def test_device_negf_with_scattering_against_drift_diffusion():
    # a 2DEG resistor: with Buttiker probes (tau), the resistance grows with
    # length at the sheet resistance of drift-diffusion with mu = q tau / m
    # (the contact resistance of the ballistic ends comes on top)
    tau = 1e-13
    Rn, Rd = [], []
    for L in (100, 200):
        dev = _bar(L, 1000.0)
        st = dev.negf({"drain": 0.01}, guess=dev.solve({"drain": 0.01}), dx=0.5,
                      scattering_time=tau)
        m = float(np.mean(st.negf["modes"][0]["mass"]))
        mu = QE * tau / (m * M0) * 1e4
        dd = _bar(L, mu).solve({"drain": 0.01})
        Rn.append(0.01 / st.current["drain"])
        Rd.append(0.01 / dd.current["drain"])
        assert Rn[-1] > Rd[-1]
    # 6 percent at this grid step; it shrinks with the step (see checks.md)
    assert (Rn[1] - Rn[0]) / (Rd[1] - Rd[0]) == pytest.approx(1.0, abs=0.09)


# ------------------------------------------- quantum channel on meshes
def _well_layers():
    import dataclasses
    g = algan(0.0)
    bar = dataclasses.replace(g, name="bar", ec=g.ec + 0.3)
    return g, bar


def _region(zz):
    return "b1" if zz < 100 else ("w" if zz < 120 else "b2")


def test_quantum_channel_on_a_mesh_equals_the_tensor_grid():
    from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_grid
    g, bar = _well_layers()
    x = np.linspace(0, 500, 26)
    z = np.unique(np.r_[np.linspace(0, 90, 10), np.linspace(90, 130, 81),
                        np.linspace(130, 200, 8)])
    mob, hmob = Mobility(500.0, "t"), Mobility(20.0, "t")
    el = [Electrode("source", "ohmic", 0, 100),
          Electrode("gate", "schottky", 200, 300, barrier=0.8),
          Electrode("drain", "ohmic", 400, 500)]
    # the window inside the well, its walls on the well's edges: every sample
    # on a line is a node of one material, as on the tensor grid
    qc = QuantumChannel((100.5, 119.5), x_range=(120, 380), line_spacing=20.0,
                        holes=(1.0, 1.0, "test"))
    layers = [StackLayer(bar, 100.0, donors=1e-4), StackLayer(g, 20.0, donors=1e-3),
              StackLayer(bar, 80.0, donors=1e-4)]
    T = Device(layers, 500.0, el, mob, Surface("states", fixed_charge=0.0),
               grid=dict(x=x, z=z), quantum=qc, hole_mass=1.0, hole_mobility=hmob)
    mesh = mesh_from_grid(x, z, lambda xx, zz: _region(zz))
    P = mesh.points
    top = np.isclose(P[:, 1], 200.0)
    con = {"source": np.flatnonzero(top & (P[:, 0] <= 100 + 1e-9)),
           "gate": np.flatnonzero(top & (P[:, 0] >= 200 - 1e-9) & (P[:, 0] <= 300 + 1e-9)),
           "drain": np.flatnonzero(top & (P[:, 0] >= 400 - 1e-9))}
    R = {k: MeshRegion(material=m, mobility=mob, donors=d, hole_mobility=hmob)
         for k, m, d in (("b1", bar, 1e-4), ("w", g, 1e-3), ("b2", bar, 1e-4))}
    M = MeshDevice(mesh, R, el, con, quantum=qc, hole_mass=1.0)
    assert len(M._qruns) == len(T._qcols)
    b = {"drain": 0.5, "gate": -0.5}
    a, c = T.solve(b), M.solve(b)
    assert np.max(np.abs(a.quantum_potential)) > 0.1             # a real correction
    assert c.current["drain"] == pytest.approx(a.current["drain"], rel=1e-10)
    assert np.max(np.abs(a.psi.ravel() - c.psi)) < 1e-10
    assert np.max(np.abs(a.quantum_potential.ravel() - c.quantum_potential)) < 1e-10
    # since 0.32.0 the ballistic limit (and NEGF, see test_v032) work on
    # the mesh's lines too
    assert M.ballistic_current(c)[0] == pytest.approx(T.ballistic_current(a)[0], rel=1e-10)


def test_quantum_channel_on_an_unstructured_mesh_converges():
    import warnings
    from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_points
    g, bar = _well_layers()
    mob, W = Mobility(500.0, "t"), 20.0
    layers = [StackLayer(bar, 100.0, donors=1e-4), StackLayer(g, 20.0, donors=1e-3),
              StackLayer(bar, 80.0, donors=1e-4)]
    R = {k: MeshRegion(material=m, mobility=mob, donors=d)
         for k, m, d in (("b1", bar, 1e-4), ("w", g, 1e-3), ("b2", bar, 1e-4))}
    el = [Electrode("top", "ohmic", 0, W)]
    err = []
    for dz in (1.0, 0.5):
        zr = np.unique(np.r_[np.linspace(0, 90, 10), np.arange(90, 130 + 1e-9, dz),
                             np.linspace(130, 200, 8)])
        T = Device(layers, W, el, mob, Surface("states", fixed_charge=0.0),
                   grid=dict(x=np.linspace(0, W, 3), z=zr),
                   quantum=QuantumChannel((95.0, 125.0), x_range=(0, W)))
        ref = np.mean(T.solve({}).quantum["sheet_cm2"])
        # rows of nodes, every other one shifted by half a spacing: a mesh of
        # nearly equilateral triangles whose edges the lines cross between nodes
        pts = []
        for j, zz in enumerate(zr):
            hx = 1.15 * dz if 90 <= zz <= 130 else 10.0
            xs = np.linspace(0, W, int(round(W / hx)) + 1)
            if j % 2 and 90 < zz < 130:
                xs = np.r_[0.0, 0.5 * (xs[1:] + xs[:-1]), W]
            pts += [(xx, zz) for xx in xs]
        m = mesh_from_points(np.array(pts), lambda xx, zz: _region(zz))
        P = m.points
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            M = MeshDevice(m, R, el, {"top": np.flatnonzero(np.isclose(P[:, 1], 200))},
                           quantum=QuantumChannel((95.0, 125.0), x_range=(0, W),
                                                  line_spacing=W / 4))
        sh = np.array(M.solve({}).quantum["sheet_cm2"])
        assert sh.size >= 4
        err.append(np.max(np.abs(sh / ref - 1)))
    # the confined sheet against the tensor grid: second order in the step
    assert err[1] < 6e-3 and err[0] / err[1] > 3.0


# ------------------------------------------ implant ranges by Monte Carlo
def test_bca_scattering_and_stopping_against_exact_results():
    from kpenvelope import bca
    coul = lambda x: np.ones_like(x)                      # noqa: E731
    # the Coulomb potential: theta = 2 atan(1 / (2 eps b)) exactly
    E, B = np.meshgrid([1e-3, 0.1, 1.0, 10.0, 100.0], [0.01, 0.5, 2.0, 10.0, 100.0])
    assert np.max(np.abs(bca.reduced_angle(E, B, screening=coul)
                         - 2 * np.arctan(1 / (2 * E * B)))) < 1e-9
    # its stopping up to p_max: gamma E pi (d^2 / 4) ln(1 + (2 p_max / d)^2)
    Z1, M1, Z2, M2 = 5, 11.009, 14, 28.0855
    for En in (1e3, 1e5):
        a = bca.zbl_length(Z1, Z2)
        d = Z1 * Z2 * bca._E2 / (En * M2 / (M1 + M2))
        g = 4 * M1 * M2 / (M1 + M2) ** 2
        ex = g * En * math.pi * d * d / 4 * math.log(1 + (2 * 50 * a / d) ** 2)
        assert bca.nuclear_stopping(En, Z1, M1, Z2, M2, screening=coul, b_max=50.0,
                                    n_p=800) == pytest.approx(ex, rel=1e-5)
    # the ZBL stopping against the universal fit where it is accurate (eps <= 1)
    for Z1_, M1_ in ((5, 11.009), (33, 74.92)):
        Es = np.array([1e3, 1e4])
        assert np.allclose(bca.nuclear_stopping(Es, Z1_, M1_, Z2, M2),
                           bca.nuclear_stopping_fit(Es, Z1_, M1_, Z2, M2), rtol=1e-2)
    # the interpolation table against the direct quadrature: the stopping
    # it gives (the energy transfer averaged over impact parameters)
    tab = bca._table()
    for eps in (1e-3, 0.1, 3.0):
        b = np.exp(np.linspace(math.log(1e-4), math.log(100.0), 3000))
        f_t = tab(np.full(b.size, eps), b) * b * b
        f_d = np.sin(0.5 * bca.reduced_angle(np.full(b.size, eps), b)) ** 2 * b * b
        lb = np.log(b)
        assert _trap(f_t, lb) == pytest.approx(_trap(f_d, lb), rel=2e-3)


def test_bca_monte_carlo_ranges():
    from kpenvelope import bca
    from kpenvelope.process import Process
    Si = bca.Target((bca.Element(14, 28.0855),), 4.994e22, "test density")
    N = 4.994e22 * 1e-21
    # electronic stopping alone: the path is 2 (sqrt(E0) - sqrt(E_cut)) / (N k)
    r = bca.simulate(5, 11.009, 3e4, Si, n_ions=200, nuclear=False)
    k = bca.electronic_stopping(1.0, 5, 11.009, 14)
    assert r.path.mean() == pytest.approx(2 * (math.sqrt(3e4) - math.sqrt(5)) / (N * k),
                                          rel=2e-3)
    assert r.projected_range == pytest.approx(r.path.mean(), rel=1e-9)
    # with collisions: the mean path against the continuous-slowing-down
    # path int dE / (N (S_n + S_e)) (ions much heavier than the target atoms
    # lose their energy in many small steps)
    for Z1, M1, E0 in ((51, 121.76, 1e5), (33, 74.92, 1e5)):
        Es = np.logspace(math.log10(5.0), math.log10(E0), 300)
        S = bca.nuclear_stopping(Es, Z1, M1, 14, 28.0855) + bca.electronic_stopping(
            Es, Z1, M1, 14)
        csda = _trap(1 / (N * S), Es)
        rr = bca.simulate(Z1, M1, E0, Si, n_ions=600, seed=1)
        assert rr.path.mean() == pytest.approx(csda, rel=0.03)
        assert rr.projected_range < rr.path.mean()
    # two independent runs agree within their statistics
    a = bca.simulate(5, 11.009, 3e4, Si, n_ions=1500, seed=2)
    b = bca.simulate(5, 11.009, 3e4, Si, n_ions=1500, seed=3)
    se = a.straggle / math.sqrt(1500)
    assert abs(a.projected_range - b.projected_range) < 4 * math.sqrt(2) * se
    # as an implant step: the profile holds the dose of the ions that stayed
    imp = a.implant("acceptor", 1e14, surface_z=1000.0)
    pr = Process([imp], width=10.0, z_bottom=0.0, z_top=1000.0, nx=3, nz=4001)
    assert pr.sheet("acceptor")[1] == pytest.approx(1e14 * (1 - a.backscattered / 1500),
                                                     rel=2e-3)


# ------------------------------------------- ion-enhanced etching kinetics
def _kin(**kw):
    from kpenvelope.process import EtchKinetics
    base = dict(neutral_flux=1e18, ion_flux=1e16, site_density=1e15,
                sticking={"Si": 0.2}, ion_yield={"Si": 5.0}, atoms_cm3={"Si": 5e22},
                reference="test", sputter_yield={"Si": 0.1}, thermal_rate={"Si": 0.0})
    base.update(kw)
    return EtchKinetics(**base)


def test_etch_kinetics_flat_surface_and_limits():
    from kpenvelope.process import Structure
    k = _kin()
    th, v = k.flat("Si")
    a, b = 0.2 * 1e18, 5.0 * 1e16
    assert th == pytest.approx(a / (a + b), rel=1e-12)
    assert v == pytest.approx((5.0 * 1e16 * th + 0.1 * 1e16 * (1 - th)) / 5e22 * 1e7, rel=1e-12)
    # a flat, open surface sees the fluxes as given (what it re-emits leaves)
    st = Structure(100.0, 0.0, 100.0, 2.0, "Si", 50.0)
    for q in st.kinetic_rates(k):
        if q["face"][2] == 0:
            assert q["neutral"] == pytest.approx(1e18, rel=1e-9)
            assert q["ion"] == pytest.approx(1e16, rel=1e-9)
            assert q["speed"] == pytest.approx(v, rel=1e-9)
    # the two limits: plenty of neutrals -> ion-limited (Y G_i); plenty of
    # ions -> neutral-limited (s0 G_n)
    _, v1 = _kin(neutral_flux=1e24, sputter_yield={}).flat("Si")
    assert v1 == pytest.approx(5.0 * 1e16 / 5e22 * 1e7, rel=1e-6)
    _, v2 = _kin(ion_flux=1e22, sputter_yield={}).flat("Si")
    assert v2 == pytest.approx(0.2 * 1e18 / 5e22 * 1e7, rel=1e-5)


def test_etch_kinetics_in_a_trench():
    from kpenvelope.process import Structure
    w, d = 40.0, 60.0
    st = Structure(200.0, 0.0, 200.0, 2.0, "Si", 150.0)
    st.etch({"Si": 1.0}, d, mode="directional", window=(100 - w / 2, 100 + w / 2))
    # every face bare (ions take the cover at once): the neutrals are those of
    # the re-emission with a fixed sticking (checked against Monte Carlo in
    # 0.25.0)
    q = st.kinetic_rates(_kin(ion_yield={"Si": 1e9}))
    ref = {(i, k, sd): f for (i, k, sd, f) in st._face_fluxes(1.0, 0.2)}
    for r in q:
        assert r["neutral"] == pytest.approx(1e18 * ref[r["face"]], rel=1e-6, abs=1e6)
    # neutrals are conserved: what enters = what sticks + what leaves
    k = _kin()
    q = st.kinetic_rates(k)
    faces, F = st.view_factors()
    idx = {(f[0], f[1], f[2]): n for n, f in enumerate(faces)}
    direct = {(i, kk, sd): f for (i, kk, sd, f) in st.fluxes(1.0)}
    s_eff = np.array([0.2 * (1 - r["coverage"]) for r in q])
    G = np.array([r["neutral"] for r in q])
    order = np.array([idx[r["face"]] for r in q])
    esc = ((1 - s_eff) * G * (1 - F[order].sum(1))).sum()
    stuck = (s_eff * G).sum()
    came = sum(1e18 * direct.get(r["face"], 0.0) for r in q)
    assert stuck + esc == pytest.approx(came, rel=3e-3)
    # aspect-ratio dependent etching: the bottom of the trench etches slower
    # than the open surface, much slower when the neutrals limit the rate
    # than when the (directional) ions do
    def bottom_ratio(kin):
        rr = st.kinetic_rates(kin)
        kb = int(np.argmin(np.abs(st.z - (150 - d)))) - 1
        bot = np.mean([r["speed"] for r in rr if r["face"][2] == 0 and r["face"][1] == kb
                       and abs(st.xc[r["face"][0]] - 100) < 4])
        return bot / kin.flat("Si")[1]
    neutral_lim = bottom_ratio(_kin(ion_flux=1e19, sputter_yield={}))
    ion_lim = bottom_ratio(_kin(neutral_flux=1e21, sputter_yield={}))
    assert neutral_lim < 0.7 and ion_lim > 0.95
    # and it etches: the trench deepens, the rest of the surface goes down too
    st2 = Structure(200.0, 0.0, 200.0, 4.0, "Si", 150.0)
    st2.etch({"Si": 1.0}, d, mode="directional", window=(100 - w / 2, 100 + w / 2))
    before = st2.thickness("Si", 100.0)
    st2.plasma_etch(_kin(neutral_flux=1e19), 4.0)
    assert st2.thickness("Si", 100.0) < before


# ------------------------------------------------ viscous flow of oxide
def test_stokes_flow_against_exact_solutions():
    from kpenvelope.process import Structure
    eta = 1e9
    # a flat oxide on a rigid substrate, new volume in its bottom row: it
    # rises at q h, sigma_zz = 0, and the bottom row is in compression
    # sigma_xx = -2 eta q (pressure 2 eta q)
    st = Structure(40.0, 0.0, 120.0, 2.0, "Si", 50.0)
    st.deposit("O", 30.0)
    nx, nz = st.mat.shape
    ko = [k for k in range(nz) if st.mat[0, k] == "O"]
    q = np.zeros((nx, nz))
    q[:, ko[0]] = 1e-3
    r = st.stokes({"O": eta}, source=q)
    assert np.allclose(r["w"][:, ko[0] + 1:ko[-1] + 2], 1e-3 * 2.0, rtol=1e-9)
    assert np.nanmax(np.abs(r["szz"][:, ko])) < 1e-6 * eta * 1e-3
    assert np.allclose(r["sxx"][:, ko[0]], -2 * eta * 1e-3, rtol=1e-9)
    assert np.allclose(r["p"][:, ko[0]], 2 * eta * 1e-3, rtol=1e-9)
    # a rigid cylinder in a ring of oxide that grows from a shell at its
    # surface: v_r = Q / (2 pi r), sigma_tt - sigma_rr = 4 eta (Q / 2 pi) / r^2
    out = []
    for h in (2.0, 1.0):
        L = 120.0
        st = Structure(L, 0.0, L, h, "Si", 0.0)
        X, Z = np.meshgrid(st.xc, st.zc, indexing="ij")
        R = np.hypot(X - L / 2, Z - L / 2)
        st.mat[:] = None
        st.mat[R < 10.0] = "C"
        st.mat[(R >= 10.0) & (R < 30.0)] = "O"
        q = np.where((R >= 10.0) & (R < 14.0), 1e-3, 0.0)
        Q = q.sum() * h * h
        r = st.stokes({"O": eta}, source=q)
        c, s_ = (X - L / 2) / np.maximum(R, 1e-9), (Z - L / 2) / np.maximum(R, 1e-9)
        srr = r["sxx"] * c * c + r["szz"] * s_ * s_ + 2 * r["sxz"] * c * s_
        stt = r["sxx"] * s_ * s_ + r["szz"] * c * c - 2 * r["sxz"] * c * s_
        m = np.abs(R - 20.0) < 0.75 * h
        uc = 0.5 * (r["u"][1:] + r["u"][:-1])
        wc = 0.5 * (r["w"][:, 1:] + r["w"][:, :-1])
        vr = np.mean((uc * c + wc * s_)[m]) / (Q / (2 * math.pi) / 20.0) - 1
        dev = np.nanmean((stt - srr)[m]) / (4 * eta * Q / (2 * math.pi) / 400.0) - 1
        out.append((abs(vr), abs(dev)))
    assert out[1][0] < 2e-3 and out[1][1] < 2e-3 and out[0][0] > out[1][0]


def test_oxidation_with_viscous_stress():
    from kpenvelope.process import Structure
    B, BA = 0.5, 0.05

    def run(geom, stress):
        st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
        if geom == "step":
            st.etch({"Si": 1.0}, 40.0, mode="directional", window=(40, 80))
        kw = dict(viscosity={"O": 1e12}, stress_volume_nm3=0.025,
                  temperature_K=1273.0) if stress else {}
        st.oxidize("Si", "O", B, BA, 2000.0, 0.44, "test", mode="2d", **kw)
        return st
    # flat: the stress normal to the interface is zero, Deal-Grove is kept
    a, b = run("flat", False), run("flat", True)
    assert np.all(a.mat == b.mat)
    assert np.nanmin(b.oxide_stress["sxx"]) < 0               # in-plane compression
    # a step: the corners are compressed and grow less (Kao et al. 1988)
    a, b = run("step", False), run("step", True)
    assert b.area("O") < a.area("O")
    with pytest.raises(ValueError, match="mode='2d'"):
        Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0).oxidize(
            "Si", "O", B, BA, 10.0, 0.44, "t", viscosity={"O": 1e12})
