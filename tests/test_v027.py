"""Checks of the features new in 0.27.0: NEGF in coupled mode space and the
quality mesh generator."""

import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope.negf import (_H2M, Chain, Slab, chain_transport,  # noqa: E402
                             lead_self_energy, lead_surface_green, slab_transport)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
KT = 0.025852


# ------------------------------------------------- coupled mode space
def test_lead_decimation_against_the_exact_chain_lead():
    t = _H2M / (0.2 * 0.25)
    E = np.array([-0.1, 0.05, 0.3, 1.0])
    g = lead_surface_green(E, np.array([[2 * t]]), np.array([[-t]]))
    sig, _ = lead_self_energy(E, 0.0, t)
    assert np.max(np.abs(t * t * g[:, 0, 0] - sig)) < 1e-7


def _dense(sl, U, m, z, zr, E):
    """The same 2D problem in real space, inverted as one dense matrix."""
    Nx, Nz = U.shape
    w = sl.w
    h = np.diff(np.concatenate([[0.0], z, [zr]]))

    def Hz(j):
        mm = np.concatenate([[m[j, 0]], m[j], [m[j, -1]]])
        c = _H2M / (0.5 * (mm[1:] + mm[:-1]) * h)
        d = (U[j] * w + c[:-1] + c[1:]) / w
        o = -c[1:-1] / np.sqrt(w[:-1] * w[1:])
        return np.diag(d) + np.diag(o, 1) + np.diag(o, -1)
    N = Nx * Nz
    H = np.zeros((N, N))
    for j in range(Nx):
        tl = sl.tL if j == 0 else sl.tx[j - 1]
        tr = sl.tR if j == Nx - 1 else sl.tx[j]
        H[j * Nz:(j + 1) * Nz, j * Nz:(j + 1) * Nz] = Hz(j) + np.diag(tl + tr)
        if j < Nx - 1:
            H[j * Nz:(j + 1) * Nz, (j + 1) * Nz:(j + 2) * Nz] = -np.diag(sl.tx[j])
            H[(j + 1) * Nz:(j + 2) * Nz, j * Nz:(j + 1) * Nz] = -np.diag(sl.tx[j])
    DL, CL = Hz(0) + np.diag(2 * sl.tL), -np.diag(sl.tL)
    DR, CR = Hz(Nx - 1) + np.diag(2 * sl.tR), -np.diag(sl.tR)
    gL = lead_surface_green(np.array([E]), DL, CL)[0]
    gR = lead_surface_green(np.array([E]), DR, CR)[0]
    SL, SR = CL @ gL @ CL, CR @ gR @ CR
    Mx = (E + 1e-9j) * np.eye(N) - H
    Mx[:Nz, :Nz] -= SL
    Mx[-Nz:, -Nz:] -= SR
    G = np.linalg.inv(Mx)
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    G0N = G[:Nz, -Nz:]
    T = np.real(np.trace(GL @ G0N @ GR @ G0N.conj().T))
    AL = np.real(np.diag(G[:, :Nz] @ GL @ G[:, :Nz].conj().T)).reshape(Nx, Nz) / w
    return T, AL


def test_coupled_mode_space_against_real_space_and_mode_counting():
    # a waveguide with a constriction and a heavier layer: with all modes the
    # recursive solution equals the dense real-space inverse
    Nx, Nz = 14, 8
    z = np.linspace(1, 8, Nz)
    U = np.zeros((Nx, Nz))
    U[5:9, :3] = U[5:9, 6:] = 0.4
    m = np.full((Nx, Nz), 0.2)
    m[:, -2:] = 0.4
    sl = Slab(U, m, 0.5, z, 0.0, 9.0)
    Es = np.array([0.08, 0.2, 0.35])
    T = sl.solve(Es)["T"]
    AL, _ = sl.node_spectral(Es)
    for k, e in enumerate(Es):
        Td, ALd = _dense(sl, U, m, z, 9.0, e)
        assert T[k] == pytest.approx(Td, rel=1e-10)
        assert np.max(np.abs(AL[k] - ALd)) < 1e-10 * ALd.max()
    # fewer modes: the transmission approaches the exact one
    err = [np.max(np.abs(Slab(U, m, 0.5, z, 0.0, 9.0, n_modes=M).solve(Es)["T"] - T))
           for M in (2, 4, 6)]
    assert err[0] > err[1] > err[2] and err[2] < 0.02 * T.max()
    # a uniform waveguide: the transmission counts the open modes
    Nz = 12
    z = np.linspace(1, 12, Nz)
    sl = Slab(np.zeros((30, Nz)), 0.2, 0.5, z, 0.0, 13.0)
    Es = np.array([0.005, 0.07, 0.14, 0.2, 0.33])
    T = sl.solve(Es)["T"]
    open_ = np.array([(sl.Ez[0] < e).sum() for e in Es])
    far = np.min(np.abs(sl.Ez[0][None, :] - Es[:, None]), axis=1) > 2e-3
    assert np.allclose(T[far], open_[far], atol=1e-6)


def test_coupled_modes_without_mixing_equal_the_uncoupled_chains():
    # the potential changes only along x: the modes do not mix, so the slab
    # carries the sum of the chains' currents
    Nz = 10
    z = np.linspace(1, 10, Nz)
    x = np.arange(0, 60, 0.5)
    bar = 0.1 * np.exp(-((x - 30) / 8) ** 2)
    sl = Slab(bar[:, None] + np.zeros((x.size, Nz)), 0.2, 0.5, z, 0.0, 11.0, n_modes=4)
    r = slab_transport(sl, 0.3, 0.2, KT)
    tot = sum(chain_transport(Chain(bar + sl.Ez[0][i], 0.2, 0.5), 0.3, 0.2, KT,
                              mass_y=0.2)["I"] for i in range(4))
    assert r["I"] == pytest.approx(tot, rel=1e-5)


def test_device_negf_in_coupled_mode_space():
    from test_v026 import _short_hemt
    dev = _short_hemt()
    b = {"gate": -1.0, "drain": 0.05}
    dd = dev.solve(b)
    u = dev.negf(b, guess=dd)
    c = dev.negf(b, guess=dd, coupled_modes=4)
    # in this channel the subbands hardly mix: within 1 percent of the
    # uncoupled result (0.45 percent with 6 modes, 0.7 percent with 12)
    assert c.negf["current"] == pytest.approx(u.negf["current"], rel=0.01)
    assert c.current["drain"] == -c.current["source"] == c.negf["current"]
    # (since 0.32.0 coupled modes also take scattering_time: test_v032)
    with pytest.raises(ValueError, match="integer"):
        dev.negf(b, coupled_modes=0)


# ------------------------------------------------ quality mesh generator
def _edges(T):
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    return set(map(tuple, e.tolist()))


def test_quality_mesh_angles_conformity_and_areas():
    from scipy.spatial import cKDTree
    from kpenvelope.mesh import _tri_quality, quality_mesh
    # an L-shaped substrate under a slanted layer, a hole-free pair of
    # regions with an interface that is not on any axis
    regs = {"sub": [(0, 0), (100, 0), (100, 40), (60, 40), (60, 70), (0, 50)],
            "cap": [(0, 50), (60, 70), (60, 40), (100, 40), (100, 90), (0, 90)]}
    # the sub: a 100 x 40 slab, a 60 x 10 step and a 60 x 20 triangle on it
    area = {"sub": 100 * 40 + 60 * 10 + 0.5 * 60 * 20}
    area["cap"] = 100 * 90 - area["sub"]
    h = lambda x, z: np.where(x < 50, 4.0, 8.0)                   # noqa: E731
    m = quality_mesh(regs, h=h, min_angle=28)
    P, T = m.points, m.triangles
    amin, _, A, longest = _tri_quality(P, T)
    assert np.all(A > 0)
    assert amin.min() >= 28.0 - 1e-6
    for lab in regs:
        assert A[m.regions == lab].sum() == pytest.approx(area[lab], rel=1e-12)
    # the size field holds: no edge much longer than h where it sits
    c = P[T].mean(axis=1)
    assert np.all(longest <= h(c[:, 0], c[:, 1]) * 1.0001 + 1e-9)
    # every boundary and interface segment is an edge, and no node lies
    # inside a segment's diametral circle (conforming Delaunay)
    E = _edges(T)
    assert all(tuple(sorted(s)) in E for s in m.segments.tolist())
    tree = cKDTree(P)
    for s in m.segments:
        mid = 0.5 * (P[s[0]] + P[s[1]])
        r = 0.5 * np.hypot(*(P[s[1]] - P[s[0]]))
        assert set(tree.query_ball_point(mid, r * (1 - 1e-9))) <= set(s.tolist())
    with pytest.raises(ValueError, match="min_angle"):
        quality_mesh(regs, min_angle=40)
    with pytest.raises(ValueError, match="empty"):
        quality_mesh({})


def test_quality_mesh_resistor_converges():
    import warnings
    from kpenvelope import algan
    from kpenvelope.device import Electrode, Mobility
    from kpenvelope.mesh import MeshDevice, MeshRegion, quality_mesh
    from kpenvelope.nitride import QE
    g, a, b, mu, nd = algan(0.0), 200.0, 600.0, 500.0, 1e18
    # a quarter annulus between two ohmic faces: R = (pi/2) / (q n mu ln(b/a) t)
    R = 1 / (QE * nd * mu) * (np.pi / 2) / (np.log(b / a) * 0.1)
    err = []
    for n, h in ((40, 60.0), (80, 30.0), (160, 15.0)):
        th = np.linspace(0, np.pi / 2, n + 1)
        ring = ([(a * np.cos(t), a * np.sin(t)) for t in th]
                + [(b * np.cos(t), b * np.sin(t)) for t in th[::-1]])
        m = quality_mesh({"gan": ring}, h=h, min_angle=28)
        P = m.points
        con = {"a": np.flatnonzero(np.isclose(P[:, 1], 0.0)),
               "b": np.flatnonzero(np.isclose(P[:, 0], 0.0, atol=1e-9))}
        with warnings.catch_warnings():
            warnings.simplefilter("error")                      # no negative dual faces
            M = MeshDevice(m, {"gan": MeshRegion(material=g, mobility=Mobility(mu, "t"),
                                                 donors=nd * 1e-21)},
                           [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)],
                           con, statistics="boltzmann")
        err.append(abs(abs(M.solve({"b": 0.01}).current["b"]) / (0.01 / R) - 1))
    # second order in the spacing (the polygon's chords included)
    assert err[-1] < 5e-4
    assert err[0] / err[1] > 2.5 and err[1] / err[2] > 2.5


def test_quantum_channel_on_a_quality_mesh():
    import dataclasses
    from kpenvelope import StackLayer, algan
    from kpenvelope.device import Device, Electrode, Mobility, QuantumChannel, Surface
    from kpenvelope.mesh import MeshDevice, MeshRegion, quality_mesh
    g = algan(0.0)
    bar = dataclasses.replace(g, name="bar", ec=g.ec + 0.3)
    mob, W = Mobility(500.0, "t"), 20.0
    layers = [StackLayer(bar, 100.0, donors=1e-4), StackLayer(g, 20.0, donors=1e-3),
              StackLayer(bar, 80.0, donors=1e-4)]
    el = [Electrode("top", "ohmic", 0, W)]
    qc = QuantumChannel((95.0, 125.0), x_range=(0, W), line_spacing=W / 4)
    z = np.unique(np.round(np.r_[np.linspace(0, 90, 10), np.arange(90, 130 + 1e-9, 0.1),
                                 np.linspace(130, 200, 8)], 6))
    ref = np.mean(Device(layers, W, el, mob, Surface("states", fixed_charge=0.0),
                         grid=dict(x=np.linspace(0, W, 3), z=z), quantum=qc)
                  .solve({}).quantum["sheet_cm2"])
    regs = {"b1": [(0, 0), (W, 0), (W, 100), (0, 100)],
            "w": [(0, 100), (W, 100), (W, 120), (0, 120)],
            "b2": [(0, 120), (W, 120), (W, 200), (0, 200)]}
    R = {k: MeshRegion(material=mm, mobility=mob, donors=d)
         for k, mm, d in (("b1", bar, 1e-4), ("w", g, 1e-3), ("b2", bar, 1e-4))}
    m = quality_mesh(regs, h=lambda x, zz: np.where((zz > 88) & (zz < 132), 0.5, 10.0),
                     min_angle=28)
    M = MeshDevice(m, R, el, {"top": np.flatnonzero(np.isclose(m.points[:, 1], 200))},
                   quantum=qc)
    sh = np.array(M.solve({}).quantum["sheet_cm2"])
    # every line, its samples where it crosses the triangles' edges and the
    # well's edges between nodes, against a fine tensor grid
    assert sh.size == 5
    assert np.max(np.abs(sh / ref - 1)) < 5e-3


# ---------------------------------------------------------- NEGF holes
def _mirror_fet(holes, mh=(0.2, 0.2), gate=True):
    # a gated well for electrons, or its mirror image for holes (band
    # profile upside down, donors -> acceptors, the gate's barrier measured
    # from the other band)
    import dataclasses
    from kpenvelope import StackLayer, algan
    from kpenvelope.device import Device, Electrode, Mobility, QuantumChannel, Surface
    base = dataclasses.replace(algan(0.0), eg=3.4, me=0.2, eps_r=9.0)

    def mk(c, name):
        return dataclasses.replace(base, name=name, ec=(-c + 3.4) if holes else c)
    dop = dict(acceptors=1e-3) if holes else dict(donors=1e-3)
    layers = [StackLayer(mk(0.0, "b1"), 50.0, **dop), StackLayer(mk(-0.3, "w"), 10.0),
              StackLayer(mk(0.0, "b2"), 20.0, **dop)]
    el = [Electrode("source", "ohmic", 0, 40),
          Electrode("gate", "schottky", 85, 115, barrier=2.4 if holes else 1.0),
          Electrode("drain", "ohmic", 160, 200)]
    if not gate:
        el = [el[0], el[2]]
    kw = dict(hole_mass=mh[0], hole_mobility=Mobility(300.0, "t")) if holes else {}
    q = (QuantumChannel((44.0, 66.0), x_range=(45, 155), electrons=False,
                        holes=(mh[0], mh[1], "test")) if holes
         else QuantumChannel((44.0, 66.0), x_range=(45, 155)))
    return Device(layers, 200.0, el, Mobility(300.0, "t"), Surface("states", fixed_charge=0.0),
                  dx_fine=5, dx_max=10, dz_fine=0.5, dz_max=10, quantum=q, **kw)


def test_negf_holes_mirror_negf_electrons():
    out = {}
    for h in (False, True):
        d = _mirror_fet(h)
        s = -1.0 if h else 1.0
        b = {"gate": -0.3 * s, "drain": 0.02 * s}
        g = d.solve(b)
        car = "holes" if h else "electrons"
        out[h] = (d.negf(b, guess=g, dx=0.5, carrier=car),
                  d.negf(b, guess=g, dx=1.0, coupled_modes=2, tol=1e-3, carrier=car))
    (a, ac), (b_, bc) = out[False], out[True]
    assert a.current["drain"] > 0
    assert -b_.current["drain"] == pytest.approx(a.current["drain"], rel=1e-10)
    assert np.allclose(b_.quantum["hole_sheet_cm2"], a.quantum["sheet_cm2"], rtol=1e-10)
    assert np.max(np.abs(a.psi + b_.psi - (a.psi + b_.psi).mean())) < 1e-9
    assert np.allclose(b_.efp[1:-1], -a.efn[1:-1], atol=1e-9)
    assert np.allclose(b_.negf["modes"][0]["E"], -a.negf["modes"][0]["E"], atol=1e-12)
    assert -bc.current["drain"] == pytest.approx(ac.current["drain"], rel=1e-6)
    with pytest.raises(ValueError, match="carrier"):
        _mirror_fet(False).negf({}, carrier="ions")
    with pytest.raises(ValueError, match="holes"):
        _mirror_fet(False).negf({}, carrier="holes")


def test_negf_holes_with_their_own_masses_at_equilibrium():
    # m_z sets the subbands, m_inplane the motion along x and the width: at
    # equilibrium the chains hold what the QuantumChannel's holes hold
    # (without a gate, so that the bands hardly change along x)
    d = _mirror_fet(True, mh=(0.3, 0.6), gate=False)
    qc = d.solve({})
    eq = d.negf({}, guess=qc, dx=0.5, carrier="holes")
    assert eq.current["drain"] == 0.0
    a, b = np.array(eq.quantum["hole_sheet_cm2"]), np.array(qc.quantum["hole_sheet_cm2"])
    k = [i for i, (x, _) in enumerate(qc.quantum["hole_subbands"]) if 45 <= x <= 155]
    assert a.size == len(k)
    assert np.max(np.abs(a[2:-2] / b[k][2:-2] - 1)) < 1e-3
    assert np.all(np.array(eq.negf["modes"][0]["mass"]) == 0.6)


def test_slab_with_its_own_mass_along_x():
    # a barrier along x only, with one mass across (0.3) and another along x
    # and the width (0.7): the slab carries the sum of the chains of its
    # column states, each with the mass along x
    Nz = 10
    z = np.linspace(1, 10, Nz)
    x = np.arange(0, 60, 0.5)
    bar = 0.1 * np.exp(-((x - 30) / 8) ** 2)
    sl = Slab(bar[:, None] + np.zeros((x.size, Nz)), 0.3, 0.5, z, 0.0, 11.0, n_modes=4,
              mass_x=0.7)
    r = slab_transport(sl, 0.3, 0.2, KT, mass_y=0.7)
    tot = sum(chain_transport(Chain(bar + sl.Ez[0][i], 0.7, 0.5), 0.3, 0.2, KT,
                              mass_y=0.7)["I"] for i in range(4))
    assert r["I"] == pytest.approx(tot, rel=1e-5)
    # and the column states are those of the mass across
    ref = Slab(bar[:, None] + np.zeros((x.size, Nz)), 0.3, 0.5, z, 0.0, 11.0, n_modes=4)
    assert np.allclose(sl.Ez, ref.Ez, atol=1e-13)
    # tunneling through the barrier follows the mass along x
    E = np.array([float(sl.Ez[0][0]) + 0.05])
    assert sl.solve(E)["T"][0] < 0.5 * ref.solve(E)["T"][0]
    with pytest.raises(ValueError, match="positive"):
        Slab(np.zeros((4, Nz)), 0.3, 0.5, z, 0.0, 11.0, mass_x=-1.0)


# --------------------------------------------- scattering lifetimes
def _fang_howard(b, z):
    return 0.5 * b ** 3 * z ** 2 * np.exp(-b * z)


def test_acoustic_and_alloy_rates_against_closed_forms():
    from kpenvelope.electrons import HBAR, KB, M0
    from kpenvelope.nitride import QE
    from kpenvelope.scattering import Subband
    b, m, T = 1.2, 0.2, 300.0
    z = np.linspace(0, 40, 8001)
    sb = Subband(z, _fang_howard(b, z), m, 9.5, sheet_cm2=8e12, temperature_K=T)
    # int psi^4 of the Fang-Howard form is 3 b / 16
    assert sb.overlap() == pytest.approx(3 * b / 16, rel=1e-9)
    # P. J. Price's rate for it: 3 m D^2 kT b / (16 hbar^3 rho v^2)
    D, rho, v = 8.3, 6150.0, 6560.0
    price = 3 * m * M0 * (D * QE) ** 2 * KB * T * QE * b * 1e9 / (16 * HBAR ** 3 * rho * v ** 2)
    ac = sb.acoustic(D, rho, v)
    assert ac(0.05)[0] == pytest.approx(price, rel=1e-9)
    assert ac(0.05, quantum=True)[0] == ac(0.2)[0]
    # the alloy rate on half the gas: m Omega V^2 x (1 - x) int_half psi^4 / hbar^3
    half = z > 2.0
    al = sb.alloy(0.25, 1.5, 0.0228, mask=half)(0.1)[0]
    ref = m * M0 * 0.0228e-27 * (1.5 * QE) ** 2 * 0.25 * 0.75 * sb.overlap(half) * 1e9 / HBAR ** 3
    assert al == pytest.approx(ref, rel=1e-12)
    # an infinite square well of width W: int psi^4 = 3 / (2 W)
    W = 10.0
    zw = np.linspace(0, W, 4001)
    sw = Subband(zw, 2 / W * np.sin(np.pi * zw / W) ** 2, m, 9.5, fermi=0.05)
    assert sw.overlap() == pytest.approx(1.5 / W, rel=1e-6)
    with pytest.raises(ValueError, match="normalized"):
        Subband(zw, np.sin(np.pi * zw / W) ** 2, m, 9.5, fermi=0.05)
    with pytest.raises(ValueError, match="fermi or sheet"):
        Subband(zw, 2 / W * np.sin(np.pi * zw / W) ** 2, m, 9.5)


def test_coulomb_rates_against_an_independent_integral():
    import math
    from scipy.integrate import quad
    from kpenvelope.electrons import EPS0, HBAR, M0
    from kpenvelope.nitride import QE
    from kpenvelope.scattering import Subband
    b, m, er, d, Ns = 1.2, 0.2, 9.5, 5.0, 1e12
    z = np.linspace(0, 40, 8001)
    sb = Subband(z, _fang_howard(b, z), m, er, sheet_cm2=8e12)
    eps = er * EPS0

    def indep(E, quantum):
        # the Fang-Howard form factors in closed form (Ando, Fowler and
        # Stern 1982): exp(-q d) (b / (b + q))^3 and the screening one
        k = math.sqrt(2 * m * M0 * E * QE) / HBAR

        def f(th):
            q = 2 * k * math.sin(th / 2)
            qn = q * 1e-9
            F = math.exp(-qn * d) * (b / (b + qn)) ** 3
            G = (8 * b ** 3 + 9 * b ** 2 * qn + 3 * b * qn ** 2) / (8 * (b + qn) ** 3)
            V = QE ** 2 / (2 * eps * q) * F / (1 + sb.q_s / q * G)
            return Ns * 1e4 * V ** 2 * (1 if quantum else 1 - math.cos(th))
        return m * M0 / (2 * math.pi * HBAR ** 3) * 2 * quad(f, 0, math.pi, limit=400,
                                                             epsabs=0, epsrel=1e-10)[0]
    imp = sb.impurities(z0=-d, sheet_cm2=Ns)
    for E in (0.01, 0.05, 0.2):
        assert imp(E)[0] == pytest.approx(indep(E, False), rel=2e-6)
        assert imp(E, quantum=True)[0] == pytest.approx(indep(E, True), rel=2e-6)
    # a density profile is the sum of its sheets
    prof = np.where((z > 2) & (z < 4), 1e16, 0.0)
    dz = z[1] - z[0]
    sheets = sum(sb.impurities(z0=zj, sheet_cm2=1e16 * dz * 1e-7)(0.05)[0]
                 for zj in z[(z > 2) & (z < 4)])
    assert sb.impurities(profile_cm3=prof)(0.05)[0] == pytest.approx(sheets, rel=1e-10)


def test_remote_impurities_far_away():
    # a thin, strongly screened, cold gas: tau_t / tau_q -> (2 k_F d)^2 and
    # mu -> 8 e (k_F d)^3 / (pi hbar N) (S. Das Sarma and F. Stern 1985),
    # with corrections of order 1 / (q_s d)
    import math
    from kpenvelope.electrons import HBAR
    from kpenvelope.nitride import QE
    from kpenvelope.scattering import Subband
    z = np.linspace(-0.5, 0.5, 201)
    p = np.exp(-z ** 2 / (2 * 0.03 ** 2))
    p /= np.trapezoid(p, z) if hasattr(np, "trapezoid") else np.trapz(p, z)
    out = []
    for d in (25.0, 100.0):
        sb = Subband(z, p, 1.0, 5.0, sheet_cm2=1.5e12, temperature_K=1.0, n_theta=2000)
        kd = math.sqrt(2 * math.pi * sb.sheet) * d * 1e-9
        imp = sb.impurities(z0=-d, sheet_cm2=1e11)
        L = sb.lifetimes(imp)
        mu = sb.mobility(imp)["mobility_cm2Vs"]
        out.append((L["transport_s"] / L["quantum_s"] / (2 * kd) ** 2 - 1,
                    mu / (8 * QE * kd ** 3 / (math.pi * HBAR * 1e15) * 1e4) - 1))
    assert all(abs(a) < 4e-3 and 0 < c < 1.2e-2 for a, c in out)
    assert out[1][1] < out[0][1]


def test_roughness_rate_and_its_matrix_element():
    import math
    from scipy.integrate import quad
    from kpenvelope.device import column_states
    from kpenvelope.electrons import HBAR, M0
    from kpenvelope.nitride import QE
    from kpenvelope.scattering import Subband
    zc = np.linspace(0, 30, 601)
    hz = zc[1] - zc[0]

    def well(zi):
        # a 0.4 eV well from 10 nm to zi, the step put on the nodes by
        # their share of it
        zb = np.r_[zc[0] - hz, zc, zc[-1] + hz]
        lo, hi = 0.5 * (zb[:-2] + zb[1:-1]), 0.5 * (zb[1:-1] + zb[2:])
        fw = np.clip(np.minimum(hi, zi) - np.maximum(lo, 10.0), 0, None) / (hi - lo)
        E, p2, w = column_states(zc, 0.4 * (1 - fw), np.full(zc.size, 0.2), 1,
                                 zc[0] - hz, zc[-1] + hz)
        return E[0], p2[:, 0], w
    zi = 20.0
    _, p2, w = well(zi)
    # moving the interface by Delta moves the level by -step psi^2(z_i) Delta
    dE = (well(zi + 1e-4)[0] - well(zi - 1e-4)[0]) / 2e-4
    assert dE == pytest.approx(-0.4 * np.interp(zi, zc, p2), rel=1e-6)
    sb = Subband(zc, p2, 0.2, 9.5, sheet_cm2=5e11, weights=w)
    M = 0.4 * QE * np.interp(zi, zc, p2) * 1e9
    D, L = 0.3e-9, 2e-9
    for corr in ("gaussian", "exponential"):
        rr = sb.roughness(zi, 0.4, 0.3, 2.0, correlation=corr)

        def indep(E, quantum):
            k = math.sqrt(2 * 0.2 * M0 * E * QE) / HBAR

            def f(th):
                q = 2 * k * math.sin(th / 2)
                S = (math.pi * D * D * L * L * math.exp(-(q * L) ** 2 / 4)
                     if corr == "gaussian" else
                     2 * math.pi * D * D * L * L / (1 + (q * L) ** 2) ** 1.5)
                e = 1 + sb.q_s / q * sb.form_factor_ee(np.array([q * 1e-9]))[0]
                return M * M * S / e ** 2 * (1 if quantum else 1 - math.cos(th))
            return 0.2 * M0 / (2 * math.pi * HBAR ** 3) * 2 * quad(f, 0, math.pi,
                                                                   epsrel=1e-10)[0]
        for E in (0.01, 0.1):
            assert rr(E)[0] == pytest.approx(indep(E, False), rel=1e-8)
            assert rr(E, quantum=True)[0] == pytest.approx(indep(E, True), rel=1e-8)
    # short correlation, no screening: pi m M^2 Delta^2 L^2 / hbar^3 for both
    s0 = Subband(zc, p2, 0.2, 9.5, sheet_cm2=5e11, weights=w, screening=False)
    r0 = s0.roughness(zi, 0.4, 0.3, 0.01)
    lim = math.pi * 0.2 * M0 * M * M * 0.3e-9 ** 2 * 0.01e-9 ** 2 / HBAR ** 3
    assert r0(0.01)[0] == pytest.approx(lim, rel=1e-4)
    assert r0(0.01, quantum=True)[0] == pytest.approx(lim, rel=1e-4)


def test_mobility_average_over_the_thermal_distribution():
    import math
    from scipy.integrate import quad
    from scipy.special import gamma
    from kpenvelope.electrons import KB, M0
    from kpenvelope.nitride import QE
    from kpenvelope.scattering import Subband
    z = np.linspace(0, 10, 401)
    p = 0.2 * np.sin(np.pi * z / 10) ** 2
    kT = KB * 300.0
    for sheet in (1e8, 5e12):                       # non-degenerate and degenerate
        sb = Subband(z, p, 0.2, 9.5, sheet_cm2=sheet)
        const = lambda E, quantum=False: np.full(np.shape(E), 1e13)      # noqa: E731
        assert sb.mobility(const)["mobility_cm2Vs"] == pytest.approx(
            QE * 1e-13 / (0.2 * M0) * 1e4, rel=1e-12)
        for s in (-0.5, 0.5, 1.0):
            f = lambda E, quantum=False, s=s: 1e13 / np.asarray(E) ** s       # noqa: E731
            num = quad(lambda e: 1e-13 * e ** s * e * 0.25 / kT
                       / math.cosh((e - sb.fermi) / (2 * kT)) ** 2, 0, 80 * kT + sb.fermi,
                       points=[kT, max(sb.fermi, kT)], limit=200)[0]
            ref = num / (kT * math.log1p(math.exp(sb.fermi / kT)))
            assert sb.mobility(f)["tau_s"] == pytest.approx(ref, rel=2e-5)
            if sheet < 1e9:
                assert sb.mobility(f)["tau_s"] == pytest.approx(
                    1e-13 * gamma(s + 2) * kT ** s, rel=5e-5)
    # Matthiessen's rule at each energy
    a = lambda E, quantum=False: np.full(np.shape(E), 1e13)              # noqa: E731
    r = sb.mobility(a, a)
    assert r["mobility_cm2Vs"] == pytest.approx(0.5 * r["each"][0], rel=1e-12)
