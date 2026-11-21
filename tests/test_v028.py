"""Checks of the features new in 0.28.0."""

import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
Q = 1.602176634e-19


# ------------------------------------------- polar optical phonons
def _fh(b, z):
    return 0.5 * b ** 3 * z ** 2 * np.exp(-b * z)


def test_polar_optical_angular_integrals_in_closed_form():
    # a gas of zero thickness (H = 1): int dtheta / q = 4 K(m) / (k + k')
    # and int dtheta cos(theta) / q = (4 A K / (k + k') - 4 (k + k') E) / B,
    # A = k^2 + k'^2, B = 2 k k', m = 4 k k' / (k + k')^2
    from scipy.special import ellipe, ellipk
    from kpenvelope.scattering import Subband
    z = np.linspace(-0.05, 0.05, 3)
    sb = Subband(z, np.array([0.0, 20.0, 0.0]), 0.2, 9.5, sheet_cm2=5e12)
    pol = sb.polar_optical(0.092, 9.5, 5.35)
    for E, dw in ((0.05, 0.092), (0.2, 0.092), (0.3, 0.001)):
        k, kp = sb.k(np.array([E])), sb.k(np.array([E + dw]))
        I0, I1 = pol.angular(k, kp)
        A, B = k * k + kp * kp, 2 * k * kp
        m = 2 * B / (A + B)
        assert I0[0] == pytest.approx((4 * ellipk(m) / (k + kp))[0], rel=1e-10)
        ref1 = (4 * A * ellipk(m) / (k + kp) - 4 * (k + kp) * ellipe(m)) / B
        assert I1[0] == pytest.approx(ref1[0], rel=1e-10)
    # the spline of the form factor against the direct sum
    b = 1.2
    zz = np.linspace(0, 40, 801)
    s2 = Subband(zz, _fh(b, zz), 0.2, 9.5, sheet_cm2=8e12)
    q = np.exp(np.random.default_rng(1).uniform(np.log(1e-5), np.log(500), 500))
    assert np.allclose(s2.polar_optical(0.092, 9.5, 5.35).H(q), s2.form_factor_ee(q),
                       rtol=1e-8)


def test_polar_optical_boltzmann_solution():
    import math
    from kpenvelope.scattering import Subband
    b = 1.2
    z = np.linspace(0, 40, 801)
    for sheet in (1e10, 8e12):                     # non-degenerate and degenerate
        sb = Subband(z, _fh(b, z), 0.2, 9.5, sheet_cm2=sheet)
        ac = sb.acoustic(8.3, 6150, 6560)
        imp = sb.impurities(z0=-5, sheet_cm2=1e12)
        # a vanishing coupling: the exact solution is the elastic one
        weak = sb.polar_optical(0.092, 9.5, 9.5 * (1 - 1e-9))
        r = sb.mobility(ac, imp, weak)
        assert r["mobility_cm2Vs"] == pytest.approx(sb.mobility(ac, imp)["mobility_cm2Vs"],
                                                    rel=1e-5)
        # a small phonon energy: the phonons scatter nearly elastically, with
        # the momentum rate C (2N + 1) int dtheta (1 - cos theta) H / q
        err = []
        for hw in (2e-3, 5e-4):
            pol = sb.polar_optical(hw, 9.5, 5.35)

            def elastic(E, quantum=False, pol=pol):
                I0, I1 = pol.angular(sb.k(E), sb.k(np.asarray(E) + 1e-9))
                return pol.C * (2 * pol.N + 1) * (I0 - I1)
            err.append(abs(sb.mobility(pol)["mobility_cm2Vs"]
                           / sb.mobility(elastic)["mobility_cm2Vs"] - 1))
        assert err[1] < 2e-3 and err[1] < err[0] / 4
        # the linearized collision operator is symmetric with the weight
        # f0 (1 - f0) (detailed balance with equilibrium phonons)
        pol = sb.polar_optical(0.092, 9.5, 5.35)
        n_sub = 16
        dE = 0.092 / n_sub
        nE = int(math.ceil((max(sb.fermi, 0) + 40 * sb.kT) / dE))
        E = (np.arange(nE) + 0.5) * dE
        f0 = 0.5 * (1 - np.tanh(0.5 * (E - sb.fermi) / sb.kT))
        So, cm, cp = pol._operator(E, f0, n_sub)
        M = np.diag(So)
        for j in range(n_sub, nE):
            M[j, j - n_sub] = -cm[j]
        for j in range(nE - n_sub):
            M[j, j + n_sub] = -cp[j]
        S = M * (f0 * (1 - f0))[None, :]
        assert np.max(np.abs(S - S.T)) < 1e-10 * np.max(np.abs(S))
        # the energy grid: steps of hbar w / 64 and / 128 agree
        a = sb.mobility(pol, n_energy=64)["mobility_cm2Vs"]
        c = sb.mobility(pol, n_energy=128)["mobility_cm2Vs"]
        assert a == pytest.approx(c, rel=2e-4)
        assert sb.lifetimes(pol, ac)["transport_s"] is None
        with pytest.raises(ValueError, match="inelastic"):
            pol(0.1)


# ------------------------------------------------ meshes that adapt
def test_recovered_hessian_is_exact_for_quadratics():
    from kpenvelope.mesh import mesh_from_grid, quality_mesh, recovered_hessian
    for m in (mesh_from_grid(np.linspace(0, 10, 21), np.linspace(0, 10, 21),
                             lambda x, z: "a"),
              quality_mesh({"a": [(0, 0), (10, 0), (10, 10), (0, 10)]}, h=0.4,
                           min_angle=28)):
        P = m.points
        u = 1.5 * P[:, 0] ** 2 - 0.7 * P[:, 0] * P[:, 1] + 0.3 * P[:, 1] ** 2 + P[:, 0] - 4
        H = recovered_hessian(m, u)                   # every node, the boundary too
        assert np.max(np.abs(H - [3.0, -0.7, 0.6])) < 1e-6


def test_adaptive_mesh_of_an_l_shaped_resistor():
    # current turning the inner corner of an L: the field is singular there,
    # and a uniform mesh converges slowly. Meshes that follow the
    # potential's curvature reach a given error with far fewer nodes.
    # Reference 0.31938 ohm mm: adaptive meshes up to 75 000 nodes and
    # uniform ones up to 51 000, both extrapolated (agree within 1e-5).
    import warnings
    from kpenvelope import algan
    from kpenvelope.device import Electrode, Mobility
    from kpenvelope.mesh import MeshDevice, MeshRegion, quality_mesh, size_from_solution
    L = [(0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)]

    def resistance(m):
        P = m.points
        con = {"a": np.flatnonzero(np.isclose(P[:, 0], 200.0)),
               "b": np.flatnonzero(np.isclose(P[:, 1], 200.0))}
        with warnings.catch_warnings():
            warnings.simplefilter("error")                     # no negative dual faces
            M = MeshDevice(m, {"s": MeshRegion(material=algan(0.0),
                                               mobility=Mobility(500.0, "t"), donors=1e-3)},
                           [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)],
                           con, statistics="boltzmann")
        st = M.solve({"b": 0.001})
        return 0.001 / abs(st.current["b"]), st
    ref = 0.31938
    mu = quality_mesh({"s": L}, h=2.5, min_angle=28)
    err_u = abs(resistance(mu)[0] / ref - 1)
    m = quality_mesh({"s": L}, h=20, min_angle=28)
    for it in range(5):
        R, st = resistance(m)
        hf = size_from_solution(m, st.psi, tolerance=2e-6 * 0.5 ** it, h_min=0.01, h_max=20)
        m = quality_mesh({"s": L}, h=hf, min_angle=28)
    R, _ = resistance(m)
    err_a = abs(R / ref - 1)
    assert len(m.points) < 0.5 * len(mu.points)
    assert err_a < 0.5 * err_u
    with pytest.raises(ValueError, match="tolerance"):
        size_from_solution(m, np.zeros(len(m.points)), 0.0, 1.0, 2.0)
    # a really obtuse triangle on a boundary still gets the warning
    from kpenvelope.mesh import mesh_from_points
    bad = mesh_from_points(np.array([(0, 0), (10, 0), (5, 0.5), (0, 6), (10, 6)], float),
                           lambda x, z: "s")
    with pytest.warns(RuntimeWarning, match="negative dual"):
        MeshDevice(bad, {"s": MeshRegion(material=algan(0.0), mobility=Mobility(500.0, "t"),
                                         donors=1e-3)},
                   [Electrode("a", "ohmic", 0, 1)], {"a": np.array([3, 4])},
                   statistics="boltzmann")


# ------------------------------------- Fermi-Dirac thermoelectric effects
def test_fermi_dirac_three_halves_and_the_peltier_factor():
    import math
    from scipy.integrate import quad
    from kpenvelope.device import _fd32, _seebeck_fd
    for eta in (-50.0, -10.0, -1.3, 0.0, 2.7, 15.0, 39.99, 60.0):
        ref = quad(lambda x: x ** 1.5 / (1 + math.exp(min(x - eta, 700))), 0,
                   max(eta, 0) + 60, points=[max(eta, 0)] if eta > 0 else None,
                   epsabs=0, epsrel=1e-13, limit=400)[0] / math.gamma(2.5)
        assert _fd32(np.array([eta]))[0][0] == pytest.approx(ref, rel=1e-8)
    G, Gd, dGd = _seebeck_fd(np.array([4.0, -30.0]))
    # 4 kT above the band edge: 1.09 kT (the Boltzmann-like form gives 0.63 kT)
    assert G[0] == pytest.approx(1.0911, abs=1e-4)
    assert G[0] - Gd[0] == pytest.approx(0.63, abs=0.01)
    assert G[1] == pytest.approx(32.5, rel=1e-12)
    # far below the band edge the excess is exp(eta) / 2^(7/2) (the first
    # terms of the series of F_1/2 and F_3/2), and its slope the same
    assert Gd[1] == pytest.approx(np.exp(-30.0) * 2 ** -3.5, rel=1e-9)
    assert dGd[1] == pytest.approx(np.exp(-30.0) * 2 ** -3.5, rel=1e-9)
    # the table and the series meet at -4
    a, b = _seebeck_fd(np.array([-4.0 - 1e-12, -4.0 + 1e-12]))[1:]
    assert a[0] == pytest.approx(a[1], rel=1e-7) and b[0] == pytest.approx(b[1], rel=1e-6)


def test_peltier_factor_against_arbitrary_precision():
    # F_j(eta) = -Li_(j+1)(-e^eta) with mpmath (skipped where it is absent)
    from kpenvelope.device import _seebeck_fd
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40

    def F(j, x):
        return -mp.re(mp.polylog(j + 1, -mp.e ** x))
    for x in (-30.0, -4.1, -3.9, 0.5, 6.0):
        X = mp.mpf(x)
        f12, f32, fm = F(0.5, X), F(1.5, X), F(-0.5, X)
        gd = 2.5 * f32 / f12 - X - 2.5 + mp.log(f12)
        dgd = 2.5 * (1 - f32 * fm / f12 ** 2) - 1 + fm / f12
        g, a, b = _seebeck_fd(np.array([x]))
        assert a[0] == pytest.approx(float(gd), rel=1e-6, abs=1e-15)
        assert b[0] == pytest.approx(float(dgd), rel=1e-5, abs=1e-15)


def _fd_bar(te, nd=2e19, L=4000.0, tm="full", heat="joule"):
    from kpenvelope import StackLayer, algan
    from kpenvelope.device import Device, Electrode, Mobility, Surface
    z = np.linspace(0, L, 201)
    return Device([StackLayer(algan(0.0), L, donors=nd * 1e-21)], 20.0,
                  [Electrode("top", "ohmic", 0, 20),
                   Electrode("bot", "ohmic", 0, 20, face="bottom")],
                  Mobility(1000.0, "t"), Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0., 10., 20.]), z=z), statistics="fermi",
                  temperature_model=tm, heat_model=heat, thermoelectric=te)


def test_fermi_dirac_thermoelectric_current_of_a_degenerate_bar():
    # Gamma = -mu n [grad F + k G(eta) grad T]; both ends grounded, n = N_d:
    # Gamma L / (mu N_d) = k int G(eta(T)) dT whatever the profile, with
    # G = (5/2) F_3/2 / F_1/2 - eta (fermi-dirac) or 5/2 - ln F_1/2 (boltzmann)
    import math
    from scipy.integrate import quad
    from kpenvelope.device import _seebeck_fd
    from kpenvelope.electrons import KB, fermi_dirac_half
    L, nd = 4000.0, 2e19
    out = {}
    for te, G in (("boltzmann", lambda e: 2.5 - math.log(float(fermi_dirac_half(e)[0]))),
                  ("fermi-dirac", lambda e: float(_seebeck_fd(np.array([e]))[0][0]))):
        dev = _fd_bar(te)
        z = dev.axes[-1]
        T = np.tile(300.0 + 100.0 * z / L + 60 * np.sin(np.pi * z / L) ** 2, 3)
        st = dev.solve({}, lattice_temperature=T)
        Nc0 = dev._lat["Nc"][0] * 1e21

        def eta_of(t, dev=dev, Nc0=Nc0):
            return float(dev._inv_occ(np.array([nd / (Nc0 * (t / 300) ** 1.5)]))[0])
        assert eta_of(300.0) > 4.5                                  # degenerate
        I = Q * 1000.0 * nd * quad(lambda t: G(eta_of(t)), 300, 400, limit=200)[0] * KB \
            / (L * 1e-7) * 20e-7 * 0.1
        assert st.current["bot"] == pytest.approx(I, rel=1e-5)
        out[te] = st.current["bot"]
    assert out["fermi-dirac"] > 1.5 * out["boltzmann"]
    # at one temperature the two are the same model
    a, b = _fd_bar("boltzmann"), _fd_bar("fermi-dirac")
    T0 = np.full(a.N, 350.0)
    assert a.solve({"bot": 0.3}, lattice_temperature=T0).current["bot"] == \
        b.solve({"bot": 0.3}, lattice_temperature=T0).current["bot"]
    with pytest.raises(ValueError, match="statistics='fermi'"):
        from kpenvelope import StackLayer, algan
        from kpenvelope.device import Device, Electrode, Mobility, Surface
        Device([StackLayer(algan(0.0), 100.0, donors=1e-3)], 20.0,
               [Electrode("top", "ohmic", 0, 20)], Mobility(1000.0, "t"),
               Surface("pinned", barrier=1.0), statistics="boltzmann",
               thermoelectric="fermi-dirac")


def test_fermi_dirac_peltier_heat_at_the_contacts():
    from kpenvelope.device import _seebeck_fd
    from kpenvelope.electrons import KB
    a, b = _fd_bar("fermi-dirac", tm="mobility", heat="joule"), \
        _fd_bar("fermi-dirac", tm="mobility", heat="thermoelectric")
    sa, sb = a.solve({"bot": 0.2}), b.solve({"bot": 0.2})
    assert sb.heat.sum() == pytest.approx(sb.power, rel=1e-12)
    d = sb.heat - sa.heat
    eta = float(b._inv_occ(np.array([2e19 / (b.Nc[0] * 1e21)]))[0])
    Pi = KB * 300 * float(_seebeck_fd(np.array([eta]))[0][0])
    I = abs(sa.current["bot"])
    assert d[:, :3].sum() == pytest.approx(Pi * I, rel=1e-6)
    assert d[:, -3:].sum() == pytest.approx(-Pi * I, rel=1e-6)


def test_jacobian_with_fermi_dirac_thermal_diffusion():
    from kpenvelope import StackLayer, algan
    from kpenvelope.device import Device, Electrode, Mobility, Surface, Trap, Varshni
    layers = [StackLayer(algan(0.0), 300.0, acceptors=3e-2),
              StackLayer(algan(0.2), 30.0, donors=1e-3), StackLayer(algan(0.0), 300.0, donors=3e-2)]
    z = np.unique(np.r_[np.linspace(0, 630, 40), [300, 330]])
    el = [Electrode("top", "ohmic", 0, 40), Electrode("bot", "ohmic", 0, 40, face="bottom")]
    tr = [Trap("acceptor", 1.0, "ec", 1e16, 1e-8, 1e-8, "t", "GaN")]
    for force in ("field", "quasi-fermi"):
        dev = Device(layers, 40.0, el, Mobility(300.0, "t", saturation_velocity=1e7,
                                                temperature_exponent=1.5),
                     Surface("pinned", barrier=1.0), grid=dict(x=np.linspace(0, 40, 4), z=z),
                     hole_mass=1.0, hole_mobility=Mobility(20.0, "t", saturation_velocity=1e7),
                     statistics="fermi", temperature_model="full", traps=tr,
                     driving_force=force, thermoelectric="fermi-dirac",
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
