"""Checks of the features new in 0.29.0."""

import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])


# --------------------------------------------- scattering between subbands
def _well(z, U, n):
    from kpenvelope.device import column_states
    hz = z[1] - z[0]
    return column_states(z, U, np.full(z.size, 0.2), n, z[0] - hz, z[-1] + hz, signed=True)


def test_column_states_signed():
    z = np.linspace(0, 40, 401)
    E, p2, w, psi = _well(z, np.where((z > 10) & (z < 20), 0.0, 0.4), 3)
    assert np.allclose(psi ** 2, p2, rtol=0, atol=1e-14 * p2.max())
    assert np.max(np.abs(psi.T @ (w[:, None] * psi) - np.eye(3))) < 1e-12


def test_one_subband_is_the_single_subband_model():
    from kpenvelope.scattering import Subband, Subbands
    z = np.linspace(0, 40, 801)
    E, p2, w, psi = _well(z, np.where((z > 10) & (z < 20), 0.0, 0.4), 1)
    s1 = Subband(z, p2[:, 0], 0.2, 9.5, fermi=0.02, weights=w)
    sm = Subbands(z, psi, E, 0.2, 9.5, fermi=E[0] + 0.02, weights=w)
    for name, rel in (("impurities", 1e-4), ("roughness", 1e-4), ("acoustic", 1e-4),
                      ("polar_optical", 1e-9)):
        args = dict(impurities=dict(z0=5.0, sheet_cm2=1e12),
                    roughness=dict(z_interface=10.0, step_eV=0.4, rms_nm=0.3,
                                   correlation_nm=1.5),
                    acoustic=dict(deformation_eV=8.3, density_kg_m3=6150,
                                  sound_velocity_m_s=6560),
                    polar_optical=dict(phonon_eV=0.092, eps_static=9.5, eps_high=5.35))[name]
        a, b = getattr(s1, name)(**args), getattr(sm, name)(**args)
        assert sm.mobility(b)["mobility_cm2Vs"] == pytest.approx(
            s1.mobility(a)["mobility_cm2Vs"], rel=rel)
        if name != "polar_optical":
            la, lb = s1.lifetimes(a), sm.lifetimes(b)
            assert lb["transport_s"][0] == pytest.approx(la["transport_s"], rel=1e-9)
            assert lb["quantum_s"][0] == pytest.approx(la["quantum_s"], rel=1e-9)


def test_subbands_that_do_not_overlap_do_not_mix():
    # two wells far apart, one subband in each (envelopes with no common
    # node): the gas is two independent gases, the mobility their
    # density-weighted mean, each subband's that of the single-subband model
    from kpenvelope.scattering import Subband, Subbands
    z = np.linspace(0, 80, 1601)
    hz = z[1] - z[0]
    U = np.where(((z > 10) & (z < 20)) | ((z > 55) & (z < 67)), 0.0, 0.4)
    psi = np.zeros((z.size, 2))
    Es = []
    for k, msk in enumerate((z < 40, z >= 40)):
        e, _, _, ps = _well(z[msk], U[msk], 1)
        psi[msk, k] = ps[:, 0]
        Es.append(e[0])
    w = np.zeros(z.size)
    w[:-1] += 0.5 * hz
    w[1:] += 0.5 * hz
    psi /= np.sqrt((w[:, None] * psi ** 2).sum(0))
    Es = np.array(Es)
    F = Es[0] + 0.03
    sm = Subbands(z, psi, Es, 0.2, 9.5, fermi=F, weights=w, screening=False)
    r = sm.mobility(sm.impurities(z0=40.0, sheet_cm2=1e12), sm.roughness(55.0, 0.4, 0.3, 1.5),
                    sm.polar_optical(0.092, 9.5, 5.35))
    mus = []
    for k in range(2):
        s = Subband(z, psi[:, k] ** 2, 0.2, 9.5, fermi=F - Es[k], weights=w, screening=False)
        mus.append(s.mobility(s.impurities(z0=40.0, sheet_cm2=1e12),
                              s.roughness(55.0, 0.4, 0.3, 1.5),
                              s.polar_optical(0.092, 9.5, 5.35))["mobility_cm2Vs"])
    mus = np.array(mus)
    assert np.allclose(r["subbands"], mus, rtol=2e-4)
    assert r["mobility_cm2Vs"] == pytest.approx((mus * sm.sheets).sum() / sm.sheets.sum(),
                                                rel=2e-4)


def _three(T, sheet):
    from kpenvelope.scattering import Subbands
    z = np.linspace(0, 40, 801)
    U = np.where((z > 5) & (z < 30), 0.0, 0.5) + 0.004 * (z - 17.5)
    E, p2, w, psi = _well(z, U, 3)
    return Subbands(z, psi, E, 0.2, 9.5, sheet_cm2=sheet, temperature_K=T, weights=w)


def test_scattering_between_subbands_against_independent_solutions():
    from scipy.integrate import quad
    from kpenvelope.electrons import HBAR
    from kpenvelope.nitride import QE
    for T, sheet in ((300.0, 5e12), (77.0, 8e12)):
        sb = _three(T, sheet)
        E = sb.E
        assert sb.sheets.sum() * 1e-4 == pytest.approx(sheet, rel=1e-9)
        assert np.all(sb.sheets > 1e11)                    # three subbands in use
        # acoustic phonons scatter the same way in every direction: the
        # exact times are 1 / sum_j nu_ij, nu_ij from int psi_i^2 psi_j^2
        r = sb.mobility(sb.acoustic(8.3, 6150, 6560))
        c = sb.m * (8.3 * QE) ** 2 * sb.kT * QE * 1e9 / (HBAR ** 3 * 6150 * 6560 ** 2)
        O = np.array([[sb.overlap(i, j) for j in range(3)] for i in range(3)])
        for i in range(3):
            def f(e, i=i):
                tau = 1.0 / sum(c * O[i, j] for j in range(3) if e > E[j])
                return (tau * (e - E[i]) * 0.25 / sb.kT
                        / math.cosh(0.5 * (e - sb.fermi) / sb.kT) ** 2)
            pts = [float(x) for x in E if x > E[i]]
            val = quad(f, E[i], max(sb.fermi, E[-1]) + 40 * sb.kT, points=pts or None,
                       limit=400, epsrel=1e-11)[0]
            mu = QE * val / sb.m / (sb.kT * math.log1p(math.exp((sb.fermi - E[i]) / sb.kT))) * 1e4
            assert r["subbands"][i] == pytest.approx(mu, rel=1e-4)
        # Coulomb, roughness and phonons together: the cell solution against
        # the times of the exact solution at each energy (Siggia and Kwok)
        # integrated by Gauss quadrature
        mechs = [sb.impurities(z0=3.0, sheet_cm2=5e11), sb.roughness(5.0, 0.5, 0.3, 1.5),
                 sb.acoustic(8.3, 6150, 6560)]
        r = sb.mobility(*mechs)
        xg, wg = np.polynomial.legendre.leggauss(60)
        num = np.zeros(3)
        segs = [E[0], E[1], E[2], max(sb.fermi, E[2]) + 40 * sb.kT]
        for a, b in zip(segs[:-1], segs[1:]):
            u = 0.5 * (xg + 1)
            for e, we in zip(a + (b - a) * u ** 2, (b - a) * 2 * u * 0.5 * wg):
                L = sb.lifetimes(*mechs, energy=e)["transport_s"]
                for i in range(3):
                    if L[i] is not None:
                        num[i] += we * L[i] * (e - E[i]) * 0.25 / sb.kT \
                            / math.cosh(0.5 * (e - sb.fermi) / sb.kT) ** 2
        nf = sb.kT * np.logaddexp(0, (sb.fermi - E) / sb.kT)
        assert np.allclose(r["subbands"], QE * num / sb.m / nf * 1e4, rtol=1e-5)


def test_polar_optical_between_subbands_keeps_detailed_balance():
    sb = _three(300.0, 5e12)
    r = sb.mobility(sb.impurities(z0=3.0, sheet_cm2=5e11), sb.polar_optical(0.092, 9.5, 5.35))
    M, Em = sb._last_system
    M = M.toarray()
    f = 0.5 * (1 - np.tanh(0.5 * (Em - sb.fermi) / sb.kT))
    S = M * (f * (1 - f))[None, :]
    assert np.max(np.abs(S - S.T)) < 1e-10 * np.max(np.abs(S))
    # the energy cells: steps of hbar w / 32 and / 64 agree
    a = sb.mobility(sb.polar_optical(0.092, 9.5, 5.35), n_energy=32)["mobility_cm2Vs"]
    b = sb.mobility(sb.polar_optical(0.092, 9.5, 5.35), n_energy=64)["mobility_cm2Vs"]
    assert a == pytest.approx(b, rel=1e-4)
    assert r["mobility_cm2Vs"] < min(r["each"])
    with pytest.raises(ValueError, match="orthonormal"):
        from kpenvelope.scattering import Subbands
        Subbands(sb.z, 2 * sb.psi, sb.E, 0.2, 9.5, fermi=0.0, weights=sb.w)


# ------------------------------------------------------- the app
def test_app_mobility_of_the_electron_layer():
    from kpenvelope.device import column_states
    from kpenvelope.electrons import KB
    from kpenvelope.gui.api import _charges, _window, build_layers, handle
    from kpenvelope.electrons import solve_electrons
    from kpenvelope.phonons import GAN_BARKER1973 as ph
    from kpenvelope.scattering import Subbands
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [{"material": "GaN", "x": 0, "thickness": 1000, "relaxation": 0,
                        "doping": "n", "doping_cm3": 1e16},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0}],
            "surface_barrier": 1.0}
    r = handle("mobility", dict(spec, scattering={"polar": True}))
    assert "error" not in r, r.get("error")
    # the same calculation written out with the library
    layers = build_layers(spec)
    ic, _ = _charges(layers, spec)
    win = _window(layers, spec)
    e = solve_electrons(layers, phi_B=1.0, temperature_K=300.0, interfaces=ic, dz=0.2,
                        quantum_window=win, classical_outside=True)
    inw = (e.z >= win[0] - 1e-9) & (e.z <= win[1] + 1e-9)
    z = e.z[inw]
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    lay = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, 1)
    mass = np.array([layers[i].material.me for i in lay])
    hz = z[1] - z[0]
    E, p2, w, psi = column_states(z, e.ec[inw], mass, 8, z[0] - hz, z[-1] + hz,
                                  e_max=12 * KB * 300.0, signed=True)
    occ = np.logaddexp(0.0, -E / (KB * 300.0))
    keep = np.flatnonzero(occ >= 0.005 * occ.sum())[:5]
    sb = Subbands(z, psi[:, keep], E[keep], layers[0].material.me, layers[0].material.eps_r,
                  fermi=0.0, weights=w)
    mu = sb.mobility(sb.polar_optical(ph.w_LO_perp / 1000, ph.eps0_perp,
                                      ph.eps_inf_perp))["mobility_cm2Vs"]
    assert r["mobility_cm2Vs"] == pytest.approx(mu, rel=1e-12)
    assert r["sheet_all_cm2"] == pytest.approx(e.sheet_quantum_cm2, rel=1e-12)
    assert 0.9 < r["sheet_cm2"] / r["sheet_all_cm2"] < 1.0
    r2 = handle("mobility", dict(spec, scattering={
        "polar": True, "donors": True,
        "roughness": {"rms_nm": 0.3, "corr_nm": 1.5, "source": "test"}}))
    assert r2["mobility_cm2Vs"] < min(x["mobility_cm2Vs"] for x in r2["each"])
    assert "come from" in handle("mobility", dict(spec, scattering={
        "roughness": {"rms_nm": 0.3, "corr_nm": 1.5}}))["error"]
    assert "Tick" in handle("mobility", dict(spec, scattering={}))["error"]
