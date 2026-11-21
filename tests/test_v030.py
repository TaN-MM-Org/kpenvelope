"""Checks of the features new in 0.30.0: the app's new process steps, its
quantum current and its heat options."""

import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def _proc(recipe, **kw):
    from kpenvelope.gui import api
    spec = dict(width=80, height=140, cell=2, substrate="Si", substrate_top=100,
                source="test", recipe=recipe)
    spec.update(kw)
    return api.run_process({"process": spec})


# ------------------------------------------------------- process steps
def test_app_oxide_flow_is_the_library_one():
    from kpenvelope.process import Structure
    r = _proc("etch Si=1 time=40 mode=directional from=40 to=80\n"
              "oxidize material=Si oxide=O B=0.5 BA=0.05 time=2000 consumed=0.44 "
              "mode=2d viscosity=1e12 stress_volume=0.025 temperature=1273")
    st = Structure(80.0, 0.0, 140.0, 2.0, "Si", 100.0)
    st.etch({"Si": 1.0}, 40.0, mode="directional", window=(40, 80))
    st.oxidize("Si", "O", 0.5, 0.05, 2000.0, 0.44, "t", mode="2d", viscosity={"O": 1e12},
               stress_volume_nm3=0.025, temperature_K=1273.0)
    ox = st.mat == "O"
    smin = min(st.oxide_stress["sxx"][ox].min(), st.oxide_stress["szz"][ox].min()) / 1e9
    assert r["areas"]["O"] == pytest.approx(st.area("O"), rel=1e-12)
    assert r["stress_GPa"] == pytest.approx(smin, rel=1e-9)
    assert round(st.area("O")) == 2892 and round(smin, 1) == -11.1     # example 61
    # the flow needs the 2D oxidation
    from kpenvelope.gui.api import SpecError
    with pytest.raises(SpecError, match="mode=2d"):
        _proc("oxidize material=Si oxide=O B=0.5 BA=0.05 time=10 viscosity=1e12")


def test_app_simulated_implant_and_lost_dose():
    from kpenvelope import bca
    recipe = ("implant ranges=simulate ion_Z=5 ion_mass=11 energy=30000 target_Z=14 "
              "target_mass=28.09 atoms=5e22 species=acceptor dose=1e13 ions=1000")
    deep = _proc(recipe, height=600, substrate_top=550)
    rr = bca.simulate(5, 11.0, 30000.0, bca.Target((bca.Element(14, 28.09),), 5e22, "t"),
                      n_ions=1000, seed=0)
    assert f"range {rr.projected_range:.1f} nm" in deep["notes"][0]
    assert len(deep["notes"]) == 1                     # the whole dose is in the picture
    kept = 1 - rr.backscattered / rr.n_ions
    assert deep["acceptor_dose"] == pytest.approx(1e13 * kept, rel=0.01)
    # 100 nm of silicon for a 130 nm range: most ions go past the bottom
    short = _proc(recipe)
    assert "of the dose stayed" in short["notes"][-1]
    # a window keeps its share of the dose without a note
    r = _proc("implant species=donor dose=1e13 range=30 straggle=10 from=20 to=60")
    assert r["notes"] == [] and r["donor_dose"] == pytest.approx(0.5e13, rel=0.01)


def test_app_plasma_etch_is_the_library_one():
    from kpenvelope.process import EtchKinetics, Structure
    r = _proc("etch Si=1 time=30 mode=directional from=30 to=50\n"
              "plasmaetch neutrals=1e18 ions=1e16 stick_Si=0.2 yield_Si=5 atoms_Si=5e22 "
              "time=4", width=80, height=100, substrate_top=80, cell=4)
    st = Structure(80.0, 0.0, 100.0, 4.0, "Si", 80.0)
    st.etch({"Si": 1.0}, 30.0, mode="directional", window=(30, 50))
    st.plasma_etch(EtchKinetics(1e18, 1e16, 1e15, {"Si": 0.2}, {"Si": 5.0},
                                {"Si": 5e22}, "t"), 4.0)
    assert r["areas"]["Si"] == pytest.approx(st.area("Si"), rel=1e-12)
    from kpenvelope.gui.api import SpecError
    with pytest.raises(SpecError, match="atoms_X"):
        _proc("plasmaetch neutrals=1e18 ions=1e16 stick_Si=0.2 yield_Si=5 time=1")


# ------------------------------------------------------ quantum current
_SHORT = dict(contact="50", gap_sg="35", gate_length="30", gap_gd="35", mobility="1500")


def test_app_quantum_current():
    from test_v020 import _spec
    from kpenvelope.electrons import M0
    from kpenvelope.gui import api
    from kpenvelope.nitride import QE
    with pytest.raises(api.SpecError, match="600 nm"):
        api.run_quantum(_spec(transistor={}))
    spec = _spec(transistor=dict(_SHORT), quantum_transport={"gate": -1, "drain": 0.05})
    r = api.run_quantum(spec)
    # with nothing to bump into, the wave current is the ballistic limit to
    # within the tunneling through the top of the gate's hill
    assert r["scattering_time_s"] is None
    assert abs(r["negf"] / r["ballistic"] - 1) < 0.05
    assert r["drift_diffusion"] < 0.5 * r["negf"]      # capped by the top speed
    # with bumping at the rate that gives the tab's mobility (tau = mu m / q)
    spec["quantum_transport"]["scattering"] = True
    rs = api.run_quantum(spec)
    dev, info = api._transistor_device(dict(spec, transistor=dict(_SHORT, quantum=True)))
    m = info["layers"][0].material.me
    assert rs["scattering_time_s"] == pytest.approx(1500e-4 * m * M0 / QE, rel=1e-12)
    assert 0 < rs["negf"] < 0.9 * r["negf"]


# ------------------------------------------------------ heat and sources
def test_app_dense_heat_and_mobility_source():
    from test_v020 import _spec
    from kpenvelope.gui import api
    base = _spec(transistor={"heating": True, "peltier": True})
    dev, info = api._transistor_device(base)
    assert dev.heat_model == "thermoelectric" and dev.thermoelectric == "boltzmann"
    assert any("1704 cm²/Vs" in s for s in info["sources"])
    dev, _ = api._transistor_device(_spec(transistor={"heating": True, "peltier": True,
                                                          "dense_heat": True}))
    assert dev.thermoelectric == "fermi-dirac"
    # a mobility of your own, without a source, is not credited to the paper
    _, info = api._transistor_device(_spec(transistor={"mobility": "1500"}))
    assert not any("Alpert" in s for s in info["sources"])
    assert any("your own numbers" in s for s in info["sources"])
    _, info = api._transistor_device(_spec(transistor={"mobility": "1500",
                                                           "mobility_source": "my Hall data"}))
    assert any("my Hall data" in s for s in info["sources"])


# --------------------------------------- screening by all pairs of levels
def _square_well(screening="full"):
    from kpenvelope.device import column_states
    from kpenvelope.scattering import Subbands
    z = np.linspace(0, 40, 401)
    hz = z[1] - z[0]
    E, p2, w, psi = column_states(z, np.where((z > 10) & (z < 30), 0.0, 0.5),
                                  np.full(z.size, 0.2), 3, z[0] - hz, z[-1] + hz, signed=True)
    return Subbands(z, psi, E, 0.2, 9.5, sheet_cm2=1.5e13, weights=w, screening=screening)


def test_polarization_of_all_pairs():
    from kpenvelope.electrons import HBAR
    from kpenvelope.nitride import QE
    sb = _square_well()
    pr = sb.pairs()
    P0 = sb.polarization()
    # long wavelength: the density change of the n x n Hamiltonian with a
    # small potential added, at a fixed Fermi level
    Ue = np.random.default_rng(1).normal(size=sb.z.size) * 0.01
    Ukl = np.array([[np.sum(sb.w * sb.psi[:, a] * sb.psi[:, b] * Ue) for b in range(3)]
                    for a in range(3)])

    def dens(d):
        ev, vec = np.linalg.eigh(np.diag(sb.E) + d * Ukl)
        ns = sb.dos * sb.kT * np.logaddexp(0, (sb.fermi - ev) / sb.kT)
        return (sb.psi @ vec) ** 2 @ ns
    fd = (dens(1e-4) - dens(-1e-4)) / 2e-4
    lin = sum(P0[a] * QE * sb.psi[:, k] * sb.psi[:, l] * Ukl[k, l]
              for a, (k, l) in enumerate(pr))
    assert np.max(np.abs(lin - fd)) < 1e-8 * np.max(np.abs(fd))
    # the Lindhard form goes over into it, and matches the 2D integral at q
    assert np.max(np.abs(sb.polarization([1e-6])[0] / P0 - 1)) < 1e-9
    kT, F, Es, m = sb.kT * QE, sb.fermi * QE, sb.E * QE, sb.m

    def f(e):
        return 0.5 * (1 - np.tanh(0.5 * (e - F) / kT))
    q = 1.5e9
    K, TH = np.meshgrid(np.linspace(0, 3e9, 3001)[1:], (np.arange(2000) + 0.5) * np.pi / 1000,
                        indexing="ij")
    Pq = sb.polarization([q * 1e-9])[0]
    for a, (k, l) in enumerate(pr):
        ea = Es[k] + HBAR ** 2 * K ** 2 / (2 * m)
        eb = Es[l] + HBAR ** 2 * (K ** 2 + q * q + 2 * K * q * np.cos(TH)) / (2 * m)
        D = ea - eb
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(D) > 0, (f(ea) - f(eb)) / D, -f(ea) * (1 - f(ea)) / kT)
        chi = 2 / (2 * np.pi) ** 2 * np.sum(g * K) * 1e6 * (np.pi / 1000)
        assert Pq[a] == pytest.approx(chi * (1 if k == l else 2), rel=2e-5, abs=0)


def test_full_screening_against_a_real_space_solve():
    from kpenvelope.nitride import QE
    sb = _square_well()
    z, w, psi, pr = sb.z, sb.w, sb.psi, sb.pairs()
    imp = sb.impurities(z0=5.0, sheet_cm2=1e12)
    pre = (QE ** 2 / (2 * sb.eps)) ** 2 * 1e16
    rough = sb.roughness(10.0, 0.5, 0.3, 1.5)
    pz = np.array([np.interp(10.0, z, psi[:, i]) for i in range(3)])
    for qn in (0.05, 0.4, 2.0):
        q = qn * 1e9
        c = QE ** 2 / (2 * sb.eps * q)
        Pq = sb.polarization([qn])[0]
        G = np.exp(-qn * np.abs(z[:, None] - z[None, :])) * w[None, :] * 1e-9
        Pm = sum(Pq[a] * np.outer(psi[:, k] * psi[:, l] * 1e9, psi[:, k] * psi[:, l] * w)
                 for a, (k, l) in enumerate(pr))
        K = np.linalg.inv(np.eye(z.size) - c * G @ Pm)
        # the sheet of charges: U = U_ext + induced, self-consistently
        U = K @ np.exp(-qn * np.abs(z - 5.0))
        for a, (i, j) in enumerate(pr):
            Uij = np.sum(w * psi[:, i] * psi[:, j] * U)
            assert imp.W(np.array([q]), i, j)[0] == pytest.approx(pre * Uij ** 2 / q ** 2,
                                                                  rel=1e-6, abs=0)
        # the roughness: a bare perturbation with the elements V_kl
        V = {p: 0.5 * QE * pz[p[0]] * pz[p[1]] * 1e9 for p in pr}
        dn = sum(Pq[a] * psi[:, k] * psi[:, l] * 1e9 * V[(k, l)] for a, (k, l) in enumerate(pr))
        Uind = K @ (c * G @ dn)
        L = 1.5e-9
        S = np.pi * (0.3e-9) ** 2 * L * L * np.exp(-(q * L) ** 2 / 4)
        for i, j in pr:
            Vs = V[(i, j)] + np.sum(w * psi[:, i] * psi[:, j] * Uind)
            assert rough.W(np.array([q]), i, j)[0] == pytest.approx(Vs ** 2 * S, rel=1e-6,
                                                                    abs=0)
    # one level: the matrix is the single number 1 - e^2 H Pi / (2 eps q)
    one = _square_well()
    from kpenvelope.scattering import Subbands
    s1 = Subbands(one.z, one.psi[:, :1], one.E[:1], 0.2, 9.5, fermi=one.fermi, weights=one.w,
                  screening="full")
    qn = 0.3
    eps1 = 1 - QE ** 2 / (2 * s1.eps * qn * 1e9) * s1.form_factor(0, 0, qn) * \
        s1.polarization([qn])[0, 0]
    assert s1.screened([1.0], qn)[0, 0] == pytest.approx(1 / eps1, rel=1e-8)
    with pytest.raises(ValueError, match="screening"):
        Subbands(one.z, one.psi, one.E, 0.2, 9.5, fermi=0.0, weights=one.w, screening="yes")
