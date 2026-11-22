"""Checks of the features new in 0.32.0: scattering in coupled mode space,
and NEGF on meshes."""

import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, __file__.rsplit("/", 1)[0])


# ------------------------------------------- probes in coupled mode space
def test_slab_probes_against_the_chain():
    from kpenvelope.negf import Chain, Slab, chain_transport, slab_transport
    kT, a, m = 0.02585, 0.5, 0.2
    x = np.arange(60) * a
    u = 0.05 * np.exp(-((x - 15) / 4) ** 2)
    z = np.linspace(0.5, 9.5, 19)
    # a potential u(x) + 0 across: every column has the same modes, so one
    # mode with its probes is a chain with its probes
    slab = Slab(u[:, None] + np.zeros(z.size)[None, :], m, a, z, 0.0, 10.0, n_modes=1,
                eta=0.01)
    e0 = slab.Ez[0, 0] - u[0]
    ch = Chain(u + e0, m, a, eta=0.01)
    E = np.linspace(e0 - 0.02, 0.4, 900)
    rs = slab_transport(slab, 0.05, 0.03, kT, energies=E)
    rc = chain_transport(ch, 0.05, 0.03, kT, energies=E)
    assert rs["I"] == pytest.approx(rc["I"], rel=1e-6)
    assert rs["I_source"] == pytest.approx(rs["I"], rel=1e-9)        # no current lost
    assert np.max(np.abs((rs["n"] * slab.w[None, :]).sum(1) / rc["n"] - 1)) < 1e-6
    assert np.max(np.abs(rs["probe_F"] - rc["probe_F"])) < 1e-8
    # two modes that mix (a potential that tilts across the slab along x):
    # the source and the drain still carry the same current
    U = (u[:, None] + 0.02 * (z[None, :] - 5.0) / 5.0 * np.tanh((x[:, None] - 15) / 5)
         + np.zeros(z.size)[None, :])
    s2 = Slab(U, m, a, z, 0.0, 10.0, n_modes=2, eta=0.01)
    r2 = slab_transport(s2, 0.05, 0.03, kT)
    assert r2["I_source"] == pytest.approx(r2["I"], rel=1e-8) and r2["I"] > 0


def test_device_negf_with_mixing_and_scattering():
    from test_v026 import _bar
    dev = _bar(60, 1000.0)
    g = dev.solve({"drain": 0.01})
    a = dev.negf({"drain": 0.01}, guess=g, dx=1.0, scattering_time=1e-13, coupled_modes=2)
    b = dev.negf({"drain": 0.01}, guess=g, dx=1.0, scattering_time=1e-13)
    assert a.current["drain"] == -a.current["source"]
    # in this uniform channel the modes hardly mix: the two pictures of
    # the probes agree on the resistance (measured 0.6 percent)
    assert a.current["drain"] == pytest.approx(b.current["drain"], rel=0.02)
    assert a.negf["probe_levels"][0].size > 10


# ------------------------------------------------------- NEGF on meshes
def test_negf_on_a_mesh_equals_the_tensor_grid():
    from kpenvelope import StackLayer
    from kpenvelope.device import Device, Electrode, Mobility, QuantumChannel, Surface
    from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_grid
    from test_v026 import _region, _well_layers
    g, bar = _well_layers()
    x = np.linspace(0, 300, 16)
    z = np.unique(np.r_[np.linspace(0, 90, 10), np.linspace(90, 130, 41),
                        np.linspace(130, 200, 8)])
    mob = Mobility(500.0, "t")
    el = [Electrode("source", "ohmic", 0, 60),
          Electrode("gate", "schottky", 120, 180, barrier=0.8),
          Electrode("drain", "ohmic", 240, 300)]
    qc = QuantumChannel((100.5, 119.5), x_range=(80, 220), line_spacing=20.0)
    layers = [StackLayer(bar, 100.0, donors=1e-4), StackLayer(g, 20.0, donors=1e-3),
              StackLayer(bar, 80.0, donors=1e-4)]
    T = Device(layers, 300.0, el, mob, Surface("states", fixed_charge=0.0),
               grid=dict(x=x, z=z), quantum=qc)
    mesh = mesh_from_grid(x, z, lambda xx, zz: _region(zz))
    P = mesh.points
    top = np.isclose(P[:, 1], 200.0)
    con = {"source": np.flatnonzero(top & (P[:, 0] <= 60 + 1e-9)),
           "gate": np.flatnonzero(top & (P[:, 0] >= 120 - 1e-9) & (P[:, 0] <= 180 + 1e-9)),
           "drain": np.flatnonzero(top & (P[:, 0] >= 240 - 1e-9))}
    R = {k: MeshRegion(material=m, mobility=mob, donors=d)
         for k, m, d in (("b1", bar, 1e-4), ("w", g, 1e-3), ("b2", bar, 1e-4))}
    M = MeshDevice(mesh, R, el, con, quantum=qc)
    b = {"drain": 0.05, "gate": -0.3}
    for kw in (dict(scattering_time=1e-13), dict(coupled_modes=2)):
        a = T.negf(b, guess=T.solve(b), dx=2.0, **kw)
        c = M.negf(b, guess=M.solve(b), dx=2.0, **kw)
        # (measured 1e-15 with probes and 1e-7 with coupled modes, whose
        # self-consistent loops stop at their tolerance)
        assert c.negf["current"] == pytest.approx(a.negf["current"], rel=1e-6)
        assert np.max(np.abs(a.psi.ravel() - c.psi)) < 1e-6
