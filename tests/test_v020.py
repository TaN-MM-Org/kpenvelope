"""Tests for v0.20.0 device shapes: a dielectric cap, metal boxes (recessed
gates, field plates, contacts through the cap) and regions cut out of the
semiconductor."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from kpenvelope import StackLayer, algan, threshold_voltage  # noqa: E402
from kpenvelope.device import (Device, Dielectric, Electrode, Mobility,  # noqa: E402
                               Region, Surface)
from kpenvelope.electrons import EPS0  # noqa: E402


def test_shape_inputs_refused():
    with pytest.raises(ValueError):
        Dielectric("SiN", 0.0, 7.5, "r")
    with pytest.raises(ValueError):
        Dielectric("SiN", 10.0, 0.5, "r")
    with pytest.raises(ValueError):
        Region(10, 5, 0, 1)
    with pytest.raises(ValueError):
        Region(0, 5, 0, 1, eps_r=0.5)
    layers = [StackLayer(algan(0.0), 100.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    el = [Electrode("s", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20)]
    # with a cap, a top contact has to be a box through it
    with pytest.raises(ValueError, match="box"):
        Device(layers, 300.0, el, Mobility(1000.0, "r"), Surface("pinned", barrier=1.0),
               cap=[Dielectric("SiN", 50.0, 7.5, "r")])


def test_parallel_plate_through_a_cap():
    # a metal sheet on a dielectric over a very heavily doped layer: the
    # capacitance is eps0 eps_r / d per area (the screening length in the
    # semiconductor is a fraction of an angstrom here)
    layers = [StackLayer(algan(0.0), 200.0, donors=1e-1, name="n+")]
    el = [Electrode("body", "ohmic", 0, 1000, face="bottom"),
          Electrode("plate", "schottky", 0, 1000, barrier=1.0, z0=250.0, z1=250.0)]
    d = Device(layers, 1000.0, el, Mobility(300.0, "t"), Surface("pinned", barrier=1.0),
               cap=[Dielectric("SiN", 50.0, 7.5, "test")], dx_fine=100, dx_max=200,
               dz_fine=1.0, dz_max=10, fine_depth=60)
    p, Y = d.small_signal(d.solve({"plate": 0.0}), [1e3])
    C = Y[0, p.index("plate"), p.index("plate")].imag / (2 * np.pi * 1e3)
    plate = EPS0 * 7.5 / 50e-9 * 1e-6 * 1e-3          # F per mm of width
    assert C == pytest.approx(plate, rel=1e-3)


def _recessed(recess, layers, top, barrier_name="barrier"):
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
          Electrode("gate", "schottky", 200, 700, barrier=1.0,
                    z0=top - recess, z1=top),
          Electrode("drain", "ohmic", 900, 1000, implant_depth=40, implant_donors=1e20)]
    reg = [Region(200, 700, top - recess, top, 1.0, "recess", "test")]
    return Device(layers, 1000.0, el, Mobility(1800.0, "t", saturation_velocity=1e7),
                  Surface("pinned", barrier=1.0), regions=reg, dx_fine=10, dx_max=50,
                  dz_fine=1.0, dz_max=40, fine_depth=40)


def _one_percent_gate(d):
    """Gate voltage where the current at 0.1 V drain falls to 1 % of its
    value at 0 V gate (log interpolation between 0.25 V steps)."""
    st = d.solve({"drain": 0.1, "gate": 0.0})
    I0 = prev = st.current["drain"]
    vprev = 0.0
    for v in np.arange(-0.25, -12.0, -0.25):
        st = d.solve({"drain": 0.1, "gate": v}, guess=st)
        I = st.current["drain"]
        if I < 0.01 * I0:
            f = np.log(0.01 * I0 / prev) / np.log(I / prev)
            return vprev + f * (v - vprev)
        prev, vprev = I, v
    raise AssertionError("no threshold")


def test_recessed_gate_threshold_matches_1d_thinned_barrier():
    # a long gate in a 10 nm recess: the current switches off where the
    # 1D solver puts the threshold of the stack with a 10 nm barrier
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    thin = [StackLayer(algan(0.0), 200.0, name="buffer"),
            StackLayer(algan(0.25), 10.0, name="barrier")]
    vt = threshold_voltage(thin, 1.0)
    assert _one_percent_gate(_recessed(10.0, layers, 220.0)) == pytest.approx(vt, abs=0.03)


def test_recess_through_a_cap_layer_refers_the_barrier_to_the_layer_below():
    # etching away a 4 nm GaN cap leaves the gate metal on the AlGaN: its
    # barrier is to the AlGaN, as for the 1D stack without the cap
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier"),
              StackLayer(algan(0.0), 4.0, name="cap")]
    bare = layers[:2]
    vt = threshold_voltage(bare, 1.0)
    assert _one_percent_gate(_recessed(4.0, layers, 224.0)) == pytest.approx(vt, abs=0.03)


def test_field_plate_lowers_the_field_at_the_gate_edge():
    # a gate-connected plate over the cap moves the drop from the gate
    # edge to the plate edge (the classic result); the device is pinched
    # off and held off by a 5 % AlGaN back barrier
    layers = [StackLayer(algan(0.05), 300.0, name="back barrier"),
              StackLayer(algan(0.0), 60.0, name="channel"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    top = 380.0
    sin = Dielectric("SiN", 100.0, 7.5, "test")
    out = {}
    for plate in (False, True):
        el = [Electrode("source", "ohmic", 0, 300, implant_depth=40, implant_donors=1e20,
                        z0=top, z1=top + 100),
              Electrode("gate", "schottky", 800, 1000, barrier=1.0, z0=top, z1=top + 100),
              Electrode("drain", "ohmic", 2200, 2500, implant_depth=40, implant_donors=1e20,
                        z0=top, z1=top + 100)]
        b = {"gate": -6.0}
        if plate:
            el.append(Electrode("plate", "schottky", 1010, 1500, barrier=1.0,
                                z0=top + 100, z1=top + 100))
            b["plate"] = -6.0
        d = Device(layers, 2500.0, el, Mobility(1704.0, "t", saturation_velocity=1e7),
                   Surface("states", fixed_charge=None), cap=[sin], dx_fine=10, dx_max=100,
                   dz_fine=1.0, dz_max=50)
        st = d.solve(dict(b, drain=0.0))
        _, I, S = d.sweep("drain", [5.0, 10.0, 20.0], bias=b, guess=st)
        s = S[-1]
        # pinched off: the currents are at the solver's floor (about 1e-11
        # A/mm, as without the cap), far below any measurable leakage
        assert max(abs(v) for v in s.current.values()) < 1e-10
        Ex, _ = d.field(s)
        x, z = d.axes[0], d.axes[-1]
        row = np.abs(Ex[:, np.argmin(abs(z - 358.0))])
        out[plate] = (row[np.argmin(abs(x - 1030.0))], row[np.argmin(abs(x - 1250.0))],
                      row[np.argmin(abs(x - 1510.0))])
    # much lower at the gate edge and under the plate, higher at its edge
    assert out[True][0] < 0.5 * out[False][0]
    assert out[True][1] < 0.5 * out[False][1]
    assert out[True][2] > 1.2 * out[False][2]


def test_contacts_through_a_cap_conserve_current():
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    top = 220.0
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20,
                    z0=top, z1=top + 50),
          Electrode("gate", "schottky", 300, 400, barrier=1.0, z0=top, z1=top + 50),
          Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20,
                    z0=top, z1=top + 50)]
    d = Device(layers, 700.0, el, Mobility(1800.0, "t", saturation_velocity=1e7),
               Surface("states", fixed_charge=None), cap=[Dielectric("SiN", 50.0, 7.5, "t")],
               dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40, fine_depth=40)
    _, I, S = d.sweep("drain", [0.5, 2.0, 5.0], bias={"gate": 0.0})
    assert np.all(np.diff(I["drain"]) > 0)
    for s in S:
        assert abs(sum(s.current.values())) < 1e-9 * abs(s.current["drain"])
        assert s.heat.sum() == pytest.approx(s.power, rel=1e-8)
        # no carriers inside the dielectric
        ins = d.insulator_nodes
        assert np.all(s.n.ravel()[ins] == 0.0)


def test_gate_tunneling_under_a_recessed_gate_matches_1d():
    # a wide gate in a 10 nm recess of a 20 nm barrier: in the middle the
    # current density is the 1D one of the stack with a 10 nm barrier
    from kpenvelope import gate_leakage
    layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    thin = [StackLayer(algan(0.0), 300.0, name="buffer"),
            StackLayer(algan(0.25), 10.0, name="barrier")]
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
          Electrode("gate", "schottky", 400, 2400, barrier=1.0, z0=310.0, z1=320.0),
          Electrode("drain", "ohmic", 2700, 2800, implant_depth=30, implant_donors=1e20)]
    dev = Device(layers, 2800.0, el, Mobility(1800.0, "test"), Surface("pinned", barrier=1.0),
                 regions=[Region(400, 2400, 310.0, 320.0)],
                 dx_fine=10, dx_max=100, dz_fine=0.5, dz_max=40, fine_depth=40)
    st = dev.solve({"gate": -3.0})
    I, Jc = dev.gate_tunneling(st)
    nz = st.axes[-1].size
    xs = np.unique(st.axes[0][dev.enodes["gate"] // nz])
    k = int(np.argmin(np.abs(xs - 1400.0)))
    _, J1, _ = gate_leakage(thin, 1.0, [-3.0], dz=0.5)
    assert Jc[k] == pytest.approx(J1[0], rel=5e-3)
    assert I < 0


def test_gate_on_a_dielectric_refuses_tunneling_and_hot_electrons_run():
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    top = 220.0
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20,
                    z0=top, z1=top + 20),
          Electrode("gate", "schottky", 300, 400, barrier=1.0, z0=top + 20, z1=top + 20),
          Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20,
                    z0=top, z1=top + 20)]
    d = Device(layers, 700.0, el, Mobility(1800.0, "t", saturation_velocity=1e7),
               Surface("states", fixed_charge=None), cap=[Dielectric("SiN", 20.0, 7.5, "t")],
               dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40, fine_depth=40)
    s = d.solve({"gate": 0.0, "drain": 2.0})
    with pytest.raises(ValueError, match="dielectric"):
        d.gate_tunneling(s)
    Te = d.electron_temperature(s, 0.2e-12)
    semi = d.node_semi.reshape(d.shape)
    assert np.all(np.isfinite(Te[semi])) and Te[semi].min() > 299.0


def test_gate_in_pieces_equals_two_electrodes_at_one_voltage():
    # an L-shaped gate (the gate box plus a plate on the cap, one metal)
    # gives the same solution as two electrodes held at the same voltage,
    # and its current is the sum of theirs
    layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    top = 220.0

    def build(pieces):
        el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20,
                        z0=top, z1=top + 50),
              Electrode("gate", "schottky", 300, 400, barrier=1.0, z0=top, z1=top + 50),
              Electrode("gate" if pieces else "plate", "schottky", 410, 520, barrier=1.0,
                        z0=top + 50, z1=top + 50),
              Electrode("drain", "ohmic", 700, 800, implant_depth=40, implant_donors=1e20,
                        z0=top, z1=top + 50)]
        return Device(layers, 800.0, el, Mobility(1800.0, "t", saturation_velocity=1e7),
                      Surface("states", fixed_charge=None),
                      cap=[Dielectric("SiN", 50.0, 7.5, "t")],
                      dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40, fine_depth=40)
    one, two = build(True), build(False)
    a = one.solve({"gate": -1.0, "drain": 3.0})
    b = two.solve({"gate": -1.0, "plate": -1.0, "drain": 3.0})
    assert np.max(np.abs(a.psi - b.psi)) < 1e-9
    assert a.current["drain"] == pytest.approx(b.current["drain"], rel=1e-9)
    assert a.charge["gate"] == pytest.approx(b.charge["gate"] + b.charge["plate"], rel=1e-9)
    with pytest.raises(ValueError, match="pieces"):
        Device(layers, 800.0, [Electrode("s", "ohmic", 0, 100, implant_depth=40,
                                         implant_donors=1e20),
                               Electrode("g", "schottky", 300, 400, barrier=1.0, z0=210.0,
                                         z1=220.0),
                               Electrode("g", "schottky", 500, 600, barrier=0.8, z0=210.0,
                                         z1=220.0)],
               Mobility(1800.0, "t"), Surface("pinned", barrier=1.0))


# ------------------------------------------------------------ the app
def _spec(**extra):
    spec = {"parameter_set": "pbesol-mahim2026", "temperature": 300,
            "layers": [{"material": "GaN", "x": 0, "thickness": 300, "relaxation": 0,
                        "doping": "none"},
                       {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0,
                        "doping": "none"}],
            "surface_barrier": 1.0, "device": {"barrier": 1.0},
            "line_charge": {"mode": "measured"}}
    spec.update(extra)
    return spec


_FET = {"gate_length": 200, "gap_sg": 300, "gap_gd": 500, "contact": 200,
        "sweep": "drain", "from": 0, "to": 6, "step": 3, "fixed": 0}


def test_app_passivated_surface_keeps_the_bare_surface_charge():
    from kpenvelope.gui.api import _transistor_device
    bare, info = _transistor_device(_spec(transistor=dict(_FET)))
    pas, _ = _transistor_device(_spec(transistor=dict(_FET, passivation=100)))
    a = bare.solve({"gate": 0.0, "drain": 0.0})
    b = pas.solve({"gate": 0.0, "drain": 0.0})
    # electron sheet in the access region between source and gate
    for d, s in ((bare, a), (pas, b)):
        x, z = d.axes
        k = int(np.argmin(np.abs(x - 350.0)))
        m = z <= d.top
        nz_, zz = s.n[k, m], z[m]
        s.ns = float(np.sum(0.5 * (nz_[1:] + nz_[:-1]) * np.diff(zz))) * 1e-7
    assert b.ns == pytest.approx(a.ns, rel=0.02)
    assert b.ns == pytest.approx(info["ns"], rel=0.02)


def test_app_transistor_with_plate_leakage_and_hot_electrons():
    from kpenvelope.gui.api import handle
    r = handle("transistor", _spec(transistor=dict(_FET, passivation=100, plate=200,
                                                   contact_res=0.3, tau_e=0.47)))
    assert "error" not in r, r.get("error")
    for k in range(3):
        assert abs(r["drain"][k] + r["gate"][k] + r["source"][k]) < 1e-8
    assert r["plate_x"] == [700.0, 900.0]
    assert r["leak"]["current"] < 0              # electrons enter from the gate
    assert r["hot"]["peak"] > 300.0
    assert any("Robertson" in x for x in r["sources"])
    bad = handle("transistor", _spec(transistor=dict(_FET, plate=200)))
    assert "passivation" in bad["error"]


def test_app_speed_matches_its_own_quasi_static_numbers():
    from kpenvelope.gui.api import handle
    r = handle("rf", _spec(transistor=dict(_FET, contact_res=0.3),
                           rf={"gate": -2, "drain": 6, "gate_res": 0.5}))
    assert "error" not in r, r.get("error")
    # at 100 MHz f |h21| is gm / (2 pi C_gate) to within the tiny Re(Y11)
    assert r["fT_extrapolated"] == pytest.approx(r["gm"] / (2 * np.pi * r["cg"]), rel=1e-3)
    assert r["fT"] > 1e9 and r["fmax"] > r["fT"]
    r0 = handle("rf", _spec(transistor=dict(_FET), rf={"gate": -2, "drain": 6, "gate_res": 0}))
    assert r0["fmax"] is None


def test_app_breakdown_and_pulse_run():
    from kpenvelope.gui.api import handle
    r = handle("breakdown", _spec(transistor=dict(_FET),
                                  breakdown={"gate": -10, "to": 20, "step": 10}))
    assert "error" not in r, r.get("error")
    assert len(r["voltages"]) == len(r["integral"]) >= 1
    assert np.all(np.diff(r["integral"]) > 0)
    assert any("GaN layers only" in x for x in r["sources"])
    p = handle("pulse", _spec(transistor=dict(_FET, substrate="SiC", substrate_um=50,
                                              pitch_um=20),
                              pulse={"gate": 0, "drain": 6, "length": 1}))
    assert "error" not in p, p.get("error")
    assert np.all(np.diff(p["tmax"]) >= -1e-6) and p["tmax"][-1] <= p["steady_t"] + 0.5
    assert p["drain"][-1] < p["drain"][0]


def _jacobian_error(dev, st, sparse_rows=False):
    x, b, T = st._x, st.bias, np.full(dev.N, 300.0)
    F0 = dev._residual(x, b, T)
    J = dev._jacobian(x, F0, b, T, None, None).toarray()
    scale = np.abs(J).max(axis=1) + 1e-300
    keep = np.ones(x.size, bool)
    if sparse_rows:
        # impact ionization uses a smoothed |Gamma|, whose exact slope a
        # central difference cannot reproduce where a flux crosses zero
        # within the difference step. That happens only for carriers far
        # too sparse to matter, so their rows are left out (more than 30
        # orders of magnitude below the density floor), and every other
        # row is measured against at least 1e-12 of the largest entry
        scale = scale + 1e-12 * np.abs(J).max()
        _, n, p, _, _ = dev._carriers(*dev._unpack(x)[:3])
        keep[np.flatnonzero(n < 1e-30 * dev.floor) * dev.nv + 1] = False
        keep[np.flatnonzero(p < 1e-30 * dev.floor) * dev.nv + 2] = False
    worst = 0.0
    for c in range(x.size):
        h = 1e-7 * max(1.0, abs(x[c]))
        xp, xm = x.copy(), x.copy()
        xp[c] += h
        xm[c] -= h
        cd = (dev._residual(xp, b, T) - dev._residual(xm, b, T)) / (2 * h)
        worst = max(worst, float(np.max((np.abs(J[:, c] - cd) / scale)[keep])))
    return worst


def test_jacobian_with_a_cap_boxes_and_a_region():
    layers = [StackLayer(algan(0.0), 60.0, name="buffer"),
              StackLayer(algan(0.25), 20.0, name="barrier")]
    top = 80.0
    el = [Electrode("source", "ohmic", 0, 40, implant_depth=30, implant_donors=1e20,
                    z0=top, z1=top + 20),
          Electrode("gate", "schottky", 80, 120, barrier=1.0, z0=top - 5, z1=top + 20),
          Electrode("drain", "ohmic", 160, 200, implant_depth=30, implant_donors=1e20,
                    z0=top, z1=top + 20)]
    dev = Device(layers, 200.0, el, Mobility(1500.0, "test", saturation_velocity=1e7),
                 Surface("states", fixed_charge=None),
                 cap=[Dielectric("SiN", 20.0, 7.0, "test")],
                 regions=[Region(80, 120, top - 5, top)],
                 dx_fine=20, dx_max=40, dz_fine=5.0, dz_max=20, fine_depth=20)
    st = dev.solve({"drain": 1.0, "gate": -0.5})
    assert _jacobian_error(dev, st) < 1e-5


def test_jacobian_with_impact_ionization():
    from kpenvelope import gan
    from kpenvelope.device import IMPACT_GAN
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 20.0, acceptors=1e-2, name="p"), StackLayer(g, 100.0, name="i"),
              StackLayer(g, 20.0, donors=1e-2, name="n")]
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    dev = Device(layers, 20.0, el, Mobility(300.0, "test"), Surface("pinned", barrier=1.0),
                 hole_mass=1.0, hole_mobility=Mobility(20.0, "test"), statistics="boltzmann",
                 ionization=IMPACT_GAN,
                 generation=lambda xx, zz: 1e24 if 22 <= zz <= 30 else 0.0,
                 grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 140.0, 36)))
    st = dev.solve({"top": 40.0})
    # strong enough ionization to matter in the rows
    from kpenvelope.device import IMPACT_GAN as ion
    assert dev.ionization_integral(st, ion)["I_n"] > 0.1
    assert _jacobian_error(dev, st, sparse_rows=True) < 1e-5
