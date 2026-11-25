"""Checks of the features new in 0.38.0: the local band-to-band rate inside
Newton's method, so that tunneling and self-heating are one heated solve;
masks pushed sideways by the oxide that flows into their sides."""

import numpy as np
import pytest

pytest.importorskip("scipy")

Q = 1.602176634e-19


def _junction(tunneling=True, generation=None):
    # the GaN p+/n+ junction of example 39, with a mobility that falls with
    # temperature so that the heat changes the current
    from kpenvelope import StackLayer, gan
    from kpenvelope.device import BandToBand, Device, Electrode, Mobility, Surface
    g = gan(polarization="hse-dreyer2016")
    layers = [StackLayer(g, 40.0, acceptors=5e-2, name="p+"),
              StackLayer(g, 40.0, donors=5e-2, name="n+")]
    el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
    return Device(layers, 20.0, el, Mobility(300.0, "t", temperature_exponent=1.5),
                  Surface("pinned", barrier=1.0), hole_mass=1.0,
                  hole_mobility=Mobility(20.0, "t"), statistics="boltzmann",
                  grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 80, 401)),
                  band_to_band=BandToBand("t") if tunneling else None,
                  generation=generation)


def test_local_band_to_band_derivatives():
    import scipy.sparse as sp
    dev = _junction()
    st = dev.solve({"top": 10.0})
    dev._bt_live = True
    try:
        dev._set_bt_spread({"top": 10.0})
        psi = st._x.reshape(dev.N, dev.nv)[:, 0].copy()
        r, c, v = dev._band_to_band_jacobian(psi, dev.nv)
        J = sp.coo_matrix((v, (r, c)), shape=(dev.N * dev.nv,) * 2).tocsr()
        G0 = dev._band_to_band_nodes(psi, dev._bt_dv)
        worst = 0.0
        for j in list(np.argsort(-G0)[:4]) + [dev.N // 2, 5]:
            for jj in (j - 1, j, j + 1):
                h = 1e-6
                up, dn = psi.copy(), psi.copy()
                up[jj] += h
                dn[jj] -= h
                d = -(dev._band_to_band_nodes(up, dev._bt_dv)
                      - dev._band_to_band_nodes(dn, dev._bt_dv)) / (2 * h)
                for row in (1, 2):
                    a = np.asarray(J[np.arange(dev.N) * dev.nv + row, jj * dev.nv]
                                   .todense()).ravel()
                    worst = max(worst, np.max(np.abs(a - d)) / max(np.max(np.abs(d)), 1e-300))
        assert worst < 1e-6                                                  # 3e-10
    finally:
        dev._bt_live = False


def test_local_band_to_band_and_heat_in_one_solve():
    from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel
    th = ThermalModel(ThermalMaterial("d", 1.0, 0.0, "t"),
                      substrate=(ThermalLayer("s", 1e5, ThermalMaterial("s", 1.0, 0.0, "t")),))
    dev = _junction()
    count = [0]
    newton = dev._newton

    def counted(*a, **k):
        r = newton(*a, **k)
        count[0] += r[2]
        return r
    dev._newton = counted
    cold = dev.solve({"top": 5.0})
    count[0] = 0
    st = dev.solve({"top": 5.0}, thermal=th)
    one = count[0]
    assert st.temperature.max() > 330
    # all the current is the tunneling's: q times the pairs it makes (to
    # the Newton tolerance, as without heat)
    assert st.current["top"] == pytest.approx(st.band_to_band_pairs * Q * 1e6,
                                              rel=1e-4)                     # 4.6e-6
    # a fixed point: solving at its temperature gives its current, its heat
    # gives its temperature
    chk = dev.solve({"top": 5.0}, lattice_temperature=st.temperature.ravel())
    assert chk.current["top"] == pytest.approx(st.current["top"], rel=1e-8)
    Tn = dev._temperature(dev._thermal_setup(th), st.heat, st.temperature.ravel())
    assert np.max(np.abs(Tn - st.temperature.ravel())) < 0.01
    # the same junction without the tunneling model, given the tunneling's
    # pairs as a fixed generation and the temperature as a set one, solved
    # by the plain solver (no rate inside Newton's method), gives the same
    # state
    vol = np.where(dev.vol > 0, dev.vol, 1.0)
    G = dev._band_to_band_nodes(st._x.reshape(dev.N, dev.nv)[:, 0], 5.0) / vol * 1e21
    fixed = _junction(tunneling=False, generation=G.reshape(dev.shape))
    ref = fixed.solve({"top": 5.0}, lattice_temperature=st.temperature.ravel())
    assert ref.current["top"] == pytest.approx(st.current["top"], rel=1e-6)
    assert np.max(np.abs(ref.psi - st.psi)) < 1e-9
    # (the rounds of heated solves of 0.23.0 took 518 Newton steps here
    # with current NumPy and SciPy, and did not settle with the oldest)
    assert one < 200
    # a set lattice temperature equal to the device's, and hot electrons that
    # relax at once, change nothing
    same = dev.solve({"top": 5.0}, lattice_temperature=300.0)
    assert same.current["top"] == pytest.approx(cold.current["top"], rel=1e-10)
    hot = dev.solve({"top": 5.0}, energy_relaxation=1e-16)
    assert hot.current["top"] == pytest.approx(cold.current["top"], rel=1e-5)  # 3.5e-6


def _trench(mirror, viscosity):
    from kpenvelope.process import Structure
    st = Structure(120.0, 0.0, 160.0, 2.0, "Si", 100.0)
    trench, block = ((60, 120), (64, 84)) if not mirror else ((0, 60), (36, 56))
    st.etch({"Si": 1.0}, 40.0, mode="directional", window=trench)
    st.deposit("SiO2", 4.0, mode="conformal")
    st.deposit("Si3N4", 20.0, mode="directional", window=block)
    st.oxidize("Si", "SiO2", 0.5, 0.05, 1500.0, 0.44, "t", mode="2d", shape="flow",
               viscosity=viscosity)
    return st


def test_masks_pushed_sideways_by_the_oxide():
    # a block of nitride on the floor of a trench, against the oxide on the
    # trench's wall: the wall's oxide flows into the block's side and
    # pushes it away from the wall
    flows = {"SiO2": 1e12, "Si3N4": 1e13}
    a, b = _trench(False, flows), _trench(True, flows)
    bal = a.oxidation_balance
    assert bal["whole_cells_nm2"] + bal["parts_nm2"] == pytest.approx(bal["made_nm2"], rel=1e-9)
    assert abs(bal["rounding_nm2"]) < bal["parts_nm2"]
    assert np.array_equal(a.mat, b.mat[::-1])               # its mirror image
    nit = np.argwhere(a.mat == "Si3N4")
    assert nit.shape[0] == 100                              # no nitride made or lost
    assert (nit[:, 0].max() + 1) * 2.0 > 84.0               # moved away from the wall
    rigid = _trench(False, {"SiO2": 1e12})
    nr = np.argwhere(rigid.mat == "Si3N4")
    assert nr[:, 0].min() * 2.0 == 64.0 and (nr[:, 0].max() + 1) * 2.0 == 84.0
    # with nothing to push into (here the domain's edge) the row is not moved
    assert a._side_run(a.mat.shape[0] - 1, 0, 1, {"Si3N4"}, "SiO2") is None
