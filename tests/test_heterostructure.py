"""Finite-barrier heterostructures: exact reduction to the uniform
assembly, textbook finite-well closed forms with second-order grid
convergence, the deep-barrier (hard-wall) limit, analytic envelope decay,
and exact Hermiticity for mixed materials."""
import numpy as np
import pytest
import scipy.optimize as so

from kpenvelope.hamiltonian import HBAR2_OVER_2M0 as C
from kpenvelope.hamiltonian import assemble_hamiltonian
from kpenvelope.heterostructure import (
    assemble_heterostructure,
    layered_profile,
    solve_heterostructure,
)
from kpenvelope.params import aln_rinke2008, demo_single_band, gan_rinke2008
from kpenvelope.solver import solve_subbands

P = demo_single_band()          # E = -2 C k^2: hbar^2/2m* = 2C
COEF = 2.0 * C
V0, WELL = 0.3, 10.0


def textbook_depths(v0=V0, a=WELL):
    """Bound-state depths of the finite square well, from the standard
    transcendental equations (independent of the solver)."""
    def even(e):
        k = np.sqrt(e / COEF)
        kap = np.sqrt((v0 - e) / COEF)
        return k * np.sin(k * a / 2) - kap * np.cos(k * a / 2)

    def odd(e):
        k = np.sqrt(e / COEF)
        kap = np.sqrt((v0 - e) / COEF)
        return k * np.cos(k * a / 2) + kap * np.sin(k * a / 2)

    out = []
    for eq in (even, odd):
        es = np.linspace(1e-9, v0 - 1e-9, 40000)
        v = np.array([eq(e) for e in es])
        for i in np.where(np.diff(np.sign(v)) != 0)[0]:
            out.append(so.brentq(eq, es[i], es[i + 1]))
    return sorted(out)


def _well(npts, v0=V0):
    z = np.linspace(0.0, 30.0, npts)
    params, edge = layered_profile(
        z, [(10.0, P, -v0), (10.0, P, 0.0), (10.0, P, -v0)])
    return z, params, edge


def test_uniform_profile_reduces_to_uniform_assembly():
    z = np.linspace(0.0, 12.0, 61)
    Hu = assemble_hamiltonian(gan_rinke2008(), z, kx=0.1, ky=-0.05)
    Hh = assemble_heterostructure(z, [gan_rinke2008()] * z.size,
                                  kx=0.1, ky=-0.05)
    assert np.abs(Hu - Hh).max() == 0.0


def test_finite_well_matches_textbook_with_second_order_convergence():
    tb = textbook_depths()
    errs = []
    for npts in (151, 301):
        z, params, edge = _well(npts)
        E, _ = solve_heterostructure(z, params, edge, n_states=30)
        uniq = -np.unique(np.round(E, 9))[::-1]
        bound = [u for u in uniq if 0 < u < V0][:4]
        errs.append(max(abs(b - t) for b, t in zip(bound, tb)))
    assert errs[1] < 1e-3                      # accurate on the fine grid
    assert errs[1] < 0.35 * errs[0]            # ~O(dz^2) convergence


def test_deep_barrier_approaches_hard_wall():
    """Ground depth increases monotonically with barrier height and lands
    within the O(dz) interface-placement offset of the hard-wall value."""
    z = np.linspace(0.0, 30.0, 301)
    dz = z[1] - z[0]
    zw = z[(z >= 10.0) & (z <= 20.0)]
    Ehw, _ = solve_subbands(P, zw, n_states=6)
    hw = -np.unique(np.round(Ehw, 9))[::-1][0]
    prev = -np.inf
    for v in (1.0, 5.0, 50.0):
        params, edge = layered_profile(
            z, [(10.0, P, -v), (10.0, P, 0.0), (10.0, P, -v)])
        E, _ = solve_heterostructure(z, params, edge, n_states=8)
        g = -np.unique(np.round(E, 9))[::-1][0]
        assert g > prev
        prev = g
    assert abs(prev - hw) < 4.0 * dz * hw      # O(dz) placement offset


def test_envelope_decays_with_the_analytic_constant():
    z, params, edge = _well(301)
    E, F = solve_heterostructure(z, params, edge, n_states=6)
    depth = -E[0]
    kappa = np.sqrt((V0 - depth) / COEF)
    prob = (np.abs(F[0]) ** 2).sum(axis=0)
    mask = (z > 21.5) & (z < 27.0)
    slope = np.polyfit(z[mask], np.log(prob[mask]), 1)[0]
    assert abs(-slope / 2 - kappa) / kappa < 0.01


def test_mixed_material_assembly_is_exactly_hermitian():
    z = np.linspace(0.0, 30.0, 121)
    params, edge = layered_profile(
        z, [(10.0, aln_rinke2008(), -0.5),
            (10.0, gan_rinke2008(), 0.0),
            (10.0, aln_rinke2008(), -0.5)])
    H = assemble_heterostructure(z, params, edge, kx=0.15, ky=-0.08)
    assert np.abs(H - H.conj().T).max() == 0.0


def test_finite_barrier_softens_confinement_for_the_demo_well():
    """A finite barrier lowers the confinement depth relative to the
    infinitely hard wall (the envelope relaxes into the barrier)."""
    z, params, edge = _well(301, v0=1.0)
    E, _ = solve_heterostructure(z, params, edge, n_states=8)
    depth_fin = -np.unique(np.round(E, 9))[::-1][0]
    zw = z[(z >= 10.0) & (z <= 20.0)]
    Ehw, _ = solve_subbands(P, zw, n_states=6)
    depth_hw = -np.unique(np.round(Ehw, 9))[::-1][0]
    assert depth_fin < depth_hw


def test_layer_profile_validation():
    z = np.linspace(0.0, 30.0, 61)
    with pytest.raises(ValueError):
        layered_profile(z, [(10.0, P, 0.0), (10.0, P, 0.0)])   # 20 != 30
    with pytest.raises(ValueError):
        assemble_heterostructure(z, [P] * 5)                    # wrong count
