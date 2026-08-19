"""Tests of the cited GaN and AlN parameter sets.

The closed-form anchors come from the quasi-cubic limits of the six-band
Hamiltonian and are quoted in the Supplemental Material (Sec. S9) of
Mahim, Mohsin and Rahman, "Origin of the conflicting hole masses in the
GaN/AlN two-dimensional hole gas" (under review):

  * zone-center splittings 5.20 and 21.80 meV for the GaN set
    (closed form: E1 = d1+d2 against the eigenvalues of [[d1-d2, sqrt2 d3],
    [sqrt2 d3, 0]]), against accepted experimental values near 5-6 and
    22 meV;
  * asymptotic in-plane masses m0/|A2+A4-A5| = 1.89 m0 and
    m0/|A2+A4+A5| = 0.18 m0.

If a transcription error ever enters the parameter tables, these tests
fail against the closed forms.
"""
import math

import numpy as np
import pytest

from kpenvelope import (HBAR2_OVER_2M0, aln_rinke2008, gan_rinke2008,
                        solve_self_consistent)
from kpenvelope.hamiltonian import bulk_blocks


def _trapz(y, x):
    """Trapezoidal integral on a uniform grid (works on NumPy 1 and 2)."""
    dz = x[1] - x[0]
    return dz * (y.sum() - 0.5 * (y[0] + y[-1]))


def test_gan_values_locked():
    p = gan_rinke2008()
    assert (p.A1, p.A2, p.A3) == (-5.947, -0.528, 5.414)
    assert (p.A4, p.A5, p.A6) == (-2.512, -2.510, -3.202)
    assert p.delta1 == 0.010
    assert p.delta2 == p.delta3 == pytest.approx(0.017 / 3.0)
    assert p.eps_r == 10.4
    assert "Rinke" in p.reference and "Barker" in p.reference


def test_aln_values_locked():
    p = aln_rinke2008()
    assert (p.A1, p.A2, p.A3) == (-3.991, -0.311, 3.671)
    assert (p.A4, p.A5, p.A6) == (-1.147, -1.329, -1.952)
    assert p.delta1 == -0.295
    assert p.delta2 == p.delta3 == pytest.approx(0.022 / 3.0)
    assert math.isnan(p.eps_r)
    assert "Rinke" in p.reference and "Carvalho" in p.reference


def test_gan_zone_center_splittings():
    p = gan_rinke2008()
    H0, _, _ = bulk_blocks(p, 0.0, 0.0)
    e = np.sort(np.linalg.eigvalsh(H0))[::-1]
    # Kramers degeneracy at k = 0
    assert e[0] == pytest.approx(e[1], abs=1e-12)
    assert e[2] == pytest.approx(e[3], abs=1e-12)
    # closed-form quasi-cubic eigenvalues
    d1, d2, d3 = p.delta1, p.delta2, p.delta3
    top = d1 + d2
    g = d1 - d2
    pair_hi = 0.5 * (g + math.sqrt(g * g + 8 * d3 * d3))
    pair_lo = 0.5 * (g - math.sqrt(g * g + 8 * d3 * d3))
    assert e[0] == pytest.approx(top, abs=1e-12)
    assert e[2] == pytest.approx(pair_hi, abs=1e-12)
    assert e[4] == pytest.approx(pair_lo, abs=1e-12)
    # the published values (Supplemental Material, Sec. S9)
    assert (e[0] - e[2]) * 1e3 == pytest.approx(5.20, abs=0.01)
    assert (e[0] - e[4]) * 1e3 == pytest.approx(21.80, abs=0.01)


def test_gan_asymptotic_inplane_masses():
    p = gan_rinke2008()
    c = HBAR2_OVER_2M0
    k1, k2 = 4.0, 4.4
    E1 = np.sort(np.linalg.eigvalsh(bulk_blocks(p, k1, 0.0)[0]))[::-1]
    E2 = np.sort(np.linalg.eigvalsh(bulk_blocks(p, k2, 0.0)[0]))[::-1]
    slope_top = (E1[0] - E2[0]) / (c * (k2 ** 2 - k1 ** 2))
    slope_bot = (E1[5] - E2[5]) / (c * (k2 ** 2 - k1 ** 2))
    # heavy branch -> m0/|A2+A4-A5| = 1.89 m0 (slope 0.53); at these
    # wavevectors the top sorted band interleaves with the A2-only band
    # (slope 0.528), which lies within the tolerance
    assert 1.0 / slope_top == pytest.approx(1.89, abs=0.03)
    # light branch -> m0/|A2+A4+A5| = 0.18 m0
    assert 1.0 / slope_bot == pytest.approx(0.180, abs=0.003)


def test_aln_refuses_poisson_without_permittivity():
    z = np.linspace(0.0, 4.0, 33)
    with pytest.raises(ValueError, match="permittivity"):
        solve_self_consistent(aln_rinke2008(), z, ps=0.1)


def test_gan_self_consistent_hard_wall():
    """Hard-wall self-consistent GaN gas at the measured sheet density.

    Compared against the hard-wall row of Table S1 of the source paper
    (centroid 0.568 nm, light-pair sheet density 0.46e13 cm^-2). The
    tolerances are loose on purpose: kpenvelope fills subbands with
    parabolic edge masses (a documented approximation), while the source
    fills the computed non-parabolic dispersions.
    """
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 97)
    res = solve_self_consistent(p, z, ps=0.46, n_states=4, mixing=0.5,
                                tol=2e-5)
    # charge neutrality, exact by construction and preserved to the end
    sheet = _trapz(res.density, z)
    assert sheet == pytest.approx(0.46, rel=5e-3)
    # the gas sits within a nanometer of the interface
    centroid = _trapz(z * res.density, z) / sheet
    assert 0.4 < centroid < 0.8
    # two heavy branches dominate; the light branch is a minority
    occ = np.sort(res.occupations)[::-1]
    assert occ[0] + occ[1] > 3.0 * occ[2]
    assert res.residual < 2e-5
