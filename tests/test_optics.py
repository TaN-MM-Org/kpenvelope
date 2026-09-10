"""v0.7 anchors: intersubband dipoles and oscillator strengths against
the infinite-well closed forms -- z12 = 16 L / (9 pi^2),
f12 = 256/(27 pi^2), the Thomas-Reiche-Kuhn sum rule -- plus exact
Hermiticity, parity selection, and the exact origin-shift gauge
identity on the fully coupled six-band well. The demo set is six-fold
degenerate, so the closed forms pin the basis-invariant
multiplet-summed quantities."""
import numpy as np

from kpenvelope import (demo_single_band, dipole_matrix, gan_rinke2008,
                        oscillator_strengths, solve_subbands)

L = 8.0
Z = np.linspace(0.0, L, 401)


def _demo_states(n_states=60):
    p = demo_single_band(A=-2.0)
    return solve_subbands(p, Z, n_states=n_states)


def test_dipole_matrix_is_hermitian_and_centered():
    e, env = _demo_states(12)
    d = dipole_matrix(Z, env)
    assert np.abs(d - d.conj().T).max() == 0.0
    # ground multiplet sits at the well center exactly (symmetry)
    assert abs(d[0, 0].real - L / 2.0) < 1e-9


def test_infinite_well_dipole_and_oscillator_closed_forms():
    e, env = _demo_states(60)
    d = dipole_matrix(Z, env)
    f = oscillator_strengths(e, d, 0.5)          # m*/m0 = 1/|A| = 1/2
    # n=1 -> n=2 multiplet (indices 6..11): |z| and f, multiplet-summed
    z12 = np.sqrt((np.abs(d[0, 6:12]) ** 2).sum())
    assert abs(z12 - 16.0 * L / (9.0 * np.pi ** 2)) / z12 < 0.01
    f12 = f[6:12].sum()
    assert abs(f12 - 256.0 / (27.0 * np.pi ** 2)) < 1e-3
    # parity: n=1 -> n=3 multiplet vanishes
    z13 = np.sqrt((np.abs(d[0, 12:18]) ** 2).sum())
    assert z13 < 1e-9


def test_trk_sum_rule_on_the_hard_wall_demo():
    """Sum over 10 well levels of the complete same-component ladder:
    the infinite-well f-sum converges to 1 from below (f12 alone is
    0.96)."""
    e, env = _demo_states(60)
    d = dipole_matrix(Z, env)
    f = oscillator_strengths(e, d, 0.5)
    total = f.sum()
    assert 0.995 < total < 1.0005


def test_origin_shift_gauge_identity_on_the_coupled_well():
    """Shifting the z origin by a shifts every diagonal dipole by
    exactly a and leaves every off-diagonal element invariant -- an
    exact identity that holds for the fully coupled six-band states,
    not just the decoupled demo."""
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 161)
    e, env = solve_subbands(p, z, kx=0.2, n_states=6)
    d0 = dipole_matrix(z, env)
    a = 3.7
    d1 = dipole_matrix(z + a, env)
    off = ~np.eye(6, dtype=bool)
    assert np.abs((d1 - d0)[off]).max() < 1e-12
    assert np.abs(np.diag(d1 - d0) - a).max() < 1e-12


def test_oscillator_strengths_shape_and_ground_zero():
    e, env = _demo_states(12)
    f = oscillator_strengths(e, dipole_matrix(Z, env), 0.5)
    assert f.shape == (12,) and f[0] == 0.0
