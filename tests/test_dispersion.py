import numpy as np
import pytest

from kpenvelope import (demo_single_band, gan_rinke2008, local_mass,
                        solve_subbands, subband_dispersion)
from kpenvelope.hamiltonian import bulk_blocks


def test_single_k_point_equals_solve_subbands():
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 49)
    disp = subband_dispersion(p, z, [0.4], theta=0.3, n_states=4)
    e_ref, _ = solve_subbands(p, z, kx=0.4 * np.cos(0.3),
                              ky=0.4 * np.sin(0.3), n_states=4)
    assert np.allclose(disp[0], e_ref, atol=1e-14)


def test_demo_set_local_mass_is_exactly_one_over_A():
    # decoupled demo set: E = A c k^2 exactly, so m*/m0 = 1/|A| at every
    # momentum and for every well level
    A = -2.0
    p = demo_single_band(A)
    z = np.linspace(0.0, 8.0, 65)
    kts = np.linspace(0.0, 1.5, 7)
    disp = subband_dispersion(p, z, kts, n_states=3)
    _, masses = local_mass(kts, disp)
    assert np.allclose(masses, 1.0 / abs(A), atol=1e-10)


def test_bulk_gan_quasicubic_asymptotic_masses():
    # applied to the bulk Rinke 2008 GaN bands at large k_t, local_mass
    # reproduces the closed-form quasi-cubic asymptotes
    # m0/|A2+A4-A5| = 1.89 and m0/|A2+A4+A5| = 0.180
    p = gan_rinke2008()
    kts = np.array([4.0, 4.4])
    E = np.array([np.sort(np.linalg.eigvalsh(
        bulk_blocks(p, kt, 0.0)[0]))[::-1] for kt in kts])
    _, masses = local_mass(kts, E)
    # asymptotic, not exact, at finite k_t: the k-linear spin-orbit terms
    # still contribute at 4 nm^-1, so the tolerance is the observed
    # residual of the asymptote, not machine precision
    assert abs(masses[0, 0] - 1.0 / abs(p.A2 + p.A4 - p.A5)) < 5e-3
    assert abs(masses[0, 5] - 1.0 / abs(p.A2 + p.A4 + p.A5)) < 1e-4
    assert abs(masses[0, 0] - 1.89) < 0.03
    assert abs(masses[0, 5] - 0.180) < 0.003


def test_subband_masses_are_finite_and_positive_in_a_well():
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 49)
    kts = np.linspace(0.05, 1.2, 6)
    disp = subband_dispersion(p, z, kts, n_states=2)
    _, masses = local_mass(kts, disp)
    assert np.all(np.isfinite(masses)) and np.all(masses > 0.0)


def test_guards():
    p = gan_rinke2008()
    z = np.linspace(0.0, 6.0, 33)
    with pytest.raises(ValueError):
        subband_dispersion(p, z, [-0.1, 0.2])
    with pytest.raises(ValueError):
        local_mass([0.5], [0.0])
    with pytest.raises(ValueError):
        local_mass([0.2, 0.1], [0.0, 1.0])
    with pytest.raises(ValueError):
        local_mass([0.1, 0.2], [0.0, 1.0, 2.0])
    # a locally flat branch yields inf, not an exception
    _, m = local_mass([0.1, 0.2], [1.0, 1.0])
    assert np.isinf(m[0])
