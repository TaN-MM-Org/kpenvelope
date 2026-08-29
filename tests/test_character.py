"""Character-analysis tests: exact zone-center block structure, machine
precision normalization, and finite-k mixing."""
import numpy as np
import pytest

from kpenvelope import gan_rinke2008, solve_subbands
from kpenvelope.character import (band_character, character_vs_k,
                                  dominant_character)


def _grid(width_nm=5.0, n=141):
    return np.linspace(0.0, width_nm, n)


def test_fractions_sum_to_one_machine_precision():
    p = gan_rinke2008()
    z = _grid()
    _, envs = solve_subbands(p, z, kx=0.3, ky=0.1, n_states=6)
    fr = band_character(envs, z)
    assert fr.shape == (6, 3)
    assert np.allclose(fr.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(fr >= -1e-15)


def test_zone_center_hh_is_exactly_decoupled():
    # at kt = 0 the six-band Hamiltonian couples only LH and CH (via
    # delta3); HH fractions must be exactly 0 or 1, not approximately
    p = gan_rinke2008()
    z = _grid()
    _, envs = solve_subbands(p, z, kx=0.0, ky=0.0, n_states=8)
    fr = band_character(envs, z)
    hh = fr[:, 0]
    assert np.all((hh > 1.0 - 1e-9) | (hh < 1e-9))
    # both kinds of state exist among the top eight
    assert np.any(hh > 0.5) and np.any(hh < 0.5)


def test_zone_center_lh_ch_do_mix():
    # delta3 of the GaN set is nonzero, so some kt = 0 state carries
    # both LH and CH weight
    p = gan_rinke2008()
    assert p.delta3 != 0.0
    z = _grid()
    _, envs = solve_subbands(p, z, kx=0.0, ky=0.0, n_states=8)
    fr = band_character(envs, z)
    mixed = (fr[:, 1] > 0.5) & (fr[:, 2] > 1e-3)
    assert np.any(mixed)


def test_finite_k_mixes_the_groups():
    p = gan_rinke2008()
    z = _grid()
    _, envs0 = solve_subbands(p, z, kx=0.0, ky=0.0, n_states=1)
    _, envs1 = solve_subbands(p, z, kx=0.6, ky=0.0, n_states=1)
    hh0 = band_character(envs0, z)[0, 0]
    hh1 = band_character(envs1, z)[0, 0]
    assert hh0 > 1.0 - 1e-9          # pure at the zone center
    assert hh1 < 1.0 - 1e-3          # visibly mixed at finite kt


def test_character_vs_k_matches_direct_solution():
    p = gan_rinke2008()
    z = _grid(n=101)
    kts = np.array([0.0, 0.25, 0.5])
    fr_path = character_vs_k(p, z, kts, n_states=3)
    assert fr_path.shape == (3, 3, 3)
    _, envs = solve_subbands(p, z, kx=0.5, ky=0.0, n_states=3)
    assert np.allclose(fr_path[2], band_character(envs, z), atol=1e-12)


def test_dominant_character_labels():
    fr = np.array([[0.9, 0.06, 0.04], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    assert dominant_character(fr) == ["HH", "LH", "CH"]
    assert dominant_character(fr[0]) == "HH"


def test_input_validation():
    z = _grid(n=11)
    with pytest.raises(ValueError):
        band_character(np.zeros((4, 5, 11), dtype=complex), z)
    with pytest.raises(ValueError):
        band_character(np.zeros((4, 6, 12), dtype=complex), z)
    with pytest.raises(ValueError):
        character_vs_k(gan_rinke2008(), z, np.array([-0.1, 0.2]))
    with pytest.raises(ValueError):
        dominant_character(np.array([[0.5, 0.5]]))
