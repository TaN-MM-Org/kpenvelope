"""Bir-Pikus strain terms against exact statements: the structural
identity with the kinetic template, closed-form eigenvalues at k = 0,
exact Hermiticity, the exact shift of a strained well, and the refusal
of parameter sets without cited deformation potentials."""
import dataclasses

import numpy as np
import pytest

from kpenvelope import (HBAR2_OVER_2M0 as C, demo_single_band,
                        gan_rinke2008, solve_subbands, strain_blocks)
from kpenvelope.hamiltonian import bulk_blocks


def _demo_with_D():
    return dataclasses.replace(demo_single_band(), D1=1.1, D2=-0.7,
                               D3=2.2, D4=-0.4, D5=0.9, D6=1.3)


def test_strain_template_equals_the_kinetic_template():
    """The whole Bir-Pikus structure at once: with D_i = c A_i and
    eps_ij = k_i k_j, the strain matrix must reproduce
    H0 + H1 kz + H2 kz^2 entry for entry (splittings off)."""
    p0 = gan_rinke2008()
    p = dataclasses.replace(p0, D1=C * p0.A1, D2=C * p0.A2, D3=C * p0.A3,
                            D4=C * p0.A4, D5=C * p0.A5, D6=C * p0.A6,
                            delta1=0.0, delta2=0.0, delta3=0.0)
    rng = np.random.default_rng(2)
    for _ in range(4):
        kx, ky, kz = rng.uniform(-1.0, 1.0, 3)
        H0, H1, H2 = bulk_blocks(p, kx, ky)
        Hk = H0 + H1 * kz + H2 * kz * kz
        kvec = np.array([kx, ky, kz])
        assert np.abs(Hk - strain_blocks(p, np.outer(kvec, kvec))).max() \
            < 1e-14


def test_zero_strain_changes_nothing_and_hermiticity_is_exact():
    p = _demo_with_D()
    assert np.abs(strain_blocks(p, np.zeros((3, 3)))).max() == 0.0
    rng = np.random.default_rng(3)
    e = rng.uniform(-0.01, 0.01, (3, 3))
    e = 0.5 * (e + e.T)
    H = strain_blocks(p, e)
    assert np.abs(H - H.conj().T).max() == 0.0


def test_diagonal_strain_closed_form_eigenvalues():
    """Splittings off, no shear: eigenvalues are lambda_eps + theta_eps
    (four-fold) and lambda_eps (two-fold), exactly."""
    p = _demo_with_D()
    e = np.diag([0.004, 0.004, -0.002])
    lam = p.D1 * e[2, 2] + p.D2 * (e[0, 0] + e[1, 1])
    th = p.D3 * e[2, 2] + p.D4 * (e[0, 0] + e[1, 1])
    ev = np.sort(np.linalg.eigvalsh(strain_blocks(p, e)))
    assert np.abs(ev - np.sort([lam + th] * 4 + [lam] * 2)).max() < 1e-15


def test_pure_shear_closed_form_eigenvalues():
    """eps_xz alone mixes exactly the H_t positions: eigenvalues
    0 (twice) and +- sqrt(2) |D6 eps_xz| (twice each)."""
    p = _demo_with_D()
    e = np.zeros((3, 3))
    e[0, 2] = e[2, 0] = 0.003
    h = p.D6 * 0.003
    ev = np.sort(np.linalg.eigvalsh(strain_blocks(p, e)))
    expect = np.sort([0.0, 0.0, np.sqrt(2) * h, -np.sqrt(2) * h,
                      np.sqrt(2) * h, -np.sqrt(2) * h])
    assert np.abs(ev - expect).max() < 1e-15


def test_strained_well_levels_shift_by_the_exact_edge_shift():
    """Diagonal strain commutes with the decoupled kinetic operator, so
    every well level of the demo set shifts rigidly by the strain edge
    shift -- machine precision."""
    p = _demo_with_D()
    e = np.diag([0.004, 0.004, -0.002])
    lam = p.D1 * e[2, 2] + p.D2 * (e[0, 0] + e[1, 1])
    th = p.D3 * e[2, 2] + p.D4 * (e[0, 0] + e[1, 1])
    z = np.linspace(0.0, 8.0, 61)
    e0, _ = solve_subbands(demo_single_band(), z, n_states=2)
    es, _ = solve_subbands(p, z, n_states=2, strain=e)
    assert abs((es[0] - e0[0]) - max(lam + th, lam)) < 1e-12


def test_missing_deformation_potentials_are_refused():
    with pytest.raises(ValueError):
        strain_blocks(gan_rinke2008(), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        strain_blocks(_demo_with_D(), np.array([[0.0, 1e-3, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0]]))
