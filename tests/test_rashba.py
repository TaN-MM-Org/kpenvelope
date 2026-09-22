"""Rashba anchors: the splitting is exactly 2|alpha|k at every
momentum (identity of the closed form AND of full diagonalisation --
two code paths); the spin texture is exactly chiral (in-plane,
perpendicular to k, unit length, opposite branches); Kramers
degeneracy at k = 0 is exact and the direction there is refused; the
Hamiltonian is Hermitian and traceless; units are internally
consistent (1 meV A = 1e-4 eV nm); and no coefficient ships -- a
missing reference is refused."""
import numpy as np
import pytest

from kpenvelope import (rashba_hamiltonian, rashba_spins,
                        rashba_splitting)

REF = "Stefanowicz et al., PRB 89, 205201 (2014): bulk n-GaN alpha"
ALPHA = 4.5e-4                       # 4.5 meV A in eV nm


def test_splitting_exact_two_paths():
    rng = np.random.default_rng(6)
    for _ in range(20):
        kx, ky = rng.uniform(-2.0, 2.0, 2)
        kt = float(np.hypot(kx, ky))
        if kt < 1e-6:
            continue
        h = rashba_hamiltonian(ALPHA, kx, ky, REF)
        e = np.linalg.eigvalsh(h)
        assert abs((e[1] - e[0]) - rashba_splitting(ALPHA, kt, REF)) \
            < 1e-15 + 1e-12 * kt
        assert abs(e[1] + e[0]) < 1e-18            # traceless
        assert np.allclose(h, h.conj().T)          # Hermitian


def test_chiral_texture_exact():
    rng = np.random.default_rng(8)
    for _ in range(10):
        kx, ky = rng.uniform(-2.0, 2.0, 2)
        if np.hypot(kx, ky) < 1e-6:
            continue
        out = rashba_spins(ALPHA, kx, ky, REF)
        for i in range(2):
            s = out["spins"][i]
            assert abs(s @ np.array([kx, ky, 0.0])) \
                < 1e-12 * np.hypot(kx, ky)         # spin  |  k
            assert abs(np.linalg.norm(s) - 1.0) < 1e-12
            assert abs(s[2]) < 1e-12               # in-plane
        assert np.allclose(out["spins"][0], -out["spins"][1],
                           atol=1e-12)             # opposite branches


def test_kramers_at_zero():
    e = np.linalg.eigvalsh(rashba_hamiltonian(ALPHA, 0.0, 0.0, REF))
    assert e[0] == e[1] == 0.0
    with pytest.raises(ValueError, match="Kramers"):
        rashba_spins(ALPHA, 0.0, 0.0, REF)


def test_units_internally_consistent():
    """4.5 meV A at k = 0.1 nm^-1: 2 * 4.5e-4 eV nm * 0.1 nm^-1
    = 9e-5 eV = 0.09 meV, exactly."""
    assert abs(rashba_splitting(4.5e-4, 0.1, REF) - 9.0e-5) < 1e-19


def test_refusals():
    with pytest.raises(ValueError, match="reference"):
        rashba_splitting(ALPHA, 0.1, "short")
    with pytest.raises(ValueError, match="finite"):
        rashba_splitting(np.inf, 0.1, REF)
    with pytest.raises(ValueError, match="nm"):
        rashba_splitting(ALPHA, -0.1, REF)


def test_zero_alpha_spin_direction_refused():
    """With alpha = 0 the two branches are degenerate at every k, so no
    spin direction is defined; before v0.11.1 rashba_spins returned
    spins along z, contradicting the in-plane texture it documents."""
    with pytest.raises(ValueError, match="degenerate"):
        rashba_spins(0.0, 0.3, 0.1, REF)
