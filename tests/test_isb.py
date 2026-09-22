"""v0.9 collective-intersubband anchors: the depolarization geometry
integral against independent adaptive quadrature over analytic
infinite-well wavefunctions, its exact origin invariance and width
scaling, the exact linearity of alpha in density, and the lineshape
sum rule -- following Allen/Tsui/Vinter 1976 and Ando/Fowler/Stern
RMP 54, 437 (1982)."""
import numpy as np
import pytest

from kpenvelope import (demo_single_band, depolarization_shift,
                        isb_lineshape, overlap_geometry_integral,
                        solve_subbands)

L = 6.0
Z = np.linspace(0.0, L, 601)


def _hard_wall_states(n_states=12):
    return solve_subbands(demo_single_band(A=-2.0), Z,
                          n_states=n_states)


def test_geometry_integral_matches_analytic_quadrature():
    """Path A: solver envelopes on the grid. Path B: adaptive
    quadrature over the ANALYTIC hard-wall wavefunctions
    sqrt(2/L) sin(n pi z / L) -- no shared arithmetic."""
    from scipy.integrate import quad
    # scalar (1, 2) case built from the analytic hard-wall functions
    xi1 = np.sqrt(2.0 / L) * np.sin(np.pi * Z / L)
    xi2 = np.sqrt(2.0 / L) * np.sin(2.0 * np.pi * Z / L)
    S_grid = overlap_geometry_integral(Z, xi1[None, :], xi2[None, :])

    def inner(zv):
        val, _ = quad(lambda t: (2.0 / L) * np.sin(np.pi * t / L)
                      * np.sin(2.0 * np.pi * t / L), 0.0, zv,
                      limit=200)
        return val

    S_quad, _ = quad(lambda zv: inner(zv) ** 2, 0.0, L, limit=200)
    assert abs(S_grid - S_quad) / S_quad < 1e-3     # grid-limited
    # exact linear width scaling: S(2L) = 2 S(L)
    Z2 = np.linspace(0.0, 2 * L, 1201)
    x1 = np.sqrt(1.0 / L) * np.sin(np.pi * Z2 / (2 * L))
    x2 = np.sqrt(1.0 / L) * np.sin(2.0 * np.pi * Z2 / (2 * L))
    S2 = overlap_geometry_integral(Z2, x1[None, :], x2[None, :])
    assert abs(S2 - 2.0 * S_grid) / S2 < 1e-3
    # exact origin invariance (orthogonality zeroes the inner
    # integral at both ends, so the origin cannot matter)
    S_shift = overlap_geometry_integral(Z + 17.3, xi1[None, :],
                                        xi2[None, :])
    assert abs(S_shift - S_grid) < 1e-12 * max(S_grid, 1.0)


def test_depolarization_shift_exact_structure():
    energies, env = _hard_wall_states()
    i, j = 0, 6                       # across the first two multiplets
    ns, er = 0.2, 10.0
    out = depolarization_shift(Z, env[i], env[j], energies[i],
                               energies[j], ns, er)
    dE = abs(energies[i] - energies[j])
    assert out["E_bare"] == dE
    # exact defining identity, and the shift is upward
    assert abs(out["E_shifted"] - dE * np.sqrt(1 + out["alpha"])) < 1e-15
    assert out["E_shifted"] > dE and out["alpha"] > 0
    # alpha exactly linear in density; zero density exact
    out2 = depolarization_shift(Z, env[i], env[j], energies[i],
                                energies[j], 2 * ns, er)
    assert abs(out2["alpha"] - 2 * out["alpha"]) < 1e-12 * out["alpha"]
    out0 = depolarization_shift(Z, env[i], env[j], energies[i],
                                energies[j], 0.0, er)
    assert out0["E_shifted"] == out0["E_bare"] and out0["alpha"] == 0.0
    # alpha inversely proportional to eps_r, exactly
    out3 = depolarization_shift(Z, env[i], env[j], energies[i],
                                energies[j], ns, 2 * er)
    assert abs(out3["alpha"] - 0.5 * out["alpha"]) < 1e-12 * out["alpha"]
    with pytest.raises(ValueError):
        depolarization_shift(Z, env[i], env[j], energies[i],
                             energies[i], ns, er)      # degenerate
    with pytest.raises(ValueError):
        depolarization_shift(Z, env[i], env[j], energies[i],
                             energies[j], -1.0, er)


def test_lineshape_sum_rule_and_peaks():
    Ei = np.array([0.08, 0.15])
    fi = np.array([0.9, 0.1])
    g = 0.004
    E = np.linspace(-4.0, 4.6, 400001)
    A = isb_lineshape(Ei, fi, g, E)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2
    total = trapezoid(A, E)
    assert abs(total - fi.sum()) < 1e-3               # tail-truncation only
    # peaks at the transition energies
    assert abs(E[np.argmax(A)] - Ei[0]) < 2 * (E[1] - E[0])
    with pytest.raises(ValueError):
        isb_lineshape(Ei, fi, -1.0, E)
    with pytest.raises(ValueError):
        isb_lineshape(Ei, fi[:1], g, E)
