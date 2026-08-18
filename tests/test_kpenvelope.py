import numpy as np

from kpenvelope import (assemble_hamiltonian, demo_single_band,
                        solve_subbands, solve_self_consistent,
                        HBAR2_OVER_2M0)
from kpenvelope.poisson import hole_potential, E2_OVER_EPS0


def test_hamiltonian_is_hermitian():
    p = demo_single_band()
    # switch on every coupling to test the discretization, not the demo
    p.A3, p.A4, p.A5, p.A6 = 0.5, -0.3, 0.4, 0.7
    p.delta1, p.delta2, p.delta3 = 0.02, 0.005, 0.005
    z = np.linspace(0.0, 5.0, 40)
    H = assemble_hamiltonian(p, z, kx=0.3, ky=0.2)
    assert np.allclose(H, H.conj().T)


def test_square_well_matches_analytic():
    # decoupled bands: E_n = A * (hbar^2/2m0) * (n pi / L)^2, A < 0 so the
    # HIGHEST states are n = 1, 2, ... from the top.
    A = -2.0
    L = 8.0
    p = demo_single_band(A)
    z = np.linspace(0.0, L, 400)
    energies, _ = solve_subbands(p, z, n_states=12)
    # the discrete Dirichlet walls sit one grid spacing outside the
    # endpoints, so the effective well width is L + 2 dz
    dz = z[1] - z[0]
    Leff = L + 2 * dz
    analytic = [A * HBAR2_OVER_2M0 * (n * np.pi / Leff) ** 2 for n in (1, 2)]
    # each level is 6-fold degenerate in the decoupled limit
    np.testing.assert_allclose(energies[:6], analytic[0], rtol=1e-4)
    np.testing.assert_allclose(energies[6:12], analytic[1], rtol=1e-4)


def test_poisson_uniform_slab():
    # uniform density filling 0..L exactly cancels ps: field decreases
    # linearly to zero, potential is parabolic with V'(0) set by ps
    L, ps, eps_r = 4.0, 0.1, 10.0
    z = np.linspace(0.0, L, 2001)
    density = np.full_like(z, ps / L)
    vh = hole_potential(z, density, ps, eps_r)
    slope0 = (vh[1] - vh[0]) / (z[1] - z[0])
    assert np.isclose(slope0, E2_OVER_EPS0 / eps_r * ps, rtol=1e-3)
    # analytic: V(L) = (e^2 ps L / (2 eps)) for the uniform slab
    assert np.isclose(vh[-1], E2_OVER_EPS0 / eps_r * ps * L / 2.0, rtol=1e-3)


def test_self_consistent_converges_and_is_neutral():
    p = demo_single_band()
    z = np.linspace(0.0, 6.0, 120)
    ps = 0.046  # nm^-2, i.e. 4.6e12 cm^-2
    res = solve_self_consistent(p, z, ps, n_states=4, tol=1e-5)
    assert res.residual < 1e-5
    dz = z[1] - z[0]
    assert np.isclose(res.density.sum() * dz, ps, rtol=1e-6)
    # the gas must be held toward the interface, not centered in the well
    centroid = (res.z * res.density).sum() / res.density.sum()
    assert centroid < 0.5 * z[-1]
