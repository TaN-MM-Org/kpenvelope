"""v0.12 anchors.

1. The Chuang-Chang Hamiltonian fix (lower-block H/H* placement): bulk
   eigenvalues against the published block-diagonal 3 x 3 form, whose
   entries depend on |k_t| only; exact Kramers degeneracy of the bulk
   bands with any strain; rotation covariance about the c axis; and
   in-plane isotropy of the subbands of an asymmetric heterostructure.
2. The momentum-grid filling: exact for parabolic subbands at any grid
   spacing, against the closed-form filler; second-order convergence on
   a non-parabolic two-branch dispersion, against an independent root
   finder.
3. The k-resolved self-consistent loop: reproduces the parabolic loop
   on the exactly parabolic demo set, and closes Gauss's law.
4. Poisson: per-point permittivity and the fixed-sheet position against
   closed forms.
5. Input refusals.
"""
import dataclasses

import numpy as np
import pytest
from scipy.optimize import brentq

from kpenvelope import (HBAR2_OVER_2M0 as C, aln_rinke2008,
                        demo_single_band, fill_subbands_kgrid,
                        fill_subbands_thermal, gan_rinke2008,
                        layered_profile, solve_heterostructure,
                        solve_self_consistent, solve_self_consistent_hetero,
                        solve_subbands, spin_splitting, strain_blocks)
from kpenvelope.hamiltonian import assemble_hamiltonian, bulk_blocks
from kpenvelope.poisson import E2_OVER_EPS0, hole_potential

GAN = gan_rinke2008()
ALN = aln_rinke2008()


# ---------------------------------------------------------------- 1

def _cc_block_eigs(p, kt, kz):
    """Eigenvalues of the block-diagonalized upper 3 x 3 Hamiltonian of
    Chuang and Chang, PRB 54, 2491 (1996), Eq. (45):
        [[F, Kt, -i Ht], [Kt, G, Delta - i Ht], [i Ht, Delta + i Ht, lam]]
    with Kt = c A5 k_t^2, Ht = c A6 k_t k_z, Delta = sqrt(2) delta3.
    Its entries depend on k_t = |k_t| only, and the lower block has the
    same eigenvalues, so every bulk band is two-fold degenerate and
    isotropic in the plane."""
    lam = C * (p.A1 * kz ** 2 + p.A2 * kt ** 2)
    th = C * (p.A3 * kz ** 2 + p.A4 * kt ** 2)
    F = p.delta1 + p.delta2 + lam + th
    G = p.delta1 - p.delta2 + lam + th
    Kt = C * p.A5 * kt ** 2
    Ht = C * p.A6 * kt * kz
    D = np.sqrt(2.0) * p.delta3
    M = np.array([[F, Kt, -1j * Ht],
                  [Kt, G, D - 1j * Ht],
                  [1j * Ht, D + 1j * Ht, lam]])
    return np.sort(np.linalg.eigvalsh(M))[::-1]


def _bulk_eigs(p, kx, ky, kz, strain=None):
    H0, H1, H2 = bulk_blocks(p, kx, ky)
    H = H0 + H1 * kz + H2 * kz * kz
    if strain is not None:
        H = H + strain_blocks(p, strain)
    return np.sort(np.linalg.eigvalsh(H))[::-1]


@pytest.mark.parametrize("p", [GAN, ALN])
def test_bulk_bands_match_the_block_diagonal_chuang_chang_form(p):
    """Before v0.12.0 this failed: with this sampling (seed 12) the old
    matrix was off by 0.327 eV for GaN and 0.191 eV for AlN. Along kx
    the old and new matrices are identical."""
    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(40):
        kt, kz = rng.uniform(0.0, 1.5, 2)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        e = _bulk_eigs(p, kt * np.cos(phi), kt * np.sin(phi), kz)
        ref = np.repeat(_cc_block_eigs(p, kt, kz), 2)
        worst = max(worst, np.abs(e - ref).max())
    assert worst < 1e-12


def _gan_with_D():
    # D values are arbitrary test numbers (symmetry holds for any D),
    # not material constants
    return dataclasses.replace(GAN, D1=1.1, D2=-0.7, D3=2.2, D4=-0.4,
                               D5=0.9, D6=1.3)


def test_bulk_kramers_degeneracy_with_any_strain():
    """Strain is even under time reversal and the six-band model has no
    k-linear terms, so every bulk level is two-fold degenerate for any
    k and any symmetric strain (the old lower block split them)."""
    p = _gan_with_D()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(30):
        e = rng.uniform(-3e-3, 3e-3, (3, 3))
        e = 0.5 * (e + e.T)
        kx, ky, kz = rng.uniform(-1.0, 1.0, 3)
        ev = _bulk_eigs(p, kx, ky, kz, e)
        worst = max(worst, np.abs(ev[0::2] - ev[1::2]).max())
    assert worst < 1e-12


def test_rotation_about_c_leaves_energies_unchanged():
    """Rotating k and the strain tensor together about the c axis is a
    symmetry of the model: eigenvalues must not change."""
    p = _gan_with_D()
    rng = np.random.default_rng(5)
    for _ in range(10):
        e = rng.uniform(-3e-3, 3e-3, (3, 3))
        e = 0.5 * (e + e.T)
        k = rng.uniform(-1.0, 1.0, 3)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        R = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                      [np.sin(phi), np.cos(phi), 0.0],
                      [0.0, 0.0, 1.0]])
        k2 = R @ k
        e2 = R @ e @ R.T
        a = _bulk_eigs(p, k[0], k[1], k[2], e)
        b = _bulk_eigs(p, k2[0], k2[1], k2[2], e2)
        assert np.abs(a - b).max() < 1e-12


def _asymmetric_stack(npts=49):
    z = np.linspace(0.0, 8.0, npts)
    # offsets illustrative, not a cited GaN/AlN alignment
    params, edge = layered_profile(z, [(3.0, ALN, -0.3), (5.0, GAN, 0.0)])
    tilt = 0.05 * z / z[-1]
    return z, params, edge, tilt


def test_subbands_depend_only_on_the_momentum_magnitude():
    """In-plane isotropy of the subbands of an asymmetric GaN/AlN
    stack with a tilted potential (a unitary phase transformation maps
    direction theta onto theta = 0, so the agreement is exact up to
    rounding). The old matrix gave direction-dependent subbands."""
    z, params, edge, tilt = _asymmetric_stack()
    kt = 0.4
    ref, _ = solve_heterostructure(z, params, edge, kx=kt, potential=tilt,
                                   n_states=6)
    for th in (0.7, np.pi / 2, 2.5):
        e, _ = solve_heterostructure(z, params, edge, kx=kt * np.cos(th),
                                     ky=kt * np.sin(th), potential=tilt,
                                     n_states=6)
        assert np.abs(e - ref).max() < 1e-10


def test_symmetric_well_has_no_spin_splitting_in_any_direction():
    """Inversion-symmetric well: Kramers pairs stay degenerate at every
    k, along ky as along kx (along ky the old matrix split them)."""
    z = np.linspace(0.0, 6.0, 41)
    for th in (0.0, 0.5, np.pi / 2):
        e, _ = solve_subbands(GAN, z, kx=0.3 * np.cos(th),
                              ky=0.3 * np.sin(th), n_states=4)
        assert np.abs(spin_splitting(e)).max() < 1e-9


# ---------------------------------------------------------------- 2

def test_kgrid_filling_is_exact_for_parabolic_subbands():
    """Two six-fold levels of the demo well, both occupied: the k-grid
    filler must equal the closed-form parabolic filler (itself checked
    against quadrature in test_thermal.py) at any grid spacing and
    temperature. At T = 0 the trapezoid rule used before v0.12.0 was
    off by 4.17e-2 nm^-2 per state with nk = 5 (upper states left
    empty) and 7.57e-3 nm^-2 (18 %) with nk = 23 on this set-up."""
    p = demo_single_band(A=-2.0)
    z = np.linspace(0.0, 8.0, 41)
    e0, _ = solve_subbands(p, z, n_states=12)
    ps = 0.3

    def at_k(kx, ky):
        return solve_subbands(p, z, kx=kx, ky=ky, n_states=12)[0]

    for T in (0.0, 150.0):
        occ_par = fill_subbands_thermal(e0, np.full(12, 0.5), ps, T)
        assert occ_par[6:].sum() > 0.05            # both levels filled
        for nk in (5, 23):
            occ, _ = fill_subbands_kgrid(at_k, ps, 12, kmax=2.6, nk=nk,
                                         ntheta=1, temperature_K=T)
            assert np.abs(occ - occ_par).max() < 1e-9
            assert abs(occ.sum() - ps) < 1e-14


def test_kgrid_filling_second_order_on_a_nonparabolic_dispersion():
    """Two branches, one strongly non-parabolic, at T = 0. The exact
    answer: E_F solves k1(E_F)^2 + k2(E_F)^2 = 4 pi ps, and branch n
    holds k_n^2 / (4 pi) -- found here with brentq, independently of
    the filler. The error is bounded by C / (nk - 1)^2 (second order
    in the step; its prefactor oscillates with where the Fermi crossing
    falls inside a segment, so successive halvings do not each gain
    exactly 4x). The trapezoid rule before v0.12.0 was first order:
    1.0e-3, 2.1e-4 and 1.2e-4 nm^-2 at nk = 41, 161 and 321."""
    a, b, a2, d = 0.08, 0.05, 0.03, 0.01

    def at_k(kx, ky):
        k2 = kx * kx + ky * ky
        return np.array([-a * k2 - b * k2 * k2, -d - a2 * k2])

    def kf2_1(ef):                     # -a q - b q^2 = ef, q = k^2
        return (-a + np.sqrt(a * a - 4.0 * b * ef)) / (2.0 * b)

    def kf2_2(ef):
        return max((-d - ef) / a2, 0.0)

    ps = 0.02
    ef = brentq(lambda e: kf2_1(e) + kf2_2(e) - 4 * np.pi * ps,
                -0.2, -1e-9, xtol=1e-15)
    exact = np.array([kf2_1(ef), kf2_2(ef)]) / (4 * np.pi)
    assert exact[1] > 0.1 * ps                 # both branches occupied
    for nk in (11, 41, 161, 321):
        occ, _ = fill_subbands_kgrid(at_k, ps, 2, kmax=1.2, nk=nk,
                                     ntheta=1)
        err = np.abs(occ - exact).max()
        assert err * (nk - 1) ** 2 < 1e-2
    assert err < 2e-8                          # nk = 321: 1e-6 relative


# ---------------------------------------------------------------- 3

def test_kgrid_self_consistency_reproduces_parabolic_on_the_demo_set():
    """On the demo set every subband is an exact parabola of mass
    0.5 m0 and the envelopes do not change with k, so the k-resolved
    loop must land on the parabolic loop's state."""
    p = demo_single_band(A=-2.0)
    z = np.linspace(0.0, 6.0, 30)
    kw = dict(ps=0.08, n_states=6, mixing=0.6, tol=1e-9, max_iter=200)
    par = solve_self_consistent(p, z, **kw)
    kg = solve_self_consistent(p, z, filling="kgrid", kmax=1.2, nk=4,
                               **kw)
    assert par.converged and kg.converged
    assert kg.filling == "kgrid" and par.filling == "parabolic"
    assert np.abs(kg.potential - par.potential).max() < 1e-8
    assert np.abs(kg.occupations - par.occupations).max() < 1e-8
    assert np.isfinite(kg.fermi_level) and np.isnan(par.fermi_level)


def test_kgrid_self_consistency_closes_gauss_law_on_gan():
    """GaN hard-wall gas with the k-resolved filling: the returned
    potential is the Poisson solution of the returned density (to the
    loop tolerance), the density holds exactly ps, and the returned
    occupations equal a separate `fill_subbands_kgrid` at the returned
    potential, with one direction or three (isotropy)."""
    z = np.linspace(0.0, 6.0, 31)
    dz = z[1] - z[0]
    ps = 0.3
    r = solve_self_consistent(GAN, z, ps, n_states=4, mixing=0.5,
                              tol=1e-6, filling="kgrid", kmax=1.6, nk=9)
    assert r.converged
    assert abs(r.density.sum() * dz - ps) < 1e-12
    assert abs(r.occupations.sum() - ps) < 1e-12
    vh = hole_potential(z, r.density, ps, GAN.eps_r)
    assert np.abs(vh - r.potential).max() < 5e-6

    def at_k(kx, ky):
        return solve_subbands(GAN, z, kx=kx, ky=ky, potential=r.potential,
                              n_states=4)[0]

    for ntheta in (1, 3):
        occ, ef = fill_subbands_kgrid(at_k, ps, 4, kmax=1.6, nk=9,
                                      ntheta=ntheta)
        assert np.abs(occ - r.occupations).max() < 1e-12
        assert abs(ef - r.fermi_level) < 1e-12


# ---------------------------------------------------------------- 4

def test_per_point_permittivity_against_the_two_layer_closed_form():
    L, ps, a, e1, e2 = 4.0, 0.1, 1.5, 10.0, 5.0
    n0 = ps / L
    exact = (E2_OVER_EPS0 / e1 * (ps * a - n0 * a * a / 2)
             + E2_OVER_EPS0 / e2 * (ps * (L - a) - n0 * (L * L - a * a) / 2))
    errs = []
    for n in (401, 4001):
        z = np.linspace(0.0, L, n)
        d = np.full(n, n0)
        v = hole_potential(z, d, ps, np.where(z < a, e1, e2))
        errs.append(abs(v[-1] / exact - 1.0))
        # a uniform per-point profile is exactly the scalar result
        assert np.array_equal(hole_potential(z, d, ps, np.full(n, e1)),
                              hole_potential(z, d, ps, e1))
    assert errs[1] < 1e-3                       # first order in the step
    assert errs[1] < 0.15 * errs[0]


def test_sheet_position_against_the_closed_form():
    """Fixed sheet at z_s = 1 nm, holes in a uniform slab on [2, 3] nm:
    zero field left of the sheet, slope e^2 ps / eps up to the slab,
    flat beyond it, V(4) = (e^2 ps / eps)(1 + 1/2)."""
    ps, eps = 0.1, 10.0
    z = np.linspace(0.0, 4.0, 401)
    dz = z[1] - z[0]
    m = (z >= 2.0 - 1e-9) & (z <= 3.0 + 1e-9)
    d = np.where(m, ps / (m.sum() * dz), 0.0)
    v = hole_potential(z, d, ps, eps, sheet_z=1.0)
    assert abs(v[-1] - E2_OVER_EPS0 * ps / eps * 1.5) < 1e-12
    assert np.abs(v[z < 1.0 - 1e-9]).max() == 0.0
    # sheet_z at the first point is the historical default, bit for bit
    assert np.array_equal(hole_potential(z, d, ps, eps, sheet_z=0.0),
                          hole_potential(z, d, ps, eps))


def test_hetero_loop_with_the_sheet_at_the_interface():
    """A 5 nm AlN barrier left of a GaN well. With the default (sheet at
    z = 0) the full sheet field e^2 ps / eps (0.35 eV/nm here) crosses
    the barrier, the far side of the barrier ends up above the GaN for
    holes, and the loop does not converge. With sheet_z at the
    interface it converges with the gas in the GaN. The permittivity profile uses
    GaN's cited 10.4 everywhere (no vetted AlN value is shipped)."""
    z = np.linspace(0.0, 10.0, 51)
    dz = z[1] - z[0]
    # offset illustrative, not a cited GaN/AlN alignment
    params, edge = layered_profile(z, [(5.0, ALN, -0.8), (5.0, GAN, 0.0)])
    ps = 0.2
    eps = np.full(z.size, GAN.eps_r)
    barrier = z < 5.0 - 1e-9
    good = solve_self_consistent_hetero(z, params, edge, ps, eps,
                                        n_states=4, mixing=0.4, tol=1e-6,
                                        max_iter=200, sheet_z=5.0)
    assert good.converged
    assert good.density[barrier].sum() * dz < 0.05 * ps
    centroid = (z * good.density).sum() / good.density.sum()
    assert 5.0 < centroid < 7.0
    # left of the sheet only the few tunnelled holes make a field,
    # against 1.74 eV for the full sheet field across 5 nm
    drop = abs(good.potential[np.searchsorted(z, 5.0 - 1e-9)])
    assert drop < 0.05 * E2_OVER_EPS0 * ps * 5.0 / GAN.eps_r
    # the historical default on the same stack does not settle: the
    # iterations swing between a gas in the GaN and a gas in the AlN
    # (on a finer grid, 11 nm on 67 points, it ends with all holes in
    # the AlN after 200 iterations)
    bad = solve_self_consistent_hetero(z, params, edge, ps, eps,
                                       n_states=4, mixing=0.4, tol=1e-6,
                                       max_iter=60)
    assert not bad.converged


# ---------------------------------------------------------------- 5

def test_refusals():
    z = np.linspace(0.0, 6.0, 25)
    with pytest.raises(ValueError, match="one value per grid point"):
        solve_subbands(GAN, z, potential=np.zeros(z.size + 3))
    with pytest.raises(ValueError, match="increasing"):
        solve_subbands(GAN, z[::-1])
    with pytest.raises(ValueError, match="n_states"):
        solve_subbands(GAN, z, n_states=6 * z.size + 1)
    with pytest.raises(ValueError, match="n_states"):
        solve_subbands(GAN, z, n_states=0)
    with pytest.raises(ValueError, match="at least 3"):
        assemble_hamiltonian(GAN, [0.0, 1.0])
    with pytest.raises(ValueError, match="filling"):
        solve_self_consistent(GAN, z, 0.1, filling="exact")
    with pytest.raises(ValueError, match="kmax"):
        solve_self_consistent(GAN, z, 0.1, filling="kgrid")
    params = [GAN] * z.size
    with pytest.raises(ValueError, match="finite and positive"):
        solve_self_consistent_hetero(
            z, params, None, 0.1, [ALN.eps_r] * z.size)
    with pytest.raises(ValueError, match="grid span"):
        solve_self_consistent_hetero(z, params, None, 0.1, GAN.eps_r,
                                     sheet_z=7.0, max_iter=1)
    p = dataclasses.replace(demo_single_band(), D1=1.0, D2=1.0, D3=1.0,
                            D4=1.0, D5=1.0, D6=1.0)
    shear = np.zeros((3, 3))
    shear[0, 2] = shear[2, 0] = 1e-3
    with pytest.raises(ValueError, match="ntheta"):
        solve_self_consistent_hetero(z, [p] * z.size, None, 0.1, 10.0,
                                     strain_list=shear, filling="kgrid",
                                     kmax=1.0)
    with pytest.raises(ValueError, match="edge of the k-grid"):
        solve_self_consistent(demo_single_band(), z, 0.1, n_states=4,
                              filling="kgrid", kmax=0.2, nk=5)
