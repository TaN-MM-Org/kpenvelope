"""v0.16 anchors.

Fixes (each test fails on v0.15.0; the docstrings give the old
numbers):

1. The envelope normalization could give different bits for the same
   eigenvector (NumPy 1.22 with AVX-512), which made a v0.15.0 test
   fail now and then in the oldest-dependencies CI job.
2. The parabolic filling of the finite-barrier loop used the mass
   along kx for every direction, also when a strain makes the
   subbands anisotropic in the plane; it is now refused there.
3. `dipole_matrix` read a single (6, N) state as six one-component
   states, and `oscillator_strengths` accepted a negative mass.

New, each against an independent calculation:

4. `velocity_matrix` and `inplane_transitions` (light polarized in the
   plane): the polynomial identity of the assembled matrix, the closed
   form of the demo set, finite differences of the solver's energies
   (Hellmann-Feynman, and the effective-mass sum rule of second-order
   perturbation theory), the in-plane isotropy of the subbands, the
   commutator form of the z velocity against `level_transitions`, and
   invariance under any unitary mixing inside each level.
"""
import dataclasses

import numpy as np
import pytest

from kpenvelope import (HBAR2_OVER_2M0 as C, aln_rinke2008,
                        assemble_hamiltonian, assemble_heterostructure,
                        demo_single_band, dipole_matrix, gan_rinke2008,
                        heterostructure_energies, inplane_transitions,
                        layered_profile, level_transitions,
                        oscillator_strengths, solve_heterostructure,
                        solve_self_consistent_hetero, solve_subbands,
                        subband_energies, velocity_matrix)
from kpenvelope.solver import envelopes_from_vectors, top_eigenpairs
from kpenvelope.velocity import velocity_derivative_matrix

GAN = gan_rinke2008()
ALN = aln_rinke2008()
# deformation potentials: arbitrary TEST numbers (not material
# constants), only there to switch the strain terms on (as in
# test_v015.py)
GAN_D = dataclasses.replace(GAN, D1=-3.7, D2=4.5, D3=8.2, D4=-4.1,
                            D5=-4.0, D6=-5.5)


# ---------------------------------------------------------------- 1

def test_normalization_does_not_depend_on_memory_layout():
    """The same eigenvectors, stored in three layouts (as the solver
    returns them, as the first columns of a wider array, and in Fortran
    order), must give bit-for-bit the same envelopes, each normalized
    to 1.

    v0.15.0 summed the squared norm from numpy.abs(v) ** 2. On NumPy
    1.22 with AVX-512 the complex absolute value runs either a vector
    loop or the scalar hypot, which differ in the last bit, depending
    on the stride of the input (the vector loop is used only below 256
    bytes) and on where the result is allocated. Here the wide layout
    has a row stride of 256 bytes, so it always took the scalar loop,
    and the Fortran layout always took the vector loop. With v0.15.0 on
    such a machine, state 3 came out of the two scaled differently by
    about 2e-16 (relative), so at least one comparison below failed on
    every run. The plain layout took one loop or the other depending on
    where the result was allocated: that was the intermittent failure
    of test_kgrid_loop_results_unchanged_by_the_extra_state. The
    squares of the real and imaginary parts give the same bits on every
    path.
    """
    z = np.linspace(0.0, 5.0, 31)
    n, dz = z.size, z[1] - z[0]
    kx, ky = 0.3, 0.1              # complex states (finite momentum)
    H = assemble_hamiltonian(GAN, z, kx, ky, 0.1 * z)
    _, vecs = top_eigenpairs(H, n, 4, kx, ky)
    assert np.abs(vecs.imag).max() > 0.1   # the case that matters
    ref = envelopes_from_vectors(vecs, n, dz)
    wide = np.zeros((6 * n, 16), dtype=complex)
    wide[:, :4] = vecs
    env_wide = envelopes_from_vectors(wide[:, :4], n, dz)
    env_f = envelopes_from_vectors(np.asfortranarray(vecs), n, dz)
    assert np.array_equal(env_wide, env_f)
    assert np.array_equal(env_wide, ref) and np.array_equal(env_f, ref)
    norms = (np.abs(ref) ** 2).sum(axis=(1, 2)) * dz
    assert np.abs(norms - 1.0).max() < 1e-14


# ---------------------------------------------------------------- 2

def test_parabolic_filling_refuses_an_anisotropic_strain():
    """With eps_xx != eps_yy (or a shear) the subbands have different
    masses along kx and ky, but the parabolic filling measures one mass
    along kx. v0.15.0 used it for every direction without a warning:
    for GaN with the test deformation potentials above, a 6 nm stack on
    41 points, strain diag(0.002, -0.002, 0), ps = 0.1 nm^-2, eps_r =
    10.4 and 6 states, it returned occupations [0, 0.0823, 0, 0.0030,
    0.0124, 0.0023] nm^-2 from kx masses [inf, 0.787, inf, 0.088,
    5.05, 0.948] m0, while the masses along ky at the same potential
    are [3.01, 0.126, inf, 0.558, inf, 0.152] m0 and the k-grid filling
    on 16 directions gives [0.0517, 0.0483, 0, 0, 0, 0] nm^-2. Now it
    is refused, as the k-grid filling without ntheta already was. An
    in-plane isotropic (biaxial) strain is still accepted."""
    z = np.linspace(0.0, 6.0, 21)
    pl = [GAN_D] * z.size
    for strain in (np.diag([0.002, -0.002, 0.0]),
                   np.array([[0.0, 0.003, 0.0], [0.003, 0.0, 0.0],
                             [0.0, 0.0, 0.0]]),
                   [np.diag([0.002, 0.002, -0.001])] * 20
                   + [np.diag([0.002, 0.0, -0.001])]):
        with pytest.raises(ValueError, match="in-plane isotropy"):
            solve_self_consistent_hetero(z, pl, None, 0.1, 10.4,
                                         n_states=4, max_iter=1,
                                         strain_list=strain)
    r = solve_self_consistent_hetero(z, pl, None, 0.05, 10.4, n_states=4,
                                     max_iter=2, mixing=0.5,
                                     strain_list=np.diag([0.002, 0.002,
                                                          -0.001]))
    assert abs(r.occupations.sum() - 0.05) < 1e-12


# ---------------------------------------------------------------- 3

def test_dipole_and_strength_inputs_are_checked():
    """v0.15.0 returned a 6 x 6 matrix for dipole_matrix(z, F[0]) (the
    six components of one state read as six states), and strengths of
    -1.31 for a mass ratio of -0.5 where 0.5 gives +1.31. Both are
    refused now; the normal use is unchanged (here against the formula
    written out)."""
    z = np.linspace(0.0, 8.0, 41)
    E, F = solve_subbands(demo_single_band(), z, n_states=12)
    with pytest.raises(ValueError, match="n_states, n_components, N"):
        dipole_matrix(z, F[0])
    with pytest.raises(ValueError, match="n_states, n_components, N"):
        dipole_matrix(z[:-1], F)
    d = dipole_matrix(z, F)
    assert d.shape == (12, 12)
    with pytest.raises(ValueError, match="mass_ratio"):
        oscillator_strengths(E, d, -0.5)
    with pytest.raises(ValueError, match="mass_ratio"):
        oscillator_strengths(E, d, float("nan"))
    with pytest.raises(ValueError, match="dipole matrix"):
        oscillator_strengths(E[:6], d, 0.5)
    f = oscillator_strengths(E, d, 0.5)
    ref = 0.5 * (E[0] - E) * np.abs(d[0]) ** 2 / C
    ref[0] = 0.0
    assert np.array_equal(f, ref)


# ---------------------------------------------------------------- 4

def _stack():
    z = np.linspace(0.0, 6.0, 31)
    pl, edge = layered_profile(z, [(2.0, ALN, -0.5), (2.0, GAN, 0.0),
                                   (2.0, ALN, -0.5)])
    return z, pl, edge


def test_velocity_operator_is_the_exact_derivative_of_the_matrix():
    """The assembled matrix is a polynomial of second degree in (kx, ky)
    (its third difference along any direction vanishes to rounding), so
    dH/dk_e = [H(k + e) - H(k - e)] / 2 exactly. The derivative matrix,
    built from differentiated 6 x 6 blocks, equals that difference of
    the package's own assembly (hard walls; a layered stack with a band
    edge, a potential and a strain, none of which depend on k_t) to
    1e-15 eV nm, and is Hermitian."""
    z, pl, edge = _stack()
    biax = np.diag([0.01, 0.01, -0.005])
    for kx, ky, ang in ((0.0, 0.0, 0.0), (0.3, 0.1, 0.7),
                        (-0.4, 0.25, 2.0)):
        ex, ey = np.cos(ang), np.sin(ang)

        def h_uni(t):
            return assemble_hamiltonian(GAN_D, z, kx + t * ex, ky + t * ey,
                                        potential=0.1 * z, strain=biax)

        def h_het(t):
            return assemble_heterostructure(
                z, pl, edge, kx + t * ex, ky + t * ey, potential=0.1 * z)

        for h, par in ((h_uni, GAN_D), (h_het, pl)):
            third = h(2.0) - 3.0 * h(1.0) + 3.0 * h(0.0) - h(-1.0)
            assert np.abs(third).max() < 1e-13 * np.abs(h(0.0)).max()
            D = velocity_derivative_matrix(z, par, kx, ky, ang)
            assert np.abs(D - 0.5 * (h(1.0) - h(-1.0))).max() < 1e-15
            assert np.array_equal(D, D.conj().T)


def test_velocity_matrix_of_the_demo_set_is_the_closed_form():
    """Demo set (six uncoupled bands, E = c A k^2): dH/dk_e = 2 c A k_e
    times the identity, so the velocity matrix between the normalized
    states is 2 c A k_e times the identity in any basis (to 1e-12), and
    in-plane polarized light drives no intersubband transition (every
    level strength is 0 to 1e-24)."""
    A = -2.0
    p = demo_single_band(A=A)
    z = np.linspace(0.0, 8.0, 41)
    for kx, ky, ang in ((0.0, 0.0, 0.0), (0.3, -0.2, 0.5),
                        (0.1, 0.4, 2.2)):
        E, F = solve_subbands(p, z, kx, ky, n_states=18)
        V = velocity_matrix(z, p, F, kx, ky, ang)
        ke = kx * np.cos(ang) + ky * np.sin(ang)
        assert np.abs(V - 2 * C * A * ke * np.eye(18)).max() < 1e-12
        t = inplane_transitions(z, p, E, F, kx, ky, ang)
        assert np.abs(t["v2"]).max() < 1e-24
        assert [c.size for c in t["levels"]] == [6, 6, 6]


def _richardson(f, h):
    """Central first and second differences, Richardson extrapolated."""
    def d1(s):
        return (f(s) - f(-s)) / (2 * s)

    def d2(s):
        return (f(s) + f(-s) - 2 * f(0.0)) / (s * s)
    return ((4 * d1(h / 2) - d1(h)) / 3, (4 * d2(h / 2) - d2(h)) / 3)


def test_velocity_diagonal_and_mass_sum_rule_against_finite_differences():
    """For each nondegenerate state n of the discrete matrix,
    dE_n/dk_e = V_nn (Hellmann-Feynman) and
    d^2 E_n/dk_e^2 = <n| d^2H/dk_e^2 |n> + 2 sum_{m != n} |V_mn|^2 /
    (E_n - E_m) (the effective-mass sum rule of second-order
    perturbation theory). Both are checked for the top 8 states (all
    6 N states enter the sum) against Richardson-extrapolated central
    differences of the solver's own energies along e: GaN in a tilted
    well at k_t = 0.32 nm^-1 (hard walls), and an AlN/GaN/AlN stack.
    The first derivative agrees to 1e-9 eV nm, the curvature to 1e-6
    of its largest value."""
    z = np.linspace(0.0, 5.0, 15)
    zs = np.linspace(0.0, 6.0, 16)
    pls, edges = layered_profile(zs, [(2.0, ALN, -0.5), (2.0, GAN, 0.0),
                                      (2.0, ALN, -0.5)])
    cases = [
        (z, GAN, lambda kx, ky, n: solve_subbands(
            GAN, z, kx, ky, potential=0.1 * z, n_states=n),
         lambda kx, ky, n: subband_energies(
            GAN, z, kx, ky, potential=0.1 * z, n_states=n),
         lambda kx, ky: assemble_hamiltonian(GAN, z, kx, ky, 0.1 * z)),
        (zs, pls, lambda kx, ky, n: solve_heterostructure(
            zs, pls, edges, kx, ky, potential=0.05 * zs, n_states=n),
         lambda kx, ky, n: heterostructure_energies(
            zs, pls, edges, kx, ky, potential=0.05 * zs, n_states=n),
         lambda kx, ky: assemble_heterostructure(
            zs, pls, edges, kx, ky, potential=0.05 * zs)),
    ]
    kx, ky, ang = 0.3, 0.1, 0.4
    ex, ey = np.cos(ang), np.sin(ang)
    for zz, par, solve, energies, assemble in cases:
        n = zz.size
        dz = zz[1] - zz[0]
        E, F = solve(kx, ky, 6 * n)
        assert np.min(np.abs(E[:8, None] - E[None, :])
                      + 1e9 * np.eye(6 * n)[:8]) > 1e-6
        V = velocity_matrix(zz, par, F, kx, ky, ang)
        g_fd, c_fd = _richardson(
            lambda t: energies(kx + t * ex, ky + t * ey, 8), 1e-3)
        assert np.abs(np.real(np.diag(V))[:8] - g_fd).max() < 1e-9
        Hee = (assemble(kx + ex, ky + ey) + assemble(kx - ex, ky - ey)
               - 2 * assemble(kx, ky))
        X = F.transpose(0, 2, 1).reshape(F.shape[0], -1).T
        second = np.real(np.einsum("ij,ij->j", X.conj(), Hee @ X)) * dz
        rule = np.empty(8)
        for m in range(8):
            gap = E[m] - E
            gap[m] = np.inf
            rule[m] = second[m] + 2 * np.sum(np.abs(V[:, m]) ** 2 / gap)
        assert np.abs(rule - c_fd).max() < 1e-6 * np.abs(c_fd).max()


def test_inplane_strengths_obey_the_f_sum_rule():
    """Summed over every final level, the in-plane strengths from a
    level obey the sum rule of second-order perturbation theory,
    sum_L f_L = (d^2E/dk_e^2 - <d^2H/dk_e^2>) / (2 hbar^2/2m0), with E
    and <d^2H/dk_e^2> the energy and the mean over the initial level
    (here the top Kramers pair of a symmetric GaN well, which stays
    degenerate at every k_t, so its energy is smooth). The curvature is
    a Richardson-extrapolated central difference of the solver's
    energies (to 1e-6 relative); all 6 N states are included. For one
    parabolic band both sides are 0 (previous test)."""
    z = np.linspace(0.0, 5.0, 15)
    n, dz = z.size, z[1] - z[0]
    E, F = solve_subbands(GAN, z, n_states=6 * n)
    for ang in (0.0, 0.9):
        ex, ey = np.cos(ang), np.sin(ang)
        t = inplane_transitions(z, GAN, E, F, angle=ang)
        assert t["degeneracy"][0] == 2
        _, curv = _richardson(lambda s: subband_energies(
            GAN, z, s * ex, s * ey, n_states=2)[0], 1e-3)
        Hee = (assemble_hamiltonian(GAN, z, ex, ey)
               + assemble_hamiltonian(GAN, z, -ex, -ey)
               - 2 * assemble_hamiltonian(GAN, z))
        X = F[:2].transpose(0, 2, 1).reshape(2, -1).T
        second = np.real(np.einsum("ij,ij->j", X.conj(), Hee @ X)).mean()
        rhs = (curv - second * dz) / (2 * C)
        total = t["oscillator_strength"].sum()
        assert abs(total - rhs) < 1e-6 * abs(rhs)
        assert abs(rhs) > 1e-3


def test_inplane_level_sums_symmetries_and_z_consistency():
    """Tilted 6 nm GaN well of README example 12 (61 points).

    * At k_t = 0 the level sums do not depend on the polarization angle
      (the subbands are isotropic in the plane), to 1e-12 relative.
    * Rotating k_t and the polarization by the same angle leaves every
      level sum unchanged (1e-9 relative).
    * Mixing the states inside each Kramers pair at random leaves every
      level sum unchanged (1e-10 relative).
    * At k_t = 0, HH1 -> HH2 is forbidden in the plane (both states lie
      only on heavy-hole components, which dH/dk_e at k_t = 0 does not
      connect): 0, while it is the one line light polarized along z
      sees.
    * With the z velocity written as the commutator hbar v_z = i [H, z]
      of the same matrix, the same level sums and the same definition
      of the strength give `level_transitions(mass_ratio=1)` (1e-10).
    """
    z = np.linspace(0.0, 6.0, 61)
    v = 0.08 * z
    E, F = solve_subbands(GAN, z, potential=v, n_states=8)
    ref = inplane_transitions(z, GAN, E, F)
    assert np.all(ref["degeneracy"] == 2)
    big = np.abs(ref["v2"]).max()
    assert big > 1e-6
    for ang in (0.4, np.pi / 2, 2.5, -1.0):
        t = inplane_transitions(z, GAN, E, F, angle=ang)
        assert np.abs(t["v2"] - ref["v2"]).max() < 1e-12 * big
    # per hole: by time reversal each member of the initial pair gives
    # the same sum over a complete final pair, so the level value is
    # the sum from either member alone (1e-12 relative)
    V = velocity_matrix(z, GAN, F, angle=0.3)
    for L, c in enumerate(ref["levels"][1:], start=1):
        s0 = (np.abs(V[c, 0]) ** 2).sum()
        s1 = (np.abs(V[c, 1]) ** 2).sum()
        assert abs(s0 - s1) <= 1e-12 * big
        assert abs(ref["v2"][L] - s0) <= 1e-12 * big
    # HH1 -> HH2 (level 2): forbidden in the plane, allowed along z
    lt = level_transitions(z, E, F, mass_ratio=1.0)
    assert ref["v2"][2] < 1e-25 * big + 1e-30
    assert lt["oscillator_strength"][2] > 0.3
    # common rotation of k_t and polarization at finite momentum
    base = None
    for phi in (0.0, 1.1, 2.9, -2.0):
        kx, ky = 0.3 * np.cos(phi), 0.3 * np.sin(phi)
        Ek, Fk = solve_subbands(GAN, z, kx, ky, potential=v, n_states=6)
        t = inplane_transitions(z, GAN, Ek, Fk, kx, ky, angle=phi + 0.2)
        if base is None:
            base = t["v2"]
        assert np.abs(t["v2"] - base).max() < 1e-9 * np.abs(base).max()
    # unitary mixing inside each Kramers pair at k_t = 0
    rng = np.random.default_rng(7)
    Fm = F.copy()
    for c in ref["levels"]:
        q, _ = np.linalg.qr(rng.normal(size=(2, 2))
                            + 1j * rng.normal(size=(2, 2)))
        Fm[c] = np.einsum("ab,bcz->acz", q.T, F[c])
    for ang in (0.0, 1.0):
        a = inplane_transitions(z, GAN, E, F, angle=ang)["v2"]
        b = inplane_transitions(z, GAN, E, Fm, angle=ang)["v2"]
        assert np.abs(a - b).max() < 1e-10 * big
    # z polarization through the commutator, same definitions
    H = assemble_hamiltonian(GAN, z, 0.0, 0.0, v)
    Z = np.repeat(z, 6)
    dz = z[1] - z[0]
    X = F.transpose(0, 2, 1).reshape(F.shape[0], -1).T
    Vz = 1j * (X.conj().T @ ((H * Z[None, :] - Z[:, None] * H) @ X)) * dz
    for L, c in enumerate(lt["levels"]):
        if L == 0:
            continue
        v2 = (np.abs(Vz[np.ix_(c, lt["levels"][0])]) ** 2).sum() / 2
        f = v2 / (C * lt["transition_energy"][L])
        assert abs(f - lt["oscillator_strength"][L]) < 1e-10


def test_inplane_refusals():
    z = np.linspace(0.0, 5.0, 21)
    E, F = solve_subbands(GAN, z, n_states=4)
    with pytest.raises(ValueError, match="shape"):
        velocity_matrix(z, GAN, F[0])
    with pytest.raises(ValueError, match="shape"):
        velocity_matrix(z[:-1], GAN, F)
    with pytest.raises(ValueError, match="one parameter set per grid"):
        velocity_matrix(z, [GAN] * 20, F)
    with pytest.raises(ValueError, match="WurtziteParameters"):
        velocity_matrix(z, [None] * 21, F)
    with pytest.raises(ValueError, match="angle"):
        velocity_matrix(z, GAN, F, angle=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        velocity_matrix(z, GAN, F, kx=float("inf"))
    with pytest.raises(ValueError, match="initial"):
        inplane_transitions(z, GAN, E, F, initial=2)
    with pytest.raises(ValueError, match="one state per energy"):
        inplane_transitions(z, GAN, E[:3], F)
