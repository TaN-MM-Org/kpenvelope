# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

## v0.12.0 - 2026-09-30

A fix to the Hamiltonian for in-plane momenta off the kx axis, a
momentum-grid filling that is exact for parabolic subbands, a
self-consistent loop that fills the full dispersion, and electrostatics
for layered stacks (permittivity profile, position of the fixed sheet).

### Fixed

- **Six-band Hamiltonian, lower 3 x 3 block.** The term linear in kz,
  H = (hbar^2/2m0) A6 k+ kz, appeared in rows 4-6 as H* at (4,6) and
  -H at (5,6), where the Chuang-Chang matrix has H and -H*. Along kx
  (ky = 0) H is real and both forms are identical, which is why every
  earlier check (all of them along kx) passed. In any other in-plane
  direction the bulk bands lost their two-fold (Kramers) degeneracy
  and depended on the direction. The Bir-Pikus strain matrix copied
  the same pattern, so a strain with a non-zero eps_yz also split the
  bulk levels (eps_xx != eps_yy, eps_xy and eps_xz alone did not).
  Both are corrected. The bulk eigenvalues
  now equal, each twice, those of the block-diagonal 3 x 3 form of
  Chuang and Chang (PRB 54, 2491 (1996), Eq. (45)), whose entries
  depend only on |k_t|: agreement to 1e-12 eV at random directions for
  the GaN and AlN sets. With the old matrix the largest deviation was
  0.327 eV for GaN and 0.191 eV for AlN over the test's 40 random
  samples (seed 12, k_t and k_z uniform in [0, 1.5] nm^-1), and
  0.381 eV for GaN on a grid of k_t, k_z = 0, 0.05, ..., 1.5 nm^-1 and
  19 directions from 0 to 90 degrees (at k_t = 1.5, k_z = 1.1 nm^-1,
  45 degrees).
- `fill_subbands_kgrid` integrated the step-like zero-temperature
  occupation with the trapezoid rule, a first-order method. On the
  exactly parabolic demo well of the new test (A = -2, 41-point grid
  from 0 to 8 nm, 12 states, ps = 0.3 nm^-2, kmax = 2.6 nm^-1,
  ntheta = 1, T = 0) the old per-state error was 4.17e-2 nm^-2 with
  nk = 5 (the six second-level states left empty), 7.57e-3 nm^-2
  (18 %) with nk = 23 and 1.76e-3 nm^-2 with nk = 41 (21 % of each
  second-level state, 4 % of each first-level state). The error depends
  on where the Fermi wavevector falls between grid points, so it also
  changes with kmax. It now treats each branch as linear in k_t^2
  between grid points and integrates the Fermi-Dirac factor exactly on
  each segment: exact for parabolic subbands at any grid spacing, and
  second order in the step otherwise.
- Inputs that gave silent wrong answers are now refused with a
  `ValueError`: a `potential` longer than the grid (it was silently
  truncated; a shorter one raised `IndexError`), a decreasing grid (it
  gave NaN envelopes), `n_states` above 6 N (fewer states were returned
  without notice) or below 1, a grid of fewer than three points, and a
  non-positive or non-finite `eps_r` in the Poisson step.

### Added

- `filling="kgrid"` on `solve_self_consistent` and
  `solve_self_consistent_hetero` (with `kmax`, `nk`, `ntheta`): the
  occupations and the hole density come from the states on a polar
  momentum grid, each state's |F(z)|^2 weighted by its share of the
  occupied momentum area. This removes the parabolic single-step mass
  from the loop, and with it the artefact described under v0.11.1
  (one Kramers partner with an infinite mass and no holes). On README
  example 2 the partners now hold 1.974/1.964 and 0.352/0.311 e13
  cm^-2 (pairs 3.938 and 0.662); the remaining difference within a
  pair is the spin splitting of the two branches at finite momentum in
  this asymmetric well, not an artefact. `ntheta` defaults to 1, exact because the
  subbands depend only on |k_t| (asserted); a strain that breaks
  in-plane isotropy requires an explicit `ntheta`.
- `SelfConsistentResult.filling` and `.fermi_level` (the latter NaN for
  the parabolic filling).
- `solve_self_consistent_hetero(..., sheet_z=...)` and
  `poisson.hole_potential(..., sheet_z=...)`: the position of the fixed
  negative sheet charge. The default (None) keeps the historical
  choice, the first grid point.
- `eps_r` of `solve_self_consistent_hetero` and of
  `poisson.hole_potential` may be one value per grid point: Gauss's
  law is applied to the displacement field, so each layer carries its
  own electric field.

### Behaviour changes

Results with ky = 0 and eps_yz = 0 are unchanged bit for bit. This
covers README examples 1 to 7, every self-consistent result with the
default parabolic filling (its mass step is along kx) and the
calibration tools (k = 0). What changes:

- Symmetric 5 nm hard-wall GaN well (51 points), k_t = 0.4 nm^-1 at
  45 degrees: top four energies 5.245, 1.687, -2.217, -19.199 meV
  before, 0.880, 0.880, -18.904, -18.904 meV now (the kx values);
  spin splittings 3.558 and 16.983 meV before, 0 now. Along ky: 0.859
  and -19.103 meV before (pairs degenerate), 0.880 and -18.904 meV now.
- `character_vs_k` in that well along ky at 0.4 nm^-1: HH/LH/CH
  0.6781/0.3159/0.0061 before, 0.6786/0.3149/0.0065 now.
- `fill_subbands_kgrid` at the converged potential of README example
  2 (defaults nk = 16, ntheta = 4): with kmax = 1.6 nm^-1 the old code
  refused (its wrong ky-direction energies reached the grid edge);
  with kmax = 2.5 nm^-1 it gave 1.9714, 1.9714, 0.3943, 0.2629 e13
  cm^-2 per state (pairs 3.943 and 0.657), now 1.9659, 1.9552, 0.3590,
  0.3199 (pairs 3.921 and 0.679). ntheta = 1 now gives the same
  numbers as ntheta = 4.
- The inputs listed under Fixed now raise instead of returning.

### Tests

- New file `tests/test_v012.py`, 14 tests: bulk eigenvalues against
  the Chuang-Chang block-diagonal form (1e-12 eV); Kramers degeneracy
  of the bulk bands with random strain and k (1e-12 eV); invariance
  under a rotation of k and strain about c (1e-12 eV); subbands of an
  asymmetric GaN/AlN stack independent of the in-plane direction
  (1e-10 eV); no spin splitting in a symmetric well along any direction
  (1e-9 eV); k-grid filling equal to the closed-form parabolic filler
  at 5 and 23 grid points, T = 0 and 150 K (1e-9 nm^-2); second-order
  error bound err (nk - 1)^2 < 1e-2 nm^-2 on a non-parabolic
  two-branch dispersion against an independent root finder; the
  k-resolved loop against the parabolic loop on the demo set (1e-8)
  and Gauss's law on GaN (5e-6 eV); two-layer permittivity and
  sheet-position closed forms; the sheet at the interface of a
  5 nm AlN / 5 nm GaN stack; refusals. 83 tests in total.
- All eight symmetry and filling tests fail on v0.11.1.

### Known limitation, found and not changed

- The default of `sheet_z` is still the first grid point, for backward
  compatibility. On a stack whose grid starts inside a barrier this
  puts the whole sheet field across the barrier (1.74 eV across 5 nm
  AlN at 2e13 cm^-2 with eps_r = 10.4), and in the README's example 9
  the loop then does not converge. Pass `sheet_z`.

## v0.11.1 - 2026-09-22

Bug fixes, dependency and CI checks, and a README rewrite.

### Fixed

- `rashba_spins` with `alpha_evnm = 0` returned spins along z
  ((0, 0, 1) and (0, 0, -1)), contradicting its documented in-plane
  texture. With alpha = 0 the two branches are degenerate at every k
  and no direction is defined; it now raises `ValueError`.
- `solve_self_consistent` and `solve_self_consistent_hetero` crashed
  with `UnboundLocalError` for `max_iter=0`; both now raise
  `ValueError("max_iter must be at least 1")`.
- `fit_band_offset` docstring: the suggested offset shape (+1.0 on
  barrier points) made the barriers attract the holes in this
  package's valence-electron convention. It now says -1.0 on barrier
  points. No code change.
- Docstrings corrected: `_inplane_masses` (a stray factor 2 in the
  formula; the code was right), `units` (the cm^-2 round trip can
  differ in the last bit; it is not always exact), `isb` (the
  geometry-integral anchor uses analytic well functions, not solver
  envelopes, and the width-scaling test uses a 1e-3 tolerance).

### Tests

- New: `test_zero_alpha_spin_direction_refused` (test_rashba.py) and
  `test_max_iter_below_one_refused` (test_thermal.py). 69 tests.
- `test_lineshape_sum_rule_and_peaks` used `numpy.trapezoid`, which
  needs NumPy >= 2.0; it now falls back to `numpy.trapz`. The whole
  suite passes on Python 3.10 with NumPy 1.22.0 and SciPy 1.8.0, the
  oldest versions pyproject.toml allows.

### Changed

- CI: Python 3.10 added to the matrix (3.9 to 3.14), and a new
  `oldest-dependencies` job (Python 3.10, NumPy 1.22.0, SciPy 1.8.0).
- README rewritten for non-specialists: runnable examples with their
  real output, a list of refusals, and the checks with the tolerances
  the tests actually use.

### Known limitation, now documented (no code change)

- The parabolic filling inside `solve_self_consistent` and
  `solve_self_consistent_hetero` takes each state's mass from one
  momentum step (0.02 nm^-1). In an asymmetric well the two states of
  a Kramers pair split linearly in momentum, so they get different
  masses (one can be `inf`) and different occupations, although they
  should fill equally. In README example 2 (hard-wall GaN,
  4.6e13 cm^-2) the second pair gets `inf` / 0.144 m0 and
  0 / 0.605e13 cm^-2. Averaging over each pair would restore equal
  filling but, at the same potential, puts about 2.2e13 cm^-2 into
  the second pair against about 0.7e13 cm^-2 from
  `fill_subbands_kgrid`, because the parabolic model itself misses how
  the top subband gets heavier away from k = 0. The results are
  therefore left unchanged in this patch release; the README (example
  2 and Limits) now describes this, notes that the pair totals also
  depend on the step size, and points to `fill_subbands_kgrid` for
  occupations.

### Corrections to earlier notes

- v0.7.0: the parity selection rule is asserted to 1e-9 nm, not 1e-12.
- v0.8.0: the closed-form finite-temperature filling is checked
  against quadrature as a formula in the test, not by calling
  `fill_subbands_thermal`; the cm^-2 conversions are single
  multiplications and their round trip is not exact for every value.
- v0.9.0: the geometry integral is checked on analytic infinite-well
  functions sampled on the grid (not solver envelopes), and the width
  scaling to a relative 1e-3, not exactly.
- v0.10.0: "every refusal pinned" is too strong; the refusal of an
  offset-insensitive transition and of a non-positive `sigma_e_ev`
  have no test.
- The v0.11.0 README said the tests run on "Python 3.9-3.14", but the
  CI matrix has skipped Python 3.10 since v0.6.0 (whose note lists
  3.9, 3.11, 3.12 and 3.13); 3.10 is tested from this release on.

## v0.11.0 - 2026-09-18

A stated limit overcome, and a future-proofing pass.

- `rashba.rashba_hamiltonian` / `rashba_splitting` / `rashba_spins`:
  the k-linear Rashba term of wurtzite structures,
  alpha c_hat . (sigma x k) (Stefanowicz et al., PRB 89, 205201
  (2014)), for the conduction-band companion problem -- with NO
  default coefficient, on purpose: your alpha arrives with its
  citation, the same rule as every constant here. The valence-band
  k-linear terms remain deliberately not shipped (no vetted
  coefficients in our sources).
- CI now also runs on Python 3.14.
- Anchors: the splitting is exactly 2 alpha k by two code paths
  (closed form and full diagonalisation); the chiral spin texture is
  exact (in-plane, perpendicular to k, unit length, opposite
  branches); Kramers degeneracy at k = 0 exact and its undefined
  direction refused; units internally consistent to 1e-19 eV;
  missing references refused.

## v0.10.0 - 2026-09-17

Lab adaptability: calibrate, from your own measured resonance, the
two numbers this package refuses to ship.

- `lab.fit_band_offset`: the band-offset scale that makes the
  calculated subband spacing match a measured resonance, by
  bracketing bisection through the package's own
  `solve_heterostructure`, with the error bar from the exact
  sensitivity dE/d(offset); refuses a bracket that does not straddle
  the measurement (naming the calculated values at both ends) and a
  transition insensitive to the offset. Kramers degeneracy at k = 0
  is documented and defaulted around (states (0, 2)).
- `lab.offset_sensitivity`: that sensitivity on its own, so the
  refusal can be anticipated before the measurement.
- `lab.sheet_density_from_shift`: the exact closed-form inverse of
  `depolarization_shift` (the relation is exactly linear in n_s),
  with exact error propagation and a refusal of a resonance at or
  below the bare spacing.
- Anchors: the offset round-trips through the public solver to 1e-6
  and its error bar matches an actual re-fit at one sigma; the
  density round trip is exact to 1e-10 with linearity in n_s exact to
  1e-12 and the derivative checked by finite differences; the exact
  k = 0 Kramers degeneracy asserted; every refusal pinned.

## v0.9.0 - 2026-09-13

Physics upgrade: the collective intersubband physics between the band
structure and the spectrometer.

- `depolarization_shift`: the measured intersubband resonance of a
  doped well sits ABOVE the subband spacing by the depolarization
  shift, E_tilde = E sqrt(1 + alpha) with alpha = 2 e^2 n_s S /
  (eps0 eps_r E) and S the envelope geometry integral (Allen, Tsui
  and Vinter, Solid State Commun. 20, 425 (1976); Ando, Fowler and
  Stern, Rev. Mod. Phys. 54, 437 (1982)) -- the correction every
  intersubband absorption experiment must apply before comparing
  with a band-structure calculation. Spinor-summed overlap density,
  so it applies to any solver state here. The excitonic final-state
  correction is deliberately omitted, stated with the reason (it
  needs an exchange-correlation model this package does not ship).
- `overlap_geometry_integral`: the geometry integral S on its own.
- `isb_lineshape`: oscillator-strength-weighted unit-area Lorentzian
  absorption shape at your measured linewidth; the absolute 2D
  absorbance prefactor needs experiment geometry and is deliberately
  not guessed.
- Anchors: S from solver envelopes against independent adaptive
  quadrature over the analytic infinite-well wavefunctions (two code
  paths); exact origin invariance of S (orthogonality); exact linear
  width scaling; alpha exactly linear in n_s and inversely
  proportional to eps_r, zero-density limit exact; the defining
  identity E_shifted = E sqrt(1 + alpha) at machine precision; the
  lineshape integrating to the strength sum by quadrature.

## v0.8.0 - 2026-09-12

Experimental-conditions release: the two knobs every measurement
actually has -- a temperature and lab units -- plus an honest
convergence report.

- Finite-temperature subband filling: `fill_subbands_thermal` (the
  closed form n_i = dos_i kT ln(1 + exp((E_i - E_F)/kT)) for 2D
  parabolic subbands, anchored against direct numerical integration
  of the Fermi-Dirac occupation and against the T = 0 filler in the
  cold limit), and a `temperature_K` parameter on
  `solve_self_consistent`, `solve_self_consistent_hetero` and
  `fill_subbands_kgrid` (Fermi-Dirac occupation factor on the
  k-grid, agreeing with the closed-form filler on an exactly
  parabolic model). The default 0 reproduces the historical cold
  filling exactly -- the same code path, asserted bit for bit. The
  Boltzmann constant in eV/K is computed from the two exact SI
  defining constants, not typed by hand (`KB_EV_PER_K`).
- Lab units in and out: `sheet_density_from_cm2` /
  `sheet_density_to_cm2` (exact powers of ten; the single most
  common way to be wrong by orders of magnitude when driving the
  solver from measured numbers).
- `SelfConsistentResult.converged`: an iteration-starved run now
  reports its failure instead of hiding it in a residual the caller
  must remember to inspect.
- The k-grid filler's undersized-window refusal extends to finite
  temperature (Fermi tail reaching the grid edge).
- README rewritten: organized by what the package does rather than
  by release history, in plainer language, same facts.

## v0.7.0 - 2026-09-10

- Intersubband optics: `dipole_matrix` (six-component dipole matrix
  elements <f|z|i> on the solver's own grid, Hermitian by
  construction) and `oscillator_strengths` (ground-subband oscillator
  strengths in the package's hole convention).
- Anchors: infinite-well closed forms z12 = 16L/(9 pi^2) and
  f12 = 256/(27 pi^2) on the hard-wall demo set (multiplet-summed,
  since the demo set is six-fold degenerate and only multiplet sums
  are basis-invariant); Thomas-Reiche-Kuhn f-sum rule to a fraction
  of a percent; parity selection rule at 1e-12; coordinate-origin
  gauge invariance of off-diagonal elements on the coupled Rinke 2008
  GaN well.

## v0.6.0 - 2026-09-05

- Bir-Pikus strain terms (`strain_blocks`, `strain=` on every
  assembly): validated by the exact structural identity with the
  kinetic template and closed-form k = 0 eigenvalues; deformation
  potentials must be supplied with a citation (none shipped).
- Full k-grid non-parabolic filling (`fill_subbands_kgrid`), agreeing
  with the closed-form parabolic filler on a parabolic model and
  refusing an undersized k-window.
- Spin-splitting analysis (`spin_splitting`, `splitting_vs_k`):
  exact zeros under Kramers/inversion symmetry, Rashba-type splitting
  under asymmetric confinement.
- Finite-barrier self-consistent loop
  (`solve_self_consistent_hetero`), exactly reproducing the hard-wall
  loop on a uniform stack.
- Transport groundwork: `group_velocity` and `dos_from_dispersion`
  with parabolic closed-form anchors; mechanism lifetimes stay
  designed out (no uncited scattering parameters).
- CI now tests Python 3.9, 3.11, 3.12 and 3.13.

## v0.5.0

- Finite barriers: position-dependent materials and band edges with
  the symmetrized Ben Daniel-Duke assembly; finite-square-well and
  decay-constant closed-form anchors.

## v0.4.0 and earlier

- Six-band wurtzite envelope solver, self-consistent Poisson loop,
  cited GaN/AlN parameter sets, dispersion and band-character
  utilities. See the GitHub releases for the per-version anchors.
