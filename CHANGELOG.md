# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

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
