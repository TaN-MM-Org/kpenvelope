# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

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
