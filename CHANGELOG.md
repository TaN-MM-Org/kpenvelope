# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

## 0.15.0 (2026-10-18)

Exact parameter sensitivities with error propagation, intersubband
numbers for whole levels (which, unlike numbers taken from one member
of a Kramers pair, do not depend on how the solver picks the pair),
solves at finite in-plane momentum about twice as fast, a third
tutorial, and one silent error fixed. Results of calls that were correct before do not change beyond
rounding (1e-12 eV asserted against the 0.14.0 code), and README
examples 1 to 11 print exactly what they printed with 0.14.0.

### Added

- `parameter_sensitivities(z, params, band_edge, kx, ky, potential,
  n_states, strain, sigmas, sigma_band_edge_scale)` (new module
  `kpenvelope.sensitivity`): the exact derivative of every computed
  energy with respect to each of A1..A6, delta1, delta2, delta3 of
  every material (`PARAMETER_NAMES`) and to a common scale of the
  band-edge profile. The assembled matrix is linear in each of them,
  so dE/dtheta = v^H (dH/dtheta) v (first-order perturbation theory
  for a Hermitian matrix, the Hellmann-Feynman theorem), with dH/dtheta
  the package's own assembly with that parameter set to 1 and all
  others to 0; no step size is involved. Inside a level of several
  states the derivative of the level's mean energy is returned, which
  does not depend on the basis the solver chose; a Kramers pair is
  never split by these parameters, and a level that is split (only
  possible for an accidental degeneracy, as in the demo set) is
  flagged in `level_split`. With 1-sigma parameter errors it returns
  the covariance J diag(sigma^2) J^T of the energies (independent
  errors, linear propagation). `params` is one parameter set (hard
  walls, the matrix of `solve_subbands`) or one per grid point (a
  layered stack). Checked against the closed forms of the discrete
  demo well (dE/dA1 = c (2 - 2 cos(l pi / (N + 1))) / dz^2, dE/dA2 =
  c k_t^2, 1e-12), against Richardson-extrapolated central finite
  differences of the solvers' energies for every parameter of GaN at
  k_t = 0 and 0.27 nm^-1 and of both materials of an AlN/GaN/AlN stack
  (2e-8 eV; the plain difference converges to the same value at second
  order), against `offset_sensitivity` (1e-6) and against the spread
  of 400 Monte Carlo parameter sets (12 %).
- `level_transitions(z, energies, envelopes, initial, mass_ratio,
  tol_ev)` and `degenerate_levels(energies, tol_ev)` (in
  `kpenvelope.isb`): `|z|^2`, the oscillator strength and the
  depolarization integral `S` between whole levels, summed over the
  states of the final level and averaged over those of the initial
  level. A number taken from one member of a Kramers pair depends on
  which basis of the pair the solver returns (for HH1 -> HH2 of the
  tilted 6 nm GaN well, S = 0.10378 nm for one HH2 member and 0 for
  the other); the level values do not (any unitary mixing inside each
  level leaves them unchanged to 1e-10 in the test). Checked against
  the closed-form sine eigenvectors of the discrete demo well (`|z|^2`
  and `S` to 1e-12), against the multiplet sum of
  `oscillator_strengths` (0.9606 of README example 4, 1e-12) and
  against the 0.10378 nm pinned since 0.13.0.
- README examples 12 (levels) and 13 (sensitivities), and the third
  tutorial (see Tutorials).

### Fixed

- **The check for holes in the state below the computed ones looked
  only at k_t = 0 when the momentum grid was used.**
  `fill_subbands_kgrid` (when `solve_at_k` returns more energies than
  it fills) and both self-consistent loops with `filling="kgrid"`
  compared only the k_t = 0 energy of the next state with the Fermi
  level. A branch can rise above its k_t = 0 energy before it falls (one
  member of a spin-split pair in an asymmetric well does), and then
  holds holes in a ring around k_t = 0 while its k_t = 0 energy lies
  below E_F, with no warning. On a GaN well tilted by 0.05 eV/nm (6 nm,
  61 points, ntheta = 1, nk = 61 up to 0.6 nm^-1) filled with 2 states
  at 0.04913 nm^-2, 0.14.0 gave no warning, but the third state holds
  7.0e-5 nm^-2 when 4 states are filled (its maximum, 0.040 meV above
  its edge, lies at 0.02 nm^-1). In the synthetic test (two parabolic
  branches and a third that rises by 1.2 meV) 0.14.0 gave no warning
  and `n_states_sufficient = True`, while the third branch holds
  1.11e-3 nm^-2 (2.7 % of ps) when filled. The next state is now
  checked at every grid momentum (the final iteration of the loop
  solves for one more state at each grid momentum; the first
  `n_states` of such a solve are exactly those of an `n_states`
  solve, asserted), and the warning says where. Returned numbers do
  not change. Tests: `test_fill_kgrid_warns_when_the_next_branch_rises_above_its_edge`,
  `test_kgrid_loop_checks_the_next_branch_on_the_whole_grid`.

### Changed

- The source distribution now also contains `examples/`,
  `CITATION.cff`, `CHANGELOG.md` and `CONTRIBUTING.md` (new
  `MANIFEST.in`; no new build dependency). It already contained the
  tests, but three of them read those files: run from an sdist built
  as 0.14.0 was, `test_version_is_consistent` (no `CITATION.cff`) and
  both tutorial tests (no `examples/`) failed. All 142 tests now pass
  from the unpacked sdist. The wheel is unchanged.
- Type hints on the assembly functions touched in this release.
- README: examples 12 and 13, the third tutorial, the new functions,
  refusals, checks, limits and timings.

### Performance

Measured on one core of a shared, heavily loaded machine
(OMP_NUM_THREADS=1, load average 5 to 9 on 2 cores), 0.14.0 and
0.15.0 runs interleaved, median of 7 runs (3 for example 8), GaN on a
97-point grid with a tilted potential and 8 states unless noted. The
load made single runs vary by up to a factor of two; a first series
of the same measurements gave factors within 0.3 of these.

- `solve_subbands` at k = 0.3 nm^-1 (angle 0.4 rad): 0.363 s -> 0.159 s
  (2.3 times faster). Each Chuang-Chang block is turned into a real
  symmetric matrix by one phase factor per component (read off the
  block, accepted only if the imaginary part left is at most 1e-14 of
  the largest element, which a random complex matrix fails), and real
  symmetric diagonalization is about three times faster than complex
  Hermitian diagonalization of the same size.
- `subband_dispersion` over 16 momenta: 5.69 s -> 3.29 s (1.7 times).
- `solve_self_consistent`, README example 2: 9.24 s -> 5.25 s (1.8
  times); README example 8 (`filling="kgrid"`): 128 s -> 63 s (2.0
  times), including the extra state now solved for in its final
  iteration. Same iteration counts; energies, occupations and
  potential differ from 0.14.0 by at most 8e-11.
- `solve_heterostructure` (README example 6 stack, k = 0): 0.080 s ->
  0.051 s (1.6 times); `solve_subbands` at k = 0: 0.053 s -> 0.048 s
  (1.1 times, within the noise). The assemblies place their 6 x 6
  blocks with array indexing instead of a Python loop over the grid
  (9 times faster for the uniform assembly on its own) and evaluate
  the bulk blocks once per distinct parameter set; the values placed
  come from the same operations as before.

Equality is checked in `tests/test_v015.py`, which keeps the 0.14.0
complex block diagonalization and both assembly loops as the
reference: energies to 1e-12 eV, projectors onto the top 8 states to
1e-10, residuals and orthonormality to 1e-12 over 24 cases (largest
energy difference seen 1.6e-14 eV), and the assembled matrices to
1e-15 of their largest element (identical on this machine).

### Tutorials

- `examples/tutorial_isb_mass_calibration.py` (a few seconds): a GaN
  well between AlN barriers and a synthetic intersubband line (fixed
  seed) made from a GaN with a heavy-hole mass of 2.0 m0 along c. It
  finds the line that couples to light polarized along z
  (`level_transitions`), lists what that line depends on
  (`parameter_sensitivities`: at k_t = 0 only A1 + A3 and the two
  splittings of each material, and the offset), calibrates A1 + A3 of
  GaN by Newton steps with the exact derivative, and gives the error
  bar from the line (1 %) and from illustrative errors of the other
  parameters, with the largest parts named (the AlN A parameters).
  Its output is in the README and at the top of the script;
  `test_tutorial_isb_mass_calibration` runs it and checks the result.

### Tests

- New file `tests/test_v015.py`, 16 tests (142 in total). The two
  tests of the fix fail on 0.14.0. The speed tests keep the 0.14.0
  code (the complex block diagonalization and the assembly loops) as
  the reference and compare with stated tolerances. No new test
  depends on which member of a Kramers pair the solver returns first
  or on the phase of an eigenvector.

## 0.14.0 (2026-10-17)

A faster eigen-solver with the same results, the quantum lifetime from
Shubnikov-de Haas (SdH) data, the cyclotron mass of a computed
subband, two worked tutorials, and two silent errors fixed. Results of
calls that were correct before do not change beyond rounding (1e-12 eV
is asserted against the old solver), and every README example prints
exactly what it printed with 0.13.1.

### Added

- `fit_sdh_dingle(fields_T, amplitudes, temperature_K, mass_ratio,
  sigma_amplitudes, sigma_mass_ratio)`: the quantum lifetime tau_q
  and the Dingle temperature from the field dependence of SdH
  amplitudes at one temperature (a Dingle plot). It uses the
  two-dimensional form A = 4 rho_0 R_T exp(-pi / (omega_c tau_q)) of
  P. T. Coleridge, R. Stoner and R. Fletcher, Phys. Rev. B 39, 1120
  (1989), and T_D = hbar / (2 pi k_B tau_q), the temperature whose
  Lifshitz-Kosevich factor exp(-2 pi^2 k_B T_D / (hbar omega_c)) (D.
  Shoenberg, Magnetic Oscillations in Metals, Cambridge University
  Press (1984), chapter 8) is the same damping. Weighted or unweighted
  linear least squares of ln(A / R_T) against 1/B; the error of the
  mass is propagated with the exact derivative. It refuses amplitudes
  that are not damped towards low field and a tau_q whose error is as
  large as tau_q. `DINGLE_TESLA_PS` (pi m0 / e = 17.862 T ps) and
  `DINGLE_KELVIN_PS` (hbar / (2 pi k_B) = 1.2157 K ps) live in
  `kpenvelope.lab`. Checked against scipy.constants (1e-6 and 1e-15),
  against the Lifshitz-Kosevich form of the Dingle factor (1e-12),
  noise-free data (2e-6, set by scipy's electron mass against the
  package's m0), `numpy.polyfit` with and without weights (values and
  error bars to 1e-9), a 400-sample Monte Carlo (the spread of tau_q
  equals the reported error to 12 %, mean reduced chi-square 1 to
  0.1) and the mass propagation against refits at shifted masses
  (1e-5).
- `cyclotron_mass(kts, energies, n_branch)`: the cyclotron mass of a
  computed branch at the Fermi contour that holds `n_branch` holes,
  m_c / m0 = (hbar^2/2m0) |d(k^2)/dE| at k_F = sqrt(4 pi n), the
  number `fit_sdh_mass` measures. A parabola in k^2 through the three
  nearest path points is differentiated at k_F. Exact for parabolic
  branches (1e-12), second order against the closed form
  m sqrt(1 + 4 c k^2 / (m Eg)) of a nonparabolic test branch (error
  below 0.5 / (nk - 1)^2 over 15 densities and 5 spacings), and within
  1 % (0.3 % at half the spacing) of a direct difference of two
  six-band GaN solves around k_F. It refuses a path that does not
  start at 0, a k_F beyond the path and a branch whose occupied region
  on the path is not the disk k < k_F.
- `subband_energies` and `heterostructure_energies`: the energies of
  `solve_subbands` and `solve_heterostructure`, bit for bit, without
  building the envelopes.
- Tutorials `examples/tutorial_sdh_analysis.py` and
  `examples/tutorial_sdh_prediction.py` (see Tutorials).

### Fixed

- **`dos_from_dispersion` returned 0 and a spike at the start of a
  momentum path that does not begin at k_t = 0.** Energies above
  E(kts[0]) were given an occupied area of 0. For the demo set
  (A = -2, constant DOS 1.0443 nm^-2 eV^-1) on a path from 0.2 nm^-1
  (E = -3.048 meV) it returned 2.596 at -4 meV, 2.076 at -3 meV and 0
  from -2 to +4 meV, where the band continues. Those energies are now
  NaN, as at the far end of the path since 0.13.0. A path from 0 is
  unchanged. Test: `test_dos_is_nan_before_a_path_that_starts_above_zero`.
- **`fill_subbands_thermal` filled a negative mass with negative
  holes.** Edges [0, -0.01] eV with masses [0.5, -0.3] m0 and
  ps = 0.1 nm^-2 gave [0.234, -0.134] nm^-2 at 0 K and at 77 K. A
  negative mass is now refused with a `ValueError` (the self-consistent
  loops never produce one). Test:
  `test_thermal_filler_refuses_negative_masses`.
- `tests/test_lab.py::test_sheet_density_exact_round_trip_and_linearity`
  used state 5 of a Kramers pair (states 4 and 5) as the upper state.
  Which member of a pair overlaps with state 0 depends on the basis
  the eigensolver picks inside the pair: S was 0.194 nm (state 4) and
  0.062 nm (state 5) with 0.13.1 on our machine, and is 0.256 and 0
  with 0.14.0 (the sum, 0.256 nm, is the same). With S = 0 the test
  refused its own input. It now takes the member with the larger S,
  whatever its index, as the 0.13.x fix of the geometry-integral test
  did.

### Changed

- The solvers diagonalize the assembled matrix in exact independent
  blocks where it has them (see Performance). The energies agree with
  the full diagonalization to rounding (largest difference seen
  2e-14 eV; 1e-12 eV asserted), and the states agree through the
  projector onto each degenerate level (1e-8 asserted). Inside a
  degenerate level the basis is a convention, and it changed: at
  k_t = 0 the two members of a Kramers pair now have bit-for-bit
  equal energies, and each lies in one group of components that the
  matrix does not couple. A number computed from one member of a pair
  (not summed over the pair) can therefore differ from 0.13.1, as it
  could already between computers. For HH1 -> HH2 of the tilted 6 nm
  GaN well of README example 5, S is 0.10378 nm and 0 for the two HH2
  members; summed over the pair it is 0.10378 nm, as before.
- `assemble_heterostructure` (and so `solve_heterostructure` and the
  finite-barrier loop) refuses a band edge that is not finite. Before,
  one NaN gave NaN energies and a `RuntimeWarning`. Test:
  `test_heterostructure_refuses_a_non_finite_band_edge`.
- Type hints on the public functions touched in this release.

### Performance

Measured on one core of a shared machine (OMP_NUM_THREADS=1), old and
new runs interleaved, median of 5 runs (3 for the last two lines), GaN
on a 97-point grid with a tilted potential and 8 states unless noted:

- `solve_subbands` at k_t = 0: 0.928 s -> 0.056 s (16.5 times faster).
  The matrix splits into independent groups of components (HH up, HH
  down, LH up with CH down, LH down with CH up). The groups are read
  from the zero pattern of the assembled matrix, each is diagonalized
  on its own (as a real matrix when it is real), and a group whose
  time-reversal partner gives the identical matrix is diagonalized
  once.
- `solve_subbands` at kx = 0.3 nm^-1: 0.912 s -> 0.275 s (3.3 times).
  The Chuang-Chang block diagonalization (Phys. Rev. B 54, 2491
  (1996), the upper and lower 3 x 3 blocks of their Eq. (45)) splits
  the matrix into two of half the size at every in-plane direction. It
  is used only after checking that the neglected coupling is at
  rounding level (at most 1e-14 of the largest element); a strain
  whose in-plane axes do not match the momentum direction couples the
  blocks, and the full matrix is then diagonalized exactly as before
  (bit for bit, asserted).
- `solve_heterostructure` (README example 6 stack, k_t = 0): 0.696 s
  -> 0.049 s (14.2 times).
- `subband_dispersion` over 16 momenta: 14.31 s -> 3.83 s (3.7 times).
- `solve_self_consistent`, README example 2: 31.3 s -> 6.9 s (4.5
  times). Same iterations; energies, occupations, potential and
  density agree with the old solver to 1e-9 in the test.
- The whole test suite took 1044 s (105 tests, 0.13.1) at the start
  of this work and 56 to 85 s (126 tests) at the end, on the same
  machine under a changing load; this is a rough indication only, the
  lines above are the measured factors.

Equality is checked in `tests/test_v014.py`, which keeps the 0.13.1
dense diagonalization as the reference: energies, projectors,
orthonormality and residuals over 40 cases (hard walls, a layered
stack, potentials, four strains, five momenta), the solvers and the
energies-only helpers, both self-consistent loops and the momentum-grid
filler run through the old and the new solver.

### Tutorials

- `examples/tutorial_sdh_analysis.py`: synthetic SdH data (fixed seed)
  to the density, the cyclotron mass and the quantum lifetime with the
  Dingle temperature, each with its error bar (about 1 s).
- `examples/tutorial_sdh_prediction.py`: the hole gas of README
  example 2, its dispersion filled at the converged potential, and the
  predicted SdH frequency and cyclotron mass of each occupied branch,
  with the numerical uncertainty from two path spacings (5 to 30 s).
- Their printed output is in the README (Tutorials) and at the top of
  each script; `test_tutorial_sdh_analysis` and
  `test_tutorial_sdh_prediction` run them and check the key numbers.

### Tests

- New file `tests/test_v014.py`, 21 tests (126 in total). The two
  fixes and the band-edge refusal fail on 0.13.1. One of the new
  tests covers the per-point strain path of
  `assemble_heterostructure`, which no test reached before (it was
  correct: equal to the uniform assembly bit for bit). No new test depends on which member of a
  Kramers pair the solver returns first or on the phase of an
  eigenvector.

## 0.13.1 (2026-10-10)

Citation correction and packaging.

### Fixed (documentation)

- `gan_rinke2008`: the crystal-field splitting (10 meV) and spin-orbit
  splitting (17 meV) were credited to Rinke et al., Phys. Rev. B 77,
  075202 (2008). Rinke et al. give A1..A6 (their Table V, matched
  exactly), but compute a crystal-field splitting of 34 meV and take
  10 meV from other work; 17 meV does not appear there. The docstring,
  the `reference` string and the README now credit the two splittings
  to Extended Data Table 1 of Chang et al., Nature Electronics 9, 346
  (2026), where the whole set is tabulated. The numbers are unchanged.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.13.0 (2026-10-10)

Five silent errors fixed, each with a test that fails on 0.12.0, and
tools to read Shubnikov-de Haas (SdH) measurements: the density from
an oscillation frequency and the cyclotron mass from the temperature
damping, with honest error bars. Results of calls that were correct
before do not change.

### Added

- `sheet_density_from_sdh(frequency_T, branches=..., sigma_frequency_T)`
  and `sdh_frequency(n_sheet, branches=...)`: the Onsager relation
  F = hbar A / (2 pi e) (L. Onsager, Phil. Mag. 43, 1006 (1952)). One
  spin-resolved branch holding n holes per area encloses
  A = 4 pi^2 n, so F = (h/e) n for any contour shape. `branches` (how
  many spin-resolved branches share the oscillating contour) is a
  required keyword: a wrong default would be a factor of two.
  `H_OVER_E_T_NM2` (in `kpenvelope.lab`) is h/e from the exact 2019 SI
  values. Checked against the Fermi wavevector of exactly parabolic
  subbands filled by `fill_subbands_kgrid` (relative 1e-9) and against
  `scipy.constants`.
- `fit_sdh_mass(temperatures_K, amplitudes, field_T, sigma_amplitudes)`
  and `lk_thermal_factor`: the cyclotron mass from the
  Lifshitz-Kosevich factor X / sinh(X),
  X = 2 pi^2 k_B T m* / (hbar e B) (I. M. Lifshitz and A. M. Kosevich,
  Sov. Phys. JETP 2, 636 (1956); D. Shoenberg, Magnetic Oscillations in
  Metals, Cambridge University Press (1984)). Weighted least squares
  with the amplitude refitted, error bars from the exact Jacobian
  (absolute with amplitude errors, scaled by the residual scatter
  without), and a reduced chi-square as a model test. It refuses data
  that do not fall with temperature, a mass error as large as the
  mass, and any fit whose chi-square does not rise by 0.7 to 1.3 units
  at one sigma on either side (the error bar would not be honest).
  `LK_TESLA_PER_KELVIN` = 14.6932 T/K is computed from the exact k_B,
  h and e and the package's own hbar^2/2m0. Checked against
  `scipy.optimize.curve_fit` (values to 1e-7, error bars to 1e-4,
  weighted and unweighted), against noise-free data (1e-9) and by a
  400-sample Monte Carlo (the spread of the masses equals the reported
  error bar to 12 %; mean reduced chi-square 1 to 0.1).
- `energy_from_wavenumber`, `energy_to_wavenumber`: eV and the cm^-1
  of infrared spectrometers, with h c / e from the exact SI values
  (1000 cm^-1 = 0.12398419843320026 eV, checked against
  `scipy.constants`).
- `SelfConsistentResult.n_states_sufficient` (see Fixed).
- README example 11 (SdH density, mass and predicted frequencies);
  example 5 also prints the line in cm^-1.

### Fixed

- **`layered_profile` put interface points into the wrong layer when
  the thicknesses did not add up exactly in floating point.** A point
  on an interface belongs to the layer on its right, but the test was
  an exact comparison with the running sum of thicknesses, and
  2.7 + 3.4 = 6.1000000000000005 lies above the grid point 6.1. In a
  sweep of 32 three-layer stacks on a 0.1 nm grid, 10 grid points were
  in the wrong layer. A 3.4 nm GaN well between AlN barriers
  (offset -0.5 eV, 95 points over 9.4 nm) had 34 points and a top
  level of 1.978 meV when it started at 2.0 nm, but 33 points and
  1.234 meV when it started at 2.7 nm. Points within 1e-9 grid steps
  of an interface now count as on it (the tolerance
  `poisson.sheet_mask` already used): 34 points and 1.978 meV in both
  places. Grids whose interfaces were assigned correctly are
  unchanged (README examples 6 and 9 and every existing test).
  Tests: `test_layered_profile_matches_exact_assignment`,
  `test_translated_well_has_the_same_levels`.
- **The depolarization geometry integral depended on the arbitrary
  phase of an eigenvector.** `overlap_geometry_integral` took the real
  part of the transition density, so multiplying one state by i (an
  equally valid eigenvector) changed S. For the two states of README
  example 5, S = 0.4503 nm and alpha = 2.311, but S = 0 and alpha = 0
  with i on the second state; for HH1 -> HH2 of a tilted 6 nm GaN
  well at k = 0 (S summed over the HH2 Kramers pair), 0.10378 nm and
  0. It now uses S = int dz |int^z lambda|^2, identical bit for bit
  for real states and independent of either phase (both cases now
  give 0.4503 and 0.10378 nm). Tests:
  `test_geometry_integral_does_not_depend_on_eigenvector_phase`,
  `test_geometry_integral_phase_invariance_on_solver_states`.
- **The self-consistent loops silently dropped the holes of occupied
  states beyond `n_states`.** Only the computed states were filled,
  with no check that the next state down lay below the Fermi level.
  README example 2 (GaN, 4.6e13 cm^-2, hard walls, `n_states=4`)
  ended with its parabolic Fermi level at -606.5 meV, below the edge
  of the uncomputed third pair at -458.9 meV. With 8 states, 0.79e13
  of the 4.6e13 cm^-2 sit in the third and fourth pairs and the gas
  center moves from 0.62 to 0.76 nm. On a synthetic demo well with
  two filled six-fold levels (eps_r = 1000, 0.6 nm^-2), 6 states gave
  0.1 nm^-2 per state where 12 give 0.0779 and 0.0221. Both loops now
  check the next state at the converged potential (above the Fermi
  level at T = 0; hole factor above 1e-4 at T > 0, the threshold of
  the k-grid edge check), issue a `RuntimeWarning` naming the fix, and
  set `n_states_sufficient = False`. `fill_subbands_kgrid` does the
  same when `solve_at_k` returns more energies than it fills. The
  returned numbers are unchanged. README examples 2 and 9 now use 8
  and 12 states, and `examples/demo_well.py` 6 instead of 4 (its
  levels are six-fold, so 4 states put 0.0115 nm^-2 into each of 4
  states instead of 0.0077 into each of 6). Tests:
  `test_loop_warns_when_occupied_states_were_not_computed`,
  `test_loop_warns_with_kgrid_filling_too`,
  `test_gan_hard_wall_example_needs_more_than_four_states`,
  `test_fill_kgrid_warns_when_given_more_energies_than_it_fills`.
- **`fill_subbands_thermal` returned occupations that did not add up
  to `ps`.** With no finite mass it returned [0, 0] for 0.1 nm^-2
  (T = 0 and 77 K); for ps = -0.1 it returned [-0.1, -0]; and for a
  ps whose Fermi level lies more than 5 eV down (1e3 nm^-2 in the
  test) the bisection stopped at its bound and the rescaling to ps
  hid it ([500.5, 499.5]). All three are now refused with a
  `ValueError`. Inside the self-consistent loops the historical
  behavior is kept for intermediate iterations (a far-from-converged
  iteration can pass through such a state, as the unsettled loop of
  `test_hetero_loop_with_the_sheet_at_the_interface` does), and a
  final state without
  holes gives a `RuntimeWarning`. Test:
  `test_thermal_filler_refuses_what_it_cannot_fill`.
- **`dos_from_dispersion` returned a density of states of 0 below the
  end of the momentum path.** The occupied area was held at its value
  at the last momentum, so energies below E(kts[-1]) showed a gap
  (0.0 where the demo set has the constant 1.044 nm^-2 eV^-1). They
  now give NaN, as does any grid point whose difference stencil
  reaches them. Test: `test_dos_is_nan_beyond_the_path`.

### Changed

- New input checks with plain messages: both self-consistent loops
  refuse a negative or non-finite `ps` and a `mixing` or `tol` that is
  not a positive finite number; `layered_profile` refuses negative or
  non-finite thicknesses and an empty stack; `dos_from_dispersion`
  refuses arrays of the wrong shape; `fill_subbands_thermal` refuses
  energies and masses of different shapes; `fill_subbands_kgrid`
  refuses a non-integer `n_states` and a `solve_at_k` that returns
  fewer energies than `n_states`. Test: `test_loop_input_checks` and
  the tests above.
- Type hints on the public functions touched in this release.
- Five existing tests that pin results of runs with too few states
  (unchanged results) now ignore the new warning, with a comment.
- README: examples 2, 5 and 9 updated and example 11 added (outputs
  produced by this version), the "refuses" and "checked" lists, the
  limits, and a plain-language account of these corrections.

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
