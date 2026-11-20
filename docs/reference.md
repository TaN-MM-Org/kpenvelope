# kpenvelope reference

Everything the package offers, function by function, with the units, a
glossary, the cited material numbers, and the cases where it refuses to
answer. Start with the [README](../README.md) for an overview and the
[examples](examples.md) to see it used.

## A short guide to the words used here

- **Wurtzite** -- the hexagonal crystal structure of GaN and AlN. Its
  special axis is the **c axis**; here it is the growth direction `z`.
- **Hole gas (2DHG)** -- a thin sheet of holes held against an
  interface. Its **sheet density** is the number of holes per unit
  area.
- **Subband** -- one allowed energy level of the trapped holes. Holes
  can still move freely in the plane of the layer, so each subband has
  an energy that depends on the **in-plane momentum** `k_t`. That curve
  is the **dispersion**.
- **Six-band k.p model** -- the standard way to describe the top of the
  valence band with six coupled components. The components come in
  three pairs: **heavy hole (HH)**, **light hole (LH)** and
  **crystal-field split-off hole (CH)**. A state's **band character**
  is how much of it lies in each pair.
- **Envelope function** -- the slowly varying part of the wavefunction
  along `z`, one per component. The package solves for it on a grid.
- **A1..A6, Delta_CR, Delta_SO** -- the material numbers of the model:
  six mass parameters, the crystal-field splitting and the spin-orbit
  splitting.
- **Hard wall / finite barrier** -- a hard wall forces the
  wavefunction to zero at the grid ends. A finite barrier is a real
  second material (for example AlN) that the wavefunction can leak
  into. The **band offset** sets how high that barrier is.
- **Self-consistent** -- the charge of the holes changes the potential,
  and the potential changes where the holes sit. The solver repeats
  both steps until they agree (Gauss's law for the potential).
- **Effective mass** -- how heavy a hole behaves in the plane, in units
  of the free-electron mass `m0`. The **local mass** is measured from
  the dispersion at one momentum.
- **Kramers pair** -- two states with exactly the same energy at zero
  momentum, required by time-reversal symmetry. At `k = 0` the solver's
  states come in such pairs, so the first spacing between different
  subbands is between states 0 and 2.
- **Intersubband transition** -- a hole jumping between two subbands by
  absorbing light. Its strength is set by the **dipole matrix element**
  `z_fi` and the dimensionless **oscillator strength** `f`.
- **Polarization** -- the direction of the light's electric field:
  along the growth axis `z`, or in the plane of the layers (as at
  normal incidence). The two couple the subbands differently
  (example 14).
- **Depolarization shift** -- in a layer with many carriers the
  absorption line sits above the bare subband spacing, because the
  oscillating charge screens itself (Allen, Tsui and Vinter, Solid
  State Commun. 20, 425 (1976); Ando, Fowler and Stern, Rev. Mod.
  Phys. 54, 437 (1982)).
- **Rashba splitting** -- a spin splitting that grows linearly with
  momentum in structures without inversion symmetry.
- **Strain / Bir-Pikus terms** -- how a stretched or squeezed crystal
  shifts the bands, through **deformation potentials** D1..D6.
- **Fermi-Dirac filling** -- how states are occupied at a given
  temperature. At zero temperature every state on one side of a
  dividing energy (the Fermi level) is full and every other state is
  empty; when warm, that edge is smeared out.
- **Hermitian** -- the matrix property that guarantees real energies;
  a correct Hamiltonian matrix must have it.
- **Closed form / quadrature** -- a closed form is an exact formula; a
  quadrature is a numerical integration, used here as an independent
  second calculation.
- **Lorentzian** -- the standard bell-like shape of a broadened
  absorption line.
- **Shubnikov-de Haas (SdH) oscillations** -- oscillations of the
  resistance in a magnetic field `B`, periodic in `1/B`. Each subband
  gives one **frequency** (in tesla), set by its hole density. The
  oscillations fade as the temperature rises, at a rate set by the
  **cyclotron mass**.

### Words for the nitride-stack part (new in 0.17.0)

- **Polarization reference** -- polarization can only be computed
  relative to a crystal you call zero. The **H** reference is the
  layered hexagonal structure (each nitrogen halfway between two metal
  layers); the **ZB** reference is zinc blende. They give the same
  charge for a fully strained layer and differ once it relaxes.
- **Relaxation r** -- the share of its mismatch to the layer below that
  a layer has given up: 0 is fully strained (coherent), 1 fully relaxed.
- **Misfit dislocation** -- the line where an extra (or missing) half
  plane of atoms ends at an interface; it is how a layer relaxes. Its
  charge per length is **Lambda** (e/nm).
- **Bare end, neutral, electron counting** -- three choices for Lambda:
  the full polarization charge of a crystal plane that just stops (what
  the H recipe assumes), zero (what the ZB recipe assumes), and the small
  charge left when the broken bonds are filled by the electron-counting
  rule (empty metal bonds, full nitrogen bonds).
- **2DEG** -- two-dimensional electron gas, the electron sheet of a GaN
  transistor.

### Words for the device and phonon part (new in 0.18.0)

- **Number set** (`parameter_set`) -- whose published polarization
  numbers a layer uses. The charge at an interface is a small difference
  of two large numbers, so both sides should come from one set.
- **Drift-diffusion** -- the standard model of current in a
  semiconductor: carriers drift in the electric field and spread out
  where their density changes.
- **Quasi-Fermi level** -- the energy that sets the electron (or hole)
  density when a current flows; with no current it is the Fermi level.
- **Quasi-equilibrium** -- the electrons keep the Fermi level of their
  contact while only the gate moves: the usual model of a C-V sweep.
- **Recombination** -- an electron filling a hole. SRH (through a
  defect level), radiative (giving light) and Auger (giving the energy
  to a third carrier) are included.
- **Optical phonon** -- a vibration in which the two atoms of a cell
  move against each other. In a polar crystal it carries an electric
  field. **TO** and **LO** are its transverse and longitudinal kinds; an
  **interface phonon** is one whose field is tied to an interface.

## Install, units and conventions

```
pip install kpenvelope
```

It needs Python 3.9 or newer and NumPy 1.22 or newer, and nothing
else. The tests also use SciPy 1.8 or newer and pytest
(`pip install -e .[test]` from a clone of the repository). When SciPy is
installed, the drift-diffusion solver uses its banded LAPACK solver,
which is about ten times faster than the NumPy one it falls back to; the
results are the same.

Units and conventions, used everywhere:

- Energies in eV, lengths in nm, momenta in nm^-1, sheet densities in
  nm^-2, volume densities in nm^-3, masses in units of `m0`,
  temperatures in kelvin, magnetic fields in tesla, Rashba coefficients
  in eV nm (1 meV A = 1e-4 eV nm).
- Energies are on the **valence-electron scale**: holes fill the
  **highest** eigenvalues first, and results are sorted from the
  highest energy down. A hole barrier is therefore a region with a
  **lower** band edge.
- Laboratories quote sheet densities in cm^-2.
  `sheet_density_from_cm2` multiplies by 1e-14 and
  `sheet_density_to_cm2` by 1e14 (so 4.6e13 cm^-2 = 0.46 nm^-2).
  Infrared spectrometers report lines in cm^-1:
  `energy_from_wavenumber` and `energy_to_wavenumber` convert with the
  exact SI value of h c / e (1000 cm^-1 = 0.12398 eV).
- In the hard-wall solvers the walls sit one grid step outside the
  first and last grid points, so a grid from 0 to `L` behaves like a
  well of width `L + 2 dz`.
- Nitride stacks (new in 0.17.0): layers are listed from the bottom
  (the relaxed substrate or buffer) up to the surface, z runs along
  [0001] of a metal-polar crystal, and the charge at an interface is
  P below minus P above. Polarizations are in C/m^2, interface charges
  are returned in C/m^2 and in elementary charges per nm^2 and per cm^2,
  line charges in elementary charges per nm (e/nm), donor densities in
  nm^-3 (1e18 cm^-3 = 1e-3 nm^-3). `E_PER_NM2_PER_C_M2` (6.2415) converts
  C/m^2 to e/nm^2. The electron solver puts the Fermi level at 0 eV.
- Drift-diffusion (new in 0.18.0): the inputs use the same units as
  above, and the results give carrier densities in cm^-3, current
  densities in A/cm^2 (positive along +z, from the bottom to the top),
  charges on a contact in C/cm^2 and capacitances in F/cm^2. A positive
  voltage on a contact lowers its Fermi level by that many eV. Phonon
  energies are in meV (`CM1_TO_MEV` converts from cm^-1).

## The app

```
kpenvelope-app
```

(or `python -m kpenvelope.gui`) opens the app in your web browser. It
runs on your own computer, on 127.0.0.1 only, and needs nothing beyond
kpenvelope and NumPy (the Transistor tab also needs SciPy:
`pip install "kpenvelope[device]"`). You build the layer stack on the left (from an
example or from scratch), choose whose published numbers to use, and
read the results on the right:

- Every result starts with an "In short" sentence that says what it
  means, every "?" opens a short explanation when you tap it, and when
  you change the sample the old results are marked as out of date.
- **Charges**: the fixed charge at each interface, with what it does
  ("attracts electrons"), how much the layer is stretched or squeezed,
  how much of the charge comes from dislocation lines, and the source
  of every number.
- **Electrons**: the band diagram and the electron density, the sheet
  density as one big number, and the lowest energy levels.
- **How sure?**: how sure the electron density is, from the errors you
  give for each composition, thickness and the surface barrier, with a
  bar chart of where the error comes from (or a spread of random
  samples).
- **Voltage sweep**: a C-V sweep (no transport numbers needed, with the
  electrons treated simply or by quantum mechanics) that says where the
  electron layer empties, the gate leakage (thermionic emission and
  tunneling through the barrier), or an I-V sweep (with your transport
  numbers and their source, and thermionic emission at a metal). Since
  0.23.0 the I-V sweep can include band-to-band tunneling, for tunnel
  junctions and tunnel diodes.
- **Fit my data**: paste a measured C-V curve (in nF/cm^2, or pF with
  the gate area) and/or a Hall density, tick the numbers you are unsure
  of, and the app finds them with error bars, plots measured against
  fitted, warns where two of them move together, and can write them
  back into your sample.
- **Design finder**: say what electron density and switch-off voltage
  you want; the app maps the composition and thickness of a layer,
  lists the closest designs with their error bars from your growth
  tolerances, and draws a map of how close each one comes.
- **2D cross-section**: a slice through a transistor with a gate of a
  given length, showing where the electrons are (or the band) under and
  beside the gate, and the electron density along the slice.
- **Transistor** (new in 0.19.0): a whole transistor built from your
  layers, with source, gate and drain. It gives the output curve (drain
  current against drain voltage) or the transfer curve (against gate
  voltage), with a mobility and top speed from cited sources (or
  yours), optionally with self-heating down through a SiC or other
  substrate (the hottest point and the thermal resistance are reported),
  and a picture of the electrons, the temperature or the band at the last
  point. A second button follows drain lag: iron traps in the base are
  filled at a high drain voltage, then the current is followed from a
  nanosecond to ten seconds after switching to the measuring voltages.
  Since 0.20.0 the tab also takes a contact resistance, a passivation
  film (silicon nitride by default) and a field plate joined to the
  gate; it reports the gate leakage by tunneling and, with an energy
  loss time, how hot the electrons get. Three more buttons give the
  speed (f_T and f_max with the gain curves), the avalanche breakdown
  voltage in the off state, and the current and temperature during a
  pulse. Since 0.21.0 the hot electrons can change the current (energy
  balance solved with it), and breakdown can also be read where the
  off-state drain current reaches a value you choose. Since 0.23.0 the
  heat can follow the full temperature model (leakage and the band gap
  follow the temperature too), include Peltier heat, and run together
  with the hot electrons; electrons can cross the barrier step by
  thermionic emission; and a thermal runaway button raises the drain
  voltage until no steady temperature is left. Since 0.24.0 the channel electrons can be
  treated as quantum states.
- **Process steps** (new in 0.23.0): write a short recipe, one step per
  line (deposit, etch, implant, anneal, oxidize), and see the
  cross-section it builds, with its materials or its doping, the
  thickness in the middle and the doses. Since 0.24.0 deposit and etch also
  take mode=flux (narrow openings fill and etch more slowly) and oxidize
  takes mode=2d (oxide under mask edges).
- **Defect lines**: what the published data say about the charge
  of the misfit lines, combined, and how precisely a planned holography
  measurement would pin it down.
- **Save and share**: the same calculation as a script you can rerun or
  put in a supplement, and the last curve as a CSV file.

Every box has a short explanation, the words are plain, and when an
input cannot give a trustworthy answer the app says why in a sentence
(for example, that ScAlN has no published band offset and asks you for
yours). Samples can be saved to a file and opened again. The tests run
the app's calculations against the library directly, check that the
script it writes gives the same number, and check that the local server
answers only the page it served (each start makes a random token).

## What is in the package

**Material parameters**

- `WurtziteParameters` -- A1..A6, `delta1` (crystal-field splitting),
  `delta2`, `delta3` (spin-orbit terms), `eps_r` (permittivity), a
  required `reference` string, and optional deformation potentials
  D1..D6.
- `gan_rinke2008()`, `aln_rinke2008()` -- the two cited sets (next
  section); `demo_single_band(A)` -- the synthetic test set.

**Energy levels and wavefunctions**

- `solve_subbands(p, z, kx, ky, potential, n_states, strain)` -- the
  top states of one material between hard walls, at any in-plane
  momentum; `assemble_hamiltonian` builds the matrix it solves.
  `HBAR2_OVER_2M0` is hbar^2/2m0 = 0.0380998 eV nm^2.
  `subband_energies` (new in 0.14.0) gives the same energies, bit for
  bit, without the envelopes.
- `layered_profile(z, layers)` -- per-point materials and band edges
  from a list of `(thickness_nm, params, band_edge_eV)` layers.
- `solve_heterostructure`, `assemble_heterostructure` -- the same for
  layered stacks with finite barriers. The discretization keeps the
  matrix Hermitian for any profile, and each layer may carry its own
  strain. `heterostructure_energies` gives the energies alone.
- Since version 0.14.0 the solvers split the matrix into the pieces
  that it falls apart into exactly: at zero in-plane momentum into
  groups of components that no element couples, and at any other
  momentum into the two halves of the Chuang-Chang block form (their
  Eq. (45)). The energies are those of the full matrix to rounding,
  3 to 16 times faster (see [Limits](limits.md#limits) for times). If a strain
  couples the halves, the full matrix is used, as before. Since
  version 0.15.0 each half is turned into a real matrix by one phase
  factor per component before it is diagonalized, which makes a solve
  at finite momentum about twice as fast again, and the matrix is
  built without a Python loop over the grid.
- `strain_blocks(p, strain)` -- the Bir-Pikus strain matrix for a
  symmetric 3 x 3 strain tensor (needs cited D1..D6). Every solver
  takes a `strain=` (or `strain_list=`) argument.

**Self-consistency with the charge**

- `solve_self_consistent(p, z, ps, ..., temperature_K, filling)` --
  hard walls; `solve_self_consistent_hetero(z, params_list, band_edge,
  ps, eps_r, ..., sheet_z)` -- finite barriers. Both return a
  `SelfConsistentResult` with `z`, `potential`, `energies`,
  `envelopes`, `density`, `occupations`, `masses`, `iterations`,
  `residual`, `converged` (False when the loop ran out of iterations),
  `filling`, `fermi_level` and `n_states_sufficient` (False, with a
  `RuntimeWarning`, when a state below the computed ones would hold
  holes; example 2).
- `filling="parabolic"` (default) treats each state as a parabola with
  one mass; `filling="kgrid"` with `kmax` (and optionally `nk`,
  `ntheta`) fills the computed dispersion on a momentum grid and builds
  the hole density from the states at every grid momentum (example 8).
- In the finite-barrier loop, `eps_r` may be one value per grid point,
  and `sheet_z` says where the fixed negative charge that holds the gas
  sits (example 9).
- `fill_subbands_thermal(energies, masses, ps, temperature_K)` --
  Fermi-Dirac filling of parabolic subbands in closed form;
  `fill_subbands_kgrid(solve_at_k, ps, n_states, kmax, ...)` -- filling
  from the full computed dispersion on a momentum grid (exact for
  parabolic subbands at any grid spacing; without in-plane
  anisotropic strain one direction, `ntheta=1`, is enough).
  `KB_EV_PER_K` is the Boltzmann constant in eV/K, computed from the
  exact SI values of k_B and e.
- `sheet_density_from_cm2`, `sheet_density_to_cm2` -- lab units;
  `energy_from_wavenumber`, `energy_to_wavenumber` -- eV and cm^-1.

**What the states are made of**

- `band_character(envelopes, z)` -- HH, LH and CH fractions of each
  state; `character_vs_k` -- the same along a momentum path;
  `dominant_character` -- a one-word label; `CHARACTER_GROUPS` -- which
  of the six components belong to each group.

**Dispersion, mass and transport ingredients**

- `subband_dispersion` -- energies along an in-plane path;
  `local_mass(kts, energies)` -- the local mass (a flat stretch gives
  `inf`, on purpose).
- `spin_splitting`, `splitting_vs_k` -- the splitting within each
  Kramers pair.
- `group_velocity` -- dE/dk (eV nm; divide by hbar for a velocity);
  `dos_from_dispersion` -- the 2D density of states of falling,
  isotropic dispersions.
- `cyclotron_mass(kts, energies, n_branch)` (new in 0.14.0) gives the
  cyclotron mass of a branch at the Fermi contour that holds
  `n_branch` holes per nm^2, the mass a Shubnikov-de Haas measurement
  sees (second tutorial).

**Intersubband optics**

- `dipole_matrix(z, envelopes)` -- `<f|z|i>` in nm;
  `oscillator_strengths(energies, dip, mass_ratio)` -- strengths from
  the top subband.
- `depolarization_shift`, `overlap_geometry_integral` -- the shifted
  line and its geometry integral `S`; `isb_lineshape` -- a
  strength-weighted Lorentzian line shape at your measured linewidth
  (shape only; the absolute absorbance needs your optical geometry).
- `level_transitions(z, energies, envelopes, initial, mass_ratio)`
  (new in 0.15.0) -- `|z|^2`, the oscillator strength and `S` between
  whole levels (summed over the final level, averaged over the initial
  one), which do not depend on how the solver picked the states of a
  Kramers pair; `degenerate_levels(energies)` -- the grouping into
  levels (example 12).
- `velocity_matrix(z, params, envelopes, kx, ky, angle)` (new in
  0.16.0) -- the in-plane velocity matrix elements `<f| dH/dk |i>`
  (eV nm; divide by hbar for a velocity) along the direction `angle`;
  its diagonal is the exact slope dE/dk of each state.
  `inplane_transitions(z, params, energies, envelopes, kx, ky, angle,
  initial)` -- the same level sums as `level_transitions`, for light
  polarized in the plane (example 14).

**Parameter sensitivities and error budgets**

- `parameter_sensitivities(z, params, band_edge, kx, ky, potential,
  n_states, strain, sigmas, sigma_band_edge_scale)` (new in 0.15.0)
  -- the exact derivative of every computed energy with respect to
  A1..A6, `delta1`, `delta2`, `delta3` of every material
  (`PARAMETER_NAMES`) and to the size of the band-edge profile, and,
  with your parameter errors, the covariance of the energies (example
  13 and the third tutorial). `params` is one parameter set (hard
  walls) or one per grid point (a layered stack).

**Calibration from your own measurement**

- `fit_band_offset`, `offset_sensitivity` -- example 6;
  `sheet_density_from_shift` -- example 5.
- `sheet_density_from_sdh`, `sdh_frequency` -- density from a
  Shubnikov-de Haas frequency and back (`H_OVER_E_T_NM2` is h/e in
  T nm^2); `fit_sdh_mass` -- the cyclotron mass from the temperature
  damping, with `lk_thermal_factor` the damping factor itself
  (`LK_TESLA_PER_KELVIN` = 14.69 T/K); example 11.
  `fit_sdh_dingle` (new in 0.14.0) gives the quantum lifetime and the
  Dingle temperature from the field dependence of the amplitude
  (`DINGLE_TESLA_PS` = pi m0 / e = 17.86 T ps,
  `DINGLE_KELVIN_PS` = hbar / (2 pi k_B) = 1.2157 K ps); first
  tutorial. The constants live in `kpenvelope.lab`.

**Rashba term**

- `rashba_hamiltonian`, `rashba_splitting`, `rashba_spins` -- example 7.

Each function's docstring (`help(kpenvelope.solve_subbands)`, for
example) gives its inputs, units and conventions.

**Nitride stacks: interface charges (new in 0.17.0)**

- `NitrideMaterial` -- one relaxed layer material: in-plane lattice
  constant `a` (nm, also the edge component `b` of its misfit lines),
  `P_H` (H-referenced spontaneous polarization), proper `e31`, `e33`,
  `c13_over_c33`, a required `reference`, and for the electron solver
  the gap `eg`, conduction-band edge `ec` relative to GaN, mass `me` and
  permittivity `eps_r`. `algan(x, temperature_K, polarization)` builds
  relaxed Al(x)Ga(1-x)N from the cited end points (next section).
  `POLARIZATION_SETS` holds the two shipped sets.
  `polarization_quantum(a)` and `zinc_blende_formal_polarization(a)`
  (3/4 of the quantum) give the reference charges.
- `ingan(x)`, `inaln(x)`, `scaln(x)` and `gan(pset)` (new in 0.18.0)
  -- relaxed In(x)Ga(1-x)N, In(x)Al(1-x)N, Sc(x)Al(1-x)N (0 <= x <=
  0.5) and the GaN partner of a set; `with_numbers(material, source,
  ...)` swaps any number of a record for yours and adds your source to
  its reference. Every record carries its `parameter_set`;
  `VURGAFTMAN2003` and `AMBACHER2021_SCALN` hold the copied tables.
- `StackLayer(material, thickness, relaxation, donors, donor_depth,
  acceptors, acceptor_depth)` -- one layer of a stack, listed bottom
  first (acceptors are new in 0.18.0, for the drift-diffusion solver); `layer_strains` gives the
  mismatch, strain and in-plane lattice of every layer, each relaxed
  relative to the one below.
- `interface_charges(layers, line_charge, reference)` -- an
  `InterfaceCharge` per interface: the charge in C/m^2, e/nm^2 and
  cm^-2, the mismatch, the misfit-line density and line charge, and the
  three terms of Eq. 2. `line_charge` is "bare", "neutral",
  "electron-counting", `LineCharge(bound_fraction, trapped)`, a number
  in e/nm, or a dict per interface.
- `strained_polarization`, `misfit_line_density` (Eq. 1),
  `line_charge_limits` (bare, electron-counting and neutral line
  charges), `charge_change_per_relaxation` and `trapped_from_sum_rule`
  (Eq. 3, the test of bound against trapped charge).

**Nitride stacks: electron gases (new in 0.17.0)**

- `solve_electrons(layers, phi_B, temperature_K, line_charge, ...)` --
  self-consistent Schroedinger-Poisson electrons for the stack, with
  donors (complete or incomplete ionization), a pinned surface, a
  quantum window and an optional classical electron gas outside it.
  Returns an `ElectronResult` (band edge, densities, the 2DEG and the
  classical sheet densities, subband energies).
- `fermi_dirac_half` -- F_1/2 and F_-1/2.
- `two_carrier_hall`, `split_two_carrier` -- the single-field Hall pair
  of two parallel channels, and its exact inverse, to remove a parallel
  channel from a measurement.
- `line_scattering_rate`, `line_charge_mobility_bound`,
  `line_charge_from_mobility_drop` -- scattering of the electron sheet
  by charged lines in the interface, and the line charge a measured
  mobility allows.

**Devices: drift-diffusion and C-V (new in 0.18.0)**

- `solve_drift_diffusion(layers, transport, top, bottom, ...)` -- the
  steady state with electrons and holes (Fermi-Dirac or Boltzmann),
  donors and acceptors (complete or incomplete ionization), SRH,
  radiative and Auger recombination, optional generation, and ohmic or
  metal (Schottky) contacts, for any bias. Returns a
  `DriftDiffusionResult` (bands, quasi-Fermi levels, densities, currents,
  the charge on the top contact). `Transport` holds the mobilities, hole
  mass, lifetimes and recombination constants of a layer, with a
  required reference; `Contact` a contact.
- `solve_quasi_equilibrium(layers, top, bottom, ...)` -- the
  semiconductor at the Fermi level of its bottom contact while the top
  metal is biased, with holes optional; bottom "ohmic" or "zero-field".
- `iv_curve` and `cv_curve` -- current and quasi-static capacitance
  sweeps, warm-started from point to point; `cv_curve_quantum` -- the
  C-V sweep with quantum electrons (`solve_electrons` at every voltage); `bernoulli` -- the stable
  Bernoulli function of the fluxes.

**Measured data, design, leakage and 2D (new in 0.18.0)**

- `fit_measurements(build, start, cv, sheet, fixed, bounds, method,
  lengths)` -- fit the unknowns of one sample to its measured C-V curve
  and/or Hall density; returns a `MeasurementFit` (values, errors,
  correlations, the fitted curve, warnings). `correlation_warnings`
  gives the sentences.
- `find_designs(build, variables, targets, ...)`, `design_error_bars`,
  `threshold_voltage`, `sheet_density` -- the design finder and its
  definitions; returns `Design`s.
- `richardson_constant`, `richardson_velocity`,
  `thermionic_emission_current`, `wkb_transmission`,
  `tsu_esaki_current`, `gate_leakage` -- current over and through
  barriers; `Contact(..., thermionic=True)` uses thermionic emission at
  a metal in `solve_drift_diffusion`.
- `solve_cross_section`, `cv_cross_section`, `Gate`, `device_grid` --
  the 2D cross-section (`CrossSection`); `fit_least_squares` gains
  `difference="forward"` for expensive models.

**Devices in 2D and 3D, traps and heat (new in 0.19.0; need SciPy)**

- `Device(layers, width, electrodes, mobility, surface, traps, depth,
  ...)` -- drift-diffusion of the whole device on a tensor-product grid,
  2D (x, z) or, with `depth`, 3D (x, y, z); `solve(bias, guess,
  thermal)`, `sweep(electrode, values, bias)`, `equilibrium()`,
  `transient(start, bias, times, rise_time, initial_occupancy)`,
  `emission_times()`. Results are `DeviceState`s (potential, bands,
  quasi-Fermi levels, carriers, temperature, trap occupancy, terminal
  currents and charges, heat, `sheet()`, `power`) and
  `TransientResult`s.
- `Electrode` (ohmic with an implant, or Schottky with optional
  thermionic emission, on the top or bottom face), `Surface` (pinned,
  or with states), `Mobility` (low field, saturation speed, shape,
  temperature power law, reference), `Trap` (acceptor or donor level,
  density, capture coefficients, reference, where), and
  `capture_coefficient`.
- `ThermalMaterial`, `ThermalLayer`, `ThermalModel`,
  `boundary_resistance`, `solve_heat`, and the shipped `GAN_BULK` and
  `SIC_4H_ROOM`.

**Speed, breakdown, pulses, tunneling, hot electrons and shapes (new in 0.20.0)**

- `Electrode(..., resistance=)` -- a series resistor at a terminal
  (contact or gate metal); the solve returns the contact voltages in
  `DeviceState.internal`.
- `Device.small_signal(state, frequencies)` -- the admittance matrix of
  all terminals; `with_series`, `two_port` and `rf_figures` give the
  common-source two-port, h21, Mason's U, f_T and f_max (crossing and
  20 dB-per-decade readings).
- `Device.pulse(bias, times, thermal)` and `solve_heat_transient` --
  heat that changes in time, with heat capacities in `ThermalMaterial`
  (`density`, `cp_a`, `cp_b`).
- `Ionization` and `IMPACT_GAN` -- impact ionization, inside the
  drift-diffusion solve (`Device(..., ionization=)`) or afterwards along
  a field line (`ionization_integral`, `breakdown_voltage`); `where`
  limits it to chosen layers.
- `Device.gate_tunneling(state)` -- Tsu-Esaki current with WKB
  transmission, column by column under a gate.
- `Device.electron_temperature(state, tau_energy)` -- the electrons'
  energy balance.
- `Dielectric` films on top (`cap=`), `Region` boxes cut out of the
  semiconductor (`regions=`), electrodes as metal boxes (`z0`, `z1`),
  and electrodes in several pieces (one name, one metal); `Device.field`
  gives the electric field.

**Hot electrons, band steps, tunneling, breakdown, process and meshes (new in 0.21.0)**

- `Device.solve(..., energy_relaxation=tau)` -- the electrons' energy
  balance solved with the currents (hydrodynamic model); the state carries
  `electron_temperature`.
- `Device(..., heterojunctions="thermionic")` -- thermionic emission over
  abrupt band steps, for electrons and holes.
- `TunnelBarrier(layer)` in `Device(tunnel_barriers=[...])` -- Tsu-Esaki
  tunneling and emission across a layer, coupled into the solve (also
  between a channel and a gate on top).
- `BandToBand(reference)` in `Device(band_to_band=)` -- Kane's
  band-to-band generation.
- `Device.breakdown_by_current`, `Device.field_check` (peak field per
  material against critical fields you give), `Device.thermal_runaway`.
- `Device(doping=f)` with `f(x, z) -> (N_d, N_a)`; module `process`:
  `Implant`, `Anneal`, `Process`.
- Module `mesh`: `Mesh`, `mesh_from_points`, `mesh_from_grid`,
  `MeshRegion`, `MeshDevice`. `Mobility` now also takes a negative
  temperature exponent (a mobility that rises with temperature).

**Temperature, tunnel diodes, meshes and process shapes (new in 0.22.0)**

- `Device(..., temperature_model="full")` -- the lattice temperature
  also sets the carrier statistics: kT, the densities of states (as
  T^(3/2)), the band gap (with `band_gap_temperature=Varshni(alpha,
  beta, conduction_share, reference)`, one for all layers or a dict by
  layer), trap emission and the contacts. The currents then include the
  push of a temperature gradient (thermoelectric currents). The default,
  `"mobility"`, is the earlier model, where only the mobility follows
  the temperature.
- `Device.solve(..., lattice_temperature=T)` -- a lattice temperature you
  set (one value, or one per node) instead of one from the heat equation.
- `Device.solve(..., energy_relaxation=tau, thermal=model)` -- hot
  electrons and self-heating together. The lattice gets the electrons'
  heat where they cool down (`DeviceState.lattice_heat`).
- `BandToBand(..., model="nonlocal")` -- electrons tunnel from the
  valence band at one place to the conduction band at another, along the
  grid's vertical lines. It works in both directions: it makes pairs in
  reverse bias and removes them in forward bias, which gives the current
  of a tunnel (Esaki) diode. `DeviceState.band_to_band_rates` holds the
  electron and hole rates.
- `MeshDevice` now takes `thermal=` (the heat equation on the mesh; every
  region needs a thermal material, the substrate goes into the mesh),
  `pulse`, `tunnel_barriers=` (the barrier meshed in vertical columns),
  `temperature_model` and `band_gap_temperature`.
- `process.Structure(width, z_bottom, z_top, h, substrate, substrate_top)`
  -- a cross-section built step by step on square cells:
  `deposit(material, thickness, mode="conformal" or "directional",
  window=, semiconductor=, donors=, acceptors=)`, `etch(rates, time,
  mode="isotropic" or "directional", window=)`, `oxidize(material,
  oxide, B, B/A, time, consumed_fraction, reference)` (Deal-Grove),
  `implant(Implant)`, `anneal(Anneal)`, then `mesh()`, `doping()` and
  `surface_nodes(mesh, material, x0, x1)` to build a `MeshDevice`.
  `thickness`, `area` and `top_surface` read the shape.
- Changed: trap emission is scaled by the degeneracy factor of its band
  with Fermi-Dirac statistics, so traps no longer recombine in
  equilibrium where a band is degenerate. Self-heating converges faster
  near thermal runaway, and runaway is recognized sooner.

**Peltier heat, tunneling with heat, tunneling on meshes (new in 0.23.0)**

- `Device(..., heat_model="thermoelectric")` (also `MeshDevice`) -- the
  heat source includes the Peltier and Thomson heat: each carrier flow
  carries the heat Pi = kT (5/2 + ln(N_c / n)) per electron (and the
  same with N_v / p for holes), Pi being the package's own thermoelectric
  power times T. Where the flow crosses a change of Pi, that heat is
  released or taken. It only moves heat around, so the total heat is
  still the electrical power. The default, `"joule"`, is the earlier
  model (Joule heat plus recombination heat).
- Band-to-band tunneling (local or nonlocal) now runs together with
  `thermal=`, `energy_relaxation=` and `lattice_temperature=`.
- `BandToBand(..., model="nonlocal")` on a `MeshDevice`: the paths are
  straight parallel lines across the mesh, `direction=(dx, dz)` (default
  `(0, 1)`, vertical) and `path_spacing` (nm, default the median edge
  length). On a tensor grid the paths stay the grid's columns.
- `nitride.gap_shift(alloy, x, conduction_share)` -- how the gap of
  GaN, AlN, InN, AlGaN, InGaN or InAlN changes with temperature, from
  the same cited Varshni numbers as `algan`, `ingan` and `inaln`. Use it
  as `band_gap_temperature` (one per layer).
- `Device.thermal_runaway(..., stop_above=T)` -- stop the sweep once the
  hottest point passes T (the result's `too_hot` says where).
- The app: a Process steps tab, a thermal runaway search, band-to-band
  tunneling in the I-V sweep, and in the Transistor tab the full
  temperature model, Peltier heat, thermionic emission over the barrier
  step, and hot electrons together with self-heating.

**Quantum holes, ballistic limit, bouncing particles (new in 0.25.0)**

- `QuantumChannel(..., holes=(m_z, m_inplane, reference), electrons=True)`
  -- holes as quantum states too (one band, your masses with their
  source); `DeviceState.quantum` then also holds `hole_subbands`,
  `hole_sheet_cm2` and `hole_potential`.
- The quantum channel now runs together with band-to-band tunneling
  (the tunneling sees the real bands).
- `Device.ballistic_current(state, source, drain)` -- the ballistic limit
  of the current through the channel's subbands (top-of-the-barrier
  model), with the current per subband.
- `Structure.view_factors()` -- the diffuse view factors between all
  exposed faces; `deposit(..., mode="flux", sticking=)` and
  `etch(..., mode="flux", sticking=)` -- particles that do not stick are
  re-emitted by the cosine law and land elsewhere.
- The app: `sticking=` in the flux steps of Process steps.

**Quantum channel, line-of-sight process steps, 2D oxidation (new in 0.24.0)**

- `QuantumChannel(window, x_range=None, n_states=400, tol=1e-5,
  max_outer=200, max_shift=1.0)` in `Device(quantum=...)` -- electrons
  inside the window (z_min, z_max in nm) as quantum states: along every
  grid column the one-band Schroedinger equation with the layer masses,
  filled with the local electron quasi-Fermi level, fed back into the
  current and the charge as a quantum potential until both agree.
  `DeviceState.quantum_potential` holds it and `DeviceState.quantum` the
  levels and the confined sheet density of every column; `ec` stays the
  real band edge. It runs with traps, holes, self-heating, the full
  temperature model and hot electrons; `small_signal`, `transient` and
  `pulse` keep the quantum potential of their starting state. Tensor
  grids only; not together with `band_to_band`, with tunnel barriers or
  thermionic band steps inside the window.
- `device.column_states(z, ec, mass, n_states, z_left, z_right)` -- the
  eigenstates of one column on any grid.
- `Structure.fluxes(exponent)` -- the flux of particles from above onto
  every exposed face, relative to a flat open surface, with the in-plane
  angular distribution cos^exponent (1: a gas at rest; large: an ion
  beam) and exact shadowing by the cells.
- `Structure.etch(rates, time, mode="flux", exponent=, beams=)` and
  `Structure.deposit(material, thickness, mode="flux", exponent=)` --
  etching and deposition at those fluxes (narrow openings etch and fill
  more slowly; `beams` combines a gas and a beam).
- `Structure.oxidize(..., mode="2d")` -- the oxidant diffuses through
  the oxide in two dimensions, so oxide grows under mask edges (LOCOS)
  and on trench walls.
- The app: in the Transistor tab, "treat the channel electrons as
  quantum states"; in Process steps, mode=flux (with exponent=) for
  deposit and etch, and mode=2d for oxidize.

**Phonons (new in 0.18.0)**

- `PolarPhonons` -- the E1 and A1 TO and LO phonons and high-frequency
  dielectric constants of one wurtzite material (with a reference), its
  dielectric functions and Lyddane-Sachs-Teller static constants;
  `GAN_BARKER1973` is the shipped GaN record.
- `single_interface_modes(m1, m2)` and `slab_modes(slab, outside, d,
  q)` -- interface phonons of one interface and of a slab in the
  dielectric continuum model, for uniaxial media; `frohlich_constant`
  -- the Froehlich electron-LO-phonon coupling constant.

**The misfit-line charge (new in 0.18.0)**

- `PUBLISHED_CONSTRAINTS` -- the published numbers that bear on the line
  charge, each a `Constraint` with its source and assumptions.
- `combine_line_charge(constraints)` -- one estimate from Gaussian
  estimates and upper limits, in closed form (`LineChargePosterior`).
- `plan_holography`, `plan_sum_rule`, `line_charge_error_from_density`
  -- the error a planned measurement would give.

**The app (new in 0.18.0)**

- `kpenvelope-app` (or `python -m kpenvelope.gui`) -- the app in your
  browser; `kpenvelope.gui.api` holds its calculations as plain
  functions, and `kpenvelope.gui.make_server` its local server.

**Error bars and fits (new in 0.17.0)**

- `propagate(func, values, sigmas, method)` -- the error of any
  calculation from the errors of its inputs, to first order (with an
  error budget and optional correlations) or by Monte Carlo (with
  percentiles and optional bounds). Returns a `Propagation`.
- `fit_least_squares(model, x, y, theta0, y_sigma, x_sigma)` --
  weighted least squares (Levenberg-Marquardt), with errors in the
  settings as well as the readings (orthogonal-distance regression);
  error scale from the scatter when no reading errors are given.
  Returns a `FitResult`.
- `fit_interface_series(build_layers, settings, sheet_measured, ...)`
  -- the calibration charge and misfit-line charge that reproduce a
  measured series, each prediction a full `solve_electrons` run.

## Cited parameter sets

No physical number in this package is made up, and none is accepted
without a source: the `reference` field of `WurtziteParameters` is
mandatory.

- `gan_rinke2008()`: the GW-based GaN valence parameters A1..A6 of
  Rinke et al., Phys. Rev. B 77, 075202 (2008), with Delta_CR = 10 meV
  and Delta_SO = 17 meV, all as tabulated in Extended Data Table 1 of
  Chang et al., Nature Electronics 9, 346 (2026). The two splittings
  are not Rinke's own results: Rinke et al. compute a crystal-field
  splitting of 34 meV and use 10 meV from other work. eps_r = 10.4 (field along
  the c axis) from Barker and Ilegems, Phys. Rev. B 7, 743 (1973).
- `aln_rinke2008()`: the matching AlN set (Delta_CR = -295 meV from
  Rinke et al.; Delta_SO = 22 meV from de Carvalho et al., Appl.
  Phys. Lett. 97, 232101 (2010)), intended as a barrier material. Its
  permittivity is deliberately NaN and the self-consistent solver
  refuses to run on it: no vetted value is shipped, and none is
  needed for a barrier.
- `demo_single_band()`: a synthetic, decoupled set the test suite
  uses because it has exact textbook solutions. Labeled non-physical.

For any other material, populate `WurtziteParameters` from the
literature (e.g. Vurgaftman and Meyer, J. Appl. Phys. 94, 3675
(2003)) and record the source. No default band offset, deformation
potential or Rashba coefficient is shipped either: they depend on the
material, the strain and the sample, so you supply them, with a
citation.

For nitride stacks (new in 0.17.0) two polarization sets are shipped,
because the literature has not settled the numbers:

- `"pbesol-mahim2026"` (default): Berry-phase calculations with the
  PBEsol functional in Quantum ESPRESSO 6.7: P_H = 1.33685 (GaN) and
  1.33299 C/m^2 (AlN), proper e31 = -0.43876 and -0.66484, e33 =
  0.76211 and 1.64001 C/m^2, C13/C33 = 0.24177 and 0.30257. Every input
  and output file is archived in T. M. Mahim, nitride-misfit-polarization
  v1.3.1, [doi:10.5281/zenodo.24315036](https://doi.org/10.5281/zenodo.24315036);
  the package keeps the full precision of its
  `results/dft_polarization.json`.
- `"hse-dreyer2016"`: the hybrid-functional values of Dreyer, Janotti,
  Van de Walle and Vanderbilt, Phys. Rev. X 6, 021038 (2016), Tables I
  and II: P_H = 1.312, 1.351 and 1.026 C/m^2 (GaN, AlN, InN), proper
  e31 = -0.551, -0.676 and -0.604, e33 = 1.020, 1.569 and 1.238 C/m^2.
  That paper gives no C13/C33, so `algan` takes the PBEsol ratio unless
  you pass your own, and `ingan` and `inaln` take c13 and c33 of
  Vurgaftman and Meyer; each `reference` string says which.
- `"vurgaftman2003"` (new in 0.18.0): Vurgaftman and Meyer, J. Appl.
  Phys. 94, 3675 (2003), Table I: spontaneous polarization with the
  zinc-blende reference (-0.034, -0.090 and -0.042 C/m^2 for GaN, AlN
  and InN), elastic constants c11, c12, c13, c33 (390, 145, 106, 398;
  396, 137, 108, 373; 223, 115, 92, 224 GPa) and piezoelectric moduli
  d13, d33 (-1.6, 3.1; -2.1, 5.4; -3.5, 7.6 pm/V), and the polarization
  bowings of Table III (C = -0.021, -0.037 and -0.070 C/m^2 for AlGaN,
  GaInN and AlInN). e31 and e33 follow from the moduli by e31 = (c11 +
  c12) d13 + c13 d33 and e33 = 2 c13 d13 + c33 d33 (Eqs. 40a and 40b of
  Ambacher et al. 2021), which gives e31 = -0.527, -0.536 and -0.484 and
  e33 = 0.895, 1.561 and 1.058 C/m^2.
- `"ambacher2021"` (new in 0.18.0): Ambacher et al., J. Appl. Phys.
  130, 045102 (2021), for Sc(x)Al(1-x)N with 0 <= x <= 0.5: a(x) (Eq.
  3a), P_SP(x) (Eq. 18c, zinc-blende reference), C13(x), C33(x) (Eqs.
  23c, 23d), e31(x), e33(x) (Eqs. 39b, 39c) and eps33(x) (Eq. 8), each
  copied from the printed equation and checked in the tests against the
  percentages the paper quotes in its text. Its GaN has a = 3.1986
  Angstrom and P_SP = -0.034 C/m^2 (Eqs. 1a and 18a at x = 1). The paper
  does not say whether its e31 is proper; it uses it with the
  zinc-blende reference, which is this package's ZB recipe, so it is
  taken as the proper constant.

Sets that give the polarization with the zinc-blende reference are
converted to the H branch on the lattice of the relaxed alloy, P_H =
P_ZB + e sqrt(3) / (2 a^2), so that every record means the same thing.

The rest of `algan` comes from Vurgaftman and Meyer, J. Appl. Phys. 94,
3675 (2003) (in-plane lattice constants at 300 K, 0.3189 and
0.3112 nm; gaps at 0 K, 3.510 and 6.25 eV, with Varshni alpha 0.909
and 1.799 meV/K and beta 830 and 1462 K; electron masses along c, 0.20
and 0.32 m0; AlGaN gap bowing 0.7 eV) and Ambacher et al., J. Appl.
Phys. 87, 334 (2000) (dielectric constants along c, 10.4 and 10.7;
conduction-band share of the gap difference, 0.7). Alloys follow
Vegard's rule except the gap. These are the numbers of the misfit
analysis the tests reproduce; `cbo_fraction` and `psp_bowing` exist
for sensitivity tests.

InGaN and InAlN (new in 0.18.0) add InN from the same Table I (a =
0.3545 nm, gap 0.78 eV with Varshni 0.245 meV/K and 624 K, mass 0.07
m0), the gap bowings of Table III (GaInN 1.4 eV; AlInN 2.5 eV, which
the authors call provisional) and the wurtzite InN/GaN valence-band
offset of 0.5 eV with no bowing of the offsets (their Sec. II F). For
InAlN the valence band is linear between InN and AlN, which is
transitivity: an assumption, since Vurgaftman and Meyer leave the
wurtzite InN/AlN offset open. InN's static dielectric constant along c,
14.4, is from Davydov et al., Appl. Phys. Lett. 75, 3297 (1999), as
listed by the Ioffe NSM database (the abstract of that paper gives the
ratio 0.91 of the two directions, which matches 13.1/14.4). ScAlN has
no band numbers in its source, so `scaln` leaves them NaN for you to
fill with `with_numbers`.

Phonons (new in 0.18.0): GaN only, from Barker and Ilegems, Phys. Rev.
B 7, 743 (1973) (LO phonons 746 and 744 cm^-1, static constants 9.5
and 10.4, eps_inf 5.35) with TO phonons of 559 cm^-1 (Lemos et al.
1972) and 533 cm^-1 (Manchon et al. 1970), all as listed by the Ioffe
NSM database. The listed numbers pass the Lyddane-Sachs-Teller check to
0.3 percent. No transport number (mobility, lifetime, recombination
constant, hole mass) is shipped: there is no single accepted set, so
`Transport` asks for yours with a reference.

Devices, traps and heat (new in 0.19.0). The library ships two thermal
records and nothing else: `GAN_BULK`, 230 W/(m K) at 300 K (the highest
room-temperature value for low-dislocation bulk GaN in C. Mion et al.,
Appl. Phys. Lett. 89, 092123 (2006), doi:10.1063/1.2335972) falling as
(T/300 K)^-1.43 (the 300-450 K dependence of the same measurements, as
quoted by B. A. Danilchenko et al., Solid State Commun. (2007),
doi:10.1016/j.ssc.2007.08.007); and `SIC_4H_ROOM`, 370 W/(m K)
(Goldberg et al. 2001, as listed by the Ioffe database), held constant
because that source gives no temperature dependence for 4H-SiC. Thin,
defective GaN conducts less heat than bulk; give your own number if you
have it. `Mobility` and `Trap` always take your numbers with a
reference. The app's Transistor tab fills them, and says so, from: a
mobility of 1704 cm^2/Vs measured at room temperature falling as
(T/300 K)^-2.35 (H. S. Alpert et al., Rev. Sci. Instrum. 91 (2020),
doi:10.1063/1.5139911, arXiv:1912.00977); a saturation speed of
10^7 / (0.38 + (n_s / 1.8e13)^0.45) cm/s from the electron density (the
fit of S. Bajaj et al., Appl. Phys. Lett. 107 (2015),
doi:10.1063/1.4933181, to their drifted Fermi-Dirac calculations, which
matched their pulsed measurements); and iron as an acceptor 0.5 eV below
the conduction band (D. Wickramaratne et al., Phys. Rev. B 99, 205202
(2019); M. J. Uren et al., IEEE Trans. Electron Devices 59, 3327
(2012)), with a capture cross section of 1e-15 cm^2 and 1e15 cm^-3
shallow compensating donors as in Uren et al. Carbon's acceptor level,
0.90 eV above the valence band (J. L. Lyons et al., Appl. Phys. Lett.
97, 152108 (2010)), needs holes, so it is not in the app; use `Trap`
with `hole_mass` in the library.

The misfit-line charge is not shipped as a default. The misfit analysis
measured -0.84 +- 0.53 e/nm, or 0.32 +- 0.20 of the bare-end charge
(0.22 +- 0.20 with the HSE constants), in one AlGaN/GaN series; pass it
as `LineCharge(bound_fraction=0.32)` or as a number if you want it, and
carry its error with `propagate`.

## When it refuses, and why

`kpenvelope` raises a `ValueError` instead of guessing when:

- a parameter set has no permittivity (`eps_r` is NaN, as for AlN) and
  a self-consistent solve is asked for;
- a strained calculation is asked of a set without D1..D6, or the
  strain tensor is not a symmetric 3 x 3 tensor;
- the grid is not uniform, not increasing or shorter than three
  points, the layer thicknesses do not add up to the grid span (within
  half a grid step), or the per-point parameter, band-edge, strain,
  potential or permittivity arrays do not match the grid, or the band
  edge is not finite;
- `n_states` is below 1 or above six times the number of grid points;
- a layer thickness is negative or not finite, or no layer is given;
- a sheet density `ps` is negative or not finite, or `mixing` or `tol`
  is not a positive finite number;
- `fill_subbands_thermal` gets energies and masses of different
  shapes, a negative mass, a positive `ps` with no finite mass to put
  it in, or a `ps` that does not fit within 5 eV of the subband edges;
- a permittivity is not a positive finite number, or `sheet_z` lies
  outside the grid;
- `filling="kgrid"` is asked for without `kmax`, or with a strain that
  breaks in-plane isotropy and no `ntheta`;
- a temperature is negative or not finite, or `max_iter` is below 1;
- the momentum-grid filler, or a loop with `filling="kgrid"`, finds
  holes still present at the edge of its grid (at T = 0 an occupied
  state at `kmax`; at T > 0 an occupation above 1e-4 on the outer
  ring) -- increase `kmax`;
- momenta are negative, or `local_mass` gets fewer than two points,
  momenta that do not increase, or a row count that does not match;
- `dos_from_dispersion` gets a dispersion that does not fall
  monotonically (use `fill_subbands_kgrid`), or arrays of the wrong
  shape (below the end of the momentum path, and above its start when
  it does not start at 0, it returns NaN: the path says nothing
  there);
- `cyclotron_mass` gets a path that does not start at 0, a Fermi
  wavevector beyond the path, or a branch that does not lie above the
  Fermi energy exactly inside the Fermi wavevector (then the occupied
  region is not one disk);
- envelope arrays have the wrong shape for `band_character` or the
  depolarization functions;
- `depolarization_shift` gets degenerate states, a negative sheet
  density, or a permittivity that is not a positive finite number;
  `isb_lineshape` gets a non-positive linewidth or unmatched arrays;
- `fit_band_offset` gets a bracket whose two ends do not straddle the
  measured spacing (the message gives the calculated spacing at both
  ends), a transition too insensitive to the offset (see example 6),
  an all-zero or non-finite profile shape, a profile shape that is not
  on the grid, a bracket that is not finite or whose low end is not
  below its high end, or two identical or negative state indices;
- `sheet_density_from_shift` gets a measured line at or below the bare
  spacing (the depolarization shift only pushes the line up);
- in `fit_band_offset` or `sheet_density_from_shift`, the measured
  value or its error `sigma_e_ev` is not a positive finite number;
- a Rashba call has no real `reference` (fewer than 8 characters), a
  non-finite coefficient or a negative momentum, or asks for spin
  directions at `k = 0` or with `alpha = 0`, where the two branches
  are degenerate;
- `sheet_density_from_sdh` or `sdh_frequency` gets no `branches`, or
  one that is not a positive integer, or a frequency or density that
  is not a positive finite number;
- `fit_sdh_mass` gets fewer than 3 distinct temperatures, amplitudes,
  temperatures or a field that are not positive, amplitudes that do
  not fall with temperature, or data so weakly damped that the error
  bar would not be honest (the mass error is as large as the mass, or
  the chi-square does not rise by about one unit, 0.7 to 1.3, at one
  sigma on either side);
- `fit_sdh_dingle` gets fewer than 3 distinct fields, amplitudes,
  fields or a mass that are not positive, a negative temperature,
  amplitudes that are not damped towards low field once the
  temperature factor is divided out, or data whose quantum lifetime
  has an error as large as itself;
- `degenerate_levels` or `level_transitions` get energies that are not
  sorted from the highest down, envelopes that do not match the
  energies or the grid, an initial level that does not exist, or a
  mass that is not positive;
- `parameter_sensitivities` gets an unknown parameter name, a
  negative or non-finite error, an error of the band-edge scale
  without a band edge, parameter sets that do not match the grid, or
  a single mapping of errors for a stack of several materials;
- `solve_self_consistent_hetero` is asked for the parabolic filling
  (the default) with a strain that breaks in-plane isotropy (since
  0.16.0; use `filling="kgrid"` with `ntheta`, see
  [Corrections](checks.md#corrections-in-earlier-versions));
- `dipole_matrix` gets envelopes that are not an array of states on
  the grid (one state is `envelopes[None]`), or `oscillator_strengths`
  gets a mass ratio that is not a positive finite number or a dipole
  matrix that does not match the energies (since 0.16.0);
- `velocity_matrix` or `inplane_transitions` get envelopes that do not
  match the grid or the energies, parameter sets that do not match the
  grid, a momentum or an angle that is not finite, or an initial level
  that does not exist.

It does not stop, but warns (`RuntimeWarning`), when a self-consistent
loop or `fill_subbands_kgrid` finds that a state below the computed
`n_states` would hold holes (example 2). With the momentum grid
(`filling="kgrid"`, or `fill_subbands_kgrid` given more energies than
it fills) this is checked at every grid momentum since 0.15.0, so a
branch that rises above its `k = 0` energy before it falls is caught
too.

For nitride stacks (new in 0.17.0) it also refuses:

- a relaxed interface without a `line_charge` (the error says which
  choices exist, since no polarization reference decides it), or a line
  charge it cannot read;
- a bottom layer with a relaxation other than 0 (the bottom layer is the
  relaxed substrate; put the layer it relaxed on below it);
- a relaxation outside [0, 1], a thickness or donor density that is not
  positive and finite, a `NitrideMaterial` without a reference or with a
  non-positive lattice constant, mass or permittivity, an Al fraction
  outside [0, 1] or an unknown polarization set;
- an electron solve on a material whose `ec`, `me` or `eps_r` is NaN,
  or an extra sheet at an interface that does not exist;
- a two-carrier split in which the parallel channel alone would carry
  more than the measurement;
- a fit with errors in the settings but none in the readings (their
  balance cannot come from the scatter), or with fewer readings than
  parameters; a propagation with a sigma for an unknown input, a
  correlation outside [-1, 1] or a correlation matrix that is not
  positive semidefinite.

Since 0.18.0 it also refuses (or warns):

- a strained layer whose record has no piezoelectric constants (the GaN
  of the ScAlN paper, which can only be the bottom layer), a set without
  the material you ask for (the PBEsol set has no InN), an In fraction
  outside [0, 1] or a Sc fraction outside [0, 0.5];
- `with_numbers` without a source, or with a field it cannot replace;
  it also drops the record's `parameter_set` when you change a
  polarization or lattice number, since it is no longer the published
  set;
- it warns (UserWarning) when two layers of different sets meet;
- `solve_electrons` on a layer with acceptors (it has no holes);
- a `Transport` without a reference, with only one SRH lifetime, or
  with a mobility or mass that is not positive; a Schottky contact
  without a barrier; a "zero-field" bottom in a current calculation;
  an ohmic contact on a layer with no neutral level; a quasi-equilibrium
  solve of a stack with acceptors but no hole mass;
- a drift-diffusion bias ramp that does not converge, with the likely
  reason (a region with almost no free carriers between the contacts,
  where a vertical current is not defined) and what to use instead;
- a `PolarPhonons` record without a reference or with an LO phonon
  below its TO phonon.

## Where it comes from

The package's methodological basis is

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas",
> Phys. Rev. B 114, L291304 (2026),
> https://doi.org/10.1103/zn44-nmx8; code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

and follows S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491
(1996). This package is the general-purpose tool; the paper repository
reproduces the specific published study.

The nitride-stack part (new in 0.17.0) implements Eqs. 1 to 3 of

> T. M. Mahim, "The charge of misfit dislocations decides the
> polarization charge at relaxed nitride interfaces" (2026, manuscript);
> code and data: https://github.com/Tanvir-Mahmud-Mahim/nitride-misfit-polarization,
> archived at https://doi.org/10.5281/zenodo.24315036

whose analysis it reproduces, and builds on Dreyer et al., Phys. Rev. X
6, 021038 (2016), and Wang et al., Appl. Phys. Lett. 124, 263502
(2024). The electron solver follows the self-consistency scheme of
Trellakis et al., J. Appl. Phys. 81, 7880 (1997). The material tables of
0.18.0 are copied from Vurgaftman and Meyer, J. Appl. Phys. 94, 3675
(2003), and Ambacher et al., J. Appl. Phys. 130, 045102 (2021); the
drift-diffusion fluxes are those of Scharfetter and Gummel, IEEE Trans.
Electron Devices 16, 64 (1969); the interface phonons follow the
dielectric continuum model of Fuchs and Kliewer, Phys. Rev. 140, A2076
(1965), and Mori and Ando, Phys. Rev. B 40, 6175 (1989). Currents over
and through barriers use the formula of Tsu and Esaki, Appl. Phys. Lett.
22, 562 (1973), and the thermionic-emission boundary of Crowell and Sze,
Solid-State Electron. 9, 1035 (1966).
