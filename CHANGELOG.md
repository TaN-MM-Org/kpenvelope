# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

## 0.31.0 (2026-11-22)

The vibrations of an interface join the mobility, a model of the
plasma's chemistry feeds the etch, and the app lets electrons change
level in the wave current. The examples print what they printed before;
examples 71 and 72 are new.

### Added

- `Subbands.interface_phonons`: the polar phonons of one interface
  between two materials (half-space LO modes of each side and the
  interface modes, Mori and Ando 1989), and `scattering.interface_modes`.
  `Subbands.mobility` now solves the Boltzmann equation with phonons of
  several energies on one grid.
- `kpenvelope.plasma`: `Species`, `Reaction`, `GlobalPlasma` (a
  volume-averaged model of an electropositive plasma with your rates:
  densities, electron temperature, pressure, the ion and neutral fluxes
  and the ions' energy), `fluxes_cm2` for `EtchKinetics`, and `SCCM`.
- The app, Electrons tab: the mobility can count the vibrations of the
  barrier and of the boundary (your barrier numbers, with their source).
- The app, Transistor tab: the wave current can let electrons change
  level along the channel (the four lowest, joined).

### Not included (see docs/limits.md)

- Phonon self-energies in NEGF (they need the width's wave numbers on a
  grid), scattering in coupled mode space, NEGF on triangle meshes; the
  confined phonons of thin layers and several interfaces; any shipped
  plasma rate set, negative ions; the oxide's shape following its flow,
  and elasticity; better electronic stopping for implants; tunnel paths
  that bend with the field; tunneling and heat inside one Newton solve
  (one to five rounds per voltage in the tests); the hot-electron model's
  limit at zero relaxation time (a property of the model).

### Checks (tests)

5 new tests (399 in all; docs/checks.md has the tolerances): the
interface modes against the dielectric-continuum roots and the damped
peak areas; the sum of the modes against a point charge's potential
solved on a grid; the mobility with interface phonons against bulk
(same material, far interface), detailed balance and grid convergence;
the plasma against its balances written out by hand; the app's new
options against the library. All 72 documented examples were rerun.

## 0.30.0 (2026-11-21)

The app catches up with the library, and the levels of a 2D gas screen
charges and roughness together. The examples print what they printed
before; example 70 is new.

### Added

- `Subbands(..., screening="full")`: the static dielectric matrix of all
  pairs of subbands (Ando, Fowler and Stern 1982), with the Lindhard
  polarization of every pair at the temperature (Maldague's integral),
  for the ionized impurities and the roughness. New methods `pairs`,
  `polarization` and `screened`.
- The app, Transistor tab: the current at one bias with the electrons
  as waves (NEGF) for short transistors, next to the ordinary current and
  the ballistic limit, ballistic or with scattering at the rate of the
  tab's mobility; the Fermi-Dirac heat per electron with the Peltier
  heat (on by default when Peltier heat is ticked).
- The app, Process steps: `implant ranges=simulate` (the Monte Carlo of
  the ions), `plasmaetch` (a reactive gas and ions together), and
  `oxidize ... mode=2d viscosity=` (the oxide flows; the summary gives
  the largest stress) with `stress_volume=` and `temperature=`; a note
  when part of an implant's dose leaves the picture or lands in a mask.
- The app, Electrons tab: the mobility uses the full screening.

### Changed

- The app's mobility with donors or roughness is lower than in 0.29.0
  (by a fifth to a quarter in the cases tried), because the full
  screening hides charges and bumps less than one Thomas-Fermi number.
- A mobility typed in the Transistor tab without a source is listed as
  your own number instead of being credited to the paper of the default.

### Not included (see docs/limits.md)

- Phonon self-energies in NEGF; scattering in coupled mode space; NEGF
  on triangle meshes; interface phonon modes in the lifetimes; the
  plasma's chemistry; the oxide's shape following its flow, and
  elasticity; better electronic stopping for implants; tunnel paths that
  bend with the field; tunneling and heat inside one Newton solve; the
  hot-electron model's limit at zero relaxation time (a property of the
  model). The app does not offer meshes, NEGF holes, coupled modes or 3D.

### Checks (tests)

7 new tests (394 in all; docs/checks.md has the tolerances): the pair
polarization against a finite difference of the Hamiltonian and against
a direct 2D integral; the screened elements against a separate solve on
the grid; the app's oxide flow, simulated implant and plasma etch
against the library; the app's quantum current against the ballistic
limit; the dense-heat option and the mobility's source. All 70
documented examples were rerun.

## 0.29.0 (2026-11-21)

Electrons of a 2D gas can scatter between its levels, and the app finds
the mobility of the electron layer. The examples print what they printed
before; example 69 is new.

### Added

- `kpenvelope.scattering.Subbands`: several subbands with their
  envelopes, every mechanism of `Subband` within and between them, and
  the linearized Boltzmann equation of all of them solved together
  (exactly for elastic scattering; with polar optical phonons on energy
  cells with an edge at every subband bottom). `mobility` returns the
  total, each mechanism alone and each subband; `lifetimes` the quantum
  and transport times of each subband.
- `column_states(..., signed=True)`: the envelopes themselves, for
  overlaps between states.
- The app: "Find the mobility" in the Electrons tab.

### Not included (see docs/limits.md)

- Phonon self-energies in NEGF; scattering in coupled mode space; NEGF on
  triangle meshes; interface phonon modes and polarization between
  subbands in the lifetimes. The process, tunneling and hot-electron
  limits are as in 0.28.0. The app has the mobility but not the other
  features added since 0.24.0.

### Checks (tests)

6 new tests (387 in all; docs/checks.md has the tolerances): the signed
envelopes; one subband against the single-subband model; two wells that
do not overlap against two independent gases; three coupled subbands
against the exact isotropic times and against Siggia and Kwok's solution
at each energy; detailed balance and grid convergence with polar optical
phonons; the app's mobility against the library. All 69 documented
examples were rerun.

## 0.28.0 (2026-11-21)

Polar optical phonons join the scattering lifetimes, with an exact
solution of the Boltzmann equation; meshes can adapt to the solution;
the thermoelectric current and Peltier heat can take their Fermi-Dirac
form. The examples print what they printed before; examples 66 to 68
are new.

### Added

- `Subband.polar_optical(phonon_eV, eps_static, eps_high, ...)`: LO
  phonon scattering through the Froehlich coupling with the subband's
  form factor. With it, `Subband.mobility` solves the linearized
  Boltzmann equation exactly (Rode's method in 2D, with the Pauli factors
  of a degenerate gas) together with the elastic mechanisms.
- `kpenvelope.mesh.recovered_hessian` and `size_from_solution`: second
  derivatives of a nodal field (exact for quadratics) and a size field
  for `quality_mesh` that spreads the interpolation error evenly.
- `Device(..., thermoelectric="fermi-dirac")` (also `MeshDevice`): the
  thermal diffusion and the Peltier heat of a degenerate gas, kT
  ((5/2) F_3/2 / F_1/2 - eta) per electron, for a constant scattering
  time.

### Changed

- `MeshDevice` judges a negative dual face against the edge's own
  length: on very fine meshes a right angle on a straight boundary (a
  face of zero, slightly negative after rounding) no longer warns.

### Not included (see docs/limits.md)

- Phonon self-energies in NEGF; scattering in coupled mode space; NEGF on
  triangle meshes; scattering between subbands and interface phonon
  modes in the lifetimes; an automatic adapt-and-solve loop and
  anisotropic meshes. The hot-electron model's limit at zero relaxation
  time with velocity saturation, the plasma's own chemistry, oxide shapes
  that follow the flow, elasticity, electronic stopping beyond
  Lindhard-Scharff, tunneling paths that bend with the field, and
  tunneling and heat inside one Newton solve are as in 0.27.0. The app
  does not have the new features yet.

### Checks (tests)

9 new tests (381 in all; docs/checks.md has the tolerances): the polar
optical angular integrals against their elliptic-integral closed forms;
the exact Boltzmann solution against the elastic limits (vanishing
coupling, small phonon energy), detailed balance of the collision
operator and convergence in the energy step; recovered second
derivatives exact for quadratics; an adaptive mesh of an L-shaped
conductor against uniform ones; F_3/2 against quadrature and the
Fermi-Dirac Peltier factor against its limits and arbitrary precision;
the thermoelectric current of a degenerate bar against its integral;
the Peltier heat at the contacts; the Jacobian of the new fluxes. All
68 documented examples were rerun.

## 0.27.0 (2026-11-21)

Electrons can change subband in quantum transport, holes can be solved
by it too, the package makes quality triangle meshes, the quantum channel
on meshes keeps band steps where the mesh puts them, and scattering
lifetimes and mobility of a 2D gas are computed from its own
wavefunction. The examples print what they printed before, except
example 58 (the quantum channel on a mesh, now closer to the tensor
grid); examples 62 to 65 are new.

### Added

- `Device.negf(..., coupled_modes=M)`: NEGF in coupled mode space, the
  lowest M states of every column joined by their overlaps, so electrons
  change subband where the channel's profile across it changes; with all
  modes it is the full 2D problem (ballistic only). `kpenvelope.negf`
  adds `Slab`, `slab_transport` and `lead_surface_green` (leads by
  decimation).
- `Device.negf(..., carrier="holes")`: the holes of a QuantumChannel with
  `holes=(m_z, m_inplane, ...)`, with or without scattering, uncoupled or
  coupled; `Slab(mass_x=)` for a mass along x of its own.
- `kpenvelope.mesh.quality_mesh(regions, h, min_angle)`: quality meshes
  of polygon regions by Delaunay refinement (every boundary and interface
  an edge, no node in a segment's diametral circle, smallest angle up to
  34 degrees, a size field).
- `kpenvelope.scattering.Subband`: transport and quantum lifetimes and
  the mobility of one subband from acoustic phonons, alloy disorder,
  interface roughness (Gaussian or exponential correlation) and screened
  charged impurities (sheets or profiles), with your material numbers.

### Changed

- The quantum channel on meshes: the material on each sample of a line is
  the mean over its share of the line. Before, a node on a band step took
  its whole share from the one region it belongs to, which moved the
  step by up to half a spacing, differently on each line: on a quality
  mesh the confined sheet was 1.6 to 2.0 percent off a fine tensor grid
  at every mesh size; now 0.54, 0.20 and 0.10 percent at 1, 0.5 and 0.25
  nm. On a mesh made from a tensor grid with a band step inside the
  window, the result now differs from the tensor grid's by that
  half-spacing effect of the tensor grid.
- `Device(grid=...)` refuses nodes closer than 1e-6 nm with a clear
  message (a near-duplicate node made the Poisson solve fail).

### Not included (see docs/limits.md)

- Scattering in coupled mode space; phonon self-energies in NEGF; NEGF
  on triangle meshes; both carriers in NEGF at once. Polar optical
  phonon and intersubband scattering in the lifetimes. The hot-electron
  model's limit at zero relaxation time with velocity saturation, the
  plasma's own chemistry, oxide shapes that follow the flow, elasticity,
  electronic stopping beyond Lindhard-Scharff, tunneling paths that bend
  with the field, tunneling and heat inside one Newton solve, and the
  Peltier heat in Fermi-Dirac form are as in 0.26.0. The app does not
  have the new features yet.

### Checks (tests)

15 new tests (372 in all; docs/checks.md has the tolerances): the leads
by decimation against the exact chain lead; coupled mode space against
the dense real-space inverse, mode counting, the sum of chains without
mixing, and the uncoupled device; holes against the mirror image of
electrons and the QuantumChannel's holes; a slab with a mass along x of
its own; the quality mesh's angles, areas, conformity and diametral
circles, and a ring resistor converging at second order; the quantum
channel on a quality mesh against a fine tensor grid; the scattering
rates against closed forms (Fang-Howard and square-well overlaps,
Price's acoustic rate, the alloy rate), an independent integral with
the closed-form Fang-Howard form factors, the far-away-charge limits,
the Hellmann-Feynman shift of a level, and the thermal average. All 65
documented examples were rerun.

## 0.26.0 (2026-11-20)

The channel electrons can be solved as waves by non-equilibrium Green's
functions, with or without scattering; the quantum channel runs on
triangle meshes; implant ranges come from a Monte Carlo simulation of
the ions; etching can be driven by ions and a reactive gas together; and
growing oxide flows, with its stress slowing the reaction. The examples
print what they printed before; examples 57 to 61 are new.

### Added

- `Device.negf(bias, ...)`: quantum transport of the channel electrons by
  non-equilibrium Green's functions in uncoupled mode space (each subband
  of the QuantumChannel between the contacts a 1D chain along x, the
  width free), self-consistent with the potential, with the Landauer
  current per subband; `scattering_time=` adds Buttiker probes (momentum
  and energy relaxation at one rate) whose Fermi levels keep every probe
  current-free. The module `kpenvelope.negf` holds the chain, the lead
  self-energies and the adaptive energy grid.
- `MeshDevice(..., quantum=QuantumChannel(..., line_spacing=))`: the
  quantum channel on triangle meshes, along vertical lines through the
  mesh.
- `kpenvelope.bca`: implant ranges by binary-collision Monte Carlo (ZBL
  universal potential, scattering angle by quadrature, Lindhard-Scharff
  electronic stopping), with the moments, the depth histogram and
  `RangeResult.implant`; `Implant(depth_profile=)` takes a tabulated
  profile.
- `EtchKinetics`, `Structure.kinetic_rates` and `Structure.plasma_etch`:
  ion-enhanced etching with a first-order site balance on every face,
  the neutrals re-emitted with the coverage-dependent sticking.
- `Structure.stokes`: slow viscous flow and stress of the cross-section;
  `Structure.oxidize(mode="2d", viscosity=, stress_volume_nm3=,
  temperature_K=)`: oxidation with the oxide's flow and a reaction rate
  that follows the stress normal to the interface.

### Not included (see docs/limits.md)

- The hot-electron model with velocity saturation still does not end on
  field-driven drift-diffusion as the relaxation time goes to zero: five
  ways of changing that were tried and none was robust (the limits page
  gives the numbers). Coupling between subbands in NEGF; holes in NEGF;
  phonon scattering; the plasma's own chemistry; oxide shapes that follow
  the flow and elasticity; a mesh generator with quality refinement;
  tunneling paths that bend with the field; Peltier heat in the
  Fermi-Dirac form. The app does not have the new features yet.

### Checks (tests)

13 new tests (357 in all; docs/checks.md has the tolerances): the NEGF
chain against the rectangular barrier, the 2D gas and the
top-of-the-barrier formula; the probes against the Drude resistance and
current conservation; the device NEGF against the QuantumChannel at
equilibrium, the top-of-the-barrier formula and drift-diffusion with
mu = q tau / m; the mesh quantum channel against the tensor grid; the
Monte Carlo against the Coulomb scattering and stopping, the universal
stopping fit, the electronic-only path and the continuous-slowing-down
path; the etching kinetics against the site balance, neutral
conservation and aspect-ratio dependence; the viscous flow against a
flat layer and a cylinder. All 61 documented examples were rerun.

## 0.25.0 (2026-11-20)

Holes can be quantum states too, the quantum channel runs with
band-to-band tunneling, the ballistic limit of a transistor's current is
available, and particles in the flux process steps can bounce. The
hot-electron model's behavior with velocity saturation at small
relaxation times is now documented with numbers. The examples print what
they printed before; examples 55 and 56 are new.

### Added

- `QuantumChannel(..., holes=(m_z, m_inplane, reference), electrons=)`:
  holes as quantum states (one band, your masses), with their own quantum
  potential on the valence band; trap emission keeps detailed balance.
  `DeviceState.quantum` adds `hole_subbands`, `hole_sheet_cm2` and
  `hole_potential`.
- `Device.ballistic_current(state, source, drain)`: the ballistic limit
  of the current through the channel's subbands, the flux of each
  subband's 2D Fermi gas over the top of its barrier from either contact
  (K. Natori, J. Appl. Phys. 76, 4879 (1994); M. Lundstrom, IEEE Electron
  Device Lett. 18, 361 (1997)).
- `Structure.view_factors(n_bins, points)`: diffuse view factors between
  all exposed faces (exact ray entry into the nearest cell, averaged over
  points along each face); `deposit` and `etch` with `mode="flux"` take
  `sticking`, and what does not stick is re-emitted by the cosine law
  (radiosity).
- The app: `sticking=` in flux steps.

### Changed

- The quantum channel now runs together with band-to-band tunneling: the
  tunneling uses the real bands, without the quantum potentials.

### Documented (docs/limits.md)

- With velocity saturation, the hydrodynamic model does not reduce to
  drift-diffusion as the relaxation time goes to zero (its mobility
  follows the local heating, not the field): 1 to 2 percent in the tests'
  transistor, up to 46 percent in a short, heavily driven bar.

### Not included (see docs/limits.md)

- Non-equilibrium Green's function transport; the quantum channel on
  triangle meshes; gas-phase or plasma chemistry and saturating surface
  kinetics; stress and viscous flow of growing oxide; implant ranges
  computed by the package.

### Checks (tests)

5 new tests (344 in all; docs/checks.md has the tolerances): quantum holes
against the mirror image of quantum electrons (1e-9); the quantum channel
with band-to-band tunneling (detailed balance, real-band rates, 1e-8);
the ballistic formula against a direct integral (1e-9); view factors
against the sky flux and reciprocity, and re-emission against an
independent Monte Carlo simulation; conformal films at low sticking. All
56 documented examples were rerun.

## 0.24.0 (2026-11-20)

The device solver can treat the electrons of a transistor channel as
quantum states, and the process steps gain line-of-sight etching and
deposition (shadowing in narrow openings) and two-dimensional oxidation
(oxide under mask edges). The examples print what they printed before;
examples 51 to 54 are new.

### Added

- `QuantumChannel(window, x_range, n_states, tol, max_outer, max_shift)`
  for `Device(quantum=...)`: along every grid column inside the window,
  the one-band Schroedinger equation (BenDaniel-Duke form, layer masses,
  the device's own grid) with two-dimensional Fermi-Dirac filling at the
  local electron quasi-Fermi level; the device sees the confined density
  through a quantum potential added to the electrons' band edge, solved
  in an outer, Anderson-accelerated loop until it settles. Every state up
  to 40 kT above the quasi-Fermi level is kept. Trap emission keeps
  detailed balance. `DeviceState.quantum_potential` and
  `DeviceState.quantum` (levels, confined sheet density per column,
  iterations); `ec` is the real band edge. By default the columns under
  ohmic contacts are left classical. Runs with traps, holes, self-heating,
  the full temperature model and hot electrons; `small_signal`,
  `transient` and `pulse` hold the quantum potential of their start.
- `device.column_states`: the eigenstates of one column on any grid.
- `Structure.fluxes(exponent)`: the flux onto every exposed face from
  particles with the in-plane angular distribution cos^n (n = 1 a gas at
  rest, the 2D crossed-strings view factor; large n an ion beam), with
  exact shadowing by the cells and mirror images at the sides.
- `Structure.etch(..., mode="flux", exponent=, beams=)` and
  `Structure.deposit(..., mode="flux", exponent=)`: etching and growth at
  those fluxes (aspect-ratio dependent etching, thinner films at trench
  bottoms).
- `Structure.oxidize(..., mode="2d")`: the oxidant diffuses through the
  oxide in two dimensions, with Deal and Grove's B and B/A as diffusion
  and reaction; oxide grows under mask edges through a pad oxide (LOCOS)
  and on trench walls; the new volume lifts its column (or swells into
  the nearest gas from a sideways face).
- The app: "treat the channel electrons as quantum states" in the
  Transistor tab; mode=flux (with exponent=) for deposit and etch and
  mode=2d for oxidize in Process steps. Process steps with mode=flux are
  limited to 400 cells along a side.

### Changed

- `Device.equilibrium()` with a quantum channel solves the quantum
  equilibrium.

### Not included (see docs/limits.md)

- Ballistic transport; quantum holes; a quantum channel on triangle
  meshes or together with band-to-band tunneling; re-emission of
  particles, gas or plasma chemistry and saturating surface kinetics in
  the flux modes; stress and viscous flow of growing oxide.

### Checks (tests)

10 new tests (339 in all; docs/checks.md has the tolerances): the column
eigenproblem against the infinite and the triangular well (second
order); the device with a quantum channel against the separate 1D
Schroedinger-Poisson solver at zero bias, under a biased gate and at
360 K, with differences that halve with the grid step; a transistor with
a quantum channel (detailed balance, current conservation, self-heating,
hot electrons as without it, path independence); line-of-sight fluxes against exact
view factors; flux etching and deposition; 2D oxidation against
Deal-Grove and the oxidant's closed form, with LOCOS; the app's new
options. All 54 documented examples were rerun.

## 0.23.0 (2026-11-20)

Heat can now include the Peltier and Thomson heat, band-to-band
tunneling runs together with self-heating, hot electrons and a set
lattice temperature, the nonlocal tunneling model works on triangle
meshes, and the browser app offers the features of the last two
releases. The examples print what they printed before; examples 48 to
50 are new.

### Added

- `Device(..., heat_model="thermoelectric")` and the same for
  `MeshDevice`: the heat adds -div(Pi_n Gamma_n + Pi_p Gamma_p) to the
  Joule and recombination heat, with Pi = kT (5/2 - ln(n / N_c)) for
  electrons (and the same for holes) taken from the package's own
  thermoelectric current, so heat and current agree (the thermodynamic
  heat of G. K. Wachutka, IEEE Trans. Comput.-Aided Des. 9, 1141 (1990);
  Eq. 21 of M. Kantner, arXiv:1911.00377). It gives the Peltier heat at
  contacts and junctions and the Thomson heat along a temperature
  difference; it sums to zero over the device. With hot electrons it
  adds the holes' share (the electrons' is in their energy balance).
  `"joule"` (the default) is the earlier model.
- Band-to-band tunneling (local and nonlocal) with `thermal=`,
  `energy_relaxation=` and `lattice_temperature=`: the tunneling rates
  and the heated (or hot-electron) solve take turns until both settle.
- `BandToBand(..., model="nonlocal")` on `MeshDevice`: straight,
  parallel paths in `direction` (default vertical), `path_spacing` nm
  apart (default the median edge length), with the bands sampled where
  the paths cross the triangles' edges, so the profile along a path is
  the mesh's own.
- `nitride.gap_shift(alloy, x, conduction_share)` (`AlloyGapShift`): the
  change of the gap with temperature for GaN, AlN, InN, AlGaN, InGaN and
  InAlN from the cited Varshni numbers of I. Vurgaftman and J. R. Meyer,
  J. Appl. Phys. 94, 3675 (2003), as used by `algan`, `ingan` and
  `inaln`. Use it as `band_gap_temperature`.
- `Device.thermal_runaway(..., stop_above=T)`: ends the sweep at the
  first voltage whose hottest point passes T (`too_hot` in the result).
- The app: a Process steps tab (a recipe of deposit, etch, implant,
  anneal and oxidize lines, drawn as materials or doping); a thermal
  runaway search in the Transistor tab; and in that tab the full
  temperature model (cited gap numbers of each layer), Peltier heat,
  thermionic emission over the barrier step, and hot electrons together
  with self-heating; band-to-band tunneling in the I-V sweep (the stack
  as a vertical device), for tunnel junctions and tunnel diodes. Results
  hotter than 1000 K carry a warning.

### Changed

- The docstring of `thermal_runaway` now describes the full temperature
  model (it said that only the mobility follows the temperature).

### Not included (see docs/limits.md)

- Tunneling paths that bend with the field; Pi in the Fermi-Dirac form
  for a degenerate electron gas (the non-degenerate form used comes out
  too small there); the change of the band edges with temperature in Pi.

### Checks (tests)

11 new tests (see docs/checks.md for the tolerances): Peltier heat at
the contacts of a bar against Pi I / q, none inside, total heat equal
to the power; Thomson heat along a linear temperature profile against a
hand calculation; band-to-band tunneling at a set temperature against
the device built for it; with self-heating a fixed point of both the
current and the heat equation; with fast-relaxing hot electrons equal
to drift-diffusion; mesh tunneling equal to the tensor grid and to
Kane's formula on an unstructured mesh in three directions;
`gap_shift` against the alloy records; `stop_above`; and the app's new
actions, including the vertical device against the 1D diode solver.
All 50 documented examples were rerun.

## 0.22.0 (2026-11-20)

The lattice temperature now reaches the carrier statistics, so leakage
grows with temperature and can drive a device into thermal runaway. Hot
electrons run together with self-heating, triangle meshes get
self-heating and tunnel barriers, band-to-band tunneling can run in
both directions (tunnel diodes), and a new `Structure` builds a device's
shape from deposition, etching and oxidation steps. The README is now
short; the examples, the full reference, the checks and the limits
moved to the `docs` folder. The examples print what they printed
before, except the sum of the terminal currents in example 28, which is
at the rounding level (now -9.1e-12 A/mm, was -4.7e-11); examples 43 to
47 are new.

### Added

- `Device(..., temperature_model="full", band_gap_temperature=Varshni(...))`:
  the lattice temperature sets kT, N_c and N_v (as T^(3/2)), the band gap
  (Varshni's form, your split between the bands), trap emission, the
  Richardson velocity and the ohmic contacts. The fluxes are the exact
  edge solution of Gamma = -mu [n grad E_c + grad(n k T)] for a
  temperature that changes linearly along the edge, written so that at
  one temperature they are the isothermal fluxes exactly; they carry
  thermoelectric currents. `"mobility"` (the default) is the earlier
  model.
- `Device.solve(..., lattice_temperature=T)`: a lattice temperature you
  set.
- `Device.solve(..., energy_relaxation=tau, thermal=model)`: hot electrons
  and self-heating together; the lattice gets the electrons' heat where
  they relax (`DeviceState.lattice_heat`).
- `BandToBand(..., model="nonlocal")`: tunneling from the valence band at
  one point to the conduction band at another along the grid columns,
  with the two-band WKB transmission integrated exactly on each edge and
  the supply f_v - f_c, so it generates in reverse bias and recombines in
  forward bias (Esaki diodes); separate electron and hole rates
  (`DeviceState.band_to_band_rates`). In a uniform field with full supply
  it is Kane's local formula.
- `MeshDevice`: `thermal=` and `pulse` (the heat equation on the mesh by
  the box method), `tunnel_barriers=` (barrier meshed in vertical
  columns), `temperature_model`, `band_gap_temperature`.
- `process.Structure`: a cross-section on square cells with conformal and
  directional deposition (also epitaxy with in-situ doping), isotropic
  (fast marching) and directional etching at material rates, Deal-Grove
  oxidation, implants measured from the local surface, anneals inside the
  semiconductor, and `mesh`, `doping` and `surface_nodes` to build a
  `MeshDevice`.

### Changed

- With Fermi-Dirac statistics, trap emission is multiplied by the
  degeneracy factor of its band, so traps keep detailed balance where a
  band is degenerate. Before, they recombined slightly in equilibrium,
  and a degenerate InN p-n diode with traps found no equilibrium at all.
  Results with Boltzmann statistics, or with traps only where the bands
  are not degenerate, do not change.
- Self-heating iterations use Anderson acceleration, which converges
  faster near thermal runaway, and runaway is recognized after 40
  iterations that keep heating without settling instead of running to
  `max_thermal_iter`.
- `thermal_runaway` now also bisects for negative sweeps.
- On meshes, a node shared equally by two regions within rounding goes to
  the higher region, as intended (a tie of 1e-11 was missed before).

### Not included (see docs/limits.md)

- Paid support; process simulation beyond fixed-rate deposition and
  etching, column-by-column oxidation and Gaussian implants; the nonlocal
  band-to-band model on meshes or together with heating; Peltier and
  Thomson heat.

## 0.21.0 (2026-11-20)

The remaining gaps of the device solver are closed as far as an open
package can close them: hot electrons now change the current (velocity
overshoot), electrons cross band steps by thermionic emission and
barriers by tunneling inside the solve, band-to-band tunneling,
current-criterion breakdown and thermal runaway are built in, implants
and anneals set the doping, and devices can be built on unstructured
triangle meshes with curved and slanted shapes. README examples 1 to 34
print what they printed before, except the sum of the terminal currents in
example 28, which is at the rounding level (now -4.7e-11 A/mm, was
-5.7e-11); example 35 prints lower electron temperatures because of the
correction below; examples 36 to 42 are new.

### Added

- Hydrodynamic (energy-transport) model: `Device.solve(...,
  energy_relaxation=tau)` solves the electrons' energy balance together
  with the currents. The flux on every edge is the exact solution for a
  temperature that changes linearly along the edge (Scharfetter-Gummel
  when it is constant); the hot-electron mobility is the drift-diffusion
  model's at the field that would heat uniform electrons to the local
  temperature, so a uniform field gives back the drift-diffusion
  velocity exactly. Continuation in the electron temperature and
  Anderson acceleration of the outer loop. `DeviceState.electron_temperature`.
- `Device(heterojunctions="thermionic")`: thermionic emission over abrupt
  band steps for electrons and holes, J = q v_R n_b [exp(dF/kT) - 1] with
  barrier-side quantities (the form of K. Yang, J. R. East, G. I. Haddad,
  Solid-State Electron. 36, 321 (1993)).
- `TunnelBarrier` (`Device(tunnel_barriers=[...])`): the Tsu-Esaki current
  through a layer's band profile, coupled into the Newton solve with
  finite-difference derivatives; into a gate on top it is the gate
  leakage as part of the solution.
- `BandToBand` (`Device(band_to_band=)`): Kane's two-band generation rate
  (as written in arXiv:2108.09328, Eq. 13), updated from the field until
  it settles.
- `Device.breakdown_by_current`, `Device.field_check` and
  `Device.thermal_runaway` (with bisection to the edge).
- `Device(doping=)` takes a doping profile; module `process` with
  `Implant` (Gaussian depth profile, error-function mask edges) and
  `Anneal` (constant diffusivity, solved exactly in time), and `Process`.
- Module `mesh`: `MeshDevice` on unstructured triangle meshes (box
  method), `Mesh`, `MeshRegion`, `mesh_from_points`, `mesh_from_grid`.
- The app: "let hot electrons change the current", and breakdown read at
  a drain current you choose.

### Changed

- `Mobility.temperature_exponent` may be negative.
- When light (or band-to-band tunneling) feeds sparse carriers, Newton's
  convergence test also weighs each level change by the density it leads
  to, so a large rise from a vanishing density no longer looks converged.

### Corrected

- `electron_temperature`: the energy the field gives the electrons is now
  -Gamma_n . grad E_c (J . E, as in the energy balance), not
  -Gamma_n . grad F_n; the two differ where the density changes (a
  degenerate channel), and the temperatures there come out lower
  (example 35). Nodes with practically no electrons are held at the
  lattice temperature.

### Not included (see the README's Limits)

- Process simulation beyond implant profiles and constant-diffusivity
  anneals, self-heating and tunnel barriers on meshes, the hydrodynamic
  model together with self-heating, band-to-band recombination and Esaki
  tunneling, runaway from temperature-activated leakage, and commercial
  support.

## 0.20.0 (2026-11-19)

The device solver of 0.19.0 gains what it was missing next to the
commercial device simulators: contact resistance, small-signal speed
(f_T and f_max), heat that changes in time, avalanche breakdown,
tunneling through the gate barrier, hot electrons, and shaped devices
(passivation films, recessed gates, field plates, etched regions). The
app's Transistor tab gains all of them. README examples 1 to 30 print
what they printed before, except two numbers at the solver's rounding
level: the sum of the terminal currents in example 28 (now -5.7e-11
A/mm, was 2.2e-11) and one current during the recovery in example 29
(0.9970 A/mm at 10 ms, was 0.9971). Examples 31 to 35 are new.

### Added

- `Electrode(..., resistance=)`: a series resistor (ohm mm in 2D, ohm
  in 3D) between the terminal and its contact; the solve iterates the
  contact voltages (`DeviceState.internal`) and reports the power at
  the terminals and inside the device (`power`, `device_power`).
- `Device.small_signal(state, frequencies, traps=True)`: the
  admittance matrix of every terminal, from (J + s M) dx = rhs with the
  same Jacobian as the DC solve; traps can follow the signal.
  `with_series` (exact Y_ext = (1 + Y R)^-1 Y), `two_port` (common
  source) and `rf_figures` (h21, Mason's U, f_T and f_max both where
  the gain crosses 1 and from the 20 dB-per-decade reading).
- Heat in time: `ThermalMaterial(density=, cp_a=, cp_b=)`,
  `solve_heat_transient`, and `Device.pulse(bias, times, thermal)`, a
  pulse from cold with the carriers following at once. `GAN_BULK` and
  `SIC_4H_ROOM` carry heat capacities from their sources (GaN: Barin et
  al. via the Ioffe database; SiC: the 6H value, as noted).
- Impact ionization: `Ionization` (Chynoweth form, with `where` to
  limit it to layers) and `IMPACT_GAN` (D. Ji, B. Ercan, S. Chowdhury,
  Appl. Phys. Lett. 115, 073503 (2019)), inside the drift-diffusion
  solve (`Device(..., ionization=)`) with exact derivatives, and
  afterwards as McIntyre's ionization integral along the traced field
  line (`ionization_integral`, `breakdown_voltage`). `Device.field`.
- `Device.gate_tunneling(state)`: Tsu-Esaki current with WKB
  transmission, column by column under a gate (also a recessed one).
- `Device.electron_temperature(state, tau_energy)`: the electrons'
  energy balance with exponentially fitted energy fluxes.
- Shapes: `Dielectric` films on top (`cap=`), `Region` boxes cut out of
  the semiconductor (`regions=`), electrodes as metal boxes (`z0`,
  `z1`: recessed gates, field plates, ohmic and Schottky contacts
  through a film), and electrodes in several pieces (several entries
  with one name are one metal, for example an L-shaped gate). A box
  gate's barrier refers to the layer under it.
- The app: contact resistance, a passivation film (silicon nitride,
  permittivity 7 from Robertson 2004, by default; the surface under it
  keeps the bare surface's charge), a gate-connected field plate, gate
  leakage by tunneling, the electron temperature with an energy loss
  time you give, and three new buttons: speed (f_T, f_max), breakdown
  and pulsed heating.

### Changed

- Inside a dielectric the reported carrier densities are 0 and the band
  edge and quasi-Fermi levels are NaN.
- The electrothermal solve and `pulse` step the lattice temperature in
  smaller jumps when one jump is too far for Newton's method.
- When light (a generation term) cannot be switched on at the final
  voltages by Newton's method, the solve now switches it on in time
  (backward-Euler steps growing to steady state). This made the
  avalanche test pass on processors without AVX-512, where the earlier
  path failed.

### Not included (see the README's Limits)

- Thermionic emission over internal band steps as a separate process,
  tunneling between layers inside the device or through dielectrics,
  feedback of the electron temperature into the mobility (a full
  hydrodynamic model) and velocity overshoot, breakdown other than
  avalanche, unstructured meshes and slanted walls, and process
  simulation.

## 0.19.0 (2026-11-19)

Whole devices in two and three dimensions. A transistor with source,
gate and drain carries current, its mobility saturates at high field and
falls with temperature, traps in the layers fill and empty over time,
and the current heats the device through a heat equation that reaches
down to the heat sink. The app gains a Transistor tab for all of it.
SciPy is needed for the new solvers (`pip install "kpenvelope[device]"`);
everything else still needs NumPy only. README examples 1 to 27 print
what they printed before.

### Added

- `Device` (module `device`): drift-diffusion on a tensor-product grid
  in 2D (x, z) or 3D (x, y, z), fully coupled Newton with exact
  derivatives of the Scharfetter-Gummel fluxes, sparse LU, adaptive bias
  ramps with a secant predictor. `Electrode` (ohmic with the doped region
  an alloyed contact makes, or Schottky, optionally with thermionic
  emission), `Surface` (pinned, or with its own fixed charge and traps),
  `Mobility` (Caughey-Thomas field dependence on the electric field along
  each edge, or on the quasi-Fermi gradient, with a power law in
  temperature; every number with its reference). `solve`, `sweep`,
  terminal currents in A/mm (2D) or A (3D) that sum to zero, and the heat
  at every node.
- `Trap`: acceptor- or donor-like deep levels in layers or at the free
  surface, with Shockley-Read-Hall capture and emission (steady state
  eliminated exactly), `Device.transient` (backward Euler with adaptive
  steps, displacement currents, optional starting occupancy), and
  `capture_coefficient` (sigma times the mean thermal speed).
  `Device.emission_times` gives the mean wait before emission.
- Self-heating (module `thermal`): `solve_heat` for k(T) = k_300 (T/300)^-a
  on layered tensor grids, `ThermalModel` (substrate layers, boundary
  resistance as `boundary_resistance`, lateral extension for the finger
  pitch, heat sink), coupled to `Device.solve(thermal=...)`. Shipped
  numbers: GaN 230 W/(m K) with a = 1.43 (Mion et al., Appl. Phys. Lett.
  89, 092123 (2006), doi:10.1063/1.2335972; the exponent as quoted by
  Danilchenko et al., doi:10.1016/j.ssc.2007.08.007) and 4H-SiC 370 W/(m K)
  held constant (Goldberg et al. 2001 via the Ioffe database, which gives
  no temperature dependence for 4H-SiC).
- App, Transistor tab: output and transfer curves of a transistor built
  from the sample, with self-heating on a SiC or user substrate and a
  picture of the electrons, temperature or band; and a drain-lag run with
  iron traps (Ec - 0.5 eV, Wickramaratne et al., Phys. Rev. B 99, 205202
  (2019) and Uren et al., IEEE Trans. Electron Devices 59, 3327 (2012);
  cross section 1e-15 cm^2 and 1e15 cm^-3 compensating donors from Uren
  et al.). Built-in mobility: 1704 cm^2/Vs and (T/300)^-2.35 (Alpert et
  al., Rev. Sci. Instrum. 91 (2020), doi:10.1063/1.5139911); saturation
  speed from the electron density by the fit of Bajaj et al. (Appl.
  Phys. Lett. 107 (2015), doi:10.1063/1.4933181) to their calculations,
  which matched their pulsed measurements.
- README examples 28 to 30: transistor curves with and without
  self-heating, drain lag, and a 3D transistor.

### Checks (tests)

24 new tests (264 in all): the 2D solver against the 1D drift-diffusion
solver on a p-n diode (current to 1e-8, potential to 1e-9 V); a uniformly
doped bar against q n mu(E) E for both driving forces and two shapes;
3D uniform along y against 2D (1e-9); currents that sum to zero and heat
that equals the electrical power; the transistor's switch-off against
the 1D threshold; trap occupancy against the Fermi function in
equilibrium (1e-12); trap emission against the exact backward-Euler
recursion; drain lag recovering to the steady state of the new bias;
surface donor states against a pinned surface; the heat solver against
the exact Kirchhoff solution of a slab with k(T) and the Fourier-series
solution of a strip heater; the boundary-resistance layer against q R;
traps with holes against the 1D solver's recombination lifetimes
(1e-6); the analytic Jacobian against central differences for both driving
forces; the app's transistor and drain-lag actions.

## 0.18.0 (2026-11-18)

The nitride part grows into a device tool. InGaN, InAlN and ScAlN are
built in from cited tables, a drift-diffusion solver gives current and
capacitance curves, the interface phonons of a layer follow from the
dielectric continuum model, what is known about the misfit-line charge
can be combined and planned for, and everything opens as an app in the
browser. 0.17.0 was not released to PyPI on its own; this release
includes it. NumPy stays the only dependency (SciPy, when present, only
speeds up the banded solves). README examples 1 to 18 print what they
printed before.

### Added

- `ingan`, `inaln`, `scaln` and `gan(pset)`: relaxed In(x)Ga(1-x)N,
  In(x)Al(1-x)N and Sc(x)Al(1-x)N (0 <= x <= 0.5) and the GaN partner of
  a set. New sets: `"vurgaftman2003"` (Vurgaftman and Meyer, J. Appl.
  Phys. 94, 3675 (2003), Tables I and III, e31 and e33 from the moduli)
  and `"ambacher2021"` (Ambacher et al., J. Appl. Phys. 130, 045102
  (2021), ScAlN fits). InN band numbers from the same Table I, the
  wurtzite InN/GaN valence-band offset, and InN eps33 = 14.4 (Davydov et
  al. 1999, as listed by the Ioffe database). `with_numbers` swaps any
  number of a record for yours, with your source. Every record carries
  its `parameter_set`, and `interface_charges` warns when two sets meet.
  `NitrideMaterial` accepts NaN piezoelectric constants for a layer that
  is only used unstrained.
- `kpenvelope.driftdiffusion`: `solve_drift_diffusion` (electrons and
  holes, Fermi-Dirac or Boltzmann statistics, donors and acceptors with
  complete or incomplete ionization, SRH, radiative and Auger
  recombination, generation, ohmic and Schottky contacts; coupled
  Newton on the potential and both quasi-Fermi levels with
  Scharfetter-Gummel fluxes), `solve_quasi_equilibrium` (the C-V model),
  `iv_curve`, `cv_curve`, `cv_curve_quantum` (C-V with quantum
  electrons), `Transport`, `Contact`, `bernoulli`.
  `StackLayer` gains `acceptors` and `acceptor_depth`.
- `kpenvelope.phonons`: `PolarPhonons`, `GAN_BARKER1973`,
  `single_interface_modes`, `slab_modes` (uniaxial dielectric continuum
  model) and `frohlich_constant`.
- `kpenvelope.linecharge`: `PUBLISHED_CONSTRAINTS`, `combine_line_charge`
  (Gaussian estimates and upper limits, closed form), `plan_holography`,
  `plan_sum_rule`, `line_charge_error_from_density`.
- `kpenvelope.datafit`: `fit_measurements` fits barrier heights,
  compositions, thicknesses, extra charges, the misfit-line charge and a
  parasitic capacitance to a measured C-V curve and/or Hall density,
  with error bars and warnings for correlated unknowns
  (`MeasurementFit`, `correlation_warnings`). `fit_least_squares` gains
  `difference="forward"` and `abs_step`.
- `kpenvelope.design`: `find_designs` searches compositions and
  thicknesses for a target electron density and switch-off voltage;
  `design_error_bars`, `threshold_voltage`, `sheet_density`.
- `kpenvelope.leakage`: thermionic emission and tunneling over and
  through barriers (`tsu_esaki_current`, `wkb_transmission`,
  `richardson_constant`, `richardson_velocity`,
  `thermionic_emission_current`) and `gate_leakage` from the band
  diagram; `Contact(..., thermionic=True)` in the drift-diffusion
  solver.
- `kpenvelope.twod`: `solve_cross_section`, `cv_cross_section`,
  `Gate`, `device_grid`: a 2D transistor cross-section (gate, access
  regions, fringing), exactly the 1D solver in its 1D limit.
- `kpenvelope.gui`: the app (`kpenvelope-app`, `python -m
  kpenvelope.gui`), a local page with plain-language results, error
  bars, C-V and I-V sweeps, the line-charge tools, and a Python script
  of every calculation. Each result opens with a one-sentence summary,
  the "?" explanations open on tap, and results are marked out of date
  when the sample changes. Tabs for fitting measured data, the design
  finder, gate leakage and the 2D cross-section.

### Changed

- `algan` takes `polarization="vurgaftman2003"` too, and its
  `psp_bowing` now defaults to None, meaning the set's own
  recommendation: 0 for the PBEsol and HSE sets (as before) and the
  Table III bowing for the Vurgaftman set.
- `solve_electrons` refuses layers with acceptors (it has no holes).
- The publish workflow runs on ubuntu-24.04, as the tests do (it was on
  ubuntu-latest, which has moved to a newer image).

### Corrected

- The documentation said the H and ZB references differ only at second
  order in the strain. They differ by 2 eps_c P_ZB of the film plus
  second-order terms: about 1.5 percent of the charge for AlGaN on GaN,
  and 10 percent for In0.15Ga0.85N. The numbers computed were right;
  only the statement was wrong.

### Tests

58 new tests (240 in all): the copied tables against the PDFs, the
ScAlN fits against the percentages quoted in that paper, the H and ZB
difference, drift-diffusion against Ohm's law, the built-in potential,
the short-diode current, a generation-recombination profile and the
Schottky 1/C^2 slope, against the electron solver, phonon modes
against closed forms, the line-charge combination, and the app
against the library.

## 0.17.0 (2026-11-18)

The package now covers where the fixed charge of a nitride stack comes
from, not only what it does to the holes. Four new parts: the charge at
every interface of a c-plane stack, coherent or partly relaxed, with the
charge of the misfit dislocations made explicit; a Schroedinger-Poisson
solver for electron gases; error propagation through any calculation;
and least-squares fits with errors in the settings, including the
calibration and misfit-line charges of a measured series. Nothing that
existed before changes: no earlier module was edited, and README
examples 1 to 14 print exactly what they printed with 0.16.0. NumPy
stays the only dependency.

### Added

- `kpenvelope.nitride`: `NitrideMaterial` (one layer material, every
  number cited, `reference` required) and `algan(x, temperature_K,
  polarization)` for relaxed Al(x)Ga(1-x)N, with two polarization sets:
  PBEsol Berry-phase values archived in nitride-misfit-polarization
  v1.3.1 (doi:10.5281/zenodo.24315036) and the HSE values of Dreyer et
  al., Phys. Rev. X 6, 021038 (2016), Tables I and II. Band numbers from
  Vurgaftman and Meyer (2003) and Ambacher et al. (2000).
  `polarization_quantum`, `zinc_blende_formal_polarization`,
  `E_PER_NM2_PER_C_M2`.
- `kpenvelope.interfaces`: `StackLayer`, `layer_strains`,
  `interface_charges` (Eq. 2 of the misfit paper: coherent charge,
  relaxation term and misfit-line charge, for the H or ZB reference),
  `LineCharge(bound_fraction, trapped)`, `line_charge_limits` (bare,
  electron-counting, neutral), `misfit_line_density` (Eq. 1),
  `strained_polarization`, `charge_change_per_relaxation` and
  `trapped_from_sum_rule` (Eq. 3). A relaxed interface is refused unless
  a line charge is chosen.
- `kpenvelope.electrons`: `solve_electrons` (single-band
  Schroedinger-Poisson with donors, a pinned surface, a quantum window
  and an optional classical gas; Trellakis et al. self-consistency),
  `fermi_dirac_half`, `two_carrier_hall`, `split_two_carrier`,
  `line_scattering_rate`, `line_charge_mobility_bound`,
  `line_charge_from_mobility_drop`.
- `kpenvelope.uncertainty`: `propagate` (first order with an error
  budget and correlations, or Monte Carlo with percentiles and bounds).
- `kpenvelope.fitting`: `fit_least_squares` (Levenberg-Marquardt, errors
  in readings and settings) and `fit_interface_series`.
- README examples 15 to 18, a word list for the nitride part, and new
  entries under cited parameter sets, refusals, checks and limits.

### Checked against

- The archived results of the misfit analysis, computed with a separate
  code: interface charges of Table S6 and the sum rule (1e-12
  relative), Schroedinger-Poisson densities of the Shen et al. series
  (1e-9 required, 1e-13 found), mobility bounds (1e-12), the line charge
  fitted to the relaxed sample, and (outside the suite, about a minute
  each) the joint labplan fits, to better than one part in a million.
- Dreyer et al. Tables I and II; closed forms (Eq. 2 exactly, the
  references' r-independent difference, the formula of Wang et al., the
  3/4 quantum, the two-carrier Hall relations, weighted straight-line
  and Deming fits); SciPy quadrature for F_1/2.

31 new tests; 182 in all.

## 0.16.0 (2026-11-10)

Intersubband transitions for light polarized in the plane (the
in-plane velocity matrix of the six-band model), two silent errors
fixed, the cause of a test that failed now and then in the
oldest-dependencies CI job found and removed, and the methodological
paper cited as published (Phys. Rev. B 114, L291304 (2026)). Results
of calls that were correct before do not change beyond rounding:
against 0.15.0, energies, level sums, band characters and both
self-consistent loops of README examples 2 and 8 are bit for bit the
same, and envelopes at finite in-plane momentum differ by at most
1.2e-16 (the new normalization, see Fixed). README examples 1 to 13
print exactly what they printed with 0.15.0.

### Added

- `velocity_matrix(z, params, envelopes, kx, ky, angle)` (new module
  `kpenvelope.velocity`): the matrix elements `<f| dH/dk_e |i>` (eV nm,
  hbar times the velocity) between solver states, for the in-plane
  direction e = (cos angle, sin angle). The assembled matrix is of
  second degree in (kx, ky) (its third difference vanishes to
  rounding, asserted), so the derivative is exact: each 6 x 6 bulk
  block is differentiated as [B(k + e) - B(k - e)] / 2 and placed with
  the solvers' own discretization. `params` is one parameter set (hard
  walls) or one per grid point (a layered stack). Its diagonal is the
  slope dE/dk of each state (the Hellmann-Feynman theorem), with no
  step size.
- `inplane_transitions(z, params, energies, envelopes, kx, ky, angle,
  initial, tol_ev)`: the level sums of `level_transitions` for light
  polarized in the plane, `v2 = (1/g_I) sum |<f| dH/dk_e |i>|^2` and
  the oscillator strength `f = v2 / ((hbar^2/2m0) (E_I - E_L))` with
  the free-electron mass. For light polarized along z the same
  definitions, with hbar v_z = i [H, z], give `level_transitions` with
  `mass_ratio=1`, so the two compare directly. For one parabolic band
  the velocity is a constant times the identity and no in-plane line
  exists; in the six-band model the band mixing allows one. Checked
  against the polynomial identity of the package's own assembly (hard
  walls; a stack with band edge, potential and strain; 1e-15 eV nm),
  the closed form 2 c A k_e times the identity of the demo set (1e-12),
  Richardson-extrapolated finite differences of the solver's energies
  for the slope (1e-9 eV nm) and, through the effective-mass sum rule
  of second-order perturbation theory, for the curvature of every one
  of the top 8 states of a tilted GaN well and an AlN/GaN/AlN stack
  (a relative 1e-6), the in-plane f-sum rule of the strengths of a
  symmetric GaN well (a relative 1e-6), the in-plane isotropy (no
  dependence on the polarization angle at k_t = 0, 1e-12; a common
  rotation of momentum and polarization changes nothing, 1e-9), the
  Kramers symmetry of the two initial states, invariance under any
  unitary mixing inside each level (1e-10), and the commutator form
  of the z velocity against `level_transitions` (1e-10). On the tilted
  6 nm GaN well of README example 12 the top level absorbs light
  polarized in the plane into the two light-hole levels (f = 0.0065
  and 0.0050) and not into the second heavy-hole level, which is the
  one line light polarized along z sees (f = 0.3969).
- README example 14.

### Fixed

- **The parabolic filling used the mass along kx for every in-plane
  direction, also when a strain makes the subbands anisotropic.**
  `solve_self_consistent_hetero` with `filling="parabolic"` (the
  default) and a strain with eps_xx != eps_yy or a shear measured one
  step mass per state along kx and filled every state as if it had
  that mass in all directions. Such a strain makes the masses differ
  strongly between directions. For GaN with arbitrary test
  deformation potentials (D1..D6 = -3.7, 4.5, 8.2, -4.1, -4.0, -5.5 eV,
  test numbers, not material constants), a 6 nm stack on 41 points,
  strain diag(0.002, -0.002, 0), ps = 0.1 nm^-2, eps_r = 10.4 and 6
  states, 0.15.0 returned occupations [0, 0.0823, 0, 0.0030, 0.0124,
  0.0023] nm^-2 from kx masses [inf, 0.787, inf, 0.088, 5.05, 0.948]
  m0, while the masses along ky at the same potential are [3.01,
  0.126, inf, 0.558, inf, 0.152] m0 and `filling="kgrid"` over 16
  directions gives [0.0517, 0.0483, 0, 0, 0, 0] nm^-2. This combination
  is now refused with a message that points to `filling="kgrid"` with
  `ntheta`, as the k-grid filling without `ntheta` already was.
  In-plane isotropic strain (eps_xx = eps_yy, no shear, for example
  the biaxial strain of c-plane growth) is accepted as before. Test:
  `test_parabolic_filling_refuses_an_anisotropic_strain`.
- **`dipole_matrix` and `oscillator_strengths` did not check their
  inputs.** `dipole_matrix(z, F[0])`, one state of shape (6, N), read
  the six components as six states and returned a 6 x 6 matrix
  without a warning; it now requires (n_states, n_components, N) on
  the grid. `oscillator_strengths` with mass_ratio = -0.5 returned
  -1.31 for a transition whose strength is 1.31 at 0.5; a mass ratio
  that is not a finite number above 0, or a dipole matrix whose shape
  does not match the energies, is now refused. Test:
  `test_dipole_and_strength_inputs_are_checked`.
- **The envelope normalization was not reproducible on NumPy 1.22.**
  `test_kgrid_loop_results_unchanged_by_the_extra_state` failed now
  and then in the oldest-dependencies CI job (NumPy 1.22.0, SciPy
  1.8.0): the energies of a 4-state and a 5-state solve agreed exactly,
  but the envelopes did not. The eigenvectors were the same bit for
  bit (in 19999 repeated solves of a tilted GaN well at k_t = 0.32
  nm^-1, none differed, while 3163 of the normalized envelopes did);
  the difference came from the norm, which was summed as
  `numpy.abs(v) ** 2` over the strided view of each eigenvector. On
  NumPy 1.22 with AVX-512 the complex absolute value runs a vector
  loop or the scalar `hypot`, which differ in the last bit for some
  inputs. NumPy uses the vector loop only when the input stride is
  below 256 bytes and its overlap check passes. That check takes the
  strided view of state j to reach 48 bytes past its last element,
  that is 16 j bytes past the end of the eigenvector array, and the
  allocator often puts the result 16 bytes after that end (in all 286
  of 20000 calls where the norm of state 3 differed, and in none of
  the others). The whole state was then computed with `hypot`, and for
  state 3 of that solve its norm changed by about 2e-16 (relative).
  Placing the result there on purpose reproduces this every time. The
  norm is now summed from the real and imaginary parts,
  `re ** 2 + im ** 2`, which gives the same bits on every code path:
  in the same 19999 solves no envelope differs. The old bit-for-bit
  form of the test, run outside the suite with 3000 more repeated
  solves, now passes in 20 of 20 runs on NumPy 1.22.0 on an AVX-512
  machine (it failed in 20 of 20 before). NumPy 2.5.3 does not show
  the effect. The 0.15.0 notes
  say that the first `n_states` of an `n_states + 1` solve are
  "exactly" those of an `n_states` solve: that holds for the energies,
  but for the envelopes it held only to about 1e-16 on such machines,
  and bit for bit it also needs a LAPACK that returns the same bits on
  every call, which NumPy does not promise. The test now compares what
  the loop uses (energies, each state up to its phase, |F|^2, and the
  occupations, density and Fermi level of one filling step) to 1e-12.
  Tests: `test_normalization_does_not_depend_on_memory_layout` (fails
  on 0.15.0 with NumPy 1.22 on an AVX-512 machine; with NumPy 2 the
  two inner loops agree and it passes on both versions), and the
  rewritten `test_kgrid_loop_results_unchanged_by_the_extra_state`.

### Changed

- The methodological paper is cited as published: T. M. Mahim, A. S.
  M. Mohsin and M. M. Rahman, Phys. Rev. B 114, L291304 (2026),
  doi:10.1103/zn44-nmx8, in `CITATION.cff` (journal, volume, article
  number, year and date of publication; no longer in press), the
  README, the package docstring, the `reference` strings of
  `gan_rinke2008()` and `aln_rinke2008()` (text only; no number
  changes) and a test docstring.
- README: example 14, the new functions, refusals, checks, corrections
  and limits.

### Tests

- New file `tests/test_v016.py`, 9 tests (151 in total). The three
  tests of the fixes fail on 0.15.0 (the normalization test needs
  NumPy 1.22 on an AVX-512 machine, as in the CI job where the old
  test failed). A mutation check of the new module (seven deliberate
  errors, such as a missing conjugate, a missing average over the
  initial level or a wrong factor in the strength) was caught by
  these tests in every case.

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

### Behavior changes

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
