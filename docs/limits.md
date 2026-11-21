# What kpenvelope leaves out

The limits of each part of the package, in detail. The
[README](../README.md) lists the main ones.

## Limits

- Scattering lifetimes (`kpenvelope.scattering`, since 0.27.0) are
  computed for one subband at a time, from its own envelope, with the
  material numbers you give (the package ships none: deformation
  potentials, alloy potentials and roughness parameters differ between
  sources by factors of two and more). Only scattering within the
  subband (no transitions to other subbands), in the Born approximation,
  with Thomas-Fermi screening by that subband alone and one background
  permittivity (no image charges). Acoustic phonons are in the
  equipartition limit (elastic; too fast below the Bloch-Gruneisen
  temperature). Polar optical phonons (since 0.28.0) are bulk LO phonons
  of one energy, without dispersion, without the interface and confined
  modes of the heterostructure (the dielectric continuum model of
  `kpenvelope.phonons` has those, not coupled to the rates), without hot
  phonons, unscreened by default (static screening, the option, is an
  upper bound: at the phonon frequency screening is weaker); with them
  the mobility comes from the exact solution of the linearized Boltzmann
  equation of one subband, which takes the energy steps of a fraction of
  the phonon energy and stops at 40 kT above the Fermi level. There is
  no piezoelectric scattering and no charged dislocations here (the
  separate line model of `line_scattering_rate` has them). Roughness is
  the shift of one band step (zero for a wave function that vanishes at
  the interface, as with an infinite barrier). The mobility is that of
  one subband in a small field (no hot electrons).
- The default filling of both self-consistent loops is still the
  parabolic one, kept so that earlier results do not change. It treats
  every state as a parabola with one mass taken from a single small
  momentum step (0.02 nm^-1) along `kx` away from `k = 0`, which in a
  lopsided well gives the two states of a Kramers pair different
  masses (sometimes `inf`) and different numbers of holes (example 2).
  With a strain that breaks in-plane isotropy it is refused (since
  0.16.0). Use `filling="kgrid"` (example 8) for occupations you rely
  on; it costs `nk` extra solutions per iteration.
- With `filling="kgrid"` the states are labeled by their energy order
  at each momentum, so where two subbands cross, the holes of the two
  labels are shared out by that order. The totals and the density do
  not depend on the labels.
- In the finite-barrier loop the fixed negative charge is one sheet at
  `sheet_z`, and by default (for backward compatibility) it sits at the
  first grid point. If your grid starts inside a barrier, pass the
  interface position, or the full sheet field lies across the barrier
  (example 9).
- A permittivity profile changes only the electrostatics (Gauss's
  law). Image-charge effects of a permittivity step on the holes are
  not included. No permittivity is shipped for AlN.
- Every solve builds a dense matrix of size 6 N and diagonalizes it in
  pieces (two of size 3 N at finite in-plane momentum), so the time
  still grows as N^3. On one core of a shared, loaded test machine one
  GaN solve with version 0.15.0 took about 0.06 s at 97 points, 0.7 s
  at 301 points and 3.5 s at 601 points at zero in-plane momentum, and
  about 0.2 s, 2 s and 23 s at finite momentum (times vary with the
  machine and its load); loops and momentum grids multiply that. A
  strain that couples the two halves makes a finite-momentum solve
  several times slower.
- The Rashba term is for the conduction-band companion problem. The
  valence-band k-linear terms beyond the six-band Chuang-Chang model
  are not included: no vetted coefficients in our sources.
- The depolarization shift leaves out the excitonic (final-state)
  correction, which needs an exchange-correlation model this package
  does not ship. Its standard form was written for real envelope
  functions; it holds for the solver's states at zero in-plane
  momentum (they are real), so use those.
- `fill_subbands_thermal` and `fill_subbands_kgrid` can only fill the
  states they are given. The self-consistent loops check that no
  state below them would hold holes and warn if one would;
  `fill_subbands_kgrid` can check only when your `solve_at_k` returns
  more energies than `n_states`.
- `fit_sdh_mass` assumes that the scattering (the Dingle factor) does
  not change over the temperature range, and that each amplitude is
  that of one frequency at one field. It returns the cyclotron mass of
  the Fermi contour, not the mass at the subband edge.
- `fit_sdh_dingle` assumes one frequency (no beating), a quantum
  lifetime that does not change with the field, and the low-field
  regime where the oscillations are small and sinusoidal. It fits the logarithm of the amplitudes, weighted by
  `(A / sigma_A)^2`, which is right to first order in the relative
  errors.
- `cyclotron_mass` assumes a branch that depends only on the size of
  the momentum (no in-plane anisotropic strain) and holds its holes in
  one disk around `k = 0`; when the occupied region is a ring (a
  maximum away from `k = 0` that reaches above the Fermi energy) it
  refuses.
- `parameter_sensitivities` propagates errors linearly, so the error
  bar holds while the energies change linearly over one sigma of each
  parameter, and it treats the errors you pass as independent (for
  correlated errors, apply your covariance to the returned
  derivatives). The deformation potentials D1..D6 are not among the
  parameters. Where a parameter splits a level of several states (only
  possible for a level that is degenerate by accident, as in the demo
  set; a Kramers pair is never split), it gives the derivative of the
  level's mean energy and sets `level_split`.
- `level_transitions` needs complete levels (an even `n_states` at
  `k = 0`), and groups states whose energies differ by less than
  `tol_ev` (1e-9 eV by default), so two different levels closer than
  that count as one.
- The check for holes in the next state down uses the momenta of the
  grid; a maximum between two grid momenta is seen only as far as the
  grid resolves it.
- `velocity_matrix` and `inplane_transitions` use the velocity of the
  six-band envelope model, `dH/dk`. Bands outside the six (the
  conduction band) enter only through the A parameters, as everywhere
  in the model. The strengths are for one in-plane momentum: an
  absorption spectrum of a filled hole gas would sum them over the
  occupied momenta, which the package does not do. As in
  `level_transitions`, pass complete levels.

- Nitride stacks (new in 0.17.0). Eq. 2 is first order in the strain.
  The misfit lines are a uniform sheet of charge: right for the total
  number of electrons as long as the strip each line depletes is
  narrower than the line spacing (about 13 nm against 110 nm in the
  relaxed Shen sample), not for scattering details near a line or for
  charge on threading segments. The line charge itself is the least
  known input: the only published series that measures it gives
  -0.84 +- 0.53 e/nm, an error of about 60 percent, and the
  electron-counting value comes from ideal surfaces, not real
  dislocations. So pass it with its error bar and read the spread.
- The polarization sets still differ (P_H 1.337 against 1.312 C/m^2
  for GaN, e31 and e33 of GaN 20 to 25 percent apart, and nearly a
  factor of two in the charge of lattice-matched InAlN/GaN), so compare
  sets where it matters. The two references differ by 2 eps_c P_ZB plus
  second-order terms: about 1.5 percent for AlGaN on GaN, 10 percent for
  In0.15Ga0.85N; compare both for large strains. Alloys use Vegard's
  rule for the polarization except where the source gives a bowing.
  InN and the In alloys rest on the Vurgaftman-Meyer band numbers,
  whose InN offsets and AlInN gap bowing the authors themselves call
  provisional. ScAlN comes with no band numbers, and its source covers
  only 0 <= x <= 0.5 (a cubic phase appears near 0.45). AlN, InN and
  alloy phonons are not shipped.
- Drift-diffusion: no thermionic emission or tunneling over abrupt
  offsets (an abrupt offset passes current like a very thin graded
  layer, which overestimates transport over tall barriers), no field
  dependence of the mobility or velocity saturation, no heating, no
  time dependence or small-signal AC (the capacitance is quasi-static).
  Metal contacts keep the carriers at their equilibrium densities
  (infinite surface recombination velocity). A vertical current needs
  a conducting path: across an undoped, insulating buffer it is not
  defined, and the solver says so; use the quasi-equilibrium model for
  C-V there. The electrons are classical (use `solve_electrons` for
  subbands). Check the grid by halving `dz`: recombination in a
  forward-biased junction needs about 0.5 nm (example 21).
- Data fits and the design finder use classical electrons unless you
  choose quantum ones (slower), first-order error bars, and a grid on
  which a thickness moves by whole steps (interpolated in the fit).
  Correlated unknowns are flagged, not resolved: only another
  measurement resolves them. The design search finds the best point of
  the range to the resolution of its last grid; it does not include
  critical thickness or relaxation, so check that a design can be grown.
- Leakage: one band and one mass along the path, WKB transmission, no
  image-force lowering, no defect-assisted current. Gate leakage uses
  the bands of the quasi-equilibrium model, so below the switch-off
  voltage (no electrons left) the bands and the leakage stop changing.
  Thermionic emission is applied at metal contacts only, not at abrupt
  offsets between layers.
- 2D cross-section (`solve_cross_section`): no drain voltage or current
  along the channel, classical electrons, holes only if asked, the free
  surface pinned at one barrier, and SciPy for speed (the NumPy fallback
  is slower). For current, use `Device`.
- Devices in 2D and 3D (`Device`, `MeshDevice`): carriers are classical
  unless a `QuantumChannel` is given; quantum transport is `Device.negf`
  (below). The hydrodynamic model uses one energy relaxation time (yours);
  models of this kind are known to be able to overstate velocity
  overshoot. With velocity saturation it does not go over into
  drift-diffusion as the relaxation time goes to zero. This is the model
  itself, not a coding error: its mobility follows the electron
  temperature, and as the relaxation time goes to zero the temperature
  follows the local heating J . E / n, so the limit is drift-diffusion
  with a mobility set by the heating (the effective field E_H, with
  mu(E_H) E_H^2 = J . E / (q n)), which equals the field-driven
  drift-diffusion mobility only where diffusion does not matter (a
  uniform field, where the tests show the two agree). In the tests'
  transistor near switch-off the hot-electron current is 1 to 2 percent
  above drift-diffusion at 1e-16 to 1e-17 s; in a 200 nm, 1e18 cm^-3 bar
  at 1e6 V/cm and 1e-16 s it is 46, 41 and 29 percent above with grid
  steps of 2, 1 and 0.25 nm. Without velocity saturation the two agree
  within 4e-4 in that transistor. In 0.26.0, five ways of making the
  model end exactly on field-driven drift-diffusion were tried (scaling
  the mobility by mu(T_n) / mu(T_H) with T_H the local-heating
  temperature, with node or electron-weighted edge temperatures, a cap at
  the low-field mobility, and two blends between the two models); the
  best one met the limit (the difference fell in step with the relaxation
  time, 0.66, 0.065 and 0.0065 percent at 1e-16, 1e-17 and 1e-18 s) but
  failed to converge at realistic relaxation times (0.47 ps) in a
  transistor, because at nearly empty nodes next to full ones the local
  heating is not what the energy balance gives. None was released. Check
  small relaxation times against drift-diffusion before relying on them.
  Thermionic emission at band steps uses the non-degenerate form on the
  barrier side; the single node at an abrupt interface gives an error
  first order in the grid step there, so refine the grid at band steps
  (example 37). `TunnelBarrier` works along vertical paths with WKB
  transmission and one band, and its derivatives by finite differences
  make a solve several times slower; there is no tunneling through
  dielectrics, trap-assisted tunneling or image-force lowering.
- Band-to-band tunneling: the local model is Kane's rate as a generation
  only, scaled to vanish in equilibrium by the spread of the electrode
  voltages. The nonlocal model (generation and recombination, tunnel
  diodes) uses the two-band WKB transmission along straight, parallel
  paths: the grid's columns on a tensor grid, and lines in one direction
  you choose on a mesh. Real paths follow the field and can bend; one
  direction is right where the field points mostly one way, as across a
  junction. Both models are coupled by repeated solves, not inside
  Newton's method, also when they run with self-heating, hot electrons
  or a set lattice temperature (since 0.23.0), so a solve with tunneling
  and heat repeats the heated solve several times.
- Temperature: with `temperature_model="full"` the lattice temperature
  sets kT, the densities of states, the gap (Varshni's form, with your
  split of the change between the bands), trap emission, the Richardson
  velocity and the ohmic contacts. A Schottky barrier height, the
  impact-ionization and band-to-band coefficients and the capture
  coefficients stay at their given values. Thermal runaway is found as
  the voltage beyond which no steady self-heated state exists; what
  happens after (the device in time) needs `pulse`.
- Heat: the default is Joule plus recombination heat. With
  `heat_model="thermoelectric"` (since 0.23.0) the Peltier and Thomson
  heat are added, with the heat each carrier carries taken from the
  package's own thermoelectric current, Pi = kT (5/2 + ln(N_c / n)),
  so that heat and current agree. That is the value for a
  non-degenerate gas with a constant mobility, and it is still the
  default, so earlier results do not change. Where the electrons are
  degenerate (n above N_c, as in a transistor channel) it comes out too
  small and can even change sign: with the Fermi level 4 kT above the
  band edge it gives 0.63 kT, against 1.09 kT for the same gas (same
  constant scattering time) worked out with Fermi-Dirac statistics.
  `thermoelectric="fermi-dirac"` (since 0.28.0, with
  `statistics="fermi"`) uses the Fermi-Dirac value in both the heat and
  the current; it is still for an energy-independent scattering time
  (other scattering laws change the 5/2), and it does not run with hot
  electrons, whose energy balance stays Boltzmann's. The change of the
  band edges with temperature does not enter Pi. Other
  codes write the thermoelectric power differently (the Kelvin formula
  used by M. Kantner, arXiv:1911.00377, has 3/2 in place of 5/2), so
  compare like with like.
- Breakdown of a dielectric or a barrier is a comparison with a critical
  field you give (`field_check`), not a model of the breakdown itself.
  Small-signal response is linear; f_max needs a gate resistance you
  supply.
- Shapes: `Device` builds boxes on a tensor grid; `MeshDevice` takes any
  triangle mesh. Since 0.27.0 `quality_mesh` makes quality meshes of
  regions given as polygons (Delaunay refinement with a smallest angle
  up to 34 degrees and a size field); curved boundaries are polygons
  (you choose the chords), there is no anisotropic refinement (thin
  layers need many small triangles), and nothing in 3D. Refinement
  driven by the solution (since 0.28.0, `size_from_solution`) is
  isotropic, uses the curvature of one field you choose, and needs you
  to remesh and solve again (no automatic loop, no transfer of the
  solution to the new mesh beyond starting over). The other generators are simple (Delaunay
  of your points, a rectangular grid cut into triangles, or the cells of
  a `Structure`); refine where the warning about negative dual faces
  appears. A mesh of long, thin triangles gives poor
  results even without the warning (the tests saw 30 percent and more on
  a jittered mesh of 20 nm by 0.5 nm cells). On meshes the heat sink is the bottom of
  the mesh (put the substrate into the mesh), a tunnel barrier must be
  meshed in vertical columns, the field-line breakdown tools are not
  available, and interface polarization charges are given by you as
  sheets.
- Quantum channel (`QuantumChannel`, since 0.24.0): one band with the
  layer masses (no nonparabolicity, no valley or band mixing), along
  vertical grid columns only (on a `MeshDevice`, since 0.26.0, vertical
  lines through the mesh), each column on its own (no tunneling from
  one column to the next; for that, `negf`). On a mesh, nodes between
  two lines take the line values interpolated along x, and the
  materials on a line are the means over each sample's share of it
  (since 0.27.0), so on a mesh made from a tensor grid the result
  differs from the tensor grid's where a band step lies inside the
  window (the tensor grid gives the node on the step wholly to the layer
  above). `negf` and `ballistic_current` are for tensor grids only. Each
  node is filled with its own electron quasi-Fermi level, which is the
  Schroedinger-Poisson density wherever that level does not change along
  the column, as in equilibrium and along a channel; where it does (a
  current flowing up through the barrier) it is an approximation. The
  quantum potential is held fixed in `small_signal`, `transient` and
  `pulse` (the quasi-static capacitance from charges at two voltages
  includes it; the small-signal one does not). It does not run with
  tunnel barriers or thermionic band steps inside its window. Where
  `x_range` ends inside a channel, the quantum potential drops to 0 from
  one column to the next; put the ends in the ungated access regions, or
  keep the default. Holes (since 0.25.0) are one band with one mass along
  z and one in the plane, both yours: no heavy-, light- and split-off-hole
  mixing (the six-band solvers of the package do that for single stacks,
  not inside the device). The outer iteration costs typically 5 to 30
  device solves per voltage.
- Ballistic limit (`ballistic_current`, since 0.25.0): electrons over the
  top of each subband's barrier pass, those below do not (no tunneling
  through the barrier, no reflection above it); the bands are those of
  the state you give.
- Quantum transport (`Device.negf`, since 0.26.0): non-equilibrium
  Green's functions in uncoupled mode space by default: each subband of
  the columns between the contacts is a 1D chain along x, so electrons
  do not change subband; the transverse direction is free with one mass
  per subband, and the leads continue the end columns' potential. Since
  0.27.0 `coupled_modes` lets electrons change subband (coupled mode
  space, or the full 2D problem with all modes), but only without
  scattering, only when every column between the contacts has the same
  z nodes, and at a cost that grows as the cube of the number of modes
  (in the transistor of example 57, about 40 s with 6 modes and 150 s
  with 12 on one core of our test machine). The chains are fed at their ends from the source and drain
  Fermi levels: access regions between a contact and the window (a
  contact on top of a barrier) are not in the quantum part. Without
  scattering, states in a dip of a subband below both leads' band
  bottoms stay empty, and the self-consistent loop can fail to settle
  (it says so): in long access regions, under a strong drain voltage,
  and even at equilibrium where a subband dips by about 1 meV along an
  ungated channel; `scattering_time` adds Buttiker probes, which settle
  it. The
  probes relax momentum and energy at one rate (no phonon spectrum, no
  separate energy relaxation); their broadening is cut at the local band
  bottom and the lost states near it are added back from an infinite
  uniform chain (exact there, approximate where the band changes fast).
  The probe resistance matches the Drude value q^2 n tau / m to about 1
  percent at 0.5 nm and better at finer steps; in the device the
  resistance per length matched drift-diffusion with mu = q tau / m to 6
  percent at 0.5 nm and 14 percent at 1 nm. One carrier at a time:
  electrons, or (since 0.27.0, `carrier="holes"`) holes with one band
  and your two masses (no heavy-, light- and split-off-hole mixing); the
  other carrier and the carriers outside the window keep their
  drift-diffusion quasi-Fermi levels at that voltage. Not with heat, hot electrons, traps' dynamics
  or small-signal analysis. A 100 nm channel takes seconds without
  scattering and minutes with it.
- Process (`Implant`, `Anneal`, `Process`, `Structure`): implant profiles
  are Gaussian from ranges you supply (the same range in every material),
  or (since 0.26.0) the depth histogram of a binary-collision Monte
  Carlo run (`bca`): amorphous target (no channeling), recoils not
  followed (no damage, no sputtering), the ZBL universal potential, and
  Lindhard-Scharff electronic stopping, which can be off by tens of
  percent for light ions (it has no dependence on the ion's shell
  structure); for accurate ranges compare with measured data or SRIM and
  pass `electronic_scale` or your own stopping. The package has not been
  checked against measured ranges. Diffusion is constant diffusivity in
  the semiconductor only (your values), no segregation, damage or
  activation kinetics. The fixed-rate modes move surfaces at given
  rates; the isotropic etch front comes from a fast-marching solution
  whose error shrinks with the cell size (about 1 percent of the etched
  area at 60 cells per etch depth in the tests). The flux modes (since
  0.24.0) add shadowing by the geometry, but particles stick where they
  land unless you give a sticking coefficient below 1 (then they are
  re-emitted by the cosine law; there is no surface diffusion), the
  angular distribution is one cos^n law that you choose, and the
  cross-section repeats as its mirror image at both sides. Their cost
  grows as the square of the number of surface cells. Ion-enhanced
  etching (`plasma_etch`, since 0.26.0) uses one first-order site balance
  per face with your fluxes and coefficients; the plasma itself (the
  gas-phase chemistry that sets the fluxes, the ion energy and angle
  distributions, charging) is not modeled, and ions do not bounce.
  Oxidation follows Deal and Grove, column by column or (mode="2d") with
  the oxidant diffusing in two dimensions; mode="2d" can also solve the
  oxide's slow viscous flow and stress (since 0.26.0, `stokes`: Newtonian,
  one viscosity per material, materials without one rigid, the gas a
  fluid 1e4 times less viscous in place of a free surface) and slow the
  reaction where the stress normal to the interface is compressive. The
  shape still follows the volume rule (the new volume moves cells up or
  into the nearest gas), not the flow; there is no elasticity (no stress
  in a rigid mask), no stress dependence of the oxidant's diffusion or
  of the viscosity, no dependence on crystal orientation, and the result
  is accurate to about a cell. Neither mode has the fast
  initial growth of thin dry oxides beyond what you give as a starting
  oxide. Shapes are made of square cells, so surfaces are staircases at
  the cell size.
- Support: there is no paid support. Questions and bug reports go to the
  GitHub issues of the repository.
- The mobility's field is rounded near zero (0.1 percent of its
  saturation term at most) to keep it smooth. Each Newton step is a
  sparse LU solve, so a 2D sweep takes seconds to
  a minute and a 3D one minutes. Currents below about 1e-10 A/mm are at
  the solver's numerical floor. With an undoped buffer and no traps the
  current under a pinched-off gate flows through the buffer, as it would
  in such a device. The quasi-Fermi driving force (an option) often
  fails to converge at high drain voltage; the default field driving
  force does not have that problem. Check the grid by refining it near
  the gate edges.
- Phonons: only the interface (decaying) modes of the dielectric
  continuum model, not the confined or half-space modes, and no
  phonon-limited mobility. The single-oscillator dielectric function
  ignores damping.
- `solve_electrons` uses one isotropic, parabolic conduction band with
  the mass along c, and no holes (use the six-band solvers for hole
  gases). The 2DEG is the quantum window you choose; outside it
  electrons are classical only if you ask for it. The quantum window is
  diagonalized as a dense matrix, so its cost grows as the cube of its
  number of points (about half a second for a 44 nm window at 0.1 nm on
  one core, in the Shen example). F_1/2 is tabulated for -40 <= eta <= 40
  and uses its asymptotic forms outside.
- The mobility bound uses one model of scattering by charged lines
  (Thomas-Fermi screening, Fang-Howard form factors) with the
  electron mass and permittivity you pass. Other defects created during
  relaxation lower the mobility too, so the bound stays a bound.
- `propagate` (linear) is first order: where it and the Monte Carlo
  spread disagree, the result is not linear over the error bars.
  `fit_least_squares` gives first-order (Gaussian) error bars.
