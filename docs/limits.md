# What kpenvelope leaves out

The limits of each part of the package, in detail. The
[README](../README.md) lists the main ones.

## Limits

- Scattering lifetimes (roughness, impurities, phonons) are not
  computed. They need cited screening and roughness parameters for
  each structure, and this package ships no number it cannot source;
  it gives the density-of-states and velocity ingredients instead.
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
- Devices in 2D and 3D (`Device`, `MeshDevice`): classical carriers (no
  quantum confinement of the channel, no ballistic transport). The
  hydrodynamic model uses one energy relaxation time (yours); models of
  this kind are known to be able to overstate velocity overshoot.
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
  non-degenerate gas with a constant mobility. Where the electrons are
  degenerate (n above N_c, as in a transistor channel) it comes out too
  small and can even change sign: with the Fermi level 4 kT above the
  band edge it gives 0.63 kT, against 1.09 kT for the same gas (same
  constant scattering time) worked out with Fermi-Dirac statistics. The
  change of the band edges with temperature does not enter Pi. Other
  codes write the thermoelectric power differently (the Kelvin formula
  used by M. Kantner, arXiv:1911.00377, has 3/2 in place of 5/2), so
  compare like with like.
- Breakdown of a dielectric or a barrier is a comparison with a critical
  field you give (`field_check`), not a model of the breakdown itself.
  Small-signal response is linear; f_max needs a gate resistance you
  supply.
- Shapes: `Device` builds boxes on a tensor grid; `MeshDevice` takes any
  triangle mesh, but the package's own mesh generators are simple
  (Delaunay of your points, a rectangular grid cut into triangles, or
  the cells of a `Structure`): bring a quality mesh from a mesh
  generator for complex shapes, and refine where the warning about
  negative dual faces appears. On meshes the heat sink is the bottom of
  the mesh (put the substrate into the mesh), a tunnel barrier must be
  meshed in vertical columns, the field-line breakdown tools are not
  available, and interface polarization charges are given by you as
  sheets.
- Process (`Implant`, `Anneal`, `Process`, `Structure`): Gaussian implant
  profiles from ranges you supply (the same range in every material),
  constant diffusivity in the semiconductor only, no segregation, damage
  or activation kinetics. Deposition and etching move surfaces at fixed
  rates (no gas transport, shadowing or plasma chemistry); the isotropic
  etch front comes from a fast-marching solution whose error shrinks
  with the cell size (about 1 percent of the etched area at 60 cells per
  etch depth in the tests). Oxidation follows Deal and Grove column by
  column: no oxidation under a mask edge (bird's beak), no stress, and no
  fast initial growth of thin dry oxides beyond what you give as a
  starting oxide. Shapes are made of square cells, so surfaces are
  staircases at the cell size.
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
