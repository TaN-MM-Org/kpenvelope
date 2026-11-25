# How kpenvelope is checked

What the automated tests compare against, and how closely, followed by
the corrections made in earlier versions.

## How the results are checked

357 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.10 with the
oldest NumPy (1.22.0) and SciPy (1.8.0) the package allows. Most
numerical checks compare against an exact formula, a symmetry, or a
second calculation done a different way; a few compare against
published values (stated below). The main checks, with the tolerances
the tests use:

**Matrix and solver**

- The bulk energies equal, each twice, those of the block-diagonal
  3 x 3 form of Chuang and Chang (Eq. (45) of their paper, whose
  entries depend only on the size of the in-plane momentum) to
  1e-12 eV, at random momenta in random directions, for the GaN and
  AlN sets.
- Every bulk level is two-fold degenerate (Kramers) for random
  momenta with random strain, and rotating both about the c axis
  changes no energy, to 1e-12 eV.
- The subbands of an asymmetric GaN/AlN stack in a tilted potential are
  the same in every in-plane direction to 1e-10 eV, and a symmetric
  well has no spin splitting along any direction (1e-9 eV).
- The assembled matrix is Hermitian with every coupling switched on
  (`numpy.allclose`), and a mixed AlN/GaN/AlN stack is Hermitian with
  a difference of exactly 0.
- The layered assembly equals the single-material assembly exactly
  (difference 0) when every layer is the same material.
- The hard-wall demo well reproduces the textbook square-well levels
  to a relative 1e-4.
- The finite-barrier demo well matches the textbook finite-well
  levels to 1e-3 eV on the finer grid, with the error on the finer
  grid below 0.35 times the error on the coarser one; the decay into the
  barrier matches the analytic decay constant to 1 %; deeper barriers
  approach the hard-wall level.

**Cited parameters**

- The GaN and AlN numbers are locked to their tabulated values.
- The GaN zone-center splittings equal their closed forms to 1e-12 eV
  and the published 5.20 and 21.80 meV to 0.01 meV.
- The GaN high-momentum masses reach m0/|A2+A4-A5| = 1.89 m0 and
  m0/|A2+A4+A5| = 0.180 m0 (to 0.03 and 0.003 m0).
- A hard-wall self-consistent GaN run at 0.46 nm^-2 keeps the charge
  to a relative 5e-3, puts the gas center between 0.4 and 0.8 nm, and
  has the two heavy branches dominate.

**Self-consistency and filling**

- The Poisson step matches the uniform-slab closed form to a relative
  1e-3; the self-consistent loop keeps the charge to a relative 1e-6.
- The finite-temperature closed form used by the filler agrees with
  direct numerical integration of the Fermi-Dirac occupation to 1e-10
  (the test evaluates the formula itself, not `fill_subbands_thermal`).
- `temperature_K=0` gives bit-for-bit the same result as the
  zero-temperature filler and the default; 0.05 K agrees with it to
  1e-6 nm^-2; filling keeps the total to 1e-15 nm^-2 at 4.2, 77 and
  300 K; warming moves holes into lower subbands.
- On the exactly parabolic demo set, the momentum-grid filler agrees
  with the closed-form parabolic filler to 1e-9 nm^-2 with only 5 or
  23 momentum points, at T = 0 and 150 K, with two levels occupied.
  On a non-parabolic two-branch dispersion its error, against an
  independent root finder, stays below 1e-2 / (nk - 1)^2 nm^-2.
- With `filling="kgrid"` the loop lands on the parabolic loop's state
  on the demo set (to 1e-8), and on GaN its potential solves Gauss's
  law for its own density (to 5e-6 eV) with the charge exact to
  1e-12 nm^-2.
- The Poisson step with two permittivities matches the two-layer
  closed form to a relative 1e-3 (first order in the grid step); with
  the fixed sheet moved to 1 nm it matches the closed form to 1e-12 eV
  and the field left of the sheet is exactly zero. On a 5 nm AlN /
  5 nm GaN stack the loop with `sheet_z` at the interface keeps more
  than 95 % of the holes in the GaN.
- The finite-barrier loop on a uniform stack gives exactly the same
  energies and occupations as the hard-wall loop.
- `KB_EV_PER_K` matches the CODATA value 8.617333262e-5 eV/K to 1e-14.

**Character, dispersion and transport**

- Character fractions sum to 1 to 1e-12; at zero momentum every HH
  fraction is 0 or 1 to 1e-9, while LH and CH mix; at 0.6 nm^-1 the
  top subband is visibly mixed.
- On the demo set the local mass is 1/|A| to 1e-10, the group
  velocity is 2 c A k to 1e-10 eV nm, and the density of states
  matches its constant value to 1 %.
- Kramers splittings are zero to 1e-9 eV at k = 0 and in a symmetric
  well, and nonzero (> 1e-5 eV) in a tilted well at finite k.

**Strain**

- With D_i = c A_i and strain eps_ij = k_i k_j, the strain matrix
  equals the kinetic matrix entry by entry to 1e-14; closed-form
  eigenvalues hold to 1e-15; zero strain changes nothing; a strained
  demo well shifts by the band-edge shift to 1e-12 eV.

**Optics and depolarization**

- Infinite-well dipole `16 L / 9 pi^2` to 1 % and oscillator strength
  `256 / 27 pi^2` to 1e-3; the sum rule total lies between 0.995 and
  1.0005; the 1 -> 3 dipole is below 1e-9 nm; shifting the origin
  leaves off-diagonal dipoles unchanged to 1e-12 on the coupled GaN
  well.
- The geometry integral `S` computed on the grid from the analytic
  well functions matches adaptive quadrature to a relative 1e-3,
  doubles with the well width to 1e-3, and is unchanged by an origin
  shift to 1e-12; `alpha` is linear in density and inversely
  proportional to `eps_r` to a relative 1e-12; the defining identity
  holds to 1e-15 eV; the line shape integrates to the strength sum to
  1e-3.

**Magnetotransport (new in 0.13.0)**

- h/e equals the value from `scipy.constants` to a relative 1e-15, and
  the Lifshitz-Kosevich constant equals 2 pi^2 k_B m_e / (hbar e) from
  `scipy.constants` to 1e-6 (the package's 6-digit hbar^2/2m0 sets the
  difference).
- On exactly parabolic subbands filled by the momentum-grid filler,
  each state's predicted frequency equals hbar pi k_F^2 / (2 pi e),
  with k_F from the closed-form Fermi level and SI constants, to a
  relative 1e-9.
- `lk_thermal_factor` equals `X / sinh(X)` to 1e-12 and does not
  overflow. The mass fit returns the mass of noise-free data to 1e-9,
  and agrees with `scipy.optimize.curve_fit` (an independent optimizer)
  to 1e-7 in the mass and amplitude and to 1e-4 in both error bars and
  their correlation, with and without amplitude errors. Over 400
  simulated data sets with 3 % noise, the spread of the fitted masses
  equals the reported error bar to 12 %, and the mean chi-square per
  degree of freedom is 1 to 0.1.
- The cm^-1 conversion matches h c / e from `scipy.constants` to
  1e-17 eV at 1000 cm^-1.

**New in 0.14.0**

- The split solvers give the energies of the full diagonalization of
  version 0.13.1 (kept in the tests as the reference) to 1e-12 eV, and
  the same states, compared through the projector onto each complete
  degenerate level (1e-8, so the arbitrary basis inside a Kramers pair
  does not enter), over 40 cases: hard walls, a layered stack,
  potentials, four kinds of strain, zero momentum and four directions.
  Where a strain couples the halves, the result is the full
  diagonalization bit for bit. Both self-consistent loops and the
  momentum-grid filler give the same results with the old and the new
  solver (1e-9 and 1e-10). At zero momentum the two members of each
  Kramers pair have exactly equal energies.
- `cyclotron_mass` is exact for parabolic branches (1e-12), its error
  on a closed-form nonparabolic branch stays below 0.5 / (nk - 1)^2,
  and on GaN it agrees with a direct difference of two solves around
  the Fermi wavevector to 1 % (0.3 % at half the path spacing).
- `fit_sdh_dingle`: its constants match `scipy.constants` (1e-6 for
  pi m0 / e, set by the package's 6-digit hbar^2/2m0; 1e-15 for
  hbar / (2 pi k_B)); its Dingle temperature gives the same damping as
  `exp(-pi / (omega_c tau_q))` (1e-12); it recovers the lifetime of
  noise-free data; it equals `numpy.polyfit` (values and error bars to
  1e-9, with and without weights); over 400 simulated data sets with
  3 % noise the spread of the lifetimes equals the reported error to
  12 % and the mean chi-square per degree of freedom is 1 to 0.1; and
  the mass error is propagated as refits at shifted masses say (1e-5).
- Both tutorials run. The first recovers the true values its data
  were made from within two error bars; the second prints frequencies
  that obey the Onsager relation and holes that add up to the density.

**New in 0.15.0**

- At finite momentum the solver's real form of each half of the
  matrix gives the energies of the complex form of 0.14.0 (kept in the
  tests as the reference) to 1e-12 eV and the same states (projectors
  to 1e-10) over 24 cases: GaN, AlN, a layered stack with a potential
  and a strained well, in six directions. A random complex matrix,
  which has no real form, is recognized and left complex. The matrix
  built without the loop equals the loop of 0.14.0 (kept in the
  tests) to 1e-15 of its largest element.
- `parameter_sensitivities`: on the demo set the derivatives equal the
  closed forms of the discrete well to 1e-12; on GaN (zero and finite
  momentum, and a stack of two materials) they equal central finite
  differences of the solver's own energies to 2e-8 eV (Richardson
  extrapolated); the band-edge derivative equals `offset_sensitivity`
  to 1e-6; and over 400 parameter sets drawn from 2 % errors the
  spread of the energies equals the propagated error to 12 %.
- `level_transitions`: on the demo set `|z|^2` and `S` equal the
  values from the closed-form eigenvectors of the discrete well to
  1e-12 and the oscillator strength equals the multiplet sum of
  `oscillator_strengths` to 1e-12; on the tilted GaN well `S` is the
  0.10378 nm pinned since 0.13.0, and mixing the states inside each
  Kramers pair at random leaves every level value unchanged to 1e-10.
- The check for missing holes catches a branch that rises above its
  `k = 0` energy (a synthetic three-branch test that 0.14.0 passed
  without a warning).
- The third tutorial runs and recovers the mass its line was made
  from within its error bar. The relation it uses, a heavy-hole
  energy of `delta1 + delta2 + c (A1 + A3) kz^2` at `k_t = 0` (so a
  mass `m0 / |A1 + A3|` along c), holds for the bulk matrix of GaN and
  AlN to 1e-14 eV.

**New in 0.16.0**

- The velocity matrix `dH/dk` equals half the difference of the
  package's own matrix at `k + e` and `k - e` to 1e-15 eV nm (exact,
  since the matrix is of second degree in the in-plane momentum; its
  third difference vanishes to rounding), for hard walls and for a
  layered stack with a band edge, a potential and a strain.
- On the demo set it is `2 c A k` times the identity (1e-12), so no
  in-plane intersubband line appears, as for any single band.
- Its diagonal equals the slope dE/dk from finite differences of the
  solver's energies (1e-9 eV nm), and the effective-mass sum rule of
  second-order perturbation theory (the curvature of each state from
  the velocity matrix elements to all other states) equals the
  finite-difference curvature to a relative 1e-6, for a tilted GaN
  well and an AlN/GaN/AlN stack at finite momentum. Summed over all
  final levels, the in-plane strengths obey the corresponding sum
  rule (a relative 1e-6).
- At `k = 0` the in-plane level sums do not depend on the polarization
  angle (1e-12), rotating the momentum and the polarization together
  changes nothing (1e-9), mixing the states inside each Kramers pair
  changes nothing (1e-10), and HH1 -> HH2 is forbidden in the plane.
  Written with the same definitions, the `z` velocity `i [H, z]` gives
  the strengths of `level_transitions(mass_ratio=1)` (1e-10).
- The same eigenvectors stored in three memory layouts give
  bit-for-bit the same envelopes (the cause of a test that failed now
  and then; see [Corrections](#corrections-in-earlier-versions)).

**Corrections in 0.13.0**

- Over a sweep of 32 layer stacks with decimal thicknesses, every grid
  point lands in the layer that exact rational arithmetic puts it in;
  a GaN well moved along the grid keeps its top level to 1e-6 eV.
- The geometry integral `S` does not change, to a relative 1e-12, when
  either state is multiplied by a phase factor, and for real states it
  is the textbook formula bit for bit.
- On a demo well with two filled levels, the loops warn when the
  second level was not computed; with enough states, the occupations
  equal the closed-form parabolic filling to 1e-10 nm^-2.
- The density of states is NaN below the end of the momentum path and
  the closed-form constant (to 1 %) inside it.

**Calibration and Rashba**

- The offset fit recovers the offset used to make a synthetic
  measurement to 1e-6 eV, and its error bar matches a re-fit at one
  sigma to 5 %; the density inversion recovers the density to a
  relative 1e-10, and its error bar matches a finite difference to
  0.1 %. The bracket and below-bare refusals are tested; the
  insensitive-offset refusal is not.
- The Rashba splitting equals `2 |alpha| k` by diagonalization to
  1e-15 + 1e-12 k eV; spins are in-plane, unit length, perpendicular
  to k and opposite to 1e-12; the two energies at k = 0 are equal
  exactly; 4.5e-4 eV nm at 0.1 nm^-1 gives 9e-5 eV to 1e-19.

**Devices in 2D and 3D, traps and heat (new in 0.19.0)**

- The 2D solver against the 1D drift-diffusion solver on a GaN p-n
  diode at 0, 2.5 and 2.9 V: potential to 1e-9 V, current to 1e-8.
- A uniformly doped bar carries exactly q n mu(E) E, for the field and
  the quasi-Fermi driving force and for two saturation shapes (1e-6).
- A 3D device uniform along y gives the 2D current and potential (1e-9).
- Terminal currents add up to zero, and the heat put into the lattice
  equals the electrical power sum V I (1e-8); the drain current of a
  transistor vanishes near the 1D threshold of the layers under its gate.
- Trap occupancy equals the Fermi function in equilibrium (1e-12);
  emission follows the exact backward-Euler recursion with the step
  sizes taken (1e-6); drain lag recovers to the steady state of the new
  bias (1e-6); many surface donors at 1.0 eV give the electron density of
  a surface pinned at 1.0 eV (3 percent); traps that exchange with both
  bands recombine exactly like the 1D solver's lifetimes in a p-n diode
  (1e-6).
- The heat solver matches the exact Kirchhoff solution of a slab with
  k(T) of power law (1e-6 K) and the Fourier-series solution of a strip
  heater (3e-4); a boundary-resistance layer raises the temperature by
  q R (1e-9); the self-heated state is a fixed point of the heat equation.
- The analytic Jacobian matches central differences of the residual on
  a biased transistor, for both driving forces (1e-5 of each row's
  scale); the app's transistor and drain-lag actions run, their currents
  add up to zero, and the drain lag settles at the steady state.

**Speed, breakdown, pulses, tunneling, hot electrons and shapes (new in 0.20.0)**

- Small signal: at zero frequency the admittances of a biased
  transistor equal the DC conductances from finite differences of the
  solve (1e-5), and at 1 kHz the gate capacitances equal the charge
  derivatives (1e-6); every column and row sums to zero (currents
  conserved, a common shift does nothing; 1e-8). A bar with series
  resistors carries exactly V / (R + R_bar) (1e-9), and `with_series`
  gives the conductance 1 / (R + R_bar) of the same bar (1e-7). In the quasi-static limit f |h21| = gm / (2 pi C) (1e-6).
- Heat in time: a surface heat flux into a thick slab against the
  semi-infinite solution 2 q sqrt(t / pi) / e (5e-3), the stored energy
  against the heat put in (1e-4); the heat capacities equal their
  sources; a pulse ends at the steady self-heated state (current 1e-4,
  temperature 0.05 K) and only heats up.
- Impact ionization: the current gain of a GaN p-i-n diode lit at one
  edge matches McIntyre's 1 / (1 - I) with I from the solved field (5
  percent, below breakdown); the ionization integral along the traced
  field line equals a direct 1D integration (1e-3); breakdown falls
  between the two steps where the integral crosses 1.
- Tunneling: under a 2 um gate the current density in the middle equals
  the 1D `gate_leakage` (5e-3), also with the gate in a recess (then
  against the thinned barrier).
- Hot electrons: in a uniform field the electron temperature is
  T + (2/3) tau mu E^2 / k (1e-4), and it is the lattice temperature at
  the contacts.
- Shapes: a metal sheet on a nitride film over n+ GaN has the parallel-
  plate capacitance eps0 eps_r / d (1e-3); recessed gates switch off at
  the 1D threshold of the thinned barrier (0.03 V), also when the recess
  removes a GaN cap (the barrier is then the AlGaN's); a gate in two
  pieces equals two electrodes at one voltage (1e-9); a gate-connected
  field plate lowers the channel field at the gate edge by more than half
  and raises it at its own end; contacts through a film conserve current
  (1e-9) and there are no carriers inside the film. The passivated
  surface of the app holds the bare surface's electron density (2
  percent).

**Hot electrons, band steps, tunneling, breakdown, process and meshes (new in 0.21.0)**

- Hydrodynamic model: the discrete flux on an edge is the exact solution
  of the edge's equation for a linear temperature (against SciPy's ODE
  solver, 1e-9); in a uniform field the coupled model returns the
  drift-diffusion velocity and T_n = T_L + (2/3) tau mu E^2 / k (1e-5); an
  n+ n n+ diode shows overshoot above twice the saturation speed while
  drift-diffusion never exceeds it, the same on two grids within 3
  percent.
- Thermionic emission: zero current in equilibrium; against a
  boundary-value solution of the same model with SciPy's `solve_bvp`
  (5e-3, after extrapolating the first-order grid error).
- Tunnel barriers: zero in equilibrium; the coupled current equals
  `tsu_esaki_current` on the solution's bands (1e-8); gate tunneling
  through a coupled barrier matches the 1D `gate_leakage` under a wide gate
  (2 percent).
- Band-to-band: the rate equals a hand evaluation of Kane's formula
  (1e-12) and its derivative a finite difference (1e-6); zero current in
  equilibrium; the terminal current equals q times the integrated
  generation (1e-4).
- Breakdown: the current criterion interpolates between the bracketing
  steps; the field in a capacitor's dielectric is V/d (2e-3); thermal
  runaway lies within 0.3 percent of the lumped model's critical voltage.
- Process: the anneal against the exact broadened Gaussian and the
  surface image solution (1e-4), the dose conserved (1e-9), the mask edge
  against a numerical convolution (1e-7), the dose carried into a device
  (2 percent, from the device grid).
- Meshes: a rectangular grid cut into right triangles reproduces the
  tensor-grid device (current 1e-10, potential 1e-10 V); a curved quarter
  ring converges to the exact resistance with second order.

**Tunneling and heat in one solve, masks pushed sideways (new in 0.38.0)**

- The derivatives of the local band-to-band rate in Newton's method
  against central differences, in the electron and hole rows, at the
  nodes with the most tunneling and elsewhere (1e-6; measured 3e-10).
- A self-heated junction with the local model: its current is q times
  the pairs the tunneling makes (1e-4; measured 4.6e-6, the Newton
  tolerance, as without heat); solved again at its own temperature it
  gives its current (1e-8), and its heat gives its temperature (0.01 K);
  the same junction without the tunneling model, given the tunneling's
  pairs as a fixed generation and the temperature as a set one, solved
  by the plain solver, gives the same current (1e-6; measured equal to
  rounding) and potential (1e-9 V; measured 4e-16 V); the heated solve
  takes fewer than 200 Newton steps (about 120; the rounds of heated
  solves of 0.23.0 took 518 with current NumPy and SciPy and did not
  settle with the oldest ones); a set lattice temperature equal to the device's changes nothing
  (1e-10), and hot electrons that relax at once change the current by
  less than 5e-5 (measured 3.5e-6, and 1.04e-5 on CI with the oldest
  NumPy and SciPy).
- A block of nitride against the oxide on a trench's wall: the oxide
  made equals what is in whole cells plus what is in parts of cells
  (1e-9); the end's rounding is smaller than the parts; the result is
  the exact mirror image of the mirrored trench; the block keeps its 100
  cells and moves away from the wall, while a rigid block stays.

**Wurtzite layers in the phonons (new in 0.37.0)**

- Isotropic stacks through the bins: the total weight of the exact poles
  (1e-5; measured 8e-7) at their mean energy (0.5 meV; measured 0.17 at
  worst, against energy steps of about 2.6 meV in the mobility), for one
  and two boundaries and air; the mobility through the bins equals the
  one through the poles (1e-5; measured 1.9e-7).
- Bulk GaN, uniaxial, against the exact potential exp(-kappa |z - z'|) /
  (2 eps_par kappa): the weight in the band between the TO energies, which
  only the anisotropy couples (2e-3; measured 1e-7 to 1e-3), and at small
  q in the band of LO modes (1e-3; measured 2.4e-4; at larger q the
  reference integral of the exact potential, which oscillates fast near
  the band's edges, does not converge, while the bins keep the sum rule);
  at every q the sum rule with both permittivities (1e-6; measured
  7e-7).
- In the mobility, GaN's two directions lower it by 3.6 percent against
  the vibrations across the axis alone (a three-level well), and by 2.2
  percent in the app's default stack.

**NEGF with phonons that take and give energy (new in 0.36.0)**

- The building blocks: the sums |G_ij|^2 x_j and the columns |G_i0|^2,
  |G_i,N-1|^2 of the chain with any broadening against a dense matrix
  inverse (measured 1e-14, in development).
- Without coupling, the closed-form ballistic current (1e-5; measured
  3e-7) and density (1e-2; measured 3e-3, the even energy grid).
- At equilibrium, every state filled as Fermi-Dirac says at every site and
  energy (1e-12 of the largest; measured 1e-15) and no current.
- With a bias: the current leaving equals the current entering (1e-8;
  measured 9e-10, and it falls with the iterations' tolerance: 1e-5,
  2e-7, 4e-11 at 1e-7, 1e-9, 1e-11), and the power the contacts lose
  equals hw times the net emission (1e-6; measured 4e-9).
- A uniform channel 75 nm long: the conductance against the sum over the
  width of each part's exact 1D result lambda / (lambda + L), lambda = 2
  hbar v_x / Gamma (3 percent; measured 1.1, and 1.5 and 1.6 at 150 and
  300 nm). Comparing instead the slope of the resistance with length to
  the Boltzmann mobility was 24 percent off at 150 to 300 nm, because the
  fastest electrons' mean free path (about 190 nm) is not short against
  those lengths; the 1D result itself was checked on one wire at one
  energy (the textbook (h / 2 q^2) / (2 v tau) per length, measured 5e-4).
- The density with scattering against the same Lorentzian broadening of a
  2D gas worked out by hand (measured 0.2 percent). A first version cut
  each part of the width at the band bottom and lost 3 to 11 percent of
  the electrons; the energies along x now run down to -inf.
- `phonon_coupling_for_mobility`: against the Boltzmann mobility
  integrated separately (1e-5; measured 2e-6), and with a Fermi level
  against an independent integration (1e-6) and the thin-gas closed form
  (1e-6; measured 7e-8).
- Holes with phonons mirror electrons with phonons in a mirrored device:
  current and heat (1e-8; measured 1e-13); with phonons the current is
  below the ballistic one and the heat below I V. The app passes GaN's
  optical phonon energy and the coupling of the formula.

**Alloys and air in the phonons, masks that ride on the oxide, the plasma in the app (new in 0.35.0)**

- GaN under air: the surface mode's energy (eps(w) = -1) to 1e-12, its
  weight 4 F^2 / (w deps/dw) and the half-space LO mode in closed form
  (1e-6; measured 2e-8, and 1.1e-7 with the oldest NumPy and SciPy where
  the weights are 1e-7 of the largest), with the surface between grid points and 23 to
  25 nm outside the grid on either side; two polar materials likewise
  (1e-6; measured 1.3e-7).
- An alloy with two vibrations in bulk: a mode at each LO with the weight
  H (2 / LO) Res(1 / eps) (1e-10; measured 5e-14). The sum rule with
  alloys, air and boundaries between grid points (1e-6; measured 1.6e-7).
- In the mobility: air 500 nm away gives bulk GaN (1e-5); a second
  vibration without strength gives `interface_phonons` (1e-5); air on top
  of a 10 nm barrier gives the barrier's surface mode at q = 1 / nm
  (1e-7) and changes the mobility by 1.5e-6 (1e-4).
- The oxide's balance: in the volume rule and in the flow with a rigid,
  a flowing and an elastic mask, the oxide made equals the oxide in whole
  cells plus the parts of cells (1e-9; measured 0 to 4e-16). This check
  found three leaks in a first version of the riding masks (oxide flowing
  into a mask's side, the fill of a gas cell a mask moved into, oxide
  flowing back from the gas) and, for an elastic mask, a first flow
  solved over a time step of zero, which also made the result lopsided;
  each was fixed. The masks that move are lifted at both edges, not in
  the middle, and the same on both sides; a rigid mask stays.
- The app: masks that flow equal the library's (exactly); the plasma's
  chemistry gives the library's state and the etch with its fluxes
  (exactly); the alloy barrier with a second vibration without strength
  gives the single vibration (1e-5), and the air above changes the
  mobility by less than 1e-4.

**Phonons of many layers, oxide shaped by its flow (new in 0.34.0)**

- The modes of layers of one material are the bulk LO phonon with the
  bulk weight (1e-9; measured 5e-14); of two materials, the half-space
  and interface modes in closed form (1e-7; measured 2.5e-11); a slab of
  barrier in GaN has the interface-mode energies of
  `phonons.slab_modes` (1e-9; measured 2e-15).
- The sum rule: the weights of all the modes add up to the difference
  of the potentials with the high-frequency and the static permittivities,
  for three and four layers at three wave numbers (1e-7; measured 5e-9),
  and for a channel, a 1 nm spacer, a 24 nm barrier and a cap at 16 wave
  numbers from 1e-4 to 30 per nm (1e-6; measured 2e-7). This last check
  found two faults of a first version, which scanned the dispersion for
  sign changes: at large q the modes of the barrier's two boundaries came
  so close in energy that some were lost (17 percent off the sum rule),
  and the cosh of a thick layer overflowed. The roots now come from the
  polynomial the dispersion becomes with its denominators cleared, and
  near-equal poles are taken in one contour.
- In the mobility: layers of one material give bulk GaN's phonons
  (1e-5; measured 9e-7), two materials give `interface_phonons` (1e-5;
  measured 1.1e-6), and a barrier 0.5 nm thin in GaN changes the bulk
  value by 2e-5 (1e-3).
- The app: with GaN's numbers everywhere, bulk GaN (1e-5); with a cap 25
  nm from the electrons, the same as the boundary next to them alone
  (1e-5; measured 3e-7); spacer numbers equal to the barrier's change
  nothing (exactly).
- Oxide shaped by its flow: on a flat surface the same as the volume rule
  (exactly); in a trench, a symmetric result, and the oxide's area equals
  the silicon consumed over 0.44 (2 percent). The app's oxide equals the
  library's (exactly).

**Springy oxide, masks that bend, negative ions (new in 0.33.0)**

- A uniform layer swelling at a constant rate: the Maxwell stress after
  n steps equals 2 eta q (1 - chi^n), and an elastic solid's 2 G q t
  (1e-10; measured 1e-13).
- A step of oxide with a steady random source: after 60 steps the
  Maxwell stress has settled on the viscous one, sxx, szz and sxz
  (1e-10; measured 1.5e-13). This checks the old stress kept at the
  corners and the edges: a first version, which took each corner's
  effective viscosity and kept share as plain means of the cells around
  it, settled 6 to 16 percent away from the viscous stress next to the
  gas and the rigid silicon; each corner now takes the mean viscosity
  times one minus the mean share kept.
- The oxidation with an enormous modulus gives the viscous result (area
  exact, stress 1e-9; measured 1e-11); with 30 GPa its stress is lower.
- The app's elastic mask equals the library's (1e-12).
- Negative ions: the densities and T_e equal the balances written out by
  hand and solved separately (1e-9; measured 6e-14), n_e is the positive
  less the negative ions, no negative ion reaches the wall, and a
  strongly attaching gas (18 negative ions per electron) is reached in
  steps from a guess.

**Scattering in coupled mode space, NEGF on meshes (new in 0.32.0)**

- A slab with one mode whose columns all have the same mode (a potential
  that changes along x only) is a chain: with probes, the current
  (1e-6; measured 2e-7, the decimation of the slab's leads against the
  chain's exact lead), the density (1e-6; measured 1e-7) and the probes'
  levels (1e-8; measured 2e-10) agree. The current into the drain equals
  the current out of the source with one mode (1e-9; measured 7e-11)
  and with two modes that mix (1e-8).
- A uniform 2DEG channel 60 nm long: two coupled modes with probes and
  the uncoupled subbands with theirs give the same current (2 percent;
  measured 0.6). The resistance added from 60 to 120 nm is 1.09 times
  that of drift-diffusion with mu = q tau / m at a step of 1 nm (the
  uncoupled chains: 1.16 at that step, 1.06 at 0.5 nm), a check run once
  and not in the test suite because it takes two minutes.
- NEGF on a mesh made from a tensor grid equals the tensor grid's, with
  both self-consistent loops run to 1e-6 V: with probes (1e-8; measured
  2e-15) and with two coupled modes (1e-8; measured 5e-10), current and
  potential; the ballistic limit too (1e-10). At the default tolerance
  of 1e-4 V the two loops can stop up to about 1e-6 apart (seen on one
  CI machine).

**Interface phonons, a plasma model, levels that mix in the app (new in 0.31.0)**

- Interface modes: the energies equal those of
  `phonons.single_interface_modes` for the same isotropic materials
  (1e-9; measured to the last digit), and each coupling equals the area
  of its peak in -Im 2 / (eps_1 + eps_2) with a small damping (1.5e-3;
  measured 5e-4 and 8e-4 at a damping of 1e-5 eV, ten times less than
  at 1e-4 eV). The same
  material on both sides gives one mode at w_LO with the bulk coupling
  (1e-12).
- The sum over the modes equals the potential of a point charge solved
  on a grid across the interface with damped permittivities, -Im of it
  integrated over the frequency (1e-3; measured 2.8e-4 at a damping of
  1e-4 eV, and 1.5e-4 at half of it in a separate run).
- Mobility: the same material on both sides gives the bulk phonon's
  mobility (1e-10; measured 4e-16), an interface far from the gas gives
  it too (1e-8; measured 3e-16), the operator keeps detailed balance with
  four modes of different energies (1e-10), and a four times finer grid
  agrees (1e-3; measured 7e-5).
- Plasma: a gas with one ion against the particle and power balances
  written out by hand and solved separately (1e-10; measured 2e-14); a
  molecule that breaks up against the atoms' balance in closed form and
  the count of atoms fed and pumped out (1e-10); the fluxes against
  n vbar / 4 and h n u_B (1e-12); reactions that do not keep the charge
  are refused.
- The app: the mobility with GaN's own numbers on the barrier side
  equals the bulk one (1e-10); the wave current with levels that mix
  equals `Device.negf(coupled_modes=4)` (1e-12).

**Screening by every pair of subbands, the app catches up (new in 0.30.0)**

- The long-wavelength polarization of every pair equals the density
  change of the n x n Hamiltonian with a small random potential added at
  a fixed Fermi level (1e-8; measured 2e-10). The Lindhard polarization
  goes over into it as q goes to 0 (1e-9; measured 2e-13) and equals a
  direct 2D integral over the wave vector at 300 K for every pair at
  q = 1.5 nm^-1 (2e-5; measured 1e-7).
- The screened elements of a sheet of charges and of the roughness,
  for every pair at q = 0.05, 0.4 and 2 nm^-1, equal a separate solve of
  the self-consistent potential on the grid nodes (the induced density
  of every pair fed back through the Coulomb kernel; 1e-6, measured
  1.4e-7). One subband: the matrix is the single number 1 - e^2 H Pi /
  (2 eps q) (1e-8).
- The app: the 2D oxide with flow gives the library's area and largest
  stress (1e-9; example 61: 2892 nm^2, -11.1 GPa); the simulated implant
  quotes the library's range and keeps the dose that was not bounced
  back (1 percent), and a note appears when the dose leaves the picture;
  plasma etching gives the library's area (1e-12); the wave current of
  a short transistor with no scattering is within 5 percent of the
  ballistic limit (measured 1.4 percent) and above twice the
  drift-diffusion current, and with scattering it uses tau = mu m / q
  (1e-12) and falls below 0.9 of the ballistic one; the dense-heat
  option switches the device to the Fermi-Dirac form; a typed mobility
  without a source is no longer credited to the paper of the default.

**Scattering between subbands, mobility in the app (new in 0.29.0)**

- `Subbands` with one subband is `Subband`: the same mobility for
  Coulomb, roughness and acoustic scattering (1e-4, the two energy
  grids; measured 2e-5 and better) and for polar optical phonons (1e-9;
  measured 1e-12), the same transport and quantum lifetimes (1e-9).
- Two wells far apart, one subband in each, envelopes with no common
  node: each subband's mobility is that of the single-subband model and
  the total their density-weighted mean (2e-4; measured 2e-5 to 4e-5),
  with Coulomb, roughness and polar optical phonons, the subband bottoms
  not on a common grid. (A first version on one uniform grid was 1.6
  percent off for the subband whose bottom fell inside a cell, where the
  threshold of phonon emission makes g jump; the cells now have an edge
  at every subband bottom.)
- Three coupled subbands of a tilted well at 300 K and 77 K: with
  acoustic phonons alone (the same rate in every direction) the exact
  times are 1 / sum_j nu_ij, integrated by quadrature (1e-4; measured
  3e-6 to 4e-5); with Coulomb, roughness and acoustic scattering
  together, the cell solution against the exact solution at each energy
  (Siggia and Kwok's small linear system) integrated by Gauss quadrature
  (1e-5; measured 3e-6); with polar optical phonons the collision operator
  is symmetric with the weight f0 (1 - f0) (1e-10; measured 2e-15), and
  cells of hbar w / 32 and / 64 agree (1e-4).
- The app's mobility equals the same calculation written with the
  library (1e-12), keeps more than 90 percent of the electrons of the
  exact region in its levels, and asks for the source of typed numbers.

**Polar optical phonons, adaptive meshes, Fermi-Dirac thermoelectric effects (new in 0.28.0)**

- Polar optical phonons: for a gas of zero thickness the angular
  integrals equal their closed forms in complete elliptic integrals,
  int dtheta / q = 4 K(m) / (k + k') and the one with cos(theta) (1e-10;
  measured 1e-14), and the spline of the form factor equals the direct
  sum (1e-8; measured 3e-10). The exact Boltzmann solution, with a
  vanishing coupling, equals the relaxation-time mobility of the elastic
  mechanisms (1e-5; measured 2e-6); with a small phonon energy it tends
  to the elastic limit with the momentum rate C (2N + 1) int (1 -
  cos(theta)) H / q dtheta (0.75 and 0.072 percent at 2 and 0.5 meV for
  a non-degenerate gas, 0.10 and 0.010 percent for a degenerate one);
  the linearized collision operator is symmetric with the weight f0 (1 -
  f0), detailed balance with equilibrium phonons (1e-10; measured 1e-13);
  energy steps of hbar w / 64 and / 128 agree (2e-4).
- Adaptive meshes: the recovered second derivatives of a quadratic are
  exact on a tensor mesh and on a quality mesh, boundary nodes included
  (1e-6; measured 2e-9, the regularization). An L-shaped conductor (a
  corner singularity) on adaptive meshes reaches half the error of a
  uniform mesh of 13 189 nodes with under half the nodes; the reference
  0.31938 ohm mm comes from adaptive meshes up to 74 706 nodes and
  uniform ones up to 51 103 nodes, extrapolated (they agree within 1e-5;
  errors -0.334, -0.142, -0.056 percent uniform at 890, 3354, 13 189
  nodes; -0.092, -0.042, -0.020 percent adaptive at 978, 1954, 5029). A
  really obtuse boundary triangle still warns.
- Fermi-Dirac thermoelectric effects: F_3/2 against quadrature (1e-8;
  measured 2e-12 to 2e-9); G(4) = 1.0911 against 0.63 for the
  Boltzmann-like form; the excess G - G_B against its asymptote exp(eta)
  / 2^(7/2) far below the band edge (1e-9) and, where mpmath is
  installed, against arbitrary precision (1e-6); a degenerate bar (eta =
  5.0) with grounded ends and a temperature profile carries Gamma L /
  (mu N_d) = k int G dT exactly in the continuum (1e-5 at 201 nodes;
  4.7e-6, 1.2e-6 and 2.9e-7 at 101, 201 and 401 nodes, second order),
  twice the Boltzmann-like current; at one temperature the two forms give
  the same current to the last bit; the Peltier heat at the contacts is
  kT G I / q (1e-6); the Jacobian of the new fluxes against finite
  differences across a p-n junction with traps (1e-6); a mesh made from
  the tensor grid gives the same current (3e-13, not in the suite).

**Coupled modes, holes in NEGF, quality meshes, scattering lifetimes (new in 0.27.0)**

- Leads by decimation: the surface Green's function of a uniform chain
  against the exact lead self-energy -t e^(ika) (2e-9).
- Coupled mode space: a uniform guide transmits exactly its number of
  open modes; with all modes, the transmission and the spectral function
  equal those of the dense real-space inverse (1e-13 and 4e-15); with
  fewer modes the transmission converges as modes are added (1.130,
  1.606, 1.857 and 1.8725 at 0.35 eV with 2, 4, 6 and 8 modes); where
  the potential changes only along x the modes do not mix and the slab
  carries the sum of the chains' currents (2e-7, tested to 1e-5); with a
  mass along x of its own, the same sum with that mass, while the column
  states keep the mass across (1e-13). In the transistor of example 57
  the coupled current is within 1 percent of the uncoupled one (+0.20
  percent with 4 modes, +0.45 percent with 6).
- Holes in NEGF: a gated well and its mirror image for holes (bands
  upside down, donors to acceptors) give the same current, sheets and
  potential with the sign turned (1e-10; measured 3e-15 uncoupled, 1e-7
  coupled, where the iteration's tolerance sets it); with masses of 0.3
  m0 across and 0.6 m0 in the plane, the equilibrium hole sheet equals
  the QuantumChannel's away from the leads (1e-3; measured 3.7e-4).
- Quality mesh generator: on an L-shaped pair of regions with a slanted
  interface and a size field, every angle is at least the bound asked
  for (28 degrees), each region's area is exact (1e-12), no edge is
  longer than the size field, every boundary and interface segment is an
  edge, and no node lies inside a segment's diametral circle. A quarter
  ring resistor meshed by it converges at second order to the exact
  resistance of the ring (0.19, 0.060 and 0.017 percent at h = 60, 30
  and 15 nm, the polygon's chords included), with no negative dual faces.
- Quantum channel on meshes: on a quality mesh the confined sheet of
  every line agrees with a fine tensor grid (0.1 nm) to 0.54, 0.20 and
  0.10 percent at sizes of 1, 0.5 and 0.25 nm (before this version, 1.6
  to 2.0 percent at all three: a node on the well's edge took the whole
  of its share of the line from one material, which moved the edge by up
  to half a spacing, differently on each line). On the shifted-row mesh
  of 0.26.0 the error is now 0.62, 0.15 and 0.046 percent at 1, 0.5 and
  0.25 nm (example 58). The tensor-grid twin is exact again (1e-10) with
  the window inside the well.
- Scattering lifetimes, against closed forms and an independent
  integral: the integral of psi^4 of the Fang-Howard form (3 b / 16) and
  of an infinite well (3 / 2W) (1e-9, 1e-6); the acoustic-phonon rate
  against Price's closed form for the Fang-Howard gas (1e-9; measured
  1e-11); the alloy rate against its formula (1e-12); the screened
  Coulomb rate of a sheet of charges against SciPy quadrature with the
  closed-form Fang-Howard form factors of Ando, Fowler and Stern (2e-6;
  measured 4e-7), for the transport and quantum rates; a density profile
  against the sum of its sheets (1e-10); far-away charges in a thin,
  strongly screened, cold gas against tau_t / tau_q = (2 k_F d)^2 (4e-3)
  and mu = 8 e (k_F d)^3 / (pi hbar N) (0.9 and 0.6 percent above it at
  k_F d = 7.7 and 31, which is the 1 / (q_s d) correction: the exact
  integral for a gas of zero thickness at T = 0 gives 0.94 and 0.39
  percent); the roughness matrix element against the shift of the level
  when the interface moves (Hellmann-Feynman, 1e-6; measured 8e-8); the
  roughness rate against quadrature (1e-8, Gaussian and exponential
  correlation) and its short-correlation limit pi m M^2 Delta^2 L^2 /
  hbar^3 (1e-4); the thermal average of the mobility against quadrature
  for rates that go as E^s, degenerate and not (2e-5), and the
  non-degenerate limit Gamma(s + 2) (kT)^s (5e-5); Matthiessen's rule
  (1e-12).
- Grids: a tensor grid with two nodes closer than 1e-6 nm is refused
  with the place named.

**Quantum transport, meshes, implants, etching kinetics, oxide flow (new in 0.26.0)**

- NEGF chain: the transmission of a rectangular barrier (0.3 eV, 2 nm,
  0.2 m0) against the textbook formula, second order in the step (error
  below 1e-3 at 0.05 nm, falling more than 3.5-fold when the step
  halves); a uniform chain at equilibrium holds the 2D gas density
  m kT / (pi hbar^2) ln(1 + exp(F / kT)) to second order (2.5e-3 at 0.5
  nm, 6e-4 at 0.25 nm) with exactly zero current; the current over a
  smooth barrier 0.15 eV high tends to the top-of-the-barrier formula as
  the barrier widens (21, 4.9 and 1.2 percent above it at 20, 40 and 80
  nm, the excess being tunneling through the top).
- Buttiker probes: the current leaving the source equals the current
  entering the drain (1e-8); the probes leave the equilibrium density
  unchanged (5e-3); and the resistance grows with length at the Drude
  rate q^2 n tau / m, nondegenerate and degenerate (1.2 percent at 0.25
  nm; 1.3 and 2.0 percent at 0.5 nm, 0.7 and 0.6 percent at 0.25 nm).
- `Device.negf`: at equilibrium, zero current and the confined sheet of
  the QuantumChannel solution away from the leads (2.5e-3; next to the
  leads the two differ by 2 to 4.5 percent, where the electrons' motion
  along x, which the QuantumChannel leaves out, shows); the
  self-consistent ballistic current against the top-of-the-barrier
  formula on the same potential (0.1 to 1.5 percent above it); with
  probes, a 2DEG resistor's resistance per length against
  drift-diffusion with mu = q tau / m (6 percent at 0.5 nm, 14 percent at
  1 nm).
- Quantum channel on meshes: a mesh made from the tensor grid, with lines
  on the grid's columns, gives the tensor grid's current (1e-10, measured
  5e-14) and potential and quantum potential (1e-10 V) with electrons and
  holes; on a mesh of nearly equilateral triangles the confined sheet
  approaches the tensor grid's at second order (since 0.27.0 0.62, 0.15
  and 0.046 percent at steps of 1, 0.5 and 0.25 nm; 1.9, 0.5 and 0.07
  percent in 0.26.0).
- Monte Carlo implants: the scattering angle of the Coulomb potential
  against 2 atan(1 / (2 eps b)) (1e-9; measured 3e-12); the Coulomb
  stopping up to a largest impact parameter against its closed form
  (1e-5); the ZBL nuclear stopping against the universal fit for eps <= 1
  (1 percent; 0.1 percent at 1 and 10 keV for B and As in Si); the
  interpolation table against direct quadrature (2e-3 in the stopping it
  gives); the path with electronic stopping alone against 2 (sqrt(E0) -
  sqrt(E_cut)) / (N k) (2e-3); the mean path of heavy ions against the
  continuous-slowing-down path (3 percent; Sb and As at 100 keV in Si:
  1.7 and 1.8 percent short); two runs agreeing within their statistics;
  the implant step holding the dose of the ions that stayed in (2e-3).
  The ranges are not compared with measured data.
- Etching kinetics: the coverage and speed on a flat surface against the
  site balance (1e-12) and its ion-limited and neutral-limited limits;
  with every face bare, the neutral fluxes in a trench equal the
  fixed-sticking re-emission of 0.25.0 (1e-6); the neutrals that enter a
  trench equal those that stick plus those that leave (3e-3, the
  difference between the two ways the fluxes are computed); the trench
  bottom etches at 0.65 of the open rate when the neutrals limit it and
  above 0.95 when the ions do.
- Viscous flow: a flat oxide on a rigid substrate rising at q h with
  sigma_zz = 0 and sigma_xx = -2 eta q in its growing layer (1e-9); a
  rigid cylinder in a ring of oxide fed at its surface against the exact
  radial flow v_r = Q / (2 pi r) and sigma_tt - sigma_rr = 4 eta (Q /
  2 pi) / r^2 (2e-3 at a 1 nm step; measured 6e-4 and 6e-5); a flat
  oxidation with the stress coupled grows exactly as without; at the
  corners of a step the compressive stress slows it.
- Hot electrons with velocity saturation as the relaxation time goes to
  zero: five ways of making the model end exactly on field-driven
  drift-diffusion were tried and none released (see the limits page).

**Quantum holes, ballistic limit, bouncing particles (new in 0.25.0)**

- Quantum holes: a well for holes built as the mirror image of a well for
  electrons (same mass, band profile turned over, acceptors for donors)
  gives the same confined sheet (1e-9), the same levels with the sign
  turned (1e-9 eV), the same density profile (1e-9) and the current
  scaled by the ratio of the mobilities (1e-8).
- Quantum channel with band-to-band tunneling (electrons and holes
  quantum): zero current in equilibrium, terminal currents that add up to
  zero, and tunneling rates equal to those of the same device without the
  quantum channel at the same potentials (1e-8), so the tunneling sees the
  real bands.
- Ballistic limit: the formula against a direct integral over the moving
  2D Fermi gas (1e-9); zero at zero drain voltage; more than the
  drift-diffusion current.
- View factors: what a face does not send to other faces equals the
  direct flux it receives from the sky (1e-3), and they are reciprocal
  (2e-3). With re-emission, the flux at the bottom of a trench agrees with
  an independent Monte Carlo simulation of bouncing particles (within
  four standard errors plus 1 percent) for sticking 0.3 and 0.05; with
  sticking 0.01 a film covers a trench bottom as thickly as the top.

**Quantum channel, line-of-sight process steps, 2D oxidation (new in 0.24.0)**

- The column eigenproblem on uneven grids: an infinite well against
  hbar^2 pi^2 n^2 / 2 m L^2 and a triangular well against the Airy
  zeros, with second-order convergence (the error falls fourfold when
  the step halves).
- The device solver with a `QuantumChannel` against the package's
  separate 1D Schroedinger-Poisson solver (`solve_electrons`), on the
  same grid, for an Al0.25Ga0.75N/GaN stack: band edge within 6e-5 eV,
  levels within 3e-6 eV, confined sheet density within 1e-5; the
  differences come from how the two codes put the material step on the
  grid and halve when the step halves, while the quantum effect itself
  is over 100 times larger. The same under a gate at -1 V (2e-4 eV,
  5e-5) and with the full temperature model at 360 K (1.2e-4 eV, 3e-5).
- A transistor with a quantum channel, traps and holes: zero current in
  equilibrium (detailed balance), terminal currents that add up to zero
  (1e-9), heat that equals the power with self-heating (1e-9), hot
  electrons that relax at once (constant mobility) changing the current
  by the same fraction with and without the quantum channel (1e-4), the
  same current reached directly or through a sweep (1e-5), and about one
  percent less current than the classical model near switch-off.
- Line-of-sight fluxes: at the bottom of trenches of three aspect ratios
  and for three angular distributions, equal to the exact integral over
  the visible directions (2e-6); on the walls, equal to the closed form
  (1/2)(1 - cos(atan(w / depth))) (1e-6); all particles entering an
  opening land inside it (1e-3, the faces sample the flux at their
  centers).
- Flux etching and deposition: a flat surface at the full rate, a narrow
  trench etching more slowly than a wide one, an ion beam etching
  straight down through a mask, a film whose volume is what fell in
  (1 percent).
- 2D oxidation: the oxidant at the interface of a flat oxide equal to
  A / (A + 2x) (1e-12); far from any edge, the Deal-Grove thickness and
  consumed silicon within one cell, also from a pad oxide; oxide volume
  equal to the consumed silicon over 0.44 (2 percent, partial cells);
  LOCOS with a pad oxide reaching further under the mask than without
  one, and lifting the mask.
- The app: a transistor with quantum electrons, the refusal of quantum
  electrons together with thermionic emission, and process recipes with
  mode=flux and mode=2d.

**Peltier heat, tunneling with heat and on meshes, the app (new in 0.23.0)**

- Peltier heat: in a uniformly doped bar at one temperature the extra
  heat sits only at the two contacts, Pi I / q where the electrons leave
  and -Pi I / q where they enter, Pi = kT (5/2 + ln(N_c / n)) (1e-9); the
  inside gets none (1e-12 of that), and the total heat is the electrical
  power in both heat models (1e-12).
- Thomson heat: along a bar with a linear temperature profile, the extra
  heat at each node equals -Gamma . grad Pi worked out by hand from the
  temperature there (1e-6).
- Band-to-band tunneling at a set lattice temperature of 360 K equals
  the device built for 360 K with its gap shifted by hand (1e-8). With
  self-heating the result is a fixed point: solving again at its own
  temperature gives its current (1e-6), and its heat gives its
  temperature (0.01 K); the total heat is the power (1e-10). With hot
  electrons that relax at once it equals drift-diffusion (1e-5).
- Nonlocal tunneling on a mesh: a rectangular grid cut into right
  triangles gives the tensor grid's current (1e-9). On an unstructured
  mesh, in a uniform field with full supply along three directions
  (vertical, horizontal and slanted), the electron and hole rates add
  up to Kane's formula integrated over each path's tunneling region
  (1e-7).
- `gap_shift` equals the change of the gap that `algan`, `ingan` and
  `inaln` give between two temperatures (1e-12 eV).
- `thermal_runaway(stop_above=)` stops at the first voltage past the
  temperature and repeats the earlier steps exactly.
- The app: the I-V sweep with tunneling shows a tunnel diode's dip; the
  vertical device it uses equals the 1D solver's diode without tunneling
  (1e-4); the Process steps tab gives the mask's area (1e-6), the dose
  through the window, and Deal-Grove oxide to 1e-12; the transistor runs
  with the full temperature model, Peltier heat, thermionic emission
  and hot electrons with heating together; bad recipe lines and ScAlN
  without gap numbers are refused with a message.

**Temperature, tunnel diodes, meshes and process shapes (new in 0.22.0)**

- Lattice temperature in the statistics: a device solved at a uniform
  380 K equals the device built for 380 K with its gap shifted by hand
  (current 1e-8, potential 1e-9 V), and at the device temperature it is
  the earlier model exactly; leakage grows more than thirtyfold from 300
  to 380 K in a narrow-gap diode.
- Thermoelectric current: a uniformly doped bar with both ends grounded
  and an uneven temperature profile carries the current that the flux
  equation gives in closed form, which depends only on the end
  temperatures (1e-6).
- The new fluxes' analytic derivatives against central differences, for
  Fermi-Dirac and Boltzmann statistics and both driving forces (1e-6).
- Thermal runaway from leakage: the device's last steady voltage and its
  first without one bracket the voltage where the lumped model
  T = T0 + R_th V I(V, T) stops having a solution, I taken from solves
  at a uniform temperature (0.3 percent).
- Hot electrons with self-heating: where the field is uniform the heat
  the electrons hand to the lattice equals the Joule heat (1e-4); in
  total it equals the electrical power (1e-3); a long bar heats and
  carries current as in drift-diffusion with self-heating (1 percent).
- Meshes: self-heating and pulses on a rectangular grid cut into right
  triangles equal the tensor grid (current 1e-10, temperature 1e-8 K);
  the heat equation on a randomly jittered unstructured mesh converges
  to the exact Kirchhoff solution of a uniformly heated slab with k(T);
  a tunnel barrier on a mesh equals the tensor grid (1e-10).
- Nonlocal band-to-band tunneling: in a uniform field with full supply it
  equals Kane's local formula integrated over the tunneling region
  (1e-7); zero current in equilibrium; electron and hole rates balance
  (1e-9); the terminal current equals the tunneling current (1e-4); an
  independent quadrature of the same model on the solved bands agrees
  (1e-6); a tunnel diode shows negative resistance and then the diffusion
  current.
- Process shapes: an isotropic etch through a mask opening against the
  exact area w R + pi R^2 / 2 (within 1 percent, shrinking with the
  cell size); a conformal film on a pillar against W t + 2 H t +
  (pi/2 - 2) t^2 (0.1 percent); a directional etch through two
  materials exactly; Deal-Grove oxidation against its closed form, two
  steps equal to one (1e-12), the consumed silicon 0.44 of the oxide, no
  oxide under a nitride mask; the dose kept through the anneal (1e-10);
  a process flow into a mesh device.
- Traps with Fermi-Dirac statistics: an InN p-n diode with traps, which
  did not reach equilibrium before 0.22.0, now does, with zero current.

**InGaN, InAlN, ScAlN, devices, phonons and the app (new in 0.18.0)**

- The Vurgaftman-Meyer and Ambacher numbers equal the printed tables
  and equations; e31 and e33 from the moduli equal a hand calculation;
  the polarization bowings, the lattice-matched InAlN (x = 0.178) and
  ScAlN (x = 0.202 on the GaN of that paper), and the InGaN and InAlN
  band edges follow their rules exactly. The ScAlN fits reproduce the
  numbers quoted in the paper's text (e33 = 3.641 at x = 0.5; changes of
  64, 148 and 233 percent in C33, e33 and P_SP, and 235 percent in
  eps33).
- The H recipe minus the ZB recipe equals 2 eps_c P_ZB of the film plus
  a second-order term, for AlGaN, InGaN and ScAlN on GaN: about 1.5
  percent of the charge for Al0.25Ga0.75N and 10 percent for
  In0.15Ga0.85N. The ScAlN charge equals the ZB recipe of Ambacher et
  al. (their Eq. 46) to 1e-14 C/m^2.
- Drift-diffusion against closed forms: Ohm's law for a doped resistor
  to 1e-9; zero current in equilibrium; the built-in potential of a p-n
  junction, kT ln(N_A N_D / n_i^2), to 1e-10 V; the short-diode current
  to 0.5 percent at 0.4 and 0.5 V, with ideality 1 to 0.1 percent; the
  minority-carrier profile of a bar under uniform generation with SRH
  recombination, G tau [1 - cosh(z/L) / cosh(W/2L)], to 1e-4; the slope
  of 1/C^2 of a metal on n-GaN, -2 / (q eps N_D), to 1e-3, by both
  solvers. Against the other electron solver: band edges within 1 meV
  and the sheet density to 1e-3 for an AlGaN/GaN layer, and the
  quasi-equilibrium C-V sheet densities to 1e-3 at three gate voltages.
  The quantum C-V: its sheets equal `solve_electrons`, its metal charge
  moves by q times the change of the electrons (to 2e-3), and it stays
  within 3 percent of the classical C-V.
  The NumPy banded solver equals a dense solve and gives the same
  current as SciPy's.
- Phonons: the GaN numbers equal the listed ones and pass the
  Lyddane-Sachs-Teller check (9.53 against 9.5); the slab modes equal
  the closed form of Fuchs and Kliewer to 1e-9 at every q, with the TO,
  LO and surface limits; an interface of two isotropic media equals the
  roots of its quadratic, and the GaN surface mode of the uniaxial model
  equals the root of eps_perp eps_par = 1; the Froehlich constant equals
  a second form of the same formula.
- The app: its charges and electron densities equal the library's, the
  Python script it writes gives the same density to 1e-12, its errors
  come back as sentences, its uncertainty budget sums to 1, and its
  server refuses a request without the page's token. Not in the test
  suite: the page was also driven through every tab in a headless
  browser, in light and dark mode and at phone width, with no script
  errors.
- Data fits: noise-free C-V data give back the composition and barrier
  to 1e-4 and 1e-3, and a thickness between grid points to 1e-3 nm;
  with noise, a joint C-V and Hall fit lands within four error bars of
  the truth with a sensible chi^2; the app gives the same barrier from
  data in nF/cm^2 and in pF over a gate area.
- Design finder: the switch-off voltage brackets the 1 percent point,
  a search recovers a known design (x = 0.30 to 0.01) that meets its
  targets, and its error bar equals the secant over one sigma.
- Leakage: A* = 120.17 A cm^-2 K^-2 for the free electron mass, the
  Richardson velocity equals A* T^2 / (q N_c); the Tsu-Esaki current over
  a thick barrier equals the thermionic-emission law to 0.2 percent, and
  over a triangular barrier has the Fowler-Nordheim exponent to 3
  percent (1.1 percent in practice); a metal with thermionic emission on
  n-GaN saturates at A* T^2 exp(-phi/kT) in reverse to 2 percent, has
  ideality 1 to 0.5 percent, and gives the Crowell-Sze forward current
  to 1 percent.
- 2D: with a gate over the whole width and a uniform grid the
  cross-section equals the 1D solver to 1e-9 (bands, density, gate
  charge), with SciPy or with the NumPy conjugate-gradient fallback; a
  wide gate gives the 1D gate and bare-surface densities under and beside
  it to 0.5 percent; the capacitance per gate length rises as the gate
  shortens and a 1 um gate is within 10 percent of 1D.
- The line-charge tools: the published constraints equal the archived
  numbers; the combination equals a truncated normal written out
  separately, and two Gaussians combine by inverse variance; the
  planners reproduce the archived predictions.

**Nitride stacks, electrons, error bars and fits (new in 0.17.0)**

Independent references: the archived results of the misfit analysis
(T. M. Mahim, nitride-misfit-polarization v1.3.1,
doi:10.5281/zenodo.24315036), computed there with a separate code
(SciPy eigensolvers and root searches, and the labplan fitting
package); Tables I and II of Dreyer et al. (2016); closed forms; SciPy
quadrature.

- The shipped HSE constants are those of Tables I and II of Dreyer et
  al., number by number; the zinc-blende reference is 3/4 of a
  polarization quantum to 1e-15 relative for three lattice constants;
  `algan` reproduces Vegard's rule, the Varshni gaps, the bowing and
  the offset rule to 1e-14 relative, and the mismatch of
  Al0.22Ga0.78N on GaN used throughout the misfit analysis.
- Interface charges equal the archived values of Table S6 of the misfit
  paper (coherent charge, and the fully relaxed charge for bare,
  electron-counting, neutral, bound and trapped lines) to 1e-12
  relative, for GaN on AlN with both constant sets and for Al0.3Ga0.7N
  on GaN held at the lattice of AlN; the bare and electron-counting
  line charges of Al0.22Ga0.78N (-2.645 and +0.085 e/nm), the line
  density and spacing of the relaxed Shen barrier, to 1e-10 to 1e-12.
- Eq. 2 holds exactly (coherent + relaxation + lines, to 1e-15 C/m^2)
  for both references; the two references differ by an r-independent
  constant equal to its closed form and smaller than 1e-3 C/m^2; the H
  recipe minus the ZB recipe is -2 r eps_c P_H (the formula of Wang et
  al., Appl. Phys. Lett. 124, 263502 (2024)) to 1e-12 relative.
- The sum rule (Eq. 3): the archived slopes, sum (-0.83683 C/m^2) and
  trapped charge (0.82679 e/nm) for Al0.3Ga0.7N stretched and
  compressed, to 1e-12 relative; any bound charge cancels in the sum to
  1e-14 C/m^2; the slope agrees with a full stack calculation.
- `solve_electrons` reproduces the archived 2DEG and barrier densities
  of the Shen series (a coherent 25 nm and 100 nm barrier, and the
  relaxed one with bare, electron-counting and neutral lines) to 1e-9
  relative (the agreement is 1e-13 in practice). Physics checks: the
  electrons sit next to the interface, the lowest level is occupied,
  the sheet grows with the barrier thickness and stays below the
  interface charge. F_1/2 agrees with SciPy quadrature to 1e-8 relative
  and its derivative with a finite difference; the tridiagonal solver
  with a dense solve.
- The mobility bound and the four mobility-drop estimates of the Shen
  analysis (1.2778 and 0.983 to 1.017 e/nm) to 1e-12 relative; the
  two-carrier Hall split inverts the forward relations to 1e-12.
- `propagate` is exact for linear functions (with and without a
  correlation), its budget sums to 1, Monte Carlo agrees with it within
  the sampling error, truncation respects the bounds, and the error of
  an interface charge equals the error of Lambda times L_A.
- `fit_least_squares` equals the closed-form weighted straight-line fit
  and its covariance (also when the error scale comes from the
  scatter), and the closed-form Deming slope when the settings have
  errors; the line charge fitted to the relaxed Shen sample equals the
  archived -0.8286 e/nm (found there by a Brent search with a tolerance
  of 1e-3).
- Not in the test suite, because each takes about a minute: the full
  joint fit of example 17. With the error scale from the scatter it
  gives a calibration of -2.7227374e12 +- 6.004988e11 cm^-2 and a line
  charge of -0.8448566 +- 0.5116510 e/nm, against -2.7227373e12 +-
  6.004990e11 and -0.8448566 +- 0.5116511 from labplan in the misfit
  analysis; with reading errors equal to that scatter and an error of
  0.05 on the relaxation, +- 0.5323733 e/nm against +- 0.5323734.

## Corrections in earlier versions

**0.16.0 (this release) fixed two silent errors and one source of
results that were not reproducible.** Each has a test that fails on
0.15.0 (the third only with NumPy 1.22 on a machine with AVX-512,
where the problem appeared).

- `solve_self_consistent_hetero` with the parabolic filling (the
  default) and a strain that breaks in-plane isotropy (`eps_xx` not
  equal to `eps_yy`, or a shear) measured each state's mass along `kx`
  only and used it for every direction. Such a strain makes the masses
  differ strongly between directions. In the case the test describes
  (GaN with arbitrary test deformation potentials, strain 0.002 and
  -0.002 in the plane) the masses along `kx` were [inf, 0.79, inf,
  0.088, 5.05, 0.95] m0 and along `ky` [3.01, 0.13, inf, 0.56, inf,
  0.15] m0, and the loop returned occupations of [0, 0.0823, 0,
  0.0030, 0.0124, 0.0023] nm^-2 where the momentum-grid filling over
  16 directions gives [0.0517, 0.0483, 0, 0, 0, 0] nm^-2. This
  combination is now refused, as the momentum-grid filling without
  `ntheta` already was.
- `dipole_matrix` given one state, `(6, N)`, read its six components
  as six states and returned a 6 x 6 matrix, and `oscillator_strengths`
  with a negative mass ratio returned negative strengths. Both are
  refused now.
- The normalization of the envelopes could differ in the last bit
  between two calls with the same eigenvector, on NumPy 1.22 with
  AVX-512, because NumPy's complex absolute value has two inner loops
  that round differently and picks one by the memory layout. The
  envelopes changed by about 2e-16 (relative), and a test that
  compared them bit for bit failed now and then in the
  oldest-dependencies CI job. The norm is now summed from the real and
  imaginary parts, which gives the same bits on every path. No printed
  number changes.

**0.15.0 fixed one silent error.** It has a test that
fails on 0.14.0.

- The self-consistent loops with `filling="kgrid"`, and
  `fill_subbands_kgrid` when given more energies than it fills,
  checked whether the next state down would hold holes only at
  `k = 0`. A branch can rise above its `k = 0` energy before it falls
  (one of a spin-split pair in a lopsided well does), and then holds
  holes in a ring although its `k = 0` energy lies below the Fermi
  level. On a tilted 6 nm GaN well (0.05 eV/nm, 61 points) filled
  with two states at 0.0491 nm^-2, 0.14.0 gave no warning, but the
  third state holds 7.0e-5 nm^-2 when it is filled too. The check now
  uses every grid momentum. Results do not change; only the warning
  and `n_states_sufficient` do.

**0.14.0 fixed two silent errors.** Each has a test
that fails on 0.13.1.

- `dos_from_dispersion` on a momentum path that does not start at 0
  returned a density of states of 0 above the energy of the first
  point, where the band continues, and a spike next to it (2.6 instead
  of 1.04 nm^-2 eV^-1 on the test's demo set). It now returns NaN
  there, as it has at the other end of the path since 0.13.0.
- `fill_subbands_thermal` accepted a negative mass and gave that
  subband a negative number of holes (-0.134 nm^-2 in the test). A
  negative mass is now refused.

A test of the depolarization inversion also depended on which member
of a Kramers pair the solver returns; it no longer does (see
[CHANGELOG.md](../CHANGELOG.md)).

**0.13.0 fixed five silent errors.** Each has a test
that fails on 0.12.0; [CHANGELOG.md](../CHANGELOG.md) has the numbers.

- `layered_profile` decided which layer a grid point on an interface
  belongs to by an exact comparison, so rounding in the sum of the
  thicknesses could move the point into the wrong layer. A 3.4 nm GaN
  well between AlN barriers then had 33 grid points instead of 34
  when it started at 2.7 nm, and its top level moved from 1.98 to
  1.23 meV.
- The depolarization geometry integral used the real part of the
  overlap of the two states. An eigenvector is only defined up to a
  phase, and multiplying one state by `i` gave `S = 0` and no shift
  at all. It now uses the modulus, which is the same for real states.
- The self-consistent loops filled only the `n_states` computed states,
  without checking that no other state lay above the Fermi level. Now
  they warn and set `n_states_sufficient` to False (example 2). This
  affected README examples 2 and 9 of version 0.12.0, which now use
  more states.
- `fill_subbands_thermal` returned all zeros (a total of 0, not the
  requested density) when no subband had a finite mass, and negative
  occupations for a negative density; it now refuses both. Below the
  end of its momentum path, `dos_from_dispersion` returned a density
  of states of 0, a gap that is not there; it now returns NaN.

**0.12.0 fixed a sign error in the Hamiltonian.** In
rows 4 to 6 of the six-band matrix, the term that couples the in-plane
and growth-direction momenta (`A6`) was placed as its complex
conjugate. Along `kx` that makes no difference, which is why every
earlier check (all along `kx`) passed. In any other in-plane direction
the results were wrong: in a symmetric 5 nm GaN well at 0.4 nm^-1 and
45 degrees the old code gave a spin splitting of 17 meV where there
must be none, and moved the top level from 0.88 to 5.25 meV. The
strain matrix had the same fault for a shear strain `eps_yz`. Affected were
`subband_dispersion`, `character_vs_k` and `splitting_vs_k` at an
angle `theta` other than 0 or pi, anything at `ky` other than 0,
`fill_subbands_kgrid` (it samples four directions by default), and
strained calculations with a non-zero `eps_yz`. Results along `kx`
without `eps_yz`, including every self-consistent run with the default
filling and README examples 1 to 7, are unchanged. The momentum-grid filler was also made
exact for parabolic subbands. On the demo well of the tests (two
levels filled, T = 0) the old filler misplaced 1.8e-3 nm^-2 per state
with 41 momentum points, 21 % of each upper-level state, and left the
upper level empty with 5 points. Several inputs that gave silent wrong
answers are now refused. [CHANGELOG.md](../CHANGELOG.md) lists before and after numbers.

**0.11.1 fixed two problems and several documentation errors.**

- `rashba_spins` with `alpha = 0` returned spins along `z`, although
  its documentation promises in-plane spins. With `alpha = 0` the two
  branches are degenerate and have no defined direction; it now
  refuses.
- `solve_self_consistent` and `solve_self_consistent_hetero` crashed
  with an `UnboundLocalError` for `max_iter=0`; they now raise a clear
  `ValueError`.
- The docstring of `fit_band_offset` suggested +1 on barrier points;
  in this package's convention that makes the barriers attract the
  holes. It now says -1 (example 6).
- One test used `numpy.trapezoid`, which needs NumPy 2.0, although the
  package allows NumPy 1.22; the test now works on both, and CI tests
  NumPy 1.22.0 with SciPy 1.8.0. CI also runs Python 3.10 now.
- Statements that were stronger than the tests: the lab-unit round
  trip is not always exact (it can differ in the last bit); several
  "exact" claims hold to the tolerances listed above. See
  [CHANGELOG.md](../CHANGELOG.md).
- Not changed, but now described: the self-consistent filling can
  give the two states of a Kramers pair different masses and holes
  (example 2 and [Limits](limits.md#limits)).

The full history is in [CHANGELOG.md](../CHANGELOG.md).
