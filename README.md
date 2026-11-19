# kpenvelope

[![PyPI](https://img.shields.io/pypi/v/kpenvelope)](https://pypi.org/project/kpenvelope/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015269-blue)](https://doi.org/10.5281/zenodo.22015269) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

`kpenvelope` is a Python package that computes the quantum states of
**holes** (the missing electrons that carry positive charge) trapped in
thin layers of wurtzite semiconductors such as GaN and AlN. It gives
the energy levels and the wavefunctions from the standard six-band
k.p model of Chuang and Chang (Phys. Rev. B 54, 2491 (1996)). It can
solve that model together with the electrostatics of the holes' own
charge, so the trapping potential is not guessed: it follows from the
balance between the fixed charge at the interface and the hole gas.
It was built for polarization-induced two-dimensional hole gases, and
works for any layered wurtzite structure you can supply cited material
numbers for.

Since version 0.17.0 it also works out where that fixed charge comes
from. For a stack of c-plane nitride layers (GaN, AlN, AlGaN, or any
layer you describe with cited numbers) it computes the charge at every
interface, also when a layer has partly relaxed, and it makes the one
number the usual recipes hide explicit: the charge carried by the
misfit dislocations through which the layer relaxed. It solves the
electron gases of such stacks with doping and a pinned surface,
carries error bars through any of these calculations, and fits the
calibration and misfit-line charges to measured series. It answers
questions such as:

- Where are the hole energy levels in my layer, and what are they made
  of (heavy-hole, light-hole or split-off character)?
- For a measured sheet density and temperature, how are the holes
  shared among the levels, and where does the hole gas sit?
- What is the hole mass, and how does it change with momentum? (A hole
  mass here is not one number.)
- Where will the intersubband absorption line sit, including the
  depolarization shift that a doped layer adds? How strongly does the
  layer absorb light polarized along the growth axis, and light
  polarized in the plane?
- Which band offset, or which sheet density, reproduces my measured
  absorption line?
- What hole density, mass and quantum lifetime do my magnetotransport
  (Shubnikov-de Haas) measurements give, with error bars, and which
  frequencies and masses should the computed hole gas show?
- How much does each computed level depend on each material number,
  and what error bar do the uncertainties of those numbers give it?
- What fixed charge sits at each interface of my nitride stack when a
  layer has partly relaxed, and how much of it depends on the charge of
  the misfit dislocations, which neither polarization reference decides?
- How many electrons does an AlGaN/GaN or AlN/GaN stack hold, with its
  doping and a pinned surface, and with what error bar?
- Which misfit-line charge, and which calibration charge, do my measured
  densities across a thickness or relaxation series imply?
- Would an experiment be precise enough to tell charge bound to a
  dislocation from electrons trapped on it?

Since version 0.18.0 it also covers the rest of a nitride device
stack. InGaN, InAlN and ScAlN are built in, each number copied from a
cited table or fit (Vurgaftman and Meyer 2003, Dreyer et al. 2016,
Ambacher et al. 2021), and layers from different number sets that meet
at an interface draw a warning. A one-dimensional drift-diffusion
solver adds holes, doping of both kinds, recombination, ohmic and metal
contacts, and current-voltage and capacitance-voltage curves. The
optical phonons of a layer follow from the dielectric continuum model.
What is known about the misfit-line charge can be combined into one
estimate, and a planned measurement can be judged before it is grown.
Measured C-V curves and Hall densities can be fitted directly, with
error bars and a warning where the data cannot separate two unknowns; a
design finder searches compositions and thicknesses for a target
electron density and switch-off voltage; gate leakage over and through
the barrier follows from the band diagram; and a two-dimensional
cross-section shows the gate edges and the electrons beside the gate.
Since version 0.19.0 it simulates whole devices in two and three
dimensions: a transistor with source, gate and drain carries current,
its mobility saturates at high field and falls as it warms, deep-level
traps (iron, carbon or surface states) fill and empty over time, and the
current heats the device through a heat equation that reaches down
through the substrate to the heat sink.
And everything above can be used without writing code, in an app that
opens in your web browser (`kpenvelope-app`) and says what each result
means in plain words. It also answers:

- What is the charge at an InAlN/GaN, InGaN/GaN or ScAlN/GaN interface,
  and how far apart are the published number sets?
- Where is the threshold of my transistor layer in a C-V sweep, and how
  much current does a p-n junction pass?
- At what energies do the interface phonons of a GaN layer sit?
- Which composition, barrier height and parasitic capacitance does my
  measured C-V curve imply, and how sure is each?
- Which barrier composition and thickness give 1.2e13 electrons per
  cm^2 and a switch-off at -4 V, and how much do growth tolerances move
  them?
- How much does the gate leak, and how much more does a short gate hold
  at its edges?
- Combining every published constraint, how well is the misfit-line
  charge known, and which experiment would pin it down?
- What drain current does my transistor carry, how hot does it get on
  SiC, and how much current does it lose to the heat?
- How far does the current drop after a high drain voltage when the
  buffer holds iron, and how long does it take to recover?

No material number is built in without a written source. When an input
cannot give a trustworthy answer, the package stops with an error that
says why, instead of returning a number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, units and conventions](#install-units-and-conventions)
- [The app](#the-app) (no code needed)
- [Examples](#examples) (each with the output it prints; 15 to 18 for
  nitride stacks, electrons, error bars and fits; 19 to 23 for InAlN and
  ScAlN, C-V, a p-n diode, phonons, and the misfit-line charge; 24 to 27
  for fitting measured data, the design finder, gate leakage and the 2D
  cross-section; 28 to 30 for transistor curves with self-heating, drain
  lag from traps, and a 3D transistor)
- [Tutorials](#tutorials) (from data to a result with error bars)
- [What is in the package](#what-is-in-the-package)
- [Cited parameter sets](#cited-parameter-sets)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

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
  numbers and their source, and thermionic emission at a metal).
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

## Examples

Each example below runs as written, and the output shown is what it
printed with kpenvelope 0.16.0 (examples 1 to 13 print the same as
with 0.15.0). Numbers labeled illustrative are chosen for the example,
not taken from a source.

### 1. Energy levels and what they are made of

```python
import numpy as np
from kpenvelope import gan_rinke2008, solve_subbands, band_character

p = gan_rinke2008()                  # cited GaN parameter set
z = np.linspace(0.0, 5.0, 51)        # a 5 nm layer, grid in nm

energies, envelopes = solve_subbands(p, z, n_states=4)
frac = band_character(envelopes, z)  # columns: HH, LH, CH
for e, (hh, lh, ch) in zip(energies, frac):
    print(f"E = {e*1e3:8.2f} meV   HH {hh:.3f}  LH {lh:.3f}  CH {ch:.3f}")
```

```
E =     8.26 meV   HH 1.000  LH 0.000  CH 0.000
E =     8.26 meV   HH 1.000  LH 0.000  CH 0.000
E =    -2.28 meV   HH 0.000  LH 0.990  CH 0.010
E =    -2.28 meV   HH 0.000  LH 0.990  CH 0.010
```

The states come in Kramers pairs. At zero in-plane momentum the top
pair is pure heavy-hole; the next pair is mostly light-hole with a
little split-off character, because the spin-orbit term mixes those
two.

### 2. A self-consistent hole gas at a measured density

```python
import numpy as np
from kpenvelope import (gan_rinke2008, sheet_density_from_cm2,
                        sheet_density_to_cm2, solve_self_consistent)

p = gan_rinke2008()
z = np.linspace(0.0, 6.0, 97)             # nm, z = 0 is the interface
ps = sheet_density_from_cm2(4.6e13)       # 4.6e13 cm^-2 -> 0.46 nm^-2
res = solve_self_consistent(p, z, ps, n_states=8, mixing=0.5, tol=2e-5)

print("converged:", res.converged, "after", res.iterations, "iterations")
print("enough states:", res.n_states_sufficient)
print("subband edges (meV):", np.round(res.energies * 1e3, 2))
print("edge masses (m0):   ", np.round(res.masses, 3))
print("holes per subband (1e13 cm^-2):",
      np.round(sheet_density_to_cm2(res.occupations) / 1e13, 3))
centroid = (z * res.density).sum() / res.density.sum()
print(f"centre of the hole gas: {centroid:.2f} nm from the interface")
```

```
converged: True after 20 iterations
enough states: True
subband edges (meV): [-402.37 -402.37 -413.58 -413.58 -548.47 -548.47 -559.02 -559.02]
edge masses (m0):    [0.445 0.442   inf 0.17  0.604 0.601   inf 0.206]
holes per subband (1e13 cm^-2): [1.622 1.61  0.    0.581 0.356 0.355 0.    0.076]
centre of the hole gas: 0.76 nm from the interface
```

The 4.6e13 cm^-2 density is the measured value used by the source
study (see [Where it comes from](#where-it-comes-from)). This run uses
hard walls and fills the subbands with parabolic masses measured at
the subband edge. The source paper reports a center of 0.568 nm for its
hard-wall case, filled from the computed dispersion instead; example 8
shows how much the filling model matters here.

`n_states` counts single states, so a Kramers pair counts twice, and it
must reach every subband that holds holes. Since version 0.13.0 the
loop checks this at the end: if the next state down lies above the
Fermi level, it warns (`RuntimeWarning`) and sets
`res.n_states_sufficient` to False. With this filling 4 states are not
enough here. Version 0.12.0 printed this example with `n_states=4`,
without a warning, and the third and fourth pairs, which hold
0.79e13 cm^-2 above, were left out.

Read the masses and the per-state holes with care. States 2 and 3 are
a Kramers pair, so by symmetry they should hold the same number of
holes; here one gets a mass of `inf` and no holes, the other 0.17 m0
and all of the pair's holes (states 6 and 7 likewise). That split is
an artifact of how the loop measures a mass: it takes one small step
in momentum (0.02 nm^-1) away from `k = 0`. In this lopsided well the
two states of a pair move apart in proportion to the momentum, so over
that one step one state rises (read as an infinite mass) and the other
falls too fast (read as too light a mass), and both numbers change if
the step changes. So do the totals per pair (here 3.23, 0.58, 0.71 and
0.08 in units of 1e13 cm^-2), because the lighter mass sets how many
holes the pair takes. The too-light masses also pull the Fermi level
far down, into the third and fourth pairs. Filling the full dispersion
with `fill_subbands_kgrid` at the same converged potential gives 4.01
and 0.59 (1e13 cm^-2) in the first two pairs and nothing in the
others. For occupations you rely on, run the loop itself with the full
dispersion (`filling="kgrid"`, example 8), and do not publish
self-consistent numbers without checking the barrier and filling model
against your own system. Pass `temperature_K=` for a finite-temperature
filling; the default 0 is the zero-temperature filling.

### 3. One subband, many masses

```python
import numpy as np
from kpenvelope import (gan_rinke2008, subband_dispersion, local_mass,
                        character_vs_k)

p = gan_rinke2008()
z = np.linspace(0.0, 5.0, 51)
kts = np.array([0.0, 0.2, 0.4, 0.6, 0.8])      # in-plane momentum, nm^-1

E = subband_dispersion(p, z, kts, n_states=2)  # (5 momenta, 2 states)
k_mid, m = local_mass(kts, E[:, 0])            # top subband
fr = character_vs_k(p, z, kts, n_states=1)     # (5, 1, 3)
for k, mass in zip(k_mid, m):
    print(f"k = {k:.1f} nm^-1   local mass {mass:.3f} m0")
print("HH share of the top subband:", np.round(fr[:, 0, 0], 3))
```

```
k = 0.1 nm^-1   local mass 0.482 m0
k = 0.3 nm^-1   local mass 1.085 m0
k = 0.5 nm^-1   local mass 1.708 m0
k = 0.7 nm^-1   local mass 1.851 m0
HH share of the top subband: [1.    0.921 0.679 0.58  0.541]
```

The top subband starts as pure heavy-hole and picks up other character
as the momentum grows, and its local mass changes with it. Which mass
an experiment sees depends on which momenta it probes.

### 4. Intersubband dipole and oscillator strength

```python
import numpy as np
from kpenvelope import (demo_single_band, solve_subbands, dipole_matrix,
                        oscillator_strengths)

# Synthetic test set: six identical, uncoupled bands of mass 0.5 m0,
# in an 8 nm hard-wall well. Each level is six-fold degenerate, so
# only sums over a level's six states have a fixed value.
p = demo_single_band(A=-2.0)
L = 8.0
z = np.linspace(0.0, L, 201)
E, F = solve_subbands(p, z, n_states=12)   # levels 1 and 2

d = dipole_matrix(z, F)
f = oscillator_strengths(E, d, mass_ratio=0.5)
z12 = np.sqrt((abs(d[0, 6:12]) ** 2).sum())
Leff = L + 2 * (z[1] - z[0])   # the hard walls sit one grid step outside
print(f"z12 = {z12:.4f} nm   (infinite well, 16 L / 9 pi^2: "
      f"{16 * Leff / (9 * np.pi**2):.4f} nm)")
print(f"f12 = {f[6:12].sum():.4f}      (infinite well, 256 / 27 pi^2: "
      f"{256 / (27 * np.pi**2):.4f})")
```

```
z12 = 1.4554 nm   (infinite well, 16 L / 9 pi^2: 1.4554 nm)
f12 = 0.9606      (infinite well, 256 / 27 pi^2: 0.9607)
```

`demo_single_band` is a made-up parameter set with textbook answers;
it is not a real material.

### 5. Depolarization shift, and the density from a measured line

```python
import numpy as np
from kpenvelope import (HBAR2_OVER_2M0, depolarization_shift,
                        energy_to_wavenumber, sheet_density_from_shift)

# Two subbands of an ideal 8 nm well, written down directly
# (illustrative: mass 0.5 m0, permittivity 10, 0.05 nm^-2 of carriers).
L, m = 8.0, 0.5
z = np.linspace(0.0, L, 801)
env1 = np.sqrt(2 / L) * np.sin(np.pi * z / L)[None, :]      # one component
env2 = np.sqrt(2 / L) * np.sin(2 * np.pi * z / L)[None, :]
E1 = -HBAR2_OVER_2M0 * (np.pi / L) ** 2 / m                 # valence convention
E2 = -HBAR2_OVER_2M0 * (2 * np.pi / L) ** 2 / m

out = depolarization_shift(z, env1, env2, E1, E2, n_sheet=0.05, eps_r=10.0)
print(f"subband spacing   {out['E_bare'] * 1e3:.2f} meV")
print(f"shifted resonance {out['E_shifted'] * 1e3:.2f} meV "
      f"(alpha = {out['alpha']:.3f})")
print(f"  which an infrared spectrometer reports as "
      f"{energy_to_wavenumber(out['E_shifted']):.1f} cm^-1")

inv = sheet_density_from_shift(z, env1, env2, E1, E2,
                               e_meas_ev=out["E_shifted"], eps_r=10.0)
print(f"density recovered from the resonance: {inv['n_sheet']:.6f} nm^-2")
```

```
subband spacing   35.25 meV
shifted resonance 64.15 meV (alpha = 2.311)
  which an infrared spectrometer reports as 517.4 cm^-1
density recovered from the resonance: 0.050000 nm^-2
```

The shifted line is `E_bare * sqrt(1 + alpha)`, with `alpha`
proportional to the sheet density. Because of that, the density follows
from a measured line in closed form: `sheet_density_from_shift` is the
exact inverse of `depolarization_shift`, not a fit. With solver states
(`solve_subbands`, `solve_heterostructure`) pass one state's `(6, N)`
envelope; note that the two members of a Kramers pair are not unique,
so pick states that are not degenerate with each other. Which
combination of the two members of a pair the solver returns is a
convention (it changed in version 0.14.0, and before that it could
differ between computers), and S can differ with it: for HH1 -> HH2
of a tilted 6 nm GaN well, one HH2 member gives 0.10378 nm and the
other 0. S summed over the two members of the upper pair does not
depend on the convention, and `level_transitions` (example 12) gives
that sum for every level. Use states at
zero in-plane momentum, where the solver's states are real. The
result does not depend on the arbitrary phase of either state (since
version 0.13.0; see [Corrections](#corrections-in-earlier-versions)).
A measured line in cm^-1 goes in through `energy_from_wavenumber`.

### 6. Calibrating a band offset from a measured spacing

```python
import numpy as np
from kpenvelope import (gan_rinke2008, aln_rinke2008, layered_profile,
                        solve_heterostructure, fit_band_offset,
                        offset_sensitivity)

GaN, AlN = gan_rinke2008(), aln_rinke2008()
z = np.linspace(0.0, 12.0, 97)
# 4 nm AlN | 4 nm GaN | 4 nm AlN. The SHAPE of the offset profile:
# -1 in the barriers (their valence band edge is lower), 0 in the well.
params, shape = layered_profile(z, [(4.0, AlN, -1.0), (4.0, GaN, 0.0),
                                    (4.0, AlN, -1.0)])

# A synthetic "measurement": the spacing the solver itself gives for an
# offset of 0.1 eV (illustrative; not a measured GaN/AlN offset).
E, _ = solve_heterostructure(z, params, band_edge=0.1 * shape, n_states=3)
e_meas = abs(E[0] - E[2])
print(f"'measured' spacing: {e_meas * 1e3:.3f} meV")

fit = fit_band_offset(z, params, shape, e_meas, bracket=(0.05, 0.4),
                      sigma_e_ev=1e-4)          # 0.1 meV measurement error
print(f"fitted offset: {fit['offset_ev']:.6f} eV "
      f"+/- {fit['sigma_offset_ev']:.3f} eV")

for off in (0.1, 0.5):
    s = offset_sensitivity(z, params, shape, off)
    print(f"at {off} eV the spacing moves {s * 1e3:.2f} meV per eV of offset")
```

```
'measured' spacing: 9.764 meV
fitted offset: 0.100000 eV +/- 0.017 eV
at 0.1 eV the spacing moves 5.86 meV per eV of offset
at 0.5 eV the spacing moves 0.48 meV per eV of offset
```

The fit recovers the offset that made the "measurement". The error bar
is the measurement error divided by the sensitivity. The last line
shows why a deep well is a poor probe of its barrier: at 0.5 eV the
spacing barely moves, so the same 0.1 meV error would give an error
bar about twelve times larger. When the sensitivity is so small that
the error would be amplified more than a million times, the fit
refuses. For a doped well, subtract the depolarization shift before
fitting (example 5).

### 7. The k-linear Rashba term, with your coefficient

```python
from kpenvelope import rashba_splitting, rashba_spins

alpha = 4.5e-4    # 4.5 meV A = 4.5e-4 eV nm (bulk n-GaN, Stefanowicz 2014)
ref = "W. Stefanowicz et al., Phys. Rev. B 89, 205201 (2014)"

print(f"splitting at k = 0.1 nm^-1: "
      f"{rashba_splitting(alpha, 0.1, ref) * 1e3:.3f} meV")
s = rashba_spins(alpha, 0.1, 0.0, ref)["spins"]
print("spin of each branch (x, y, z):", s.round(3).tolist())
```

```
splitting at k = 0.1 nm^-1: 0.090 meV
spin of each branch (x, y, z): [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
```

The splitting is `2 |alpha| k`, and the two spins lie in the plane,
at right angles to the momentum, pointing opposite ways. No coefficient
ships with the package; you pass yours with a `reference`. For
orientation, the measured bulk n-GaN value is 4.5 +/- 1 meV A
(Stefanowicz et al., PRB 89, 205201 (2014)), and GaN/AlGaN
two-dimensional electron gases show about 5.5-6 meV A (PRB 74, 033302
and 74, 113308 (2006)). This term is for the conduction-band companion
problem; see [Limits](#limits).

### 8. The self-consistent hole gas, filled from the full dispersion

```python
import numpy as np
from kpenvelope import (gan_rinke2008, sheet_density_from_cm2,
                        sheet_density_to_cm2, solve_self_consistent)

p = gan_rinke2008()
z = np.linspace(0.0, 6.0, 97)
ps = sheet_density_from_cm2(4.6e13)
res = solve_self_consistent(p, z, ps, n_states=4, mixing=0.5, tol=2e-5,
                            filling="kgrid", kmax=1.6, nk=16)

print("converged:", res.converged, "after", res.iterations, "iterations")
print("subband edges (meV):", np.round(res.energies * 1e3, 2))
print("holes per state (1e13 cm^-2): ",
      np.round(sheet_density_to_cm2(res.occupations) / 1e13, 3))
pairs = res.occupations[0::2] + res.occupations[1::2]
print("holes per Kramers pair (1e13 cm^-2):",
      np.round(sheet_density_to_cm2(pairs) / 1e13, 3))
print(f"Fermi level: {res.fermi_level * 1e3:.2f} meV")
centroid = (z * res.density).sum() / res.density.sum()
print(f"centre of the hole gas: {centroid:.2f} nm from the interface")
```

```
converged: True after 21 iterations
subband edges (meV): [-394.74 -394.74 -405.91 -405.91]
holes per state (1e13 cm^-2):  [1.974 1.964 0.352 0.311]
holes per Kramers pair (1e13 cm^-2): [3.938 0.662]
Fermi level: -450.10 meV
centre of the hole gas: 0.63 nm from the interface
```

This is example 2 again, but every iteration now solves the problem
at 16 in-plane momenta from 0 to `kmax` = 1.6 nm^-1 and fills those
computed energies, instead of treating each state as a parabola. The
hole density is built from the wavefunctions at every one of those
momenta, not only at `k = 0`. Each iteration needs 17 solutions
instead of 2, so expect it to take about ten times longer than example
2 (with version 0.15.0, about 5 s for example 2 and 60 s for this one
on one core of our loaded test machine). No state is left
with an infinite mass and no holes any more. The two members of a pair
still hold slightly different numbers of holes, and that is physical:
in this lopsided well they split apart as the momentum grows
(`splitting_vs_k` shows it), so they are filled to different momenta.
`kmax` must lie beyond the last occupied momentum; if it does not, the
solver stops and says so. Four states are enough here: the loop checks
that the next state down lies below the Fermi level, and would warn
otherwise. Only one momentum direction is used, which is
exact here because the subbands do not depend on the direction (a
checked property). With a strain that makes the plane anisotropic you
must pass `ntheta`.

### 9. A barrier on the left: say where the fixed charge sits

```python
import numpy as np
from kpenvelope import (gan_rinke2008, aln_rinke2008, layered_profile,
                        solve_self_consistent_hetero)

GaN, AlN = gan_rinke2008(), aln_rinke2008()
z = np.linspace(0.0, 10.0, 51)            # 5 nm AlN, then 5 nm GaN
# the -0.8 eV offset is illustrative, not a cited GaN/AlN value
params, edge = layered_profile(z, [(5.0, AlN, -0.8), (5.0, GaN, 0.0)])
# one permittivity per grid point; GaN's cited 10.4 is used everywhere
# here because no vetted AlN value ships with the package
eps = np.full(z.size, GaN.eps_r)

for sheet in (None, 5.0):
    r = solve_self_consistent_hetero(z, params, edge, ps=0.2, eps_r=eps,
                                     n_states=12, mixing=0.4, tol=1e-6,
                                     max_iter=60, sheet_z=sheet)
    print(f"sheet_z = {sheet}: converged {r.converged} "
          f"after {r.iterations} iterations")
    if r.converged:
        in_aln = r.density[z < 5.0].sum() * (z[1] - z[0]) / 0.2
        centre = (z * r.density).sum() / r.density.sum()
        print(f"  {100 * in_aln:.1f} % of the holes in the AlN, "
              f"centre of the gas at {centre:.2f} nm")
```

```
sheet_z = None: converged False after 60 iterations
sheet_z = 5.0: converged True after 30 iterations
  0.8 % of the holes in the AlN, centre of the gas at 5.75 nm
```

The holes are held by a fixed sheet of negative charge at the
interface. `sheet_z` tells the solver where that sheet is. Without it
the solver keeps its old assumption that the sheet sits at the first
grid point, which is right when the grid starts at the interface (the
hard-wall case) but not here: then the sheet's whole field, 1.74 eV
across these 5 nm of AlN, lies inside the barrier. The far side of the
barrier then looks more attractive to the holes than the GaN, and the
loop does not settle (its last iteration has 67 % of the holes in the
AlN). With the sheet at the
interface the gas sits in the GaN, against the barrier. Version 0.12.0
printed this example with 4 states, which leave out the third pair;
the loop now warns about that (see example 2). With 8 states the
converged run is already complete, but the unsettled one is not. `eps_r` may be
one value per grid point, so a stack of layers with different
permittivities is handled; supply a cited value for each layer.

### 10. Refusals

```python
import numpy as np
from kpenvelope import (aln_rinke2008, gan_rinke2008, rashba_splitting,
                        solve_self_consistent, strain_blocks)

checks = [
    ("AlN has no permittivity",
     lambda: solve_self_consistent(aln_rinke2008(), np.linspace(0, 4, 33), 0.1)),
    ("GaN set has no deformation potentials",
     lambda: strain_blocks(gan_rinke2008(), np.zeros((3, 3)))),
    ("Rashba coefficient without a source",
     lambda: rashba_splitting(4.5e-4, 0.1, reference="")),
]
for label, call in checks:
    try:
        call()
    except ValueError as err:
        print(f"{label}:\n  {err}")
```

```
AlN has no permittivity:
  this parameter set carries no vetted permittivity (eps_r is NaN; barrier-only set). Supply a cited eps_r before running a self-consistent Poisson solve.
GaN set has no deformation potentials:
  this parameter set carries no deformation potentials (D1..D6 are None). Supply cited values before building a strained Hamiltonian; none are shipped by default, on purpose.
Rashba coefficient without a source:
  a real `reference` string is required: no Rashba coefficient ships with this package, so yours must carry its source (your measurement or a paper)
```

The script `examples/demo_well.py` runs the self-consistent loop on
the synthetic demo set.

### 11. Density and mass from Shubnikov-de Haas measurements

```python
import numpy as np
from kpenvelope import (fit_sdh_mass, sdh_frequency, sheet_density_from_cm2,
                        sheet_density_from_sdh)

# Illustrative magnetotransport numbers, not from a real sample.
# One SdH frequency of 150 +/- 2 T, read as a spin-degenerate subband:
d = sheet_density_from_sdh(150.0, branches=2, sigma_frequency_T=2.0)
print(f"density: {d['n_sheet_cm2']:.3e} +/- {d['sigma_n_sheet_cm2']:.1e} cm^-2")

# Its amplitude at 15 T and six temperatures, each +/- 0.01:
T = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
amp = [0.915, 0.728, 0.483, 0.315, 0.181, 0.111]
fit = fit_sdh_mass(T, amp, field_T=15.0, sigma_amplitudes=[0.01] * 6)
print(f"cyclotron mass: {fit['mass_ratio']:.3f} +/- "
      f"{fit['sigma_mass_ratio']:.3f} m0")
print(f"chi-square per degree of freedom: {fit['chi2_red']:.2f} "
      f"({fit['dof']} degrees of freedom)")

# The frequencies the per-state densities of example 8 should show:
per_state = sheet_density_from_cm2(np.array([1.974e13, 1.964e13,
                                             0.352e13, 0.311e13]))
print("predicted frequencies (T):",
      np.round(sdh_frequency(per_state, branches=1), 1))
```

```
density: 7.254e+12 +/- 9.7e+10 cm^-2
cyclotron mass: 0.502 +/- 0.006 m0
chi-square per degree of freedom: 0.79 (4 degrees of freedom)
predicted frequencies (T): [816.4 812.2 145.6 128.6]
```

The frequency gives the density through the Onsager relation: a
contour that encloses an area `A` in momentum space oscillates with
`F = hbar A / (2 pi e)`, so one spin-resolved branch holding `n` holes
per area has `F = (h/e) n`, whatever the shape of the contour.
`branches` says how many spin-resolved branches share the contour that
oscillates (2 when one frequency stands for a spin-degenerate subband,
1 when the two spin branches show separate frequencies). It has no
default, because a wrong guess is a factor of two.

The mass comes from the Lifshitz-Kosevich temperature factor
`X / sinh(X)` with `X = 2 pi^2 k_B T m* / (hbar e B)`, about
14.69 (m*/m0) T / B with T in kelvin and B in tesla. The fit refits
the zero-temperature amplitude with the mass, and gives honest error
bars: from your amplitude errors if you pass them (then the
chi-square per degree of freedom, near 1 here, tests the model), or
else from the scatter of the points. It refuses data that do not fall
clearly with temperature, because those do not determine the mass. The
fit assumes the scattering does not change over the temperature range.

The measured mass is the cyclotron mass of the Fermi contour. For a
subband that depends only on the size of the momentum, `local_mass`
between two momenta just below and above the Fermi wavevector
(`k_F = sqrt(4 pi n)` for one branch holding `n`) is the same quantity,
as a finite difference, for the computed dispersion, so the two can be
compared directly.

### 12. Transitions between levels, not single states

```python
import numpy as np
from kpenvelope import gan_rinke2008, level_transitions, solve_subbands

p = gan_rinke2008()
z = np.linspace(0.0, 6.0, 61)        # a 6 nm GaN layer
v = 0.08 * z                         # a tilting field (illustrative)
E, F = solve_subbands(p, z, potential=v, n_states=6)

lev = level_transitions(z, E, F)     # from the top level (initial=0)
for L, states in enumerate(lev["levels"]):
    print(f"level {L} (states {', '.join(map(str, states))}): "
          f"{lev['transition_energy'][L] * 1e3:6.2f} meV down, "
          f"|z|^2 {lev['z2'][L]:.5f} nm^2, S {lev['S'][L]:.5f} nm")
```

```
level 0 (states 0, 1):   0.00 meV down, |z|^2 0.00000 nm^2, S 0.00000 nm
level 1 (states 2, 3):  11.07 meV down, |z|^2 0.00000 nm^2, S 0.00000 nm
level 2 (states 4, 5):  88.40 meV down, |z|^2 0.17109 nm^2, S 0.10378 nm
```

At zero in-plane momentum each level is a Kramers pair, and which two
states of the pair the solver returns is a convention (see example
5). A number taken from one state, such as `|z_fi|^2` or the
depolarization integral `S`, can change with that convention; for
level 2 here one state gives `S` = 0.10378 nm and the other 0.
`level_transitions` sums over the states of the final level and
averages over those of the initial level, as a transition rate
between levels does, so its numbers do not depend on the convention.
Level 1 is light-hole-like and does not couple to light polarized
along `z` from the heavy-hole top level; level 2 does. Pass
`mass_ratio=` for oscillator strengths, and `initial=` to start from
another level. `degenerate_levels(energies)` gives the grouping alone.

### 13. How much a level depends on each material number

```python
import numpy as np
from kpenvelope import gan_rinke2008, parameter_sensitivities

p = gan_rinke2008()
z = np.linspace(0.0, 5.0, 51)
s = parameter_sensitivities(z, p, kx=0.4, n_states=2,
                            sigmas={"A5": 0.1, "A6": 0.1})  # illustrative
print(f"top state at k = 0.4 nm^-1: {s['energies'][0] * 1e3:.2f} meV")
for name, d in zip(s["names"], s["derivatives"][0, 0]):
    if name.startswith("A"):
        print(f"  dE/d{name:<6s} {d * 1e3:+8.3f} meV per unit of {name}")
    else:
        print(f"  dE/d{name:<6s} {d:+8.3f} meV per meV")
print(f"error from A5 and A6 (+/- 0.1 each): "
      f"{s['sigma_energies'][0] * 1e3:.2f} meV")
```

```
top state at k = 0.4 nm^-1: 0.88 meV
  dE/dA1      +14.113 meV per unit of A1
  dE/dA2       +6.096 meV per unit of A2
  dE/dA3      +13.998 meV per unit of A3
  dE/dA4       +6.056 meV per unit of A4
  dE/dA5       -5.592 meV per unit of A5
  dE/dA6       -0.228 meV per unit of A6
  dE/ddelta1   +0.994 meV per meV
  dE/ddelta2   +0.364 meV per meV
  dE/ddelta3   +0.124 meV per meV
error from A5 and A6 (+/- 0.1 each): 0.56 meV
```

`parameter_sensitivities` gives the exact derivative of every computed
energy with respect to each of the nine parameters (A1..A6 and the
three splittings) of every material, and, with a band edge, with
respect to the size of the whole offset profile. Each derivative is
`v^H (dH/dparameter) v` for the computed state `v` (first-order
perturbation theory, exact for the discrete matrix), so no step size
is involved; the tests compare it with finite differences of the
solver. With `sigmas=`, your 1-sigma errors of the parameters (none
ship with the package; the 0.1 here is illustrative), it also returns
the covariance of the energies, assuming independent errors. Here the
in-plane terms A5 and A6 move the top level at finite momentum, while
at `k = 0` they would not enter at all. The third tutorial uses this
for an error budget.

### 14. Light polarized in the plane

```python
import numpy as np
from kpenvelope import (band_character, dominant_character, gan_rinke2008,
                        inplane_transitions, level_transitions,
                        solve_subbands, velocity_matrix)

p = gan_rinke2008()
z = np.linspace(0.0, 6.0, 61)        # the tilted 6 nm well of example 12
v = 0.08 * z
E, F = solve_subbands(p, z, potential=v, n_states=8)

kind = dominant_character(band_character(F, z))
along_z = level_transitions(z, E, F, mass_ratio=1.0)
in_plane = inplane_transitions(z, p, E, F, angle=0.0)
for L in (1, 2, 3):
    print(f"level 0 -> {L} ({kind[2 * L]}): "
          f"{along_z['transition_energy'][L] * 1e3:5.2f} meV   "
          f"f along z {along_z['oscillator_strength'][L]:.4f}   "
          f"f in the plane {in_plane['oscillator_strength'][L]:.4f}")

# the diagonal is the group velocity dE/dk (times hbar), with no step size
k = 0.4
Ek, Fk = solve_subbands(p, z, kx=k, potential=v, n_states=2)
V = velocity_matrix(z, p, Fk, kx=k)
print(f"dE/dk of the top state at k = {k} nm^-1: {V[0, 0].real:.5f} eV nm")
```

```
level 0 -> 1 (LH): 11.07 meV   f along z 0.0000   f in the plane 0.0065
level 0 -> 2 (HH): 88.40 meV   f along z 0.3969   f in the plane 0.0000
level 0 -> 3 (LH): 99.32 meV   f along z 0.0000   f in the plane 0.0050
dE/dk of the top state at k = 0.4 nm^-1: -0.02176 eV nm
```

Light polarized along the growth axis `z` couples the top heavy-hole
level only to the second heavy-hole level (example 12). Light
polarized in the plane, for example at normal incidence, couples
through the in-plane velocity, which is `dH/dk` divided by hbar, with
`H` the matrix the solver diagonalizes. For one simple band that
velocity is the same for every subband, so it cannot move a hole from
one subband to another. The six-band matrix also has terms that
couple the heavy-hole, light-hole and split-off components (at
`k = 0` the A6 term), and here they let the top level absorb into the
two light-hole levels, 60 to 80 times more weakly than the `z` line,
and not into the second heavy-hole level. Both strengths use the
free-electron mass (hence `mass_ratio=1.0` for the `z` line), so they
can be compared directly. At `k = 0` they do not depend on the
direction of the polarization (`angle`), because the subbands are the
same in every in-plane direction. For other momenta, solve at
`kx`, `ky` and pass the same `kx`, `ky` here. The diagonal of
`velocity_matrix` is the slope of each state's dispersion, exactly
(here the same number as a central difference of two solves,
-0.021761 eV nm); it needs no step size.

### 15. The charge at a relaxed AlGaN/GaN interface (new in 0.17.0)

```python
from kpenvelope import StackLayer, algan, interface_charges

gan, bar = algan(0.0), algan(0.22)                 # PBEsol polarization set
for lc in ("bare", "neutral", "electron-counting"):
    stack = [StackLayer(gan, 2000.0), StackLayer(bar, 103.0, relaxation=0.78)]
    c = interface_charges(stack, line_charge=lc)[0]
    print(f"{lc:18s} Lambda = {c.line_charge:+.3f} e/nm   sigma = {c.sigma_cm2:.3e} cm^-2")
c0 = interface_charges([StackLayer(gan, 2000.0), StackLayer(bar, 103.0)])[0]
print(f"coherent barrier                        sigma = {c0.sigma_cm2:.3e} cm^-2")
print(f"mismatch {c.eps_c:.5f}, misfit lines {c.line_density * 1e7:.3e} cm/cm^2")
```

```
bare               Lambda = -2.645 e/nm   sigma = 3.564e+12 cm^-2
neutral            Lambda = +0.000 e/nm   sigma = 1.051e+13 cm^-2
electron-counting  Lambda = +0.085 e/nm   sigma = 1.073e+13 cm^-2
coherent barrier                        sigma = 1.432e+13 cm^-2
mismatch 0.00534, misfit lines 2.626e+05 cm/cm^2
```

The barrier is the 103 nm Al0.22Ga0.78N of Shen et al., Appl. Phys.
Lett. 76, 2746 (2000), relaxed by 78 percent (their X-ray maps).
Relaxation lowers the charge below the coherent value either way, but
by how much depends on the charge of the misfit dislocations: bare
ends (what the H recipe assumes, 2.645 electron charges per nanometer
of line here) take away nearly three times as much as neutral lines
(what the ZB recipe assumes). Ends whose broken bonds follow electron
counting sit close to neutral. `interface_charges` refuses a relaxed
interface unless you choose, because no reference decides it. Each
`InterfaceCharge` also carries the three terms of Eq. 2 of the misfit
paper (`coherent`, `relaxation_term`, `lines`) and the mismatch,
strains and line density behind them.

### 16. The electron gas of that sample, with an error bar

```python
from kpenvelope import (E_PER_NM2_PER_C_M2, StackLayer, algan, propagate,
                        solve_electrons)

T = 77.0
gan, bar = algan(0.0, T), algan(0.22, T)
calib = -0.004430561884400016 * E_PER_NM2_PER_C_M2   # e/nm^2, see the text

def two_deg(lam):
    stack = [StackLayer(gan, 80.0), StackLayer(bar, 3.0, relaxation=0.78),
             StackLayer(bar, 100.0, donors=1.26e-3)]          # 1.26e18 cm^-3
    r = solve_electrons(stack, phi_B=0.5, temperature_K=T, line_charge=lam,
                        extra_sheet={0: calib}, quantum_window=(40.0, 84.0),
                        classical_outside=True)
    return r.sheet_quantum_cm2

for lc in ("bare", "neutral"):
    print(f"{lc:8s} 2DEG = {two_deg(lc):.3e} cm^-2")
p = propagate(two_deg, {"lam": -0.84}, {"lam": 0.53})
print(f"Lambda = -0.84 +- 0.53 e/nm: 2DEG = {p.value:.3e} +- {p.sigma:.1e} cm^-2")
```

```
bare     2DEG = 2.228e+12 cm^-2
neutral  2DEG = 8.380e+12 cm^-2
Lambda = -0.84 +- 0.53 e/nm: 2DEG = 6.423e+12 +- 1.2e+12 cm^-2
```

`solve_electrons` solves the Schroedinger and Poisson equations
together for the whole stack: 80 nm of GaN, a 3 nm spacer and 100 nm
of doped barrier, the surface Fermi level 0.5 eV below the barrier
conduction band, and the 2DEG treated quantum mechanically within
40 nm below and 4 nm above the interface (outside it, the undepleted
barrier holds a classical electron gas, `sheet_classical`). The extra
sheet `calib` is the fixed calibration charge that the misfit analysis
fitted to the four coherent samples of the same series (it absorbs
errors in the constants and in the Al content); without it, use your
own. The two limits bracket the measured 0.645e13 cm^-2 at 77 K, and
`propagate` carries the measured line charge, -0.84 +- 0.53 e/nm, into
an error bar on the density (to first order; `method="monte-carlo"`
gives percentiles instead). Each solve takes about half a second.

### 17. The misfit-line charge from a measured series

```python
import numpy as np
from kpenvelope import StackLayer, algan, fit_interface_series

T = 77.0
gan, bar = algan(0.0, T), algan(0.22, T)
# Shen et al. (2000), Fig. 2(b), as digitized in the misfit analysis:
# doped-barrier thickness (nm), relaxation, 77 K Hall density (1e13 cm^-2)
t = [25, 35, 50, 65, 100]
R = [0.0, 0.0, 0.0, 0.0, 0.78]
n = np.array([1.302, 1.132, 1.214, 1.059, 0.645]) * 1e13 * 1e-14   # nm^-2

def build(row):
    thickness, relaxation = row
    return [StackLayer(gan, 80.0), StackLayer(bar, 3.0, relaxation=relaxation),
            StackLayer(bar, thickness, donors=1.26e-3)]

fit = fit_interface_series(build, np.column_stack([t, R]), n,
                           theta0=(-0.027, -0.8), phi_B=0.5, temperature_K=T,
                           quantum_window=(40.0, 84.0), classical_outside=True)
cal, lam = fit.theta
dcal, dlam = fit.sigma
print(f"calibration charge {cal * 1e14:.3e} +- {dcal * 1e14:.1e} cm^-2")
print(f"misfit-line charge {lam:.3f} +- {dlam:.3f} e/nm")
print(f"fraction of the bare-end charge {lam / -2.645:.2f} +- {dlam / 2.645:.2f}")
```

```
calibration charge -2.723e+12 +- 6.0e+11 cm^-2
misfit-line charge -0.845 +- 0.512 e/nm
fraction of the bare-end charge 0.32 +- 0.19
```

`fit_interface_series` fits the calibration charge and the line charge
together to the five samples whose relaxation is known (under a
minute: every prediction is a full solve). With no reading errors
given, the error scale comes from the scatter of the readings about
the fit. Give `sheet_sigma` and `settings_sigma` (for example 0.05 on
the relaxation read from an X-ray map) and the errors of the settings
enter the error bars too: here the line charge then comes out as
-0.845 +- 0.532 e/nm. Both fits agree with the labplan fits of the
misfit analysis to better than one part in a million (see
[How the results are checked](#how-the-results-are-checked)).

### 18. Bound charge or trapped electrons? A test without constants

```python
from kpenvelope import (LineCharge, algan, charge_change_per_relaxation,
                        line_charge_limits, trapped_from_sum_rule)

gan, aln, film = algan(0.0), algan(1.0), algan(0.3)
e_t = (gan.a - film.a) / film.a        # stretched on relaxed GaN
e_c = (aln.a - film.a) / film.a        # compressed on GaN held at AlN
q = 0.313 * abs(line_charge_limits(film, e_t)["bare"])   # e/nm, if all trapped
for name, lc in (("bound", LineCharge(bound_fraction=0.313)),
                 ("trapped", LineCharge(trapped=q))):
    yp = charge_change_per_relaxation(film, e_t, lc)
    ym = charge_change_per_relaxation(film, e_c, lc)
    print(f"{name:8s} y+ = {yp:+.4f}  y- = {ym:+.4f}  sum = {yp + ym:+.4f} C/m^2"
          f"  -> q = {round(trapped_from_sum_rule(yp, ym, film.a), 3) + 0.0:.3f} e/nm")
print(f"mismatches: {e_t:+.4f} (tension), {e_c:+.4f} (compression)")
```

```
bound    y+ = -1.1913  y- = +1.1913  sum = +0.0000 C/m^2  -> q = 0.000 e/nm
trapped  y+ = -1.1913  y- = +0.3552  sum = -0.8361 C/m^2  -> q = 0.826 e/nm
mismatches: +0.0073 (tension), -0.0170 (compression)
```

`charge_change_per_relaxation` gives y, the change of the interface
charge per unit of relieved areal strain, 2 r |eps_c|. Charge bound to
partly filled bonds flips sign between a stretched and a compressed
layer, like the piezoelectric term; electrons trapped on the line do
not. So for one alloy grown both ways the sum y+ + y- cancels every
polarization constant and leaves only the trapped electrons,
-2 q e / b (Eq. 3 of the misfit paper), and `trapped_from_sum_rule`
reads q back. The compressed layer here is Al0.3Ga0.7N on a GaN channel
thin enough to keep the lattice of AlN, as grown by Elwaradi et al.,
J. Appl. Phys. 133, 145705 (2023). Build such a stack with
`StackLayer(algan(1.0), ...)`, then `StackLayer(algan(0.0), 20.0)`,
then the barrier: the strain of each layer follows from the one below
(`layer_strains`).

### 19. InAlN and ScAlN on GaN, and how far the number sets agree (new in 0.18.0)

```python
from kpenvelope import StackLayer, gan, inaln, scaln, interface_charges

# InAlN matched to GaN, with the two sets that have InN
for pset in ("hse-dreyer2016", "vurgaftman2003"):
    layers = [StackLayer(gan(pset), 1000.0),
              StackLayer(inaln(0.17, polarization=pset), 10.0)]
    c = interface_charges(layers)[0]
    print(f"In0.17Al0.83N/GaN, {pset}: {c.sigma_cm2:.3e} per cm^2")

# ScAlN on the GaN of the same paper
layers = [StackLayer(gan("ambacher2021"), 1000.0), StackLayer(scaln(0.18), 10.0)]
for ref in ("H", "ZB"):
    c = interface_charges(layers, reference=ref)[0]
    print(f"Sc0.18Al0.82N/GaN, reference {ref}: {c.sigma_cm2:.3e} per cm^2")
```

```
In0.17Al0.83N/GaN, hse-dreyer2016: 1.335e+13 per cm^2
In0.17Al0.83N/GaN, vurgaftman2003: 2.490e+13 per cm^2
Sc0.18Al0.82N/GaN, reference H: 5.977e+13 per cm^2
Sc0.18Al0.82N/GaN, reference ZB: 6.025e+13 per cm^2
```

In0.17Al0.83N is close to lattice matched to GaN, so its charge is
almost all spontaneous polarization, and there the two sets that have
InN differ by nearly a factor of two. That is the honest size of the
uncertainty of the published numbers for this interface, and why
`parameter_set` is recorded with every layer. `gan(pset)` gives the GaN
of the same set. ScAlN comes from the composition fits of Ambacher et
al. (2021), with its own GaN (`gan("ambacher2021")`, which can only be
the unstrained bottom layer: the paper gives no GaN piezoelectric
constants). The H and ZB references differ by less than 1 percent here.
ScAlN has no published band offset, gap or mass in that paper, so for
electrons add yours with `with_numbers(scaln(0.18), "your source",
ec=..., eg=..., me=...)`.

### 20. The C-V curve of a transistor layer

```python
import numpy as np
from kpenvelope import Contact, StackLayer, algan, cv_curve

layers = [StackLayer(algan(0.0), 1000.0), StackLayer(algan(0.25), 25.0)]
V, C, Q, res = cv_curve(layers, np.arange(-7.0, 0.01, 1.0),
                        top=Contact("schottky", barrier=1.0),
                        bottom=Contact("zero-field"), dz=0.25)
for v, c, r in zip(V, C, res):
    print(f"{v:5.1f} V  C = {c * 1e9:6.1f} nF/cm^2   n_s = {r.sheet():.3e} per cm^2")
```

```
 -7.0 V  C =    0.0 nF/cm^2   n_s = 2.351e+05 per cm^2
 -6.0 V  C =  323.3 nF/cm^2   n_s = 1.030e+12 per cm^2
 -5.0 V  C =  341.7 nF/cm^2   n_s = 3.125e+12 per cm^2
 -4.0 V  C =  345.5 nF/cm^2   n_s = 5.272e+12 per cm^2
 -3.0 V  C =  347.4 nF/cm^2   n_s = 7.435e+12 per cm^2
 -2.0 V  C =  348.8 nF/cm^2   n_s = 9.607e+12 per cm^2
 -1.0 V  C =  351.3 nF/cm^2   n_s = 1.179e+13 per cm^2
  0.0 V  C =  362.7 nF/cm^2   n_s = 1.401e+13 per cm^2
```

`cv_curve` moves the metal on top while the electrons keep the Fermi
level of the side contacts (quasi-equilibrium, the usual model of a C-V
sweep; no transport numbers are needed). The capacitance sits just
below that of the 25 nm barrier and drops to zero where the electron
sheet empties, between -7 and -6 V for this layer and this 1 eV barrier.
Each sheet density equals that of `solve_electrons` with the matching
surface barrier (see the checks). The electrons here are classical.
`cv_curve_quantum(layers, voltages, barrier=1.0)` does the same sweep
with the electrons' quantum states (one `solve_electrons` run per
voltage, so about ten times slower); for this layer the two
capacitances differ by -0.6 to +1.7 percent between -5 and -1 V.

### 21. A GaN p-n junction (illustrative transport numbers)

```python
import math
from kpenvelope import Contact, StackLayer, Transport, algan, iv_curve

# Illustrative transport numbers, NOT data: replace them with yours.
tr = Transport(mobility_n=1000.0, mobility_p=10.0, hole_dos_mass=1.0,
               reference="illustrative values for this example",
               tau_n=1e-9, tau_p=1e-9)
g = algan(0.0)
layers = [StackLayer(g, 500.0, donors=1e-3),        # n-GaN, 1e18 cm^-3
          StackLayer(g, 500.0, acceptors=1e-3)]     # p-GaN, 1e18 cm^-3, fully ionized
V, J, res = iv_curve(layers, tr, [2.6, 2.7, 2.8], top=Contact("ohmic"),
                     dz=0.5)
for v, j in zip(V, J):
    print(f"{v:.1f} V  J = {-j:.3e} A/cm^2")
n = 0.1 / (8.617333262e-5 * 300 * math.log(J[1] / J[0]))
print(f"ideality from 2.6 to 2.7 V: {n:.3f}")
```

```
2.6 V  J = 1.373e-05 A/cm^2
2.7 V  J = 1.056e-04 A/cm^2
2.8 V  J = 8.966e-04 A/cm^2
ideality from 2.6 to 2.7 V: 1.896
```

`solve_drift_diffusion` and `iv_curve` solve for the potential and the
two quasi-Fermi levels together (Newton, Scharfetter-Gummel fluxes),
ramping the voltage from equilibrium. With SRH recombination in the
junction the ideality factor comes out near 2, as it should; without
recombination it is 1 (checked in the tests against the closed-form
diode current). The transport numbers above are placeholders to show
the call, not data: the current scales with them directly. Check the
grid: this current changes by 11 percent between `dz=2` and `dz=1` nm
and by 0.1 percent between 0.5 and 0.25 nm.

### 22. Interface phonons of GaN

```python
from kpenvelope import GAN_BARKER1973, single_interface_modes, slab_modes

g = GAN_BARKER1973
print(f"GaN LO phonons: {g.w_LO_perp:.2f} meV (E1), {g.w_LO_par:.2f} meV (A1)")
print(f"LST check: eps_0 perp {g.eps0_perp:.3f} (listed 9.5)")
print(f"GaN surface phonon (GaN/vacuum): {single_interface_modes(g, 1.0)[0]:.2f} meV")
r = slab_modes(g, 1.0, 3.0, [0.05, 0.5, 5.0])
for name in ("even", "odd"):
    print(name, ["%.2f" % w for w in r[name][0]], "meV at q = 0.05, 0.5, 5 /nm")
```

```
GaN LO phonons: 92.49 meV (E1), 92.24 meV (A1)
LST check: eps_0 perp 9.528 (listed 9.5)
GaN surface phonon (GaN/vacuum): 88.94 meV
even ['76.65', '87.64', '88.94'] meV at q = 0.05, 0.5, 5 /nm
odd ['91.93', '89.91', '88.94'] meV at q = 0.05, 0.5, 5 /nm
```

The phonon energies are those of Barker and Ilegems (1973) as listed by
the Ioffe database, checked against each other with the
Lyddane-Sachs-Teller relation. A thin GaN slab in vacuum has its even
mode near the TO phonon and its odd mode near the LO phonon; a thick
one has both at the surface-phonon energy, as in the dielectric
continuum model of Fuchs and Kliewer. Put your own cited numbers in a
`PolarPhonons` record for AlN, InN or an alloy.

### 23. How well is the misfit-line charge known, and what would pin it down?

```python
from kpenvelope import (algan, combine_line_charge, plan_holography,
                        plan_sum_rule)

post = combine_line_charge(["shen-density-pbesol", "shen-mobility"])
print(f"Lambda = {post.mean:.3f} +- {post.sd:.3f} e/nm "
      f"(68%: {post.interval68[0]:.3f} to {post.interval68[1]:.3f})")
print(f"the mobility limit removed {100 * post.removed:.1f} percent")

film, sub = algan(0.25), algan(0.0)
eps = (sub.a - film.a) / film.a
print(f"holography, one layer relaxed 50%: +- {plan_holography(film, eps, 0.5):.3f} e/nm")
```

```
Lambda = -0.652 +- 0.403 e/nm (68%: -1.074 to -0.234)
the mobility limit removed 20.8 percent
holography, one layer relaxed 50%: +- 0.326 e/nm
```

`combine_line_charge` multiplies the Gaussian estimates and cuts the
result where an upper limit excludes it, in closed form. The measured
mobility removes the part of the density fit that it rules out, which
moves the estimate and shrinks it from +- 0.53 to +- 0.40 e/nm. The
relative error stays about 60 percent: no calculation can do better
without new data. `plan_holography` and `plan_sum_rule` say how well a
planned measurement would do; `line_charge_error_from_density` does the
same for a planned Hall density.

### 24. Fit a measured C-V curve and Hall density

```python
import numpy as np
from kpenvelope import (Contact, StackLayer, algan, cv_curve, fit_measurements,
                        sheet_density)

def build(v):          # the sample, for given values of the unknowns
    return {"layers": [StackLayer(algan(0.0), 300.0),
                       StackLayer(algan(v["x"]), 25.0)]}

# synthetic "measurements" of a sample with x = 0.27 and a 1.1 eV gate metal:
# C-V with 2 nF/cm^2 of noise, and a Hall density (free surface at 0.8 eV)
true = build({"x": 0.27})["layers"]
V = np.linspace(-7.5, -0.5, 15)
_, C, _, _ = cv_curve(true, V, top=Contact("schottky", barrier=1.1),
                      bottom=Contact("zero-field"), dz=0.5)
C = C + np.random.default_rng(0).normal(0.0, 2e-9, V.size)
ns = sheet_density(true, 0.8)[0]

f = fit_measurements(build, {"x": 0.25, "barrier": 1.0}, cv=(V, C, 2e-9))
print(f"C-V alone: x = {f.values['x']:.3f} +- {f.errors['x']:.3f}, "
      f"correlation with the barrier {f.correlation[0, 1]:+.3f}")
f = fit_measurements(build, {"x": 0.25, "barrier": 1.0}, fixed={"surface": 0.8},
                     cv=(V, C, 2e-9), sheet=(ns, 0.02 * ns))
for k in f.names:
    print(f"with Hall: {k:8s} {f.values[k]:.4f} +- {f.errors[k]:.4f}")
print(f"chi2 {f.chi2:.1f} for {f.dof} degrees of freedom")
```

```
C-V alone: x = 0.283 +- 0.009, correlation with the barrier +1.000
with Hall: x        0.2736 +- 0.0036
with Hall: barrier  1.2141 +- 0.1123
chi2 11.8 for 14 degrees of freedom
```

`fit_measurements` adjusts the unknowns you name until the computed C-V
curve (and Hall density) match the measured ones, with first-order error
bars. C-V alone cannot separate the composition from the metal barrier
here: both move the switch-off voltage, so their correlation is +1.000
and the fit warns. The Hall density, which depends on the composition
but not on the gate metal, separates them (x = 0.2736 +- 0.0036 against
the true 0.27). Layer thicknesses can be fitted too (`lengths=`), as can
an extra fixed charge, the misfit-line charge and a parasitic
capacitance.

### 25. Find the layer that gives what you want

```python
from kpenvelope import StackLayer, algan, design_error_bars, find_designs

def build(v):
    return [StackLayer(algan(0.0), 300.0), StackLayer(algan(v["x"]), v["t"])]

designs = find_designs(build, {"x": (0.15, 0.40), "t": (10.0, 35.0)},
                       {"sheet_bare": (1.2e13, 0.05e13), "threshold": (-4.0, 0.2)},
                       barrier=1.0, surface=0.8, n_grid=5, top=3)
for d in designs:
    print(f"x = {d.values['x']:.3f}, t = {d.values['t']:.1f} nm: "
          f"n_s = {d.sheet_bare:.3e} cm^-2, V_off = {d.threshold:+.2f} V, "
          f"meets targets: {d.meets_targets}")
eb = design_error_bars(build, designs[0].values, {"x": 0.01, "t": 1.0},
                       barrier=1.0, surface=0.8,
                       quantities=("sheet_bare", "threshold"))
print(f"with x +- 0.01 and t +- 1 nm: n_s +- {eb['sheet_bare'].sigma:.2e} cm^-2, "
      f"V_off +- {eb['threshold'].sigma:.2f} V")
```

```
x = 0.223, t = 18.3 nm: n_s = 1.211e+13 cm^-2, V_off = -4.05 V, meets targets: True
x = 0.223, t = 17.3 nm: n_s = 1.198e+13 cm^-2, V_off = -3.80 V, meets targets: False
x = 0.223, t = 19.7 nm: n_s = 1.202e+13 cm^-2, V_off = -4.30 V, meets targets: False
with x +- 0.01 and t +- 1 nm: n_s +- 7.06e+11 cm^-2, V_off +- 0.34 V
```

`find_designs` maps the range on a grid, refines around the best points
and returns the closest designs, marked when they meet every target
within its tolerance. `design_error_bars` then shows what growth
tolerances do: here plus or minus 0.01 in the Al fraction and 1 nm in
thickness move the switch-off by about a third of a volt, more than the
0.2 V asked for. The switch-off voltage is defined as the gate voltage
where the electron density falls to 1 percent of its value at 0 V.

### 26. Current over and through a barrier

```python
from kpenvelope import (Contact, StackLayer, Transport, algan, gate_leakage,
                        richardson_constant, solve_drift_diffusion,
                        thermionic_emission_current)

print(f"A* of GaN (m = 0.20 m0): {richardson_constant(0.20):.2f} A cm^-2 K^-2")
# a metal on n-GaN (1e17 cm^-3), forward 0.3 V, thermionic emission at the metal
tr = Transport(1000.0, 30.0, 1.0, reference="illustrative values")
r = solve_drift_diffusion([StackLayer(algan(0.0), 500.0, donors=1e-4)], tr,
                          top=Contact("schottky", barrier=1.0, voltage=0.3,
                                      thermionic=True), dz=1.0)
print(f"forward 0.3 V: {-r.current_density:.3e} A/cm^2 "
      f"(thermionic-emission law {thermionic_emission_current(1.0, 0.3, 0.2):.3e})")
# gate leakage of a transistor layer: over and through the barrier
for t in (25.0, 10.0):
    layers = [StackLayer(algan(0.0), 300.0), StackLayer(algan(0.25), t)]
    V, J, _ = gate_leakage(layers, 1.0, [-1.0, -3.0])
    print(f"{t:.0f} nm barrier: {J[0]:.2e} A/cm^2 at -1 V, {J[1]:.2e} at -3 V")
```

```
A* of GaN (m = 0.20 m0): 24.03 A cm^-2 K^-2
forward 0.3 V: 3.609e-06 A/cm^2 (thermionic-emission law 3.764e-06)
25 nm barrier: -7.22e-08 A/cm^2 at -1 V, -8.27e-03 at -3 V
10 nm barrier: -1.71e-01 A/cm^2 at -1 V, -1.37e+02 at -3 V
```

With `thermionic=True` a metal contact passes electrons by thermionic
emission instead of holding them at fixed densities; the forward current
is then the thermionic-emission law reduced by the drift-diffusion
correction of Crowell and Sze, 1/(1 + v_R/v_D), here 4 percent (checked
in the tests). `gate_leakage` adds tunneling through the barrier with the
formula of Tsu and Esaki and WKB transmission, on the bands of the
quasi-equilibrium C-V model: a 10 nm barrier leaks millions of times more
than a 25 nm one at -1 V. The barrier is ideal; real gates often leak
more, through defects.

### 27. A transistor cross-section in two dimensions

```python
import numpy as np
from kpenvelope import Gate, StackLayer, algan, cv_cross_section, solve_cross_section

layers = [StackLayer(algan(0.0), 300.0), StackLayer(algan(0.25), 25.0)]
r = solve_cross_section(layers, Gate(400.0, 600.0, barrier=1.0, voltage=-3.0),
                        width=1000.0, surface_barrier=0.8)
print(f"grid {r.x.size} x {r.z.size}, converged: {r.converged}")
for xx in (100.0, 400.0, 500.0):
    print(f"x = {xx:.0f} nm: {np.interp(xx, r.x, r.sheet_x):.3e} electrons per cm^2")
for lg in (50.0, 200.0, 1000.0):
    V, C, Q, _ = cv_cross_section(layers, Gate(400.0, 400.0 + lg, 1.0), [-2.0],
                                  lg + 800.0, 0.8)
    print(f"gate {lg:.0f} nm: C = {C[0] * 1e9:.0f} nF per cm^2 of gate")
```

```
grid 79 x 279, converged: True
x = 100 nm: 1.444e+13 electrons per cm^2
x = 400 nm: 1.066e+13 electrons per cm^2
x = 500 nm: 7.412e+12 electrons per cm^2
gate 50 nm: C = 769 nF per cm^2 of gate
gate 200 nm: C = 454 nF per cm^2 of gate
gate 1000 nm: C = 370 nF per cm^2 of gate
```

`solve_cross_section` solves the stack with a 200 nm gate at -3 V and
the pinned free surface beside it, on a grid that is fine at the
surface, the interfaces and the gate edges. Beside the gate the electron
density is that of the bare surface; under the middle of the gate it is
that of the gate (each equal to the 1D result to 0.5 percent in the
tests); at the gate edge it is in between. A short gate holds more
charge per length than a long one because the field spreads at its
edges, and a 1 um gate comes within 6 percent of the 1D capacitance.

### 28. A transistor's current, with and without self-heating (new in 0.19.0)

```python
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.thermal import GAN_BULK, SIC_4H_ROOM, ThermalLayer, ThermalModel

layers = [StackLayer(algan(0.0), 1000.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
# room-temperature mobility and its power law in T: Alpert et al. (2020); a saturation
# speed of 1e7 cm/s (a round illustrative number)
mob = Mobility(1704.0, "Alpert et al., Rev. Sci. Instrum. 91 (2020)",
               saturation_velocity=1.0e7, temperature_exponent=2.35)
el = [Electrode("source", "ohmic", 0, 500, implant_depth=40, implant_donors=1e20),
      Electrode("gate", "schottky", 1500, 1750, barrier=1.0),
      Electrode("drain", "ohmic", 3000, 3500, implant_depth=40, implant_donors=1e20)]
dev = Device(layers, 3500.0, el, mob, Surface("pinned", barrier=1.0),
             dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=100)
heat = ThermalModel(GAN_BULK, substrate=(ThermalLayer("SiC", 100e3, SIC_4H_ROOM),),
                    lateral_extension=20e3)
V, cold, _ = dev.sweep("drain", [1, 2, 5, 10], bias={"gate": 0.0})
V, hot, S = dev.sweep("drain", [1, 2, 5, 10], bias={"gate": 0.0}, thermal=heat)
for v, a, b, s in zip(V, cold["drain"], hot["drain"], S):
    print(f"Vd = {v:4.1f} V: {a:.3f} A/mm without heating, {b:.3f} A/mm with "
          f"(hottest {s.temperature.max():.0f} K, {s.power:.2f} W/mm)")
s = S[-1]
print(f"currents add up to {sum(s.current.values()):.1e} A/mm; heat put in "
      f"{s.heat.sum():.6f} W/mm = power {s.power:.6f} W/mm")
```

```
Vd =  1.0 V: 0.740 A/mm without heating, 0.719 A/mm with (hottest 307 K, 0.72 W/mm)
Vd =  2.0 V: 0.969 A/mm without heating, 0.916 A/mm with (hottest 317 K, 1.83 W/mm)
Vd =  5.0 V: 1.046 A/mm without heating, 0.933 A/mm with (hottest 353 K, 4.66 W/mm)
Vd = 10.0 V: 1.068 A/mm without heating, 0.870 A/mm with (hottest 411 K, 8.70 W/mm)
currents add up to 2.2e-11 A/mm; heat put in 8.696261 W/mm = power 8.696261 W/mm
```

`Device` builds a 3.5 um long transistor: ohmic source and drain on
heavily doped regions, a 250 nm Schottky gate, and the pinned surface in
between. The current rises and then flattens as the electrons reach
their top speed. With `thermal=` the heat made by the current flows
down through 100 um of SiC to a heat sink at 300 K; the hottest point
reaches 411 K at 8.7 W/mm (about 13 K per W/mm), and the warmer, slower
electrons carry 18.5 percent less current. The last line is a check that
holds in every solve: the terminal currents add up to zero, and the
heat put into the lattice equals the electrical power. The 4H-SiC
conductivity is held constant (its source gives no temperature
dependence), which underestimates the heating somewhat.

### 29. Drain lag from iron in the buffer

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import (Device, Electrode, Mobility, Surface, Trap,
                               capture_coefficient)

# iron in the lower buffer: an acceptor 0.5 eV below the conduction band,
# capture cross section 1e-15 cm^2, with 1e15 cm^-3 shallow donors
# (Uren et al., IEEE Trans. Electron Devices 59, 3327 (2012))
buf = algan(0.0)
layers = [StackLayer(buf, 1000.0, name="buffer", donors=1e-6),
          StackLayer(algan(0.25), 20.0, name="barrier")]
cn = capture_coefficient(1e-15, buf.me)
fe = Trap("acceptor", 0.5, "ec", 1e17, cn, 0.0, "Uren et al. 2012",
          where="buffer", z_range=(0.0, 900.0))
el = [Electrode("source", "ohmic", 0, 500, implant_depth=40, implant_donors=1e20),
      Electrode("gate", "schottky", 1500, 1750, barrier=1.0),
      Electrode("drain", "ohmic", 3000, 3500, implant_depth=40, implant_donors=1e20)]
dev = Device(layers, 3500.0, el, Mobility(1704.0, "Alpert et al. 2020",
             saturation_velocity=1.0e7), Surface("pinned", barrier=1.0),
             traps=[fe], dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=100)
print(f"c_n = {cn:.2e} cm^3/s; a trapped electron waits {dev.emission_times()[0][0]:.2e} s")
stress = dev.solve({"drain": 15.0})
settled = dev.solve({"drain": 3.0}, guess=stress)
r = dev.transient(stress, {"drain": 3.0}, np.logspace(-9, 0, 10))
for t, i in zip(r.times, r.current["drain"]):
    print(f"t = {t:.0e} s: {i:.4f} A/mm")
print(f"settled value {settled.current['drain']:.4f} A/mm")
```

```
c_n = 2.41e-08 cm^3/s; a trapped electron waits 4.65e-03 s
t = 1e-09 s: 0.9560 A/mm
t = 1e-08 s: 0.9613 A/mm
t = 1e-07 s: 0.9613 A/mm
t = 1e-06 s: 0.9614 A/mm
t = 1e-05 s: 0.9615 A/mm
t = 1e-04 s: 0.9629 A/mm
t = 1e-03 s: 0.9738 A/mm
t = 1e-02 s: 0.9971 A/mm
t = 1e-01 s: 0.9973 A/mm
t = 1e+00 s: 0.9973 A/mm
settled value 0.9973 A/mm
```

The buffer holds iron, an acceptor 0.5 eV below the conduction band.
At 15 V on the drain, electrons pushed into the buffer fill some of
these traps. Right after the switch to 3 V the trapped charge is still
there and the current is 4 percent low; it recovers between 0.1 and 10
milliseconds, as the traps let their electrons go (a trapped electron
waits 4.65 ms on average at 300 K), and settles exactly at the steady
state of the new bias. This is the drain lag behind "current collapse".

### 30. A transistor in three dimensions

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface

layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1800.0, "illustrative", saturation_velocity=1.0e7)
grid = dict(dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40, fine_depth=40)


def electrodes(gate_y=None):
    g = dict(y0=gate_y[0], y1=gate_y[1]) if gate_y else {}
    return [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
            Electrode("gate", "schottky", 300, 400, barrier=1.0, **g),
            Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20)]


flat = Device(layers, 700.0, electrodes(), mob, Surface("pinned", barrier=1.0), **grid)
s2 = flat.solve({"drain": 2.0, "gate": -3.0})
x, z = flat.axes
y = np.linspace(0.0, 400.0, 9)
# 3D, gate across the whole width: the same as 2D
full = Device(layers, 700.0, electrodes(), mob, Surface("pinned", barrier=1.0),
              depth=400.0, grid=dict(x=x, y=y, z=z))
s3 = full.solve({"drain": 2.0, "gate": -3.0})
print(f"2D: {s2.current['drain']:.5f} A/mm; 3D, full gate: "
      f"{s3.current['drain'] / 4e-4:.5f} A/mm")
# 3D, gate over half the width: the other half is an open channel
half = Device(layers, 700.0, electrodes((0.0, 200.0)), mob,
              Surface("pinned", barrier=1.0), depth=400.0, grid=dict(x=x, y=y, z=z))
s4 = half.solve({"drain": 2.0, "gate": -3.0})
print(f"3D, gate over half the width: {s4.current['drain'] / 4e-4:.5f} A/mm "
      f"(grid {s4.n.shape})")
```

```
2D: 0.84196 A/mm; 3D, full gate: 0.84196 A/mm
3D, gate over half the width: 1.23343 A/mm (grid (39, 9, 33))
```

With `depth` the same device becomes three-dimensional. A gate across
the whole width gives the 2D current to five digits (the tests check
1e-9). A gate over only half of the width leaves the other half as an
open channel, so at -3 V on the gate much more current flows. 3D solves
take minutes rather than seconds on a laptop.

## Tutorials

Three scripts in `examples/` go from data to a final result with error
bars. They use only this package and NumPy. Each prints the output
below, which is also at the top of the script. The test suite runs
all three and checks their key numbers.

### Shubnikov-de Haas data: density, mass and quantum lifetime

`examples/tutorial_sdh_analysis.py` (about one second) makes synthetic
"measured" SdH data from known, illustrative values (m* = 0.5 m0,
tau_q = 0.15 ps; not a real sample) with a fixed random seed, and
recovers them:

1. the density from the oscillation frequency (`sheet_density_from_sdh`),
2. the cyclotron mass from how the amplitude falls with temperature at
   one field (`fit_sdh_mass`),
3. the quantum lifetime and the Dingle temperature from how the
   amplitude falls towards low field at one temperature
   (`fit_sdh_dingle`, a Dingle plot), with the error of the mass from
   step 2 carried along.

```
1. density from the SdH frequency
   F = 150.0 +/- 2.0 T, spin-degenerate subband (branches=2)
   n = 7.254e+12 +/- 9.7e+10 cm^-2
2. cyclotron mass from the temperature damping at 20 T
   m* = 0.500 +/- 0.002 m0   (true 0.5)
   chi-square per degree of freedom 1.49 (4 degrees of freedom)
3. quantum lifetime from the field damping at 2 K (Dingle plot)
   tau_q = 0.147 +/- 0.002 ps   (true 0.15)
     of which 0.002 ps from the data and 0.001 ps from the mass
   Dingle temperature T_D = 8.27 +/- 0.11 K
   A0 = 4.32 +/- 0.18 (4 for a homogeneous sample, in units of rho_0)
   chi-square per degree of freedom 0.12 (6 degrees of freedom)
```

To use it on your own sample, replace the three blocks marked DATA with
your measurements. The quantum lifetime comes from the two-dimensional
form of the oscillation amplitude,
`A = 4 rho_0 (X / sinh X) exp(-pi / (omega_c tau_q))` (Coleridge,
Stoner and Fletcher, Phys. Rev. B 39, 1120 (1989)), so `ln(A / R_T)` is
a straight line in `1/B` with slope `-pi m* / (e tau_q)`. In units of
`rho_0` the line should reach 4 at `1/B = 0` for a homogeneous sample.
The Dingle temperature is `T_D = hbar / (2 pi k_B tau_q)`: the
temperature that would damp the oscillation as much (Shoenberg,
Magnetic Oscillations in Metals, chapter 8).

### A computed hole gas: which frequencies and masses to expect

`examples/tutorial_sdh_prediction.py` (5 to 30 seconds) computes the
hole gas of example 2, fills its computed dispersion at the converged
potential, and predicts for each spin-resolved branch that holds holes
its SdH frequency and its cyclotron mass at the Fermi contour
(`cyclotron_mass`), the two numbers the first tutorial measures.

```
self-consistent: converged True after 20 iterations
Fermi level -458.75 meV, 4 branches hold holes, total 4.6000e+13 cm^-2
branch  holes (1e13 cm^-2)  F (T)            m_c (m0)
  0     2.0108 +/- 0.0003   831.6 +/- 0.1    1.882 +/- 0.000
  1     2.0028 +/- 0.0003   828.3 +/- 0.1    1.885 +/- 0.000
  2     0.3062 +/- 0.0005   126.7 +/- 0.2    0.333 +/- 0.001
  3     0.2802 +/- 0.0001   115.9 +/- 0.0    0.314 +/- 0.000
```

The error bars are numerical: the change between a momentum path of 17
and of 33 points. Both path methods are second order, so the error of
the 33-point value is about a third of that change. The limits of the
model itself (hard walls, the cited parameter set, no disorder) are
not in these numbers. Each branch gives its own frequency here because
the two members of each pair hold slightly different numbers of holes
in this lopsided well (example 8).

### An intersubband line: the heavy-hole mass along c, with an error budget

`examples/tutorial_isb_mass_calibration.py` (a few seconds) takes a
GaN well between AlN barriers (3 nm each, an illustrative offset of
0.3 eV) and a synthetic "measured" intersubband line, made with a
fixed random seed from a GaN whose heavy-hole mass along c is 2.0 m0
(the cited set has 1.876 m0), and

1. finds the line a spectrometer sees: of the transitions from the top
   level, only the one to the second heavy-hole level couples to light
   polarized along `z` (`level_transitions`),
2. lists what that line depends on (`parameter_sensitivities`): at
   `k = 0` the heavy holes feel only A1 + A3 and the two splittings of
   each material, and the offset,
3. calibrates A1 + A3 of GaN, which sets the heavy-hole mass along c
   (`m0 / |A1 + A3|`, from the heavy-hole diagonal of the Chuang-Chang
   matrix), by Newton steps with the exact derivative, and
4. gives the error bar from the line and, separately, from the
   errors of the other parameters (illustrative errors, not published
   ones), with the largest parts named.

```
1. levels of the cited model (k_t = 0), from the top level
   level  energy (meV)  character  E_top - E (meV)  |z|^2 (nm^2)
     0        -0.83        HH           0.00          0.0000
     1       -11.39        LH          10.56          0.0000
     2       -50.32        HH          49.49          0.3926
     3       -61.52        LH          60.69          0.0000
   the line a spectrometer sees (light polarized along c): level 0 -> 2
2. measured line: 377.9 +/- 2.4 cm^-1 = 46.85 +/- 0.30 meV
3. what the 0 -> 2 line depends on (dE/dparameter, cited model)
   GaN A1        -0.0719 eV     AlN A1        -0.0154 eV
   GaN A3        -0.0719 eV     AlN A3        -0.0154 eV
   GaN delta1    +0.0104 eV/eV  AlN delta1    -0.0104 eV/eV
   GaN delta2    +0.0104 eV/eV  AlN delta2    -0.0104 eV/eV
   offset scale  +0.0031 eV (a 10 % change of the offset moves the line 0.31 meV)
   every other parameter: 0 (at k_t = 0 the heavy holes feel only these)
4. calibrating A1 + A3 of GaN (Newton steps with the exact derivative)
   start  A1 + A3 = -0.53300   line 49.488 meV
   step 1 A1 + A3 = -0.49634   line 46.821 meV
   step 2 A1 + A3 = -0.49678   line 46.854 meV
   step 3 A1 + A3 = -0.49678   line 46.854 meV
   dE/d(A1 + A3) at the result: exact -0.073647 eV, finite difference -0.073647 eV
5. result, with illustrative errors: 5 % on A1..A6 of AlN, 5 meV on every splitting, 10 % on the offset
   A1 + A3 = -0.4968 +/- 0.0040 (line) +/- 0.0536 (other parameters)
   m_hh^c = 2.013 +/- 0.016 (line) +/- 0.217 (other parameters) m0
   largest parts of the second error bar (in A1 + A3): AlN A1 0.0393, AlN A3 0.0362, offset 0.0038
   the cited set has m0 / |A1 + A3| = 1.876 m0; the synthetic line was made with 2.0 m0
```

The line fixes A1 + A3 to about 1 %, but with these illustrative
errors the A parameters of the AlN barrier dominate the error bar.
That is the kind of budget `parameter_sensitivities` gives before a
measurement is planned. Replace the blocks marked DATA with your own
structure, line and errors.

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
  3 to 16 times faster (see [Limits](#limits) for times). If a strain
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
  [Corrections](#corrections-in-earlier-versions));
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

## How the results are checked

264 automated tests run on every push and pull request, on Python 3.9,
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
- The GaN zone-centre splittings equal their closed forms to 1e-12 eV
  and the published 5.20 and 21.80 meV to 0.01 meV.
- The GaN high-momentum masses reach m0/|A2+A4-A5| = 1.89 m0 and
  m0/|A2+A4+A5| = 0.180 m0 (to 0.03 and 0.003 m0).
- A hard-wall self-consistent GaN run at 0.46 nm^-2 keeps the charge
  to a relative 5e-3, puts the gas centre between 0.4 and 0.8 nm, and
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
[CHANGELOG.md](CHANGELOG.md)).

**0.13.0 fixed five silent errors.** Each has a test
that fails on 0.12.0; [CHANGELOG.md](CHANGELOG.md) has the numbers.

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
answers are now refused. [CHANGELOG.md](CHANGELOG.md) lists before and after numbers.

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
  [CHANGELOG.md](CHANGELOG.md).
- Not changed, but now described: the self-consistent filling can
  give the two states of a Kramers pair different masses and holes
  (example 2 and [Limits](#limits)).

The full history is in [CHANGELOG.md](CHANGELOG.md).

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
- With `filling="kgrid"` the states are labelled by their energy order
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
- Devices in 2D and 3D (`Device`): drift-diffusion with classical
  carriers (no quantum confinement of the channel, no hot-electron
  energy balance or ballistic transport), a heterojunction as a band
  step between two nodes (no thermionic emission or tunneling at
  internal interfaces), no impact ionization or breakdown, no
  small-signal AC, and tensor-product grids only (no unstructured
  meshes). Contact resistance is not included: an ohmic contact is an
  ideal contact on a heavily doped region. The heat equation is steady
  state; the lattice temperature changes the mobility only (the carrier
  statistics stay at the ambient temperature). The mobility's field is
  rounded near zero (0.1 percent of its saturation term at most) to
  keep it smooth. The quasi-Fermi driving force (an option) often fails
  to converge at high drain voltage; the default field driving force
  does not have that problem. Each Newton step is a sparse LU solve, so a 2D sweep
  takes seconds to a minute and a 3D one minutes. With an undoped
  buffer and no traps the current under a pinched-off gate flows
  through the buffer, as it would in such a device. Check the grid by
  refining it near the gate edges.
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

## Citing, support and license

If `kpenvelope` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22015269](https://doi.org/10.5281/zenodo.22015269),
which always resolves to the latest version; every release is
archived on Zenodo. [CITATION.cff](CITATION.cff) has the details. If
you use its method, please also cite the Physical Review B paper in
[Where it comes from](#where-it-comes-from).

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/kpenvelope/issues).
Usage questions are welcome alongside bug reports; a docstring that
left a unit or a sign convention unclear is treated as a
documentation bug, not user error. While the version is below 1.0
the API may still move between minor versions; such changes are
called out in the release notes.

Licensed under Apache-2.0.
