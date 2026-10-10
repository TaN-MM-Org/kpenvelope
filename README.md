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
numbers for. It answers questions such as:

- Where are the hole energy levels in my layer, and what are they made
  of (heavy-hole, light-hole or split-off character)?
- For a measured sheet density and temperature, how are the holes
  shared among the levels, and where does the hole gas sit?
- What is the hole mass, and how does it change with momentum? (A hole
  mass here is not one number.)
- Where will the intersubband absorption line sit, including the
  depolarization shift that a doped layer adds?
- Which band offset, or which sheet density, reproduces my measured
  absorption line?
- What hole density and mass do my magnetotransport (Shubnikov-de Haas)
  measurements give, with error bars, and which frequencies should the
  computed hole gas show?

No material number is built in without a written source. When an input
cannot give a trustworthy answer, the package stops with an error that
says why, instead of returning a number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, units and conventions](#install-units-and-conventions)
- [Examples](#examples) (each with the output it prints)
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

## Install, units and conventions

```
pip install kpenvelope
```

It needs Python 3.9 or newer and NumPy 1.22 or newer, and nothing
else. The tests also use SciPy 1.8 or newer and pytest
(`pip install -e .[test]` from a clone of the repository).

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

## Examples

Each example below runs as written, and the output shown is what it
printed with kpenvelope 0.13.0. Numbers labeled illustrative are
chosen for the example, not taken from a source.

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
so pick states that are not degenerate with each other. Use states at
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
instead of 2, so expect it to take several times longer than example
2 (from under two minutes to a few minutes of CPU time in our runs,
depending on the machine and its load). No state is left
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
- `layered_profile(z, layers)` -- per-point materials and band edges
  from a list of `(thickness_nm, params, band_edge_eV)` layers.
- `solve_heterostructure`, `assemble_heterostructure` -- the same for
  layered stacks with finite barriers. The discretization keeps the
  matrix Hermitian for any profile, and each layer may carry its own
  strain.
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

**Intersubband optics**

- `dipole_matrix(z, envelopes)` -- `<f|z|i>` in nm;
  `oscillator_strengths(energies, dip, mass_ratio)` -- strengths from
  the top subband.
- `depolarization_shift`, `overlap_geometry_integral` -- the shifted
  line and its geometry integral `S`; `isb_lineshape` -- a
  strength-weighted Lorentzian line shape at your measured linewidth
  (shape only; the absolute absorbance needs your optical geometry).

**Calibration from your own measurement**

- `fit_band_offset`, `offset_sensitivity` -- example 6;
  `sheet_density_from_shift` -- example 5.
- `sheet_density_from_sdh`, `sdh_frequency` -- density from a
  Shubnikov-de Haas frequency and back (`H_OVER_E_T_NM2` is h/e in
  T nm^2); `fit_sdh_mass` -- the cyclotron mass from the temperature
  damping, with `lk_thermal_factor` the damping factor itself
  (`LK_TESLA_PER_KELVIN` = 14.69 T/K); example 11. The two constants
  live in `kpenvelope.lab`.

**Rashba term**

- `rashba_hamiltonian`, `rashba_splitting`, `rashba_spins` -- example 7.

Each function's docstring (`help(kpenvelope.solve_subbands)`, for
example) gives its inputs, units and conventions.

## Cited parameter sets

No physical number in this package is made up, and none is accepted
without a source: the `reference` field of `WurtziteParameters` is
mandatory.

- `gan_rinke2008()`: the GW-based GaN valence set of Rinke et al.,
  Phys. Rev. B 77, 075202 (2008) (A1..A6, Delta_CR = 10 meV,
  Delta_SO = 17 meV), as tabulated in Extended Data Table 1 of Chang
  et al., Nature Electronics 9, 346 (2026). eps_r = 10.4 (field along
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

## When it refuses, and why

`kpenvelope` raises a `ValueError` instead of guessing when:

- a parameter set has no permittivity (`eps_r` is NaN, as for AlN) and
  a self-consistent solve is asked for;
- a strained calculation is asked of a set without D1..D6, or the
  strain tensor is not a symmetric 3 x 3 tensor;
- the grid is not uniform, not increasing or shorter than three
  points, the layer thicknesses do not add up to the grid span (within
  half a grid step), or the per-point parameter, band-edge, strain,
  potential or permittivity arrays do not match the grid;
- `n_states` is below 1 or above six times the number of grid points;
- a layer thickness is negative or not finite, or no layer is given;
- a sheet density `ps` is negative or not finite, or `mixing` or `tol`
  is not a positive finite number;
- `fill_subbands_thermal` gets energies and masses of different
  shapes, a positive `ps` with no finite mass to put it in, or a `ps`
  that does not fit within 5 eV of the subband edges;
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
  shape (below the end of the momentum path it returns NaN: the path
  says nothing there);
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
  sigma on either side).

It does not stop, but warns (`RuntimeWarning`), when a self-consistent
loop or `fill_subbands_kgrid` finds that a state below the computed
`n_states` would hold holes (example 2).

## How the results are checked

105 automated tests run on every push and pull request, on Python 3.9,
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

## Corrections in earlier versions

**0.13.0 (this release) fixed five silent errors.** Each has a test
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
  momentum step (0.02 nm^-1) away from `k = 0`, which in a lopsided
  well gives the two states of a Kramers pair different masses
  (sometimes `inf`) and different numbers of holes (example 2). Use
  `filling="kgrid"` (example 8) for occupations you rely on; it costs
  `nk` extra solutions per iteration.
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
- Every solve builds and diagonalizes a dense matrix of size 6 N, so
  the time grows as N^3. On one core of a shared, loaded test machine
  one GaN solve took about 0.25 s at 97 points, 11 s at 301 points and
  100 s at 601 points (times vary with the machine); loops and
  momentum grids multiply that.
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

## Where it comes from

The package's methodological basis is

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas"
> (under review); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

and follows S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491
(1996). This package is the general-purpose tool; the paper repository
reproduces the specific published study.

## Citing, support and license

If `kpenvelope` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22015269](https://doi.org/10.5281/zenodo.22015269),
which always resolves to the latest version; every release is
archived on Zenodo. [CITATION.cff](CITATION.cff) has the details.

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
