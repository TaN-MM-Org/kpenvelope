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
  temperatures in kelvin, Rashba coefficients in eV nm
  (1 meV A = 1e-4 eV nm).
- Energies are on the **valence-electron scale**: holes fill the
  **highest** eigenvalues first, and results are sorted from the
  highest energy down. A hole barrier is therefore a region with a
  **lower** band edge.
- Laboratories quote sheet densities in cm^-2.
  `sheet_density_from_cm2` multiplies by 1e-14 and
  `sheet_density_to_cm2` by 1e14 (so 4.6e13 cm^-2 = 0.46 nm^-2).
- In the hard-wall solvers the walls sit one grid step outside the
  first and last grid points, so a grid from 0 to `L` behaves like a
  well of width `L + 2 dz`.

## Examples

Each example below runs as written, and the output shown is what it
printed with kpenvelope 0.11.1. Numbers labelled illustrative are
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
res = solve_self_consistent(p, z, ps, n_states=4, mixing=0.5, tol=2e-5)

print("converged:", res.converged, "after", res.iterations, "iterations")
print("subband edges (meV):", np.round(res.energies * 1e3, 2))
print("edge masses (m0):   ", np.round(res.masses, 3))
print("holes per subband (1e13 cm^-2):",
      np.round(sheet_density_to_cm2(res.occupations) / 1e13, 3))
centroid = (z * res.density).sum() / res.density.sum()
print(f"centre of the hole gas: {centroid:.2f} nm from the interface")
```

```
converged: True after 22 iterations
subband edges (meV): [-394.38 -394.38 -405.54 -405.54]
edge masses (m0):    [0.453 0.449   inf 0.144]
holes per subband (1e13 cm^-2): [2.007 1.988 0.    0.605]
centre of the hole gas: 0.62 nm from the interface
```

The 4.6e13 cm^-2 density is the measured value used by the source
study (see [Where it comes from](#where-it-comes-from)). This run uses
hard walls and fills the subbands with parabolic masses measured at
the subband edge. The source paper reports a centre of 0.568 nm for its
hard-wall case, with the same subband structure; the difference comes
from the filling model (parabolic here, the computed dispersion
there). The test suite only checks that the centre lies between 0.4
and 0.8 nm.

Read the masses and the per-state holes with care. States 2 and 3 are
a Kramers pair, so by symmetry they should hold the same number of
holes; here one gets a mass of `inf` and no holes, the other 0.144 m0
and all of the pair's holes. That split is an artefact of how the
loop measures a mass: it takes one small step in momentum
(0.02 nm^-1) away from `k = 0`. In this lopsided well the two states
of a pair move apart in proportion to the momentum, so over that one
step one state rises (read as an infinite mass) and the other falls
too fast (read as too light a mass), and both numbers change if the
step changes. So do the totals per pair (here about 4.0e13 cm^-2 in
the top pair and 0.6e13 cm^-2 in the second), because the lighter
mass sets how many holes the pair takes. Averaging the step over each pair would give both states the same
mass, but with one mass per state it puts about 2.2e13 cm^-2 into the
second pair (at the same potential), further from the full-dispersion
answer, because the top subband gets much heavier away from `k = 0`
(example 3). Filling the full dispersion with `fill_subbands_kgrid` at
the converged potential gives about 0.7e13 cm^-2 in the second pair.
Use `fill_subbands_kgrid` for occupations you rely on, and do not
publish self-consistent numbers without checking the barrier and
filling model against your own system. Pass `temperature_K=` for a
finite-temperature filling; the default 0 is the zero-temperature
filling.

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
                        sheet_density_from_shift)

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

inv = sheet_density_from_shift(z, env1, env2, E1, E2,
                               e_meas_ev=out["E_shifted"], eps_r=10.0)
print(f"density recovered from the resonance: {inv['n_sheet']:.6f} nm^-2")
```

```
subband spacing   35.25 meV
shifted resonance 64.15 meV (alpha = 2.311)
density recovered from the resonance: 0.050000 nm^-2
```

The shifted line is `E_bare * sqrt(1 + alpha)`, with `alpha`
proportional to the sheet density. Because of that, the density follows
from a measured line in closed form: `sheet_density_from_shift` is the
exact inverse of `depolarization_shift`, not a fit. With solver states
(`solve_subbands`, `solve_heterostructure`) pass one state's `(6, N)`
envelope; note that the two members of a Kramers pair are not unique,
so pick states that are not degenerate with each other.

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

### 8. Refusals

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

- `solve_self_consistent(p, z, ps, ..., temperature_K)` -- hard walls;
  `solve_self_consistent_hetero(z, params_list, band_edge, ps, eps_r,
  ...)` -- finite barriers. Both return a `SelfConsistentResult` with
  `z`, `potential`, `energies`, `envelopes`, `density`, `occupations`,
  `masses`, `iterations`, `residual` and `converged` (False when the
  loop ran out of iterations).
- `fill_subbands_thermal(energies, masses, ps, temperature_K)` --
  Fermi-Dirac filling of parabolic subbands in closed form;
  `fill_subbands_kgrid(solve_at_k, ps, n_states, kmax, ...)` -- filling
  from the full computed dispersion on a momentum grid.
  `KB_EV_PER_K` is the Boltzmann constant in eV/K, computed from the
  exact SI values of k_B and e.
- `sheet_density_from_cm2`, `sheet_density_to_cm2` -- lab units.

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
- the grid is not uniform, the layer thicknesses do not add up to the
  grid span (within half a grid step), or the per-point parameter,
  band-edge or strain lists do not match the grid;
- a temperature is negative or not finite, or `max_iter` is below 1;
- the momentum-grid filler finds holes still present at the edge of
  its grid (at T = 0 an occupied state at `kmax`; at T > 0 an
  occupation above 1e-4 on the outer ring) -- increase `kmax`;
- momenta are negative, or `local_mass` gets fewer than two points,
  momenta that do not increase, or a row count that does not match;
- `dos_from_dispersion` gets a dispersion that does not fall
  monotonically (use `fill_subbands_kgrid`);
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
  are degenerate.

## How the results are checked

69 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.10 with the
oldest NumPy (1.22.0) and SciPy (1.8.0) the package allows. Most
numerical checks compare against an exact formula, a symmetry, or a
second calculation done a different way; a few compare against
published values (stated below). The main checks, with the tolerances
the tests use:

**Matrix and solver**

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
  with the parabolic filler to 4e-3 nm^-2, at T = 0 and at 150 K.
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

**0.11.1 (this release) fixed two problems and several documentation
errors.**

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
- The Poisson step uses one permittivity for the whole structure, not
  a profile that changes with position.
- The filling inside both self-consistent loops
  (`solve_self_consistent` and `solve_self_consistent_hetero`) treats
  every state as a parabola, with one mass taken from a single small
  momentum step (0.02 nm^-1) away from `k = 0`. In a lopsided
  (asymmetric) well the two states of a Kramers pair split apart in
  proportion to the momentum, and this one step then gives them
  different masses, sometimes `inf`, and so different numbers of
  holes, although by symmetry they should hold the same number. The
  total per pair depends on that step too, and a parabola ignores how
  the mass changes with momentum. For occupations you rely on, fill the full
  dispersion with `fill_subbands_kgrid` at the converged potential
  (example 2).
- The Rashba term is for the conduction-band companion problem. The
  valence-band k-linear terms beyond the six-band Chuang-Chang model
  are not included: no vetted coefficients in our sources.
- The depolarization shift leaves out the excitonic (final-state)
  correction, which needs an exchange-correlation model this package
  does not ship.

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
