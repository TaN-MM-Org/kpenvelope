# kpenvelope examples

Worked examples, each with the output it printed, and tutorials that go
from measured data to a result with error bars. The [reference](reference.md)
describes every function used here.

## Examples

Each example below runs as written, and the output shown is what it
prints with kpenvelope 0.25.0 (all of them were rerun for this release).
Numbers labeled illustrative are chosen for the example, not taken from
a source.

Examples 1 to 14 cover hole levels, masses, light absorption and
magnetotransport; 15 to 18 nitride stacks, electron gases, error bars
and fits; 19 to 27 InAlN and ScAlN, C-V curves, a p-n diode, phonons,
fitting measured data, the design finder, gate leakage and a 2D
cross-section; 28 to 35 transistors (self-heating, drain lag from traps,
3D, speed, shapes, avalanche breakdown, pulses, gate tunneling, hot
electrons); 36 to 42 velocity overshoot, band steps, tunneling barriers,
band-to-band tunneling, breakdown by current and thermal runaway,
implants and a curved mesh; 43 to 47 leakage and runaway with
temperature, hot electrons with heating, a tunnel diode, oxidation, and
a process flow into a device; 48 to 50 Peltier heat, a tunnel diode that
heats itself, and tunneling on a turned mesh; 51 to 54 quantum electrons
in the device solver and a transistor, line-of-sight etching and
deposition, and LOCOS oxidation; 55 and 56 the ballistic limit of a
transistor and particles that bounce.

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
print(f"center of the hole gas: {centroid:.2f} nm from the interface")
```

```
converged: True after 20 iterations
enough states: True
subband edges (meV): [-402.37 -402.37 -413.58 -413.58 -548.47 -548.47 -559.02 -559.02]
edge masses (m0):    [0.445 0.442   inf 0.17  0.604 0.601   inf 0.206]
holes per subband (1e13 cm^-2): [1.622 1.61  0.    0.581 0.356 0.355 0.    0.076]
center of the hole gas: 0.76 nm from the interface
```

The 4.6e13 cm^-2 density is the measured value used by the source
study (see [Where it comes from](reference.md#where-it-comes-from)). This run uses
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
version 0.13.0; see [Corrections](checks.md#corrections-in-earlier-versions)).
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
problem; see [Limits](limits.md#limits).

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
print(f"center of the hole gas: {centroid:.2f} nm from the interface")
```

```
converged: True after 21 iterations
subband edges (meV): [-394.74 -394.74 -405.91 -405.91]
holes per state (1e13 cm^-2):  [1.974 1.964 0.352 0.311]
holes per Kramers pair (1e13 cm^-2): [3.938 0.662]
Fermi level: -450.10 meV
center of the hole gas: 0.63 nm from the interface
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
        center = (z * r.density).sum() / r.density.sum()
        print(f"  {100 * in_aln:.1f} % of the holes in the AlN, "
              f"center of the gas at {center:.2f} nm")
```

```
sheet_z = None: converged False after 60 iterations
sheet_z = 5.0: converged True after 30 iterations
  0.8 % of the holes in the AlN, center of the gas at 5.75 nm
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
[How the results are checked](checks.md#how-the-results-are-checked)).

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
currents add up to -9.1e-12 A/mm; heat put in 8.696261 W/mm = power 8.696261 W/mm
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
t = 1e-02 s: 0.9970 A/mm
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

### 31. Contact resistance and speed: f_T and f_max (new in 0.20.0)

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import (Device, Electrode, Mobility, Surface, rf_figures,
                               two_port, with_series)

layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1704.0, "Alpert et al. 2020", saturation_velocity=1.0e7)
el = [Electrode("source", "ohmic", 0, 300, implant_depth=40, implant_donors=1e20,
                resistance=0.3),
      Electrode("gate", "schottky", 800, 1000, barrier=1.0),
      Electrode("drain", "ohmic", 2000, 2300, implant_depth=40, implant_donors=1e20,
                resistance=0.3)]
dev = Device(layers, 2300.0, el, mob, Surface("pinned", barrier=1.0),
             dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=50)
st = dev.solve({"gate": -3.0, "drain": 10.0})
inner = st.internal
print(f"Id = {st.current['drain']:.4f} A/mm; at the contacts themselves the drain "
      f"sees {inner['drain']:.3f} V and the source {inner['source']:.3f} V")
f = np.logspace(8, 12.5, 37)
ports, Y = dev.small_signal(st, f)
Y2 = two_port(ports, with_series(ports, Y, {"source": 0.3, "drain": 0.3, "gate": 0.5}))
r = rf_figures(f, Y2)
gm = Y2[0, 1, 0].real
cg = Y2[0, 0, 0].imag / (2 * np.pi * f[0])
print(f"gm = {1e3 * gm:.1f} mS/mm, gate capacitance {1e12 * cg:.3f} pF/mm, "
      f"gm / (2 pi C) = {gm / (2 * np.pi * cg) / 1e9:.2f} GHz")
print(f"fT = {r['fT'] / 1e9:.2f} GHz ({r['fT_method']}), read at 100 MHz: "
      f"{r['fT_extrapolated'] / 1e9:.2f} GHz")
print(f"fmax = {r['fmax'] / 1e9:.2f} GHz ({r['fmax_method']}), read at 100 MHz: "
      f"{r['fmax_extrapolated'] / 1e9:.2f} GHz")
```

```
Id = 0.6207 A/mm; at the contacts themselves the drain sees 9.814 V and the source 0.186 V
gm = 265.5 mS/mm, gate capacitance 0.890 pF/mm, gm / (2 pi C) = 47.48 GHz
fT = 28.73 GHz (interpolated), read at 100 MHz: 47.48 GHz
fmax = 66.09 GHz (interpolated), read at 100 MHz: 439.00 GHz
```

`resistance` puts a resistor between each terminal and its contact
(here 0.3 ohm mm, a typical TLM value), and the solve finds the voltages
at the contacts themselves (`state.internal`): 0.186 V is lost at each
end. `small_signal` linearizes the device about that state and gives the
admittance of every terminal at each frequency; `with_series` adds the
contact resistors and a 0.5 ohm mm gate resistance exactly, `two_port`
picks the common-source pair, and `rf_figures` gives the current gain
h21 and Mason's unilateral gain U. Up to a few GHz the current gain
falls as 1/f, and f |h21| there equals gm / (2 pi C), the textbook
quasi-static figure (a test checks this to 1e-3). Above about 10 GHz
the gains fall faster, because the electrons need about 10 ps to cross
the 1 um between gate and drain at their top speed, so the real
crossings (28.7 and 66.1 GHz) come below the 20 dB-per-decade
extrapolation from 100 MHz. Both readings are returned; the second is
the one most measurements quote, so compare like with like.

### 32. Shapes: a recessed gate, a passivation film and a field plate

```python
import numpy as np
from kpenvelope import StackLayer, algan, threshold_voltage
from kpenvelope.device import Device, Dielectric, Electrode, Mobility, Region, Surface

layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1800.0, "illustrative", saturation_velocity=1.0e7)


def recessed(depth):
    top = 220.0
    el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
          Electrode("gate", "schottky", 200, 700, barrier=1.0, z0=top - depth, z1=top),
          Electrode("drain", "ohmic", 900, 1000, implant_depth=40, implant_donors=1e20)]
    return Device(layers, 1000.0, el, mob, Surface("pinned", barrier=1.0),
                  regions=[Region(200, 700, top - depth, top)],
                  dx_fine=10, dx_max=50, dz_fine=1.0, dz_max=40, fine_depth=40)


def off_voltage(dev):
    """gate voltage where the current at 0.1 V drain is 1 % of that at 0 V"""
    s = dev.solve({"drain": 0.1, "gate": 0.0})
    i0 = prev = s.current["drain"]
    vp = 0.0
    for v in np.arange(-0.25, -10.0, -0.25):
        s = dev.solve({"drain": 0.1, "gate": v}, guess=s)
        i = s.current["drain"]
        if i < 0.01 * i0:
            return vp + np.log(0.01 * i0 / prev) / np.log(i / prev) * (v - vp)
        prev, vp = i, v


for depth in (5.0, 10.0):
    thin = [layers[0], StackLayer(algan(0.25), 20.0 - depth, name="barrier")]
    print(f"{depth:.0f} nm recess: off at {off_voltage(recessed(depth)):.3f} V; "
          f"1D threshold of a {20 - depth:.0f} nm barrier {threshold_voltage(thin, 1.0):.3f} V")

# a passivated transistor with and without a gate-connected field plate
# (one metal in two pieces: the gate box and a strip on the nitride)
stack = [StackLayer(algan(0.05), 300.0, name="back barrier"),
         StackLayer(algan(0.0), 60.0, name="channel"),
         StackLayer(algan(0.25), 20.0, name="barrier")]
top = 380.0
sin = Dielectric("SiN", 100.0, 7.0, "Robertson, Eur. Phys. J. Appl. Phys. 28, 265 (2004)")
for plate in (False, True):
    box = dict(z0=top, z1=top + 100)
    el = [Electrode("source", "ohmic", 0, 300, implant_depth=40, implant_donors=1e20, **box),
          Electrode("gate", "schottky", 800, 1000, barrier=1.0, **box),
          Electrode("drain", "ohmic", 2200, 2500, implant_depth=40, implant_donors=1e20, **box)]
    if plate:
        el.append(Electrode("gate", "schottky", 800, 1500, barrier=1.0,
                            z0=top + 100, z1=top + 100))
    d = Device(stack, 2500.0, el, mob, Surface("states", fixed_charge=None), cap=[sin],
               dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=50)
    s = d.solve({"gate": -6.0, "drain": 0.0})
    _, _, S = d.sweep("drain", [5.0, 10.0, 20.0], bias={"gate": -6.0}, guess=s)
    Ex, _ = d.field(S[-1])
    x, z = d.axes
    row = np.abs(Ex[:, np.argmin(abs(z - 358.0))]) / 1e6
    at = lambda xx: row[np.argmin(abs(x - xx))]
    print(f"{'with' if plate else 'no  '} plate, 20 V: channel field {at(1030):.3f} MV/cm "
          f"by the gate, {at(1250):.3f} under the plate, {at(1510):.3f} at its end")
```

```
5 nm recess: off at -3.697 V; 1D threshold of a 15 nm barrier -3.697 V
10 nm recess: off at -2.293 V; 1D threshold of a 10 nm barrier -2.302 V
no   plate, 20 V: channel field 0.258 MV/cm by the gate, 0.185 under the plate, 0.170 at its end
with plate, 20 V: channel field 0.071 MV/cm by the gate, 0.064 under the plate, 0.234 at its end
```

A `Region` cuts a box out of the semiconductor (here the barrier under
the gate) and an electrode with `z0`, `z1` is a metal box, so the gate
can sit in a recess. With a long gate the current switches off where
the 1D solver puts the threshold of the thinned barrier (within 0.01 V).
`cap` lays dielectric films on top (`Dielectric`, with the source of its
permittivity), through which the contacts and gate reach down as boxes.
Several `Electrode` entries with one name are pieces of one metal: here
the gate box and a strip on the nitride make an L-shaped gate. Pinched
off at 20 V, the plate lowers the channel field next to the gate from
0.26 to 0.07 MV/cm and moves the drop to its own end, the reason field
plates raise the breakdown voltage.

### 33. Avalanche breakdown of a GaN p-i-n diode

```python
import numpy as np
from kpenvelope import StackLayer, gan
from kpenvelope.device import IMPACT_GAN, Device, Electrode, Mobility, Surface

# a GaN p-i-n diode: 300 nm undoped between heavily doped layers
g = gan(polarization="hse-dreyer2016")
layers = [StackLayer(g, 50.0, acceptors=1e-2, name="p"), StackLayer(g, 300.0, name="i"),
          StackLayer(g, 50.0, donors=1e-2, name="n")]
grid = dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0.0, 400.0, 401))
el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
dev = Device(layers, 20.0, el, Mobility(300.0, "illustrative"), Surface("pinned", barrier=1.0),
             hole_mass=1.0, hole_mobility=Mobility(20.0, "illustrative"),
             statistics="boltzmann", grid=grid)
vbr, V, I, S = dev.breakdown_voltage("top", np.arange(80.0, 121.0, 5.0),
                                     ionization=IMPACT_GAN)
for v, i in zip(V, I):
    print(f"{v:5.1f} V reverse: ionization integral {i:.3f}")
r = dev.ionization_integral(S[-1], IMPACT_GAN)
print(f"breakdown at {vbr:.1f} V; peak field there {r['peak_field'] / 1e6:.2f} MV/cm")
```

```
 80.0 V reverse: ionization integral 0.141
 85.0 V reverse: ionization integral 0.204
 90.0 V reverse: ionization integral 0.288
 95.0 V reverse: ionization integral 0.421
100.0 V reverse: ionization integral 0.664
105.0 V reverse: ionization integral 0.934
110.0 V reverse: ionization integral 1.494
breakdown at 105.6 V; peak field there 3.54 MV/cm
```

`IMPACT_GAN` holds the measured GaN coefficients (D. Ji et al., Appl.
Phys. Lett. 115, 073503 (2019)). `ionization_integral` follows the
field line through the strongest field and integrates McIntyre's
ionization integral along it; `breakdown_voltage` steps a voltage until
it reaches 1. The 300 nm undoped layer breaks down at 105.6 V, at a peak
field of 3.5 MV/cm. The tests also run the same diode with impact
ionization inside the drift-diffusion solve (`Device(...,
ionization=IMPACT_GAN)`), with light making carriers at one edge of
the undoped layer, and check that the current gain matches McIntyre's
1 / (1 - I) to 5 percent below breakdown.

### 34. Heating during a pulse

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.thermal import GAN_BULK, SIC_4H_ROOM, ThermalLayer, ThermalModel

layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1704.0, "Alpert et al. 2020", saturation_velocity=1.0e7,
               temperature_exponent=2.35)
el = [Electrode("source", "ohmic", 0, 300, implant_depth=40, implant_donors=1e20),
      Electrode("gate", "schottky", 800, 1000, barrier=1.0),
      Electrode("drain", "ohmic", 2000, 2300, implant_depth=40, implant_donors=1e20)]
dev = Device(layers, 2300.0, el, mob, Surface("pinned", barrier=1.0),
             dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=50)
heat = ThermalModel(GAN_BULK, substrate=(ThermalLayer("SiC", 100e3, SIC_4H_ROOM),),
                    lateral_extension=20e3)
times = np.logspace(-9, -3, 7)
t, I, Tmax, _ = dev.pulse({"gate": 0.0, "drain": 10.0}, times, heat)
for ti, i, T in zip(t, I["drain"], Tmax):
    print(f"t = {ti:.0e} s: {i:.4f} A/mm, hottest {T:.1f} K")
cold = dev.solve({"gate": 0.0, "drain": 10.0})
dc = dev.solve({"gate": 0.0, "drain": 10.0}, guess=cold, thermal=heat)
print(f"steady self-heated state: {dc.current['drain']:.4f} A/mm, hottest "
      f"{dc.temperature.max():.1f} K")
```

```
t = 1e-09 s: 1.2115 A/mm, hottest 325.8 K
t = 1e-08 s: 1.1910 A/mm, hottest 342.6 K
t = 1e-07 s: 1.1612 A/mm, hottest 356.1 K
t = 1e-06 s: 1.1315 A/mm, hottest 368.1 K
t = 1e-05 s: 1.0832 A/mm, hottest 387.7 K
t = 1e-04 s: 1.0173 A/mm, hottest 415.7 K
t = 1e-03 s: 1.0027 A/mm, hottest 422.1 K
steady self-heated state: 1.0024 A/mm, hottest 422.2 K
```

`pulse` switches the voltages on at t = 0 with the device at the heat
sink temperature and steps the heat equation in time, with heat
capacities from cited sources (GaN from Barin et al. via the Ioffe
database; for SiC the 6H value, as noted in `SIC_4H_ROOM`). The current
falls from 1.21 A/mm at 1 ns to 1.00 A/mm at 1 ms, where it has reached
the steady self-heated state (1.0024 A/mm and 422 K). This is why pulsed I-V curves with sub-microsecond pulses come
out higher than DC ones.

### 35. Tunneling through the gate barrier, and hot electrons

```python
import numpy as np
from kpenvelope import StackLayer, algan, gate_leakage
from kpenvelope.device import Device, Electrode, Mobility, Surface

layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
          StackLayer(algan(0.25), 10.0, name="barrier")]
el = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
      Electrode("gate", "schottky", 400, 2400, barrier=1.0),
      Electrode("drain", "ohmic", 2700, 2800, implant_depth=30, implant_donors=1e20)]
dev = Device(layers, 2800.0, el, Mobility(1800.0, "illustrative", saturation_velocity=1e7),
             Surface("pinned", barrier=1.0), dx_fine=10, dx_max=100, dz_fine=0.5,
             dz_max=40, fine_depth=40)
st = dev.solve({"gate": -3.0})
I, Jc = dev.gate_tunneling(st)
_, J1, _ = gate_leakage(layers, 1.0, [-3.0], dz=0.5)
print(f"gate current {I:.3e} A/mm; middle of the gate {Jc[len(Jc) // 2]:.4e} A/cm^2, "
      f"1D solver {J1[0]:.4e} A/cm^2")
on = dev.solve({"gate": 0.0, "drain": 10.0})
# energy relaxation time: 0.47 ps, measured at 250 K in AlGaN/AlN/GaN
# (E. Tiras et al., Superlattices Microstruct. 51, 733 (2012))
Te = dev.electron_temperature(on, 0.47e-12)
x, z = dev.axes
k = np.argmin(abs(z - 299.0))            # 1 nm below the barrier, in the electron layer
for xx in (300, 1400, 2380, 2500):
    i = np.argmin(abs(x - xx))
    print(f"x = {x[i]:6.0f} nm: electrons at {Te[i, k]:6.0f} K")
```

```
gate current -3.942e-03 A/mm; middle of the gate -1.3586e+02 A/cm^2, 1D solver -1.3566e+02 A/cm^2
x =    310 nm: electrons at    335 K
x =   1420 nm: electrons at    371 K
x =   2377 nm: electrons at    595 K
x =   2490 nm: electrons at    908 K
```

`gate_tunneling` takes the band diagram of the 2D solution column by
column under the gate and integrates the Tsu-Esaki current with WKB
transmission. In the middle of a 2 um gate it gives the 1D solver's
value to 0.15 percent; near the edges the band bends and the current
changes. `electron_temperature` solves the energy balance of the
electrons with the fields and currents of a solution, with an energy
relaxation time you supply (0.47 ps here, measured at 250 K by Tiras et
al. 2012). The electrons are a little warm under the gate and much
hotter in the high field between the gate and the drain. Here the
temperature is an output only; for the coupled model see example 36.

### 36. Hot electrons that change the current: velocity overshoot (new in 0.21.0)

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface

# an n+ / n / n+ GaN diode, 200 nm each (illustrative doping and mobility)
g = algan(0.0)
layers = [StackLayer(g, 200.0, donors=1e-2, name="n+ bottom"),
          StackLayer(g, 200.0, donors=1e-4, name="n"),
          StackLayer(g, 200.0, donors=1e-2, name="n+ top")]
mob = Mobility(1000.0, "illustrative", saturation_velocity=1.5e7)
el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
dev = Device(layers, 20.0, el, mob, Surface("pinned", barrier=1.0),
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 600, 601)),
             statistics="boltzmann")
dd = dev.solve({"bot": 2.0})
# energy relaxation time 0.47 ps: measured at 250 K in AlGaN/AlN/GaN by
# Tiras et al. (2012), used here as an example value
hot = dev.solve({"bot": 2.0}, energy_relaxation=0.47e-12)


def velocity(s):           # cm/s along the diode: J / (q n)
    return abs(s.current["bot"]) / (20e-7 * 0.1) / (1.602176634e-19 * s.n[1])


z = dev.axes[-1]
m = (z > 200) & (z < 400)
print(f"current: drift-diffusion {dd.current['bot']:.4f} A/mm, "
      f"hydrodynamic {hot.current['bot']:.4f} A/mm")
print(f"fastest electrons in the n layer: {velocity(dd)[m].max():.3e} cm/s and "
      f"{velocity(hot)[m].max():.3e} cm/s (saturation speed 1.5e7)")
print(f"hottest electrons {hot.electron_temperature.max():.0f} K")
```

```
current: drift-diffusion 0.0646 A/mm, hydrodynamic 0.0865 A/mm
fastest electrons in the n layer: 1.385e+07 cm/s and 3.968e+07 cm/s (saturation speed 1.5e7)
hottest electrons 4625 K
```

With `energy_relaxation=` the solve couples the electrons' energy balance
to the currents (the hydrodynamic or energy-transport model). The flux on
every edge is the exact solution for a temperature that changes linearly
along it (it reduces to Scharfetter-Gummel when the temperature is
constant), and the mobility is the drift-diffusion model's at the field
that would heat uniform electrons to the local temperature. So where the
field is uniform the model gives back the drift-diffusion velocity and
T_n = T_L + (2/3) tau mu E^2 / k exactly (a test checks both to 1e-5).
Where the field jumps, the electrons stay cool for a moment and run
faster than the saturation speed: 4.0e7 cm/s here against 1.4e7 cm/s in
drift-diffusion, and 34 percent more current. Constant-tau models like
this one are known to be able to overstate overshoot; compare with Monte
Carlo where it matters.

### 37. Electrons over a band step: thermionic emission

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.interfaces import interface_charges
from kpenvelope.device import Device, Electrode, Mobility, Surface

# n-GaN / n-Al0.15Ga0.85N, 1e18 cm^-3 each, polarization charge removed so
# only the band step matters (illustrative)
layers = [StackLayer(algan(0.0), 100.0, donors=1e-3, name="GaN"),
          StackLayer(algan(0.15), 100.0, donors=1e-3, name="AlGaN")]
no_pol = {c.index: -c.sigma_nm2 for c in interface_charges(layers)}
grid = dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 200, 5001))
el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
for model in ("drift-diffusion", "thermionic"):
    dev = Device(layers, 20.0, el, Mobility(300.0, "illustrative"), Surface("pinned", barrier=1.0),
                 extra_sheet=no_pol, grid=grid, heterojunctions=model, statistics="boltzmann")
    I = [dev.solve({"top": v}).current["top"] for v in (0.02, 0.1)]
    print(f"{model:16s}: {I[0]:.4e} A/mm at 0.02 V, {I[1]:.4e} A/mm at 0.1 V")
```

```
drift-diffusion : 1.7609e-03 A/mm at 0.02 V, 5.8464e-03 A/mm at 0.1 V
thermionic      : 1.7265e-04 A/mm at 0.02 V, 4.3164e-04 A/mm at 0.1 V
```

With `heterojunctions="thermionic"` the current across an abrupt band
step is the emission current over it, J = q v_R n_b [exp(dF/kT) - 1],
with the Richardson velocity and density of the barrier side (the form
of K. Yang, J. R. East and G. I. Haddad, Solid-State Electron. 36, 321
(1993)). Drift-diffusion treats the step as a very steep slope, which
here lets ten times too much current through. A test solves the same
model as a boundary-value problem with SciPy and agrees to 0.5 percent
once the grid error (first order, from the single node at the interface)
is extrapolated away.

### 38. Tunneling through a barrier inside the device

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.interfaces import interface_charges
from kpenvelope.device import Device, Electrode, Mobility, Surface, TunnelBarrier

# n+ GaN / Al0.3Ga0.7N barrier / n+ GaN, polarization charge removed
# (illustrative): the electrons cross the barrier by tunneling and over its
# top, computed with the Tsu-Esaki integral inside the Newton solve
for thick in (2.0, 4.0, 6.0):
    layers = [StackLayer(algan(0.0), 60.0, donors=1e-2, name="bottom"),
              StackLayer(algan(0.3), thick, name="barrier"),
              StackLayer(algan(0.0), 60.0, donors=1e-2, name="top")]
    z = np.unique(np.r_[np.linspace(0, 60, 121), np.linspace(60, 60 + thick, int(thick / 0.1) + 1),
                        np.linspace(60 + thick, 120 + thick, 121)])
    dev = Device(layers, 20.0,
                 [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")],
                 Mobility(300.0, "illustrative"), Surface("pinned", barrier=1.0),
                 extra_sheet={c.index: -c.sigma_nm2 for c in interface_charges(layers)},
                 grid=dict(x=np.array([0.0, 10.0, 20.0]), z=z),
                 tunnel_barriers=[TunnelBarrier("barrier", n_energy=800)])
    J = [dev.solve({"top": v}).current["top"] / (20e-7 * 0.1) for v in (0.05, 0.2)]
    print(f"{thick:.0f} nm barrier: {J[0]:.3e} A/cm^2 at 0.05 V, {J[1]:.3e} A/cm^2 at 0.2 V")
```

```
2 nm barrier: 1.075e+04 A/cm^2 at 0.05 V, 4.207e+04 A/cm^2 at 0.2 V
4 nm barrier: 2.486e+01 A/cm^2 at 0.05 V, 1.623e+02 A/cm^2 at 0.2 V
6 nm barrier: 1.291e+00 A/cm^2 at 0.05 V, 1.244e+01 A/cm^2 at 0.2 V
```

`TunnelBarrier("barrier")` takes the electrons across that layer with
the Tsu-Esaki current through the solution's own band profile (WKB
transmission, tunneling and emission over the top together), coupled
into the Newton solve with its derivatives. A layer at the top under a
gate gives the gate leakage as part of the solution; under a wide gate it
matches the 1D `gate_leakage` to 2 percent in the tests.

### 39. Band-to-band (Zener) tunneling in a reverse-biased junction

```python
import numpy as np
from kpenvelope import StackLayer, gan
from kpenvelope.device import BandToBand, Device, Electrode, Mobility, Surface

# a GaN p+/n+ junction, 5e19 cm^-3 on each side (illustrative)
g = gan(polarization="hse-dreyer2016")
layers = [StackLayer(g, 40.0, acceptors=5e-2, name="p+"), StackLayer(g, 40.0, donors=5e-2, name="n+")]
el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
dev = Device(layers, 20.0, el, Mobility(300.0, "illustrative"), Surface("pinned", barrier=1.0),
             hole_mass=1.0, hole_mobility=Mobility(20.0, "illustrative"), statistics="boltzmann",
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 80, 401)),
             band_to_band=BandToBand("Kane's formula as in arXiv:2108.09328, Eq. 13; "
                                     "reduced mass from m_e and the illustrative m_h = 1"))
for V in (5.0, 10.0, 15.0):
    s = dev.solve({"top": V})
    F = np.abs(np.gradient(s.psi[1], s.axes[-1])).max() * 1e7
    print(f"{V:4.1f} V reverse: {abs(s.current['top']):.3e} A/mm, peak field {F / 1e6:.2f} MV/cm")
```

```
 5.0 V reverse: 1.544e-06 A/mm, peak field 8.50 MV/cm
10.0 V reverse: 1.323e-04 A/mm, peak field 10.74 MV/cm
15.0 V reverse: 1.767e-03 A/mm, peak field 12.59 MV/cm
```

`BandToBand` adds Kane's two-band generation rate at the local field (as
written in Eq. 13 of arXiv:2108.09328), here at fields of 8.5 to 12.6
MV/cm. The terminal current equals q times the generation integrated over
the device (the tests check 1e-4). It is a generation term only: tunneling
recombination and forward-bias (Esaki) tunneling are not modeled.

### 40. Breakdown by current, and thermal runaway

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel

# 1) the off-state drain voltage where 1 mA/mm flows (punch-through here:
#    a short gate on an undoped 200 nm buffer)
layers = [StackLayer(algan(0.0), 200.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
el = [Electrode("source", "ohmic", 0, 100, implant_depth=40, implant_donors=1e20),
      Electrode("gate", "schottky", 300, 400, barrier=1.0),
      Electrode("drain", "ohmic", 600, 700, implant_depth=40, implant_donors=1e20)]
dev = Device(layers, 700.0, el, Mobility(1800.0, "illustrative", saturation_velocity=1e7),
             Surface("pinned", barrier=1.0), dx_fine=10, dx_max=50, dz_fine=2.0, dz_max=40,
             fine_depth=40)
r = dev.breakdown_by_current("drain", np.arange(1.0, 41.0, 1.0), bias={"gate": -8.0},
                             current=1e-3)
print(f"1 mA/mm at {r['voltage']:.2f} V; the current returns through: "
      + ", ".join(f"{k} {100 * v:.1f}%" for k, v in r["share"].items()))

# 2) thermal runaway of a bar whose mobility rises as T^2 (illustrative),
#    on a poor heat sink (1 W/(m K), 1 um)
bar = Device([StackLayer(algan(0.0), 100.0, donors=1e-4, name="bar")], 1000.0,
             [Electrode("a", "ohmic", 0, 50, z0=50.0, z1=100.0),
              Electrode("b", "ohmic", 950, 1000, z0=50.0, z1=100.0)],
             Mobility(300.0, "illustrative", temperature_exponent=-2.0),
             Surface("states", fixed_charge=0.0), dx_fine=25, dx_max=50, dz_fine=10, dz_max=10,
             fine_depth=100)
heat = ThermalModel({"bar": ThermalMaterial("bar", 5000.0, 0.0, "illustrative")},
                    substrate=(ThermalLayer("sink", 1000.0,
                                            ThermalMaterial("sink", 1.0, 0.0, "illustrative")),))
G0 = bar.solve({"b": 0.01}).current["b"] / 0.01
s1 = bar.solve({"b": 0.01}, thermal=heat)
Rth = float(np.sum(s1.heat * (s1.temperature - 300)) / np.sum(s1.heat)) / s1.power
Vc = np.sqrt(300.0 / (Rth * G0 * 4.0))     # T* = 600 K for mu ~ T^2
run = bar.thermal_runaway("b", np.linspace(0.5, 1.5, 21), heat)
print(f"lumped model: runaway at {Vc:.4f} V; device: steady up to {run['stable']:.4f} V, "
      f"none at {run['runaway']:.4f} V (hottest {run['hottest'][-1]:.0f} K)")
```

```
1 mA/mm at 1.25 V; the current returns through: source 100.0%, gate 0.0%
lumped model: runaway at 1.1938 V; device: steady up to 1.1930 V, none at 1.1938 V (hottest 579 K)
```

`breakdown_by_current` steps a voltage until a terminal current reaches
a value you choose (1 mA/mm is a common off-state criterion) and says
where the current goes; whatever the device includes counts. Here the
short gate on an undoped buffer never switches the channel off well, so
1 mA/mm flows already at 1.25 V, all of it through the source (under the
gate and through the buffer), none through the gate. `thermal_runaway`
sweeps with self-heating and bisects to the voltage where no
self-heated steady state is left. For a bar whose mobility grows as T^2
the lumped model T = T0 + R_th G0 (T/T0)^2 V^2 has its last solution at
1.1938 V; the device finds the edge between 1.1930 and 1.1938 V.

### 41. Implant, anneal, then the transistor

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.process import Anneal, Implant, Process

# Si-like donors implanted through a mask over the source and drain, then
# annealed (every number illustrative: take R_p, straggle and D from SRIM or
# SIMS and from a diffusion study of your material)
implants = [Implant("donor", 2e15, 30.0, 12.0, "illustrative", surface_z=220.0,
                    lateral_straggle_nm=10.0, window=w) for w in ((0.0, 120.0), (580.0, 700.0))]
anneal = Anneal(30.0, 1473.0, 1e-4, 3.0, "illustrative")
pr = Process(implants + [anneal], width=700.0, z_bottom=0.0, z_top=220.0, nx=351, nz=441)
print(f"D at 1200 C: {anneal.diffusivity_nm2_s:.3g} nm^2/s; dose under the source "
      f"{pr.sheet()[25]:.3e} cm^-2 (implanted 2e15)")
layers = [StackLayer(algan(0.0), 200.0, name="buffer"), StackLayer(algan(0.25), 20.0, name="barrier")]
el = [Electrode("source", "ohmic", 0, 100), Electrode("gate", "schottky", 300, 400, barrier=1.0),
      Electrode("drain", "ohmic", 600, 700)]
dev = Device(layers, 700.0, el, Mobility(1800.0, "illustrative", saturation_velocity=1e7),
             Surface("pinned", barrier=1.0), doping=pr.doping(), dx_fine=10, dx_max=50,
             dz_fine=1.0, dz_max=40, fine_depth=60)
V, I, _ = dev.sweep("drain", [0.5, 2.0, 5.0], bias={"gate": 0.0})
print("drain current:", ", ".join(f"{i:.3f} A/mm at {v:.1f} V" for v, i in zip(V, I["drain"])))
```

```
D at 1200 C: 0.544 nm^2/s; dose under the source 1.988e+15 cm^-2 (implanted 2e15)
drain current: 1.264 A/mm at 0.5 V, 1.661 A/mm at 2.0 V, 1.739 A/mm at 5.0 V
```

`Implant` gives the Gaussian depth profile of a range and straggle you
supply, with error-function edges at a mask window; `Anneal` diffuses it
with D = D0 exp(-Ea/kT), exactly in time, conserving the dose (0.6
percent of this implant's Gaussian lies above the surface, which is why
1.988e15 is left). `Process.doping()` hands the result to `Device(doping=)`.
The tests check the anneal against the exact broadened Gaussian and the
surface image solution to 1e-4, and the mask edge to 1e-7.

### 42. A curved device on a triangle mesh

```python
import warnings
import numpy as np
from kpenvelope import algan
from kpenvelope.device import Electrode, Mobility
from kpenvelope.mesh import MeshDevice, MeshRegion, mesh_from_points

# a quarter ring of n-GaN (radii 200 and 600 nm, 1e18 cm^-3, 500 cm^2/Vs,
# illustrative) with contacts on its two straight ends:
# R = rho (pi/2) / ln(b/a) per unit depth
a, b, mu, nd = 200.0, 600.0, 500.0, 1e18
R = 1 / (1.602176634e-19 * nd * mu) * (np.pi / 2) / (np.log(b / a) * 0.1)      # ohm mm
for nr, nt in ((9, 13), (17, 25), (33, 49)):
    r, t = np.meshgrid(np.linspace(a, b, nr), np.linspace(0, np.pi / 2, nt), indexing="ij")
    pts = np.column_stack([(r * np.cos(t)).ravel(), (r * np.sin(t)).ravel()])
    mesh = mesh_from_points(pts, lambda x, z: "GaN" if a - 1e-9 <= np.hypot(x, z) <= b + 1e-9
                            else None)
    P = mesh.points
    contacts = {"a": np.flatnonzero(np.isclose(P[:, 1], 0.0)),
                "b": np.flatnonzero(np.isclose(P[:, 0], 0.0))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # obtuse triangles on the arcs
        dev = MeshDevice(mesh, {"GaN": MeshRegion(material=algan(0.0),
                                                  mobility=Mobility(mu, "illustrative"),
                                                  donors=nd * 1e-21)},
                         [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)],
                         contacts, statistics="boltzmann")
    I = abs(dev.solve({"b": 0.01}).current["b"])
    print(f"{mesh.triangles.shape[0]:5d} triangles: {I:.6f} A/mm, exact {0.01 / R:.6f} A/mm, "
          f"error {I * R / 0.01 - 1:+.2e}")
```

```
  192 triangles: 0.056423 A/mm, exact 0.056028 A/mm, error +7.06e-03
  768 triangles: 0.056127 A/mm, exact 0.056028 A/mm, error +1.77e-03
 3072 triangles: 0.056053 A/mm, exact 0.056028 A/mm, error +4.42e-04
```

`MeshDevice` solves the same equations on an unstructured triangle mesh
with the box (Voronoi) method, so shapes can be curved or slanted. On this
quarter ring the error falls by four with every halving of the mesh size
(second order) toward the exact resistance. On a rectangular grid cut
into right triangles it reproduces the tensor-grid `Device` to rounding
(a test checks 1e-10).

### 43. Leakage that grows with temperature, and thermal runaway from it (new in 0.22.0)

```python
import dataclasses
import numpy as np
from kpenvelope import StackLayer, ingan
from kpenvelope.device import Device, Electrode, Mobility, Surface, Trap, Varshni
from kpenvelope.interfaces import interface_charges
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel

# an InN p-n diode (band numbers from the package's cited InN record; doping,
# traps, mobilities and the heat sink are illustrative)
inn = ingan(1.0)
p_side = StackLayer(inn, 500.0, acceptors=1e-3, name="p")
n_side = StackLayer(dataclasses.replace(inn, name="InN n"), 500.0, donors=1e-3, name="n")
layers = [p_side, n_side]
# InN gap versus temperature: Vurgaftman and Meyer (2003), the numbers the
# package's InN record uses; the even split between the bands is an assumption
gap_T = Varshni(0.245e-3, 624.0, 0.5, "I. Vurgaftman and J. R. Meyer, J. Appl. Phys. 94, 3675 (2003)")
dev = Device(layers, 20.0,
             [Electrode("top", "ohmic", 0, 20), Electrode("bottom", "ohmic", 0, 20, face="bottom")],
             Mobility(1000.0, "illustrative", temperature_exponent=1.5),
             Surface("pinned", barrier=1.0),
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 1000, 401)),
             hole_mass=0.5, hole_mobility=Mobility(50.0, "illustrative"),
             traps=[Trap("acceptor", 0.38, "ec", 1e17, 1e-8, 1e-8, "illustrative", ("p", "n"))],
             extra_sheet={c.index: -c.sigma_nm2 for c in interface_charges(layers)},
             temperature_model="full", band_gap_temperature=gap_T)

# leakage at a few lattice temperatures (set by hand)
for T in (300, 350, 400):
    st = dev.solve({"bottom": -2.0}, lattice_temperature=T)
    print(f"{T} K: leakage {abs(st.current['bottom']):.3e} A/mm")

# on a poor heat sink the leakage heats the diode, which leaks more ...
sink = ThermalModel(ThermalMaterial("InN (illustrative)", 45.0, 0.0, "illustrative"),
                    substrate=(ThermalLayer("substrate", 1e5,
                                            ThermalMaterial("poor sink", 1e-4, 0.0, "illustrative")),))
r = dev.thermal_runaway("bottom", -np.linspace(1.0, 10.0, 10), sink)
print(f"steady up to {abs(r['stable']):.3f} V, runaway by {abs(r['runaway']):.3f} V, "
      f"hottest {r['hottest'][-1]:.1f} K")
```

```
300 K: leakage 2.931e-11 A/mm
350 K: leakage 3.153e-10 A/mm
400 K: leakage 2.001e-09 A/mm
steady up to 3.395 V, runaway by 3.396 V, hottest 319.7 K
```

With `temperature_model="full"` the lattice temperature enters the
carrier statistics, not just the mobility: the band gap shrinks
(Varshni's form, with InN's cited numbers here), the densities of states
grow, and the traps let more carriers through. The diode's leakage grows
about seventy times from 300 to 400 K. On a poor heat sink that feeds on
itself: more leakage, more heat, more leakage. `thermal_runaway` sweeps
the voltage with self-heating and narrows down the point beyond which no
steady temperature exists, here near 3.4 V with the diode only about
20 K above the sink. A test checks this edge against the simple
heat-balance model built from the same leakage (within 0.3 percent).

### 44. Hot electrons and self-heating together

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.thermal import GAN_BULK, SIC_4H_ROOM, ThermalLayer, ThermalModel

# the n+ / n / n+ GaN diode of example 36, now on a substrate that warms up
g = algan(0.0)
layers = [StackLayer(g, 200.0, donors=1e-2, name="n+ bottom"),
          StackLayer(g, 200.0, donors=1e-4, name="n"),
          StackLayer(g, 200.0, donors=1e-2, name="n+ top")]
# mobility falls as T^-1.5 (illustrative exponent)
mob = Mobility(1000.0, "illustrative", saturation_velocity=1.5e7, temperature_exponent=1.5)
el = [Electrode("top", "ohmic", 0, 20), Electrode("bot", "ohmic", 0, 20, face="bottom")]
dev = Device(layers, 20.0, el, mob, Surface("pinned", barrier=1.0),
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 600, 601)),
             statistics="boltzmann")
# GaN (cited bulk numbers) on 2 um of SiC (cited) above the heat sink
sink = ThermalModel(GAN_BULK, substrate=(ThermalLayer("SiC", 2000.0, SIC_4H_ROOM),))
cold = dev.solve({"bot": 2.0}, energy_relaxation=0.47e-12)
warm = dev.solve({"bot": 2.0}, energy_relaxation=0.47e-12, thermal=sink)
print(f"current: {cold.current['bot']:.4f} A/mm at 300 K, {warm.current['bot']:.4f} A/mm "
      f"self-heated (lattice up to {warm.temperature.max():.1f} K)")
print(f"hottest electrons {warm.electron_temperature.max():.0f} K")
# where the lattice gets the heat: the electrons give it up where they cool
z = dev.axes[-1]
print(f"heat into the lattice peaks at z = {z[np.argmax(warm.lattice_heat[1, 1:-1]) + 1]:.0f} nm; "
      f"Joule heat (J.E) peaks at z = {z[np.argmax(warm.heat[1, 1:-1]) + 1]:.0f} nm")
```

```
current: 0.0865 A/mm at 300 K, 0.0811 A/mm self-heated (lattice up to 351.6 K)
hottest electrons 4562 K
heat into the lattice peaks at z = 193 nm; Joule heat (J.E) peaks at z = 242 nm
```

The electrons' energy (`energy_relaxation`) and the heat equation
(`thermal`) are now solved together. The warm lattice slows the
electrons and the current drops by about 6 percent. Where the heat
lands also changes: the field gives energy to the electrons inside the
lightly doped layer, but they hand it to the lattice only once they cool
down, a little further along their path (the electrons flow toward
lower z here). A test checks that where the field is uniform the two
heats agree, and that the totals match.

### 45. A tunnel (Esaki) diode

```python
import dataclasses
import numpy as np
from kpenvelope import StackLayer, ingan
from kpenvelope.device import BandToBand, Device, Electrode, Mobility, Surface
from kpenvelope.interfaces import interface_charges

# an InN p+/n+ tunnel (Esaki) diode: band numbers from the package's cited
# InN record; doping, hole mass and the reduced tunneling mass are illustrative
inn = ingan(1.0)
layers = [StackLayer(inn, 100.0, acceptors=3e-2, name="p+"),
          StackLayer(dataclasses.replace(inn, name="InN n"), 100.0, donors=1e-2, name="n+")]
tunnel = BandToBand("illustrative reduced mass", reduced_mass_m0=0.03, model="nonlocal")
dev = Device(layers, 20.0,
             [Electrode("top", "ohmic", 0, 20), Electrode("bottom", "ohmic", 0, 20, face="bottom")],
             Mobility(1000.0, "illustrative"), Surface("pinned", barrier=0.5),
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 200, 401)),
             hole_mass=0.5, hole_mobility=Mobility(50.0, "illustrative"),
             extra_sheet={c.index: -c.sigma_nm2 for c in interface_charges(layers)},
             band_to_band=tunnel)
prev = None
for V in (-0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    prev = dev.solve({"bottom": V}, guess=prev)
    print(f"{V:5.2f} V: {prev.current['bottom']:+.3e} A/mm")
```

```
-0.20 V: -4.638e-03 A/mm
 0.00 V: -5.336e-14 A/mm
 0.10 V: +1.419e-04 A/mm
 0.20 V: +1.991e-05 A/mm
 0.30 V: +2.619e-10 A/mm
 0.40 V: +4.097e-09 A/mm
 0.50 V: +1.869e-07 A/mm
 0.60 V: +8.918e-06 A/mm
 0.70 V: +4.241e-04 A/mm
```

`BandToBand(model="nonlocal")` lets electrons tunnel from the valence
band on one side of the junction to the conduction band on the other,
energy by energy, through the real band profile. The direction follows
the occupation of the two ends, so the same model makes the reverse
current, the forward tunneling peak near 0.1 V, the drop after it
(negative resistance: more voltage, less current), and then the normal
diode current takes over. At 0 V the current is
zero up to the convergence floor of the iteration. Every number but
InN's band numbers is illustrative, so read the shape, not the values.

### 46. Oxidizing silicon

```python
import math
from kpenvelope.process import Structure

# dry oxidation of silicon at 1000 C, with Deal and Grove's constants as
# tabulated in the EE 212 class handout #10 (1999) posted at
# dunham.ece.uw.edu/ee528/notes/Chapter6.pdf:
#   B = 7.72e2 um^2/h exp(-1.23 eV / kT),  B/A = 6.23e6 um/h exp(-2.0 eV / kT)
kT = 8.617333262e-5 * (1000 + 273.15)
B = 7.72e2 * math.exp(-1.23 / kT) * 1e6 / 3600          # nm^2/s
BA = 6.23e6 * math.exp(-2.0 / kT) * 1e3 / 3600          # nm/s
print(f"B = {B * 3600 / 1e6:.4f} um^2/h, B/A = {BA * 3600 / 1e3:.4f} um/h")

wafer = Structure(400.0, 0.0, 600.0, 1.0, "Si", 400.0)
wafer.deposit("Si3N4", 20.0, mode="directional", window=(0, 200))   # a nitride mask on the left half
x = wafer.oxidize("Si", "SiO2", B, BA, 2 * 3600.0, 0.44,
                  "Deal-Grove constants above; 0.44 of the oxide comes from consumed silicon")
print(f"after 2 h: {x[300]:.1f} nm of oxide in the open half, "
      f"{wafer.thickness('SiO2', 100):.0f} nm under the mask")
print(f"silicon left in the open half: {wafer.thickness('Si', 300):.0f} nm (it was 400)")
```

```
B = 0.0104 um^2/h, B/A = 0.0754 um/h
after 2 h: 91.0 nm of oxide in the open half, 0 nm under the mask
silicon left in the open half: 360 nm (it was 400)
```

`Structure` builds a cross-section from process steps on square cells.
`oxidize` uses the Deal-Grove law (B. E. Deal and A. S. Grove, J. Appl.
Phys. 36, 3770 (1965)) column by column; 44 percent of the oxide
thickness comes from consumed silicon, the figure several course notes
quote (for example the Radi dissertation at TU Wien, section 2.2). The
constants above give B and B/A close to the values tabulated at 1000 C
in the City University of Hong Kong notes on oxidation (0.0117 um^2/h
and 0.071 um/h). The nitride mask keeps the silicon under it from
oxidizing. The oxide thickness is exact; the cells round the shape to
1 nm. Thin dry oxides grow faster at first than this law says; if that
matters, start from an oxide layer of your own.

### 47. From process steps to a device

```python
from kpenvelope import algan
from kpenvelope.device import Dielectric, Electrode, Mobility
from kpenvelope.mesh import MeshDevice, MeshRegion
from kpenvelope.process import Anneal, Implant, Structure

# a GaN wafer: nitride film everywhere, a window opened in it, donors
# implanted through the window, an anneal, then a resistor between two
# contacts in the window (process numbers are illustrative)
s = Structure(1000.0, 0.0, 400.0, 5.0, "GaN", 300.0)
s.deposit("SiN", 50.0, mode="conformal")
s.etch({"SiN": 1.0}, 60.0, mode="directional", window=(100, 900))
s.implant(Implant("donor", 1e13, 30.0, 15.0, "illustrative range and straggle", 0.0))
s.anneal(Anneal(60.0, 1300.0, 1e-4, 3.0, "illustrative diffusivity"))
print(f"donors in the window: {s.donors.sum() * (s.h * 1e-7) ** 2 / 800e-7:.3e} cm^-2")

mesh = s.mesh()
regions = {"GaN": MeshRegion(material=algan(0.0), mobility=Mobility(300.0, "illustrative")),
           "SiN": MeshRegion(dielectric=Dielectric("SiN", 1.0, 7.0, "illustrative"))}
contacts = {"a": s.surface_nodes(mesh, "GaN", 100, 200),
            "b": s.surface_nodes(mesh, "GaN", 800, 900)}
dev = MeshDevice(mesh, regions, [Electrode("a", "ohmic", 0, 1), Electrode("b", "ohmic", 0, 1)],
                 contacts, doping=s.doping())
st = dev.solve({"b": 0.01})
print(f"{mesh.triangles.shape[0]} triangles; 10 mV drives {st.current['b'] * 1e3:.2f} mA/mm "
      f"({0.01 / st.current['b']:.2f} ohm mm)")
```

```
donors in the window: 1.000e+13 cm^-2
24800 triangles; 10 mV drives 6.95 mA/mm (1.44 ohm mm)
```

The window is opened in the nitride by a directional etch, the donors
go in through it (those that stop in the nitride are dropped), and the
anneal spreads them inside the GaN only. `mesh()` turns the cells into
triangles, `doping()` hands the doping to the device, and
`surface_nodes` picks the contacts on the exposed GaN. The whole implant
dose ends up in the window, and the device then solves like any other
`MeshDevice`.

### 48. Peltier heat at the contacts (new in 0.23.0)

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface
from kpenvelope.electrons import KB

# a 2 um bar of n-type GaN (1e18 cm^-3) between two contacts; doping and
# mobility are illustrative
def bar(heat_model):
    return Device([StackLayer(algan(0.0), 2000.0, donors=1e-3)], 20.0,
                  [Electrode("top", "ohmic", 0, 20), Electrode("bottom", "ohmic", 0, 20, face="bottom")],
                  Mobility(1000.0, "illustrative"), Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 2000, 201)),
                  statistics="boltzmann", heat_model=heat_model)

for model in ("joule", "thermoelectric"):
    dev = bar(model)
    st = dev.solve({"bottom": 0.5})
    H = st.heat.sum(axis=0)                          # W/mm along the bar, bottom first
    print(f"{model:14s} bottom end {H[:3].sum()*1e3:+6.3f} mW/mm, top end {H[-3:].sum()*1e3:+6.3f}, "
          f"total {st.heat.sum()*1e3:.3f} (power {st.power*1e3:.3f})")
Pi = KB * 300 * (2.5 + np.log(dev.Nc[0] * 1e21 / 1e18))    # eV, at 300 K
print(f"Pi = {Pi*1e3:.1f} meV; Pi I = {Pi * st.power / 0.5 * 1e3:.3f} mW/mm")
```

```
joule          bottom end +0.501 mW/mm, top end +0.501, total 40.054 (power 40.054)
thermoelectric bottom end +7.352 mW/mm, top end -6.351, total 40.054 (power 40.054)
Pi = 85.5 meV; Pi I = 6.852 mW/mm
```

Electrons carry heat with them as they flow. With
`heat_model="thermoelectric"` each one carries Pi = kT (5/2 + ln(N_c /
n)), here 85.5 meV. They enter at the top contact and leave at the
bottom one, so the top cools and the bottom warms, each by Pi times the
current (6.85 mW/mm): the ends now differ by 13.7 mW/mm, where the
Joule heat alone heats them alike. Inside the uniform bar nothing
changes, and the total is still the electrical power, because this heat
is only moved, not made. Along a temperature difference the same term
gives the Thomson heat. The value of Pi is the one for a non-degenerate
gas; in a dense electron layer it comes out too small (see the
[limits](limits.md)).

### 49. A tunnel diode that heats itself

```python
import dataclasses
import numpy as np
from kpenvelope import StackLayer, ingan
from kpenvelope.device import BandToBand, Device, Electrode, Mobility, Surface
from kpenvelope.interfaces import interface_charges
from kpenvelope.nitride import gap_shift
from kpenvelope.thermal import ThermalLayer, ThermalMaterial, ThermalModel

# the InN tunnel diode of example 45, now with the full temperature model:
# the gap follows the cited Varshni numbers of InN (its change shared evenly
# between the two bands, an assumption)
inn = ingan(1.0)
layers = [StackLayer(inn, 100.0, acceptors=3e-2, name="p+"),
          StackLayer(dataclasses.replace(inn, name="InN n"), 100.0, donors=1e-2, name="n+")]
dev = Device(layers, 20.0,
             [Electrode("top", "ohmic", 0, 20), Electrode("bottom", "ohmic", 0, 20, face="bottom")],
             Mobility(1000.0, "illustrative"), Surface("pinned", barrier=0.5),
             grid=dict(x=np.array([0.0, 10.0, 20.0]), z=np.linspace(0, 200, 401)),
             hole_mass=0.5, hole_mobility=Mobility(50.0, "illustrative"),
             extra_sheet={c.index: -c.sigma_nm2 for c in interface_charges(layers)},
             band_to_band=BandToBand("illustrative reduced mass", reduced_mass_m0=0.03, model="nonlocal"),
             temperature_model="full", band_gap_temperature=gap_shift("InN", conduction_share=0.5))
# a 2 um substrate under the diode, illustrative conductivity
sink = ThermalModel(ThermalMaterial("InN, illustrative", 45.0, 0.0, "illustrative"),
                    substrate=(ThermalLayer("substrate", 2000.0,
                                            ThermalMaterial("substrate", 5.0, 0.0, "illustrative")),))
for V in (-0.3, -0.2, 0.1, 0.3):
    I300 = dev.solve({"bottom": V}).current["bottom"]
    I400 = dev.solve({"bottom": V}, lattice_temperature=400.0).current["bottom"]
    hot = dev.solve({"bottom": V}, thermal=sink)
    print(f"{V:+.1f} V: {I300:+.3e} A/mm at 300 K, {I400:+.3e} at 400 K, "
          f"{hot.current['bottom']:+.3e} self-heated (hottest {hot.temperature.max():.0f} K)")
```

```
-0.3 V: -9.880e-03 A/mm at 300 K, -1.013e-02 at 400 K, -1.005e-02 self-heated (hottest 361 K)
-0.2 V: -4.638e-03 A/mm at 300 K, -4.620e-03 at 400 K, -4.643e-03 self-heated (hottest 319 K)
+0.1 V: +1.419e-04 A/mm at 300 K, +1.104e-04 at 400 K, +1.418e-04 self-heated (hottest 300 K)
+0.3 V: +2.619e-10 A/mm at 300 K, +4.799e-08 at 400 K, +2.619e-10 self-heated (hottest 300 K)
```

Band-to-band tunneling now runs with a lattice temperature you set
and with self-heating. At 400 K the tunneling peak at 0.1 V drops by about
a fifth, while the current in the valley at 0.3 V grows almost two
hundredfold, because more carriers get over the junction by heat; so the
ratio of peak to valley falls from about 540 000 to about 2 300. In
reverse bias the diode heats itself through the poor substrate, to
361 K at -0.3 V, and its current lies between the 300 K and the 400 K
values. `nitride.gap_shift` gives the gap's change with temperature
from the same cited numbers as the layer records. The material numbers
other than InN's band numbers are illustrative.

### 50. Tunneling on a triangle mesh, at an angle

```python
import dataclasses
import numpy as np
from kpenvelope import ingan
from kpenvelope.device import BandToBand, Electrode, Mobility
from kpenvelope.mesh import Mesh, MeshDevice, MeshRegion, mesh_from_grid

# the InN tunnel diode of example 45, turned by 30 degrees on a triangle mesh
inn = ingan(1.0)
flat = mesh_from_grid(np.array([0.0, 10.0, 20.0]), np.linspace(0, 200, 401),
                      lambda x, z: "p+" if z < 100 else "n+")
a = np.radians(30.0)
rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
mesh = Mesh(flat.points @ rot.T, flat.triangles, flat.regions)
up = rot @ [0.0, 1.0]                                   # across the junction
contacts = {"top": np.flatnonzero(np.isclose(flat.points[:, 1], 200.0)),
            "bottom": np.flatnonzero(np.isclose(flat.points[:, 1], 0.0))}
regions = {"p+": MeshRegion(material=inn, mobility=Mobility(1000.0, "illustrative"),
                            acceptors=3e-2, hole_mobility=Mobility(50.0, "illustrative")),
           "n+": MeshRegion(material=dataclasses.replace(inn, name="InN n"),
                            mobility=Mobility(1000.0, "illustrative"), donors=1e-2,
                            hole_mobility=Mobility(50.0, "illustrative"))}
tunnel = BandToBand("illustrative reduced mass", reduced_mass_m0=0.03, model="nonlocal",
                    direction=tuple(up), path_spacing=5.0)
dev = MeshDevice(mesh, regions, [Electrode("top", "ohmic", 0, 1), Electrode("bottom", "ohmic", 0, 1)],
                 contacts, hole_mass=0.5, band_to_band=tunnel)
prev = None
for V in (-0.2, 0.1, 0.2, 0.3):
    prev = dev.solve({"bottom": V}, guess=prev)
    print(f"{V:5.2f} V: {prev.current['bottom']:+.3e} A/mm")
```

```
-0.20 V: -4.638e-03 A/mm
 0.10 V: +1.419e-04 A/mm
 0.20 V: +1.991e-05 A/mm
 0.30 V: +2.619e-10 A/mm
```

The nonlocal band-to-band model now works on meshes. Its paths are
straight parallel lines in the `direction` you give, `path_spacing` nm
apart. Here the diode of example 45 is turned by 30 degrees and the
paths run across the junction, and the currents are those of example
45. Point the paths along the junction instead and they find no
tunneling. Pick the direction in which the field points.

### 51. Quantum electrons in the device solver (new in 0.24.0)

```python
import numpy as np
from kpenvelope import StackLayer, algan, solve_electrons
from kpenvelope.device import Device, Electrode, Mobility, Surface, QuantumChannel

# an Al0.25Ga0.75N/GaN stack (cited polarization and band numbers); the
# buffer doping and the mobility are illustrative
layers = [StackLayer(algan(0.0), 300.0, donors=1e-4), StackLayer(algan(0.25), 25.0)]
window = (280.0, 324.0)                 # nm, from inside the GaN to below the surface
one_d = solve_electrons(layers, 1.0, dz=0.1, quantum_window=window,
                        classical_outside=True, n_states=40)
for q in (None, QuantumChannel(window)):
    dev = Device(layers, 20.0, [Electrode("bottom", "ohmic", 0, 20, face="bottom")],
                 Mobility(1000.0, "illustrative"), Surface("pinned", barrier=1.0),
                 grid=dict(x=np.array([0.0, 10.0, 20.0]), z=one_d.z), quantum=q)
    st = dev.equilibrium()
    z, n = one_d.z, st.n[1]
    m = (z > 280) & (z < 300)
    depth = (n[m] * (300 - z[m])).sum() / n[m].sum()
    name = "quantum  " if q else "classical"
    print(f"{name}: electrons {depth:.2f} nm below the interface on average", end="")
    if q:
        E = st.quantum["subbands"][1][1]
        print(f", lowest levels {E[0]:+.4f} {E[1]:+.4f} eV")
    else:
        print()
m = (one_d.z > 280) & (one_d.z < 300)
depth = (one_d.n[m] * (300 - one_d.z[m])).sum() / one_d.n[m].sum()
print(f"1D Schroedinger-Poisson: {depth:.2f} nm, lowest levels "
      f"{one_d.subband_energies[0]:+.4f} {one_d.subband_energies[1]:+.4f} eV")
```

```
classical: electrons 1.95 nm below the interface on average
quantum  : electrons 2.15 nm below the interface on average, lowest levels -0.1385 -0.0037 eV
1D Schroedinger-Poisson: 2.15 nm, lowest levels -0.1385 -0.0037 eV
```

`QuantumChannel(window)` makes the device solver treat the electrons
inside the window as waves: along every vertical line of the grid it
finds their allowed energies (the levels are given relative to the
Fermi level), fills them, and feeds the result back into the current
and the charge until both agree. The electrons then sit a little
further from the interface than the classical model puts them, which
lowers the capacitance of the channel. Solved on the same grid, the
device solver and the package's separate 1D Schroedinger-Poisson
solver give the same levels and the same electron profile; the tests
check that their small differences, which come from how the material
step sits on the grid, halve when the grid step halves.

### 52. A transistor with a quantum channel

```python
import numpy as np
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface, QuantumChannel

# a short transistor: cited polarization and band numbers; mobility, speed
# and contacts illustrative
layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1500.0, "illustrative", saturation_velocity=1.0e7)
contacts = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
            Electrode("gate", "schottky", 300, 400, barrier=1.0),
            Electrode("drain", "ohmic", 600, 700, implant_depth=30, implant_donors=1e20)]
devs = {}
for name, q in (("classical", None), ("quantum", QuantumChannel((270.0, 319.5)))):
    devs[name] = Device(layers, 700.0, contacts, mob, Surface("pinned", barrier=1.0),
                        dx_fine=10, dx_max=50, dz_fine=0.5, dz_max=50, quantum=q)
prev = {}
for vg in (0.0, -1.0, -2.0, -3.0):
    I = {}
    for name, dev in devs.items():
        prev[name] = dev.solve({"gate": vg, "drain": 1.0}, guess=prev.get(name))
        I[name] = prev[name].current["drain"]
    print(f"gate {vg:+.0f} V: classical {I['classical']:.4f}, quantum {I['quantum']:.4f} A/mm "
          f"({100 * (I['quantum'] / I['classical'] - 1):+.1f} %)")
```

```
gate +0 V: classical 1.3889, quantum 1.3929 A/mm (+0.3 %)
gate -1 V: classical 1.2799, quantum 1.2846 A/mm (+0.4 %)
gate -2 V: classical 1.0542, quantum 1.0532 A/mm (-0.1 %)
gate -3 V: classical 0.7554, quantum 0.7466 A/mm (-1.2 %)
```

The quantum treatment runs with the current, the traps, self-heating
and hot electrons. By default it covers every column except those
under the ohmic contacts, whose implanted electrons need no
confinement. In this transistor the two models differ by less than
half a percent while the channel is open; near switch-off the energy of
the lowest level costs electrons, and the current falls by about one
percent at -3 V. Every number except the cited material numbers is
illustrative.

### 53. Etching and filling trenches with line-of-sight fluxes

```python
import math
from kpenvelope.process import Structure

# trenches etched by a gas at rest (exponent 1) and by a narrow ion beam
# (exponent 1000); the rate of 1 nm/s on an open surface is illustrative
for name, n in (("gas", 1.0), ("beam", 1000.0)):
    depths = []
    for w in (20.0, 100.0):
        s = Structure(400.0, 0.0, 300.0, 2.0, "Si", 200.0)
        s.deposit("mask", 10.0, mode="directional")
        s.etch({"mask": 1.0}, 10.0, mode="directional", window=(200 - w / 2, 200 + w / 2))
        s.etch({"Si": 1.0}, 60.0, mode="flux", exponent=n)
        depths.append(200 - s.top_surface()[100])
    print(f"{name}: 60 s etch, depth {depths[0]:.0f} nm in a 20 nm opening, "
          f"{depths[1]:.0f} nm in a 100 nm one")

# a film from a gas at rest on a 100 nm wide, 100 nm deep trench
s = Structure(400.0, 0.0, 300.0, 2.0, "Si", 200.0)
s.etch({"Si": 1.0}, 100.0, mode="directional", window=(150, 250))
s.deposit("oxide", 20.0, mode="flux")
print(f"film: {s.thickness('oxide', 20):.0f} nm on top, "
      f"{s.thickness('oxide', 200):.0f} nm at the trench bottom "
      f"(view factor {math.sin(math.atan(50 / 100)):.3f} of the top at first)")
```

```
gas: 60 s etch, depth 26 nm in a 20 nm opening, 50 nm in a 100 nm one
beam: 60 s etch, depth 60 nm in a 20 nm opening, 60 nm in a 100 nm one
film: 20 nm on top, 8 nm at the trench bottom (view factor 0.447 of the top at first)
```

With `mode="flux"` every exposed spot etches (or grows) in proportion
to the particles that reach it in a straight line from above. A gas at
rest arrives from all directions, so the bottom of a narrow opening
sees little of it and etches slowly (aspect-ratio dependent etching); a
narrow ion beam etches both openings at the full rate, straight down.
A film from a gas at rest is thinner at the bottom of a trench than on
top. The fluxes are the exact view factors of the cell geometry (the
tests compare them with the closed forms), and particles stick where
they land. The rates are illustrative.

### 54. LOCOS oxidation: the bird's beak

```python
import math
from kpenvelope.process import Structure

# LOCOS: a nitride mask on a thin pad oxide, then oxidation at 1000 C with
# the Deal-Grove constants of example 46
kT = 8.617333262e-5 * (1000 + 273.15)
B = 7.72e2 * math.exp(-1.23 / kT) * 1e6 / 3600          # nm^2/s
BA = 6.23e6 * math.exp(-2.0 / kT) * 1e3 / 3600          # nm/s
w = Structure(600.0, 0.0, 700.0, 2.0, "Si", 400.0)
w.deposit("SiO2", 10.0, mode="directional")                        # pad oxide
w.deposit("Si3N4", 40.0, mode="directional", window=(0, 300))      # mask on the left
w.oxidize("Si", "SiO2", B, BA, 4 * 3600.0, 0.44,
          "Deal-Grove constants of example 46; 0.44 of the oxide from consumed silicon",
          mode="2d")
for x in (50, 150, 200, 250, 300, 350, 500):
    print(f"x = {x:3d} nm: oxide {w.thickness('SiO2', x):5.0f} nm, "
          f"top of the stack at {w.top_surface()[int(x / 2)]:.0f} nm")
```

```
x =  50 nm: oxide    10 nm, top of the stack at 450 nm
x = 150 nm: oxide    14 nm, top of the stack at 452 nm
x = 200 nm: oxide    32 nm, top of the stack at 464 nm
x = 250 nm: oxide    70 nm, top of the stack at 480 nm
x = 300 nm: oxide    96 nm, top of the stack at 488 nm
x = 350 nm: oxide   144 nm, top of the stack at 484 nm
x = 500 nm: oxide   150 nm, top of the stack at 490 nm
```

With `mode="2d"` the oxygen diffuses through the oxide in two
dimensions. Under the edge of the nitride mask it reaches the silicon
through the pad oxide, so oxide grows there too, thinning out under the
mask (the bird's beak), and the swelling oxide lifts the mask's edge.
Far from the mask the thickness is Deal and Grove's, 150 nm against
149.9 nm from the closed form for a 10 nm start. The new oxide's extra
volume moves cells up or out; the stress and flow of real oxide are not
modeled.

### 55. The ballistic limit of a transistor (new in 0.25.0)

```python
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface, QuantumChannel

# the transistor of example 52 with quantum electrons: its current from
# drift-diffusion, and the ballistic limit through the same bands
layers = [StackLayer(algan(0.0), 300.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1500.0, "illustrative", saturation_velocity=1.0e7)
contacts = [Electrode("source", "ohmic", 0, 100, implant_depth=30, implant_donors=1e20),
            Electrode("gate", "schottky", 300, 400, barrier=1.0),
            Electrode("drain", "ohmic", 600, 700, implant_depth=30, implant_donors=1e20)]
dev = Device(layers, 700.0, contacts, mob, Surface("pinned", barrier=1.0),
             dx_fine=10, dx_max=50, dz_fine=0.5, dz_max=50,
             quantum=QuantumChannel((270.0, 319.5)))
st = None
for vg in (0.0, -2.0, -3.0):
    st = dev.solve({"gate": vg, "drain": 1.0}, guess=st)
    I_b, per = dev.ballistic_current(st)
    print(f"gate {vg:+.0f} V: drift-diffusion {st.current['drain']:.3f} A/mm, "
          f"ballistic limit {I_b:.3f} A/mm (ratio {st.current['drain'] / I_b:.2f}); "
          f"top of the lowest subband {per[0][0]:+.3f} eV")
```

```
gate +0 V: drift-diffusion 1.393 A/mm, ballistic limit 4.265 A/mm (ratio 0.33); top of the lowest subband -0.191 eV
gate -2 V: drift-diffusion 1.053 A/mm, ballistic limit 3.789 A/mm (ratio 0.28); top of the lowest subband -0.180 eV
gate -3 V: drift-diffusion 0.747 A/mm, ballistic limit 3.270 A/mm (ratio 0.23); top of the lowest subband -0.166 eV
```

`ballistic_current` asks how much current the channel would carry if no
electron scattered between the contacts: every electron that gets over
the top of the channel's energy barrier, in each subband, arrives at the
other side (the top-of-the-barrier model of Natori and Lundstrom). The
ratio of the drift-diffusion current to it says how far the transistor
is from that limit, here about a third or less. The bands are those of
the drift-diffusion solution; electrons below the top are not let
through by tunneling. The mobility, speed and contacts are illustrative.

### 56. Particles that bounce: sticking and conformal films

```python
from kpenvelope.process import Structure

# a 10 nm film on a 40 nm wide, 100 nm deep trench, from a gas whose
# particles stick at the first landing (sticking 1) or mostly bounce off
# and land elsewhere (sticking 0.1 and 0.01); rates illustrative
for s in (1.0, 0.1, 0.01):
    st = Structure(200.0, 0.0, 250.0, 2.0, "Si", 150.0)
    st.etch({"Si": 1.0}, 100.0, mode="directional", window=(80, 120))
    st.deposit("film", 10.0, mode="flux", sticking=s)
    wall = sum(1 for m in st.mat[:, 50] if m == "film") * st.h / 2       # each wall, at 101 nm
    print(f"sticking {s:4.2f}: {st.thickness('film', 20):.0f} nm on top, "
          f"{st.thickness('film', 100):.0f} nm at the bottom, {wall:.0f} nm on each wall")
```

```
sticking 1.00: 10 nm on top, 2 nm at the bottom, 2 nm on each wall
sticking 0.10: 10 nm on top, 6 nm at the bottom, 6 nm on each wall
sticking 0.01: 10 nm on top, 10 nm at the bottom, 8 nm on each wall
```

With `sticking` below 1, a particle that does not stick leaves again in a
random direction (the cosine law) and lands somewhere else. The more
often particles bounce before they stick, the more evenly they cover the
inside of a trench, which is why slow-reacting gases give conformal
films. The package works this out from the view factors between all
exposed faces (the tests compare it with an independent Monte Carlo
simulation of bouncing particles). The cells round each film to 2 nm.

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
