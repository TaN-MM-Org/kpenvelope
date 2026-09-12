# kpenvelope

[![PyPI](https://img.shields.io/pypi/v/kpenvelope)](https://pypi.org/project/kpenvelope/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015269-blue)](https://doi.org/10.5281/zenodo.22015269) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

A solver for the quantum states of holes confined in thin
semiconductor layers (wurtzite crystals such as GaN and AlN). It
computes the energy levels and wavefunctions from the standard
six-band k.p model of Chuang and Chang, solved together with the
electrostatics of the charge itself, so the confining potential is
not guessed -- it emerges from the balance between the fixed
interface charge and the hole gas. Built for polarization-induced
two-dimensional hole gases, and usable for any layered wurtzite
structure you can supply cited parameters for.

## Install

```
pip install kpenvelope
```

For development: clone the repository and `pip install -e .[test]`.

## Quick start

```python
import numpy as np
from kpenvelope import (gan_rinke2008, sheet_density_from_cm2,
                        solve_self_consistent)

p = gan_rinke2008()                        # cited GaN parameter set
z = np.linspace(0.0, 6.0, 97)              # grid in nm, from the interface
ps = sheet_density_from_cm2(4.6e13)        # measured density, lab units in
res = solve_self_consistent(p, z, ps, temperature_K=300.0)
# res.energies, res.masses, res.density, res.occupations, res.converged
```

Conventions, stated once: energies in eV on the valence-electron
scale (holes occupy the highest eigenvalues), lengths in nm, sheet
densities in nm^-2. `sheet_density_from_cm2` / `sheet_density_to_cm2`
convert to and from the cm^-2 numbers a lab quotes, exactly.

## What it can do

**Energy levels and wavefunctions.** `solve_subbands` solves the
six-band problem on a 1D grid at any in-plane momentum, with hard
walls at the grid ends. `solve_heterostructure` lifts that
restriction: layered stacks with position-dependent material
parameters and band offsets (built with `layered_profile`), using the
symmetrized discretization that keeps the problem exactly Hermitian
for any profile. Strain enters through the Bir-Pikus terms
(`strain_blocks`), per layer, when you supply cited deformation
potentials.

**Self-consistency with the charge.** `solve_self_consistent`
(hard-wall) and `solve_self_consistent_hetero` (finite barriers)
iterate the quantum problem with Gauss's law until the potential and
the charge agree, at your measured sheet density and temperature
(`temperature_K`; the default 0 reproduces the historical cold
filling exactly). The result carries a `converged` flag, so an
iteration-starved run reports its failure instead of hiding it.
Filling uses either parabolic subbands (closed-form, including the
finite-temperature closed form) or the full computed dispersion on a
k-grid (`fill_subbands_kgrid`), which refuses a momentum window the
occupied states outgrow.

**What the states are made of.** `band_character` resolves each
state into heavy-hole, light-hole and split-off fractions, and
`character_vs_k` tracks how the composition changes with momentum --
the physics behind "one subband, many masses": the top subband is
pure heavy-hole at zero momentum, and its mass moves as other
components mix in.

**Numbers an experiment measures.** `subband_dispersion` and
`local_mass` give the energy-versus-momentum curves and the local
effective mass (a hole mass is not one number, and a flat band
honestly reports an infinite mass). `spin_splitting` /
`splitting_vs_k` quantify the splitting an asymmetric potential
induces. `group_velocity` and `dos_from_dispersion` are the
ingredients every transport estimate needs. `dipole_matrix` and
`oscillator_strengths` give the intersubband optical matrix elements
and dimensionless strengths that set absorption spectra and detector
design.

## Cited parameter sets

No physical number in this package is made up, and none is accepted
without a source -- the `reference` field of `WurtziteParameters` is
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
(2003)) and record the source. No default band offset is shipped
either: alignments are material- and strain-specific, so
`layered_profile` takes them from you, with a citation.

## How it is checked

Every physics claim in the test suite (54 tests, Python 3.9-3.13, run
in CI on every push) is anchored to a closed form, an exact identity,
or two independent code paths agreeing -- never to a stored number:

- the assembled matrix is exactly Hermitian with every coupling on,
  and the layered assembly reduces to the uniform one at machine
  precision when every layer is the same material;
- the decoupled well reproduces the textbook square-well levels (hard
  wall and finite barrier, including the analytic decay into the
  barrier);
- the cited GaN set lands on its closed-form splittings (5.20 and
  21.80 meV) and asymptotic masses (1.89 and 0.18 m0);
- the finite-temperature filling matches direct numerical integration
  of the Fermi-Dirac occupation (two independent code paths) and
  reduces to the cold filling as T goes to 0;
- filling conserves charge exactly, the k-grid filler agrees with the
  closed-form filler on an exactly parabolic model, and the strain
  terms are validated by a structural identity with the kinetic
  template;
- the intersubband dipoles reproduce the textbook closed forms, obey
  the f-sum rule, respect parity selection, and are invariant under a
  shift of the coordinate origin.

One comparison against the source paper is on record and stated
honestly: a hard-wall run at the measured density puts the gas
centroid at 0.62 nm against the paper's 0.568 nm, with the same
subband structure; the difference is the filling model (parabolic
here, computed-dispersion there). Do not publish self-consistent
numbers without checking the barrier and filling model against your
own system.

## Honest limits

Deliberate scope -- designed out with reasons, not overlooked:
scattering lifetimes (roughness, impurities, phonons) need cited
screening and roughness parameters per structure, and this package
ships no number it cannot source, so it provides the DOS and velocity
factors every lifetime integral needs instead; the Poisson solve uses
one cited permittivity (not a spatially varying profile); and
k-linear bulk-inversion-asymmetry terms beyond the six-band
Chuang-Chang model are not included.

## Associated paper

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas"
> (under review); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

and S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491 (1996). This
package is the general-purpose tool; the paper repository reproduces
the specific published study.

## Support and governance

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

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22015269](https://doi.org/10.5281/zenodo.22015269),
which always resolves to the latest version.
