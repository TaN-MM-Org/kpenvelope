# kpenvelope

[![PyPI](https://img.shields.io/pypi/v/kpenvelope)](https://pypi.org/project/kpenvelope/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015269-blue)](https://doi.org/10.5281/zenodo.22015269) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

A six-band **k.p envelope-function solver** for wurtzite heterostructures,
solved **self-consistently with Poisson's equation** on a 1D grid. Built
for polarization-induced two-dimensional hole gases (GaN/AlN and related
systems), where the confining potential is not imposed but emerges from
the balance between the fixed polarization charge and the gas itself.

## Status

v0.3.0 (alpha). Implemented and tested:

- six-band wurtzite valence Hamiltonian (standard Chuang-Chang form),
  discretized with symmetrized operator ordering so the matrix is exactly
  Hermitian for position-dependent parameters
- envelope eigen-solution at arbitrary in-plane k
- Gauss-law hole potential from the density profile
- self-consistent loop with T = 0 subband filling using numeric in-plane
  edge masses, converging to charge neutrality
- cited GaN and AlN parameter sets with closed-form verification
  (new in v0.2, see below)
- **dispersion and mass utilities (new in v0.3)**: subband dispersions
  along an in-plane path (`subband_dispersion`) and the local
  finite-difference effective mass of any dispersion (`local_mass`),
  because a hole mass is not one number. Asserted in the test suite: the
  decoupled demo set returns exactly 1/|A| m0 at every momentum; applied
  to the bulk Rinke 2008 GaN bands the utility reproduces the
  quasi-cubic asymptotic masses 1.89 and 0.180 m0; and a locally flat
  branch reports an infinite mass rather than an error, because a
  diverging mass is physics.

Not yet implemented, stated plainly because they matter physically:

- **finite barriers.** Hard walls one grid spacing outside the grid ends.
  A hard wall forces the envelope to vanish at the interface and pushes
  the gas outward, which shifts subband energies and masses; a finite AlN
  barrier treated as position-dependent material parameters is the v0.3
  gate before research-grade use.
- strain terms, full k-grid (non-parabolic) filling, spin splitting
  analysis, transport lifetimes.

## Cited parameter sets (new in v0.2)

- `gan_rinke2008()`: the consistent GW-based GaN valence set of Rinke et
  al., Phys. Rev. B 77, 075202 (2008) (A1..A6, Delta_CR = 10 meV,
  Delta_SO = 17 meV, delta2 = delta3 = Delta_SO/3), as tabulated in
  Extended Data Table 1 of Chang et al., Nature Electronics 9, 346 (2026)
  and used in the source paper below. eps_r = 10.4 (E parallel to c) from
  Barker and Ilegems, Phys. Rev. B 7, 743 (1973).
- `aln_rinke2008()`: the matching AlN set (Delta_CR = -295 meV from Rinke
  et al.; Delta_SO = 22 meV from de Carvalho et al., Appl. Phys. Lett.
  97, 232101 (2010)), intended as a barrier material. Its permittivity is
  deliberately NaN, and the self-consistent solver refuses to run on it,
  because no vetted value is shipped and none is needed for a barrier.

The test-suite locks every number and checks the GaN set against closed
forms: the zone-center splittings come out at 5.20 and 21.80 meV (against
accepted experimental values near 5-6 and 22 meV), and the asymptotic
in-plane masses at m0/|A2+A4-A5| = 1.89 m0 and m0/|A2+A4+A5| = 0.18 m0,
the quasi-cubic values quoted in the source paper's Supplemental
Material.

For any other material or parameterization, populate
`WurtziteParameters` from the literature (e.g. Vurgaftman and Meyer,
J. Appl. Phys. 94, 3675 (2003)) and record the source in the mandatory
`reference` field. The shipped `demo_single_band()` set is a decoupled,
non-physical configuration used by the test-suite, chosen because it has
closed-form well solutions to test against.

## Install and use

```
pip install kpenvelope
```

For development, clone the repository and `pip install -e .[test]`.

```python
import numpy as np
from kpenvelope import gan_rinke2008, solve_self_consistent

p = gan_rinke2008()                      # cited set; or your own WurtziteParameters
z = np.linspace(0.0, 6.0, 97)            # nm, from the interface
res = solve_self_consistent(p, z, ps=0.46)   # ps in nm^-2; 4.6e13 cm^-2 = 0.46
# res.energies, res.masses, res.density, res.potential, res.occupations
```

Convention: valence-electron energies, holes occupy the highest
eigenvalues; energies in eV, lengths in nm, sheet densities in nm^-2.

## Verification

The test-suite checks Hermiticity with every coupling switched on, the
decoupled square-well limit against the analytic spectrum, the uniform
slab against the analytic Gauss-law potential, convergence plus exact
charge neutrality of the self-consistent loop, and (new in v0.2) the
cited GaN set against the closed-form quasi-cubic splittings and masses
above.

One comparison against the source paper is on record and stated honestly:
a hard-wall self-consistent run at the measured sheet density
(4.6e13 cm^-2, 97 points over 6 nm) puts the gas centroid at 0.62 nm
against 0.568 nm for the hard-wall row of the paper's Table S1, with the
same subband structure (two heavy branches filled, the light branch a
minority). The difference comes from the filling model: this package
fills parabolic edge-mass subbands, while the paper fills the computed
non-parabolic dispersions, and the light branch is strongly
non-parabolic. Together with the hard-wall boundary this is why the same
warning stands: do not use these numbers in publications before the
finite-barrier, dispersion-filled v0.3.

## Methodological basis

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas"
> (under review); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

and S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491 (1996). This
package is the general-purpose tool; the paper repository reproduces the
specific published study, including the finite-barrier physics that this
package does not yet have.

## Support and governance

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the issue tracker at
https://github.com/TaN-MM-Org/kpenvelope/issues. Usage questions are
welcome there alongside bug reports; a docstring that left a unit or a
sign convention unclear is treated as a documentation bug, not as user
error. The maintainer aims to respond within a week.

While the version is below 1.0 the API may still move between minor
versions; such changes are called out in the release notes. The
limitations named under Status are deliberate scope, recorded there
precisely so that a user can tell a designed-out feature from an
oversight.

## License

Apache-2.0
