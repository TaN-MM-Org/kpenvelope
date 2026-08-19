# kpenvelope

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015270-blue)](https://doi.org/10.5281/zenodo.22015270) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

A six-band **k.p envelope-function solver** for wurtzite heterostructures,
solved **self-consistently with Poisson's equation** on a 1D grid. Built
for polarization-induced two-dimensional hole gases (GaN/AlN and related
systems), where the confining potential is not imposed but emerges from
the balance between the fixed polarization charge and the gas itself.

## Status

v0.2.0 (alpha). Implemented and tested:

- six-band wurtzite valence Hamiltonian (standard Chuang-Chang form),
  discretized with symmetrized operator ordering so the matrix is exactly
  Hermitian for position-dependent parameters
- envelope eigen-solution at arbitrary in-plane k
- Gauss-law hole potential from the density profile
- self-consistent loop with T = 0 subband filling using numeric in-plane
  edge masses, converging to charge neutrality
- cited GaN and AlN parameter sets with closed-form verification
  (new in v0.2, see below)

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
pip install -e .
```

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

## License

Apache-2.0
