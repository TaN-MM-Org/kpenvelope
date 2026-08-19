# kpenvelope

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015270-blue)](https://doi.org/10.5281/zenodo.22015270) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

A six-band **k.p envelope-function solver** for wurtzite heterostructures,
solved **self-consistently with Poisson's equation** on a 1D grid. Built
for polarization-induced two-dimensional hole gases (GaN/AlN and related
systems), where the confining potential is not imposed but emerges from
the balance between the fixed polarization charge and the gas itself.

## Status

v0.1.0 (alpha). Implemented and tested:

- six-band wurtzite valence Hamiltonian (standard Chuang-Chang form),
  discretized with symmetrized operator ordering so the matrix is exactly
  Hermitian for position-dependent parameters
- envelope eigen-solution at arbitrary in-plane k
- Gauss-law hole potential from the density profile
- self-consistent loop with T = 0 subband filling using numeric in-plane
  edge masses, converging to charge neutrality

Not yet implemented, stated plainly because they matter physically:

- **finite barriers.** v0.1 has hard walls one grid spacing outside the
  grid ends. A hard wall forces the envelope to vanish at the interface
  and pushes the gas outward, which shifts subband energies and masses;
  a finite AlN barrier treated as position-dependent material parameters
  is the v0.2 gate before research-grade use.
- strain terms, full k-grid (non-parabolic) filling, spin splitting
  analysis, transport lifetimes.

## What this package deliberately does not include

No validated material constants. The A parameters and splittings of GaN
and AlN differ between parameterizations and materially change computed
masses. Populate `WurtziteParameters` from the literature (e.g.
Vurgaftman and Meyer 2003; Rinke et al. 2008) and record the source in
the mandatory `reference` field. The shipped `demo_single_band()` set is
a decoupled, non-physical configuration used by the test-suite, chosen
because it has closed-form well solutions to test against.

## Install and use

```
pip install -e .
```

```python
import numpy as np
from kpenvelope import WurtziteParameters, solve_self_consistent

p = WurtziteParameters(
    A1=..., A2=..., A3=..., A4=..., A5=..., A6=...,
    delta1=..., delta2=..., delta3=...,
    eps_r=...,
    reference="cite the parameter set you use",
)
z = np.linspace(0.0, 15.0, 300)          # nm, from the interface
res = solve_self_consistent(p, z, ps=0.046)  # ps in nm^-2; 4.6e13 cm^-2 = 0.46
# res.energies, res.masses, res.density, res.potential, res.occupations
```

Convention: valence-electron energies, holes occupy the highest
eigenvalues; energies in eV, lengths in nm, sheet densities in nm^-2.

## Verification

The test-suite checks Hermiticity with every coupling switched on, the
decoupled square-well limit against the analytic spectrum, the uniform
slab against the analytic Gauss-law potential, and convergence plus exact
charge neutrality of the self-consistent loop. Validation against
published GaN/AlN subband structures is the acceptance gate for v0.2 and
has not yet been performed; do not use v0.1 numbers in publications.

## Methodological basis

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas"
> (under review); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

and S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491 (1996). This
package is the general-purpose tool; the paper repository reproduces the
specific published study, including the finite-barrier physics that v0.1
of this package does not yet have.

## License

Apache-2.0
