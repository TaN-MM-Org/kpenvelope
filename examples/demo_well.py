"""Self-consistent demo on the synthetic single-band parameter set.

Physically meaningless parameters (see kpenvelope.params), but the full
pipeline runs: k.p solve -> filling -> Gauss law -> self-consistency.
"""
import numpy as np

from kpenvelope import demo_single_band, solve_self_consistent

p = demo_single_band()
z = np.linspace(0.0, 6.0, 120)
# every demo level is six-fold degenerate, so 6 states hold the whole
# first level (with 4, kpenvelope 0.13.0 warns that two are missing)
res = solve_self_consistent(p, z, ps=0.046, n_states=6)
print(f"converged in {res.iterations} iterations, residual {res.residual:.2e} eV")
print("subband edges (eV):", np.round(res.energies, 4))
print("edge masses (m0):", np.round(res.masses, 3))
print("occupations (nm^-2):", np.round(res.occupations, 4))
centroid = (res.z * res.density).sum() / res.density.sum()
print(f"gas centroid: {centroid:.2f} nm from the interface")
