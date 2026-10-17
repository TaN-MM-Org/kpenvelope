"""Tutorial: which SdH frequencies and cyclotron masses should a GaN
hole gas show?

This is the model side of `tutorial_sdh_analysis.py`. It computes the
hole gas of README example 2 (GaN between hard walls, 4.6e13 cm^-2,
the density measured in the source study), fills the computed
dispersion at the converged potential, and predicts for every
spin-resolved branch that holds holes its SdH frequency (Onsager
relation) and its cyclotron mass at the Fermi contour, the two numbers
`sheet_density_from_sdh` and `fit_sdh_mass` give from a measurement.

The error bars here are numerical, not experimental: each prediction is
made on a momentum path with 17 points and with 33 points, and the
change is quoted as the uncertainty of the 33-point value (both path
methods are second order in the spacing, so the error of the 33-point
value itself is about a third of that change).
The model's own limits (hard walls, the cited parameter set, no
disorder) are not in that number; see the README.

Run:  python examples/tutorial_sdh_prediction.py  (5 to 30 seconds,
depending on the machine)

Output (kpenvelope 0.14.0):

    self-consistent: converged True after 20 iterations
    Fermi level -458.75 meV, 4 branches hold holes, total 4.6000e+13 cm^-2
    branch  holes (1e13 cm^-2)  F (T)            m_c (m0)
      0     2.0108 +/- 0.0003   831.6 +/- 0.1    1.882 +/- 0.000
      1     2.0028 +/- 0.0003   828.3 +/- 0.1    1.885 +/- 0.000
      2     0.3062 +/- 0.0005   126.7 +/- 0.2    0.333 +/- 0.001
      3     0.2802 +/- 0.0001   115.9 +/- 0.0    0.314 +/- 0.000
"""
import warnings

import numpy as np

from kpenvelope import (cyclotron_mass, fill_subbands_kgrid, gan_rinke2008,
                        sdh_frequency, sheet_density_from_cm2,
                        sheet_density_to_cm2, solve_self_consistent,
                        subband_energies)

p = gan_rinke2008()
z = np.linspace(0.0, 6.0, 97)
ps = sheet_density_from_cm2(4.6e13)
res = solve_self_consistent(p, z, ps, n_states=8, mixing=0.5, tol=2e-5)
print(f"self-consistent: converged {res.converged} after "
      f"{res.iterations} iterations")

# energies along kx at the converged potential, computed once and
# reused by both paths (the 17-point path is every second point)
KMAX, N_STATES = 1.6, 8
cache = {}


def energies_at(kx, ky):
    key = round(kx, 12)
    if key not in cache:
        cache[key] = subband_energies(p, z, kx=kx, ky=ky,
                                      potential=res.potential,
                                      n_states=N_STATES)
    return cache[key]


out = {}
for nk in (17, 33):
    with warnings.catch_warnings():
        # the last pairs hold no holes here; the check is on purpose
        warnings.simplefilter("error", RuntimeWarning)
        occ, ef = fill_subbands_kgrid(energies_at, ps, N_STATES, KMAX,
                                      nk=nk, ntheta=1)
    kts = np.linspace(0.0, KMAX, nk)
    E = np.array([energies_at(k, 0.0) for k in kts])
    filled = np.flatnonzero(occ > 1e-6 * ps)
    m = cyclotron_mass(kts, E[:, filled], occ[filled])["mass_ratio"]
    out[nk] = (ef, filled, occ[filled], sdh_frequency(occ[filled],
                                                      branches=1), m)

ef, filled, occ, freq, mass = out[33]
_, filled17, occ17, freq17, mass17 = out[17]
assert np.array_equal(filled, filled17)
print(f"Fermi level {ef * 1e3:.2f} meV, {filled.size} branches hold "
      f"holes, total {sheet_density_to_cm2(occ.sum()):.4e} cm^-2")
print("branch  holes (1e13 cm^-2)  F (T)            m_c (m0)")
for i, b in enumerate(filled):
    n13 = sheet_density_to_cm2(occ[i]) / 1e13
    dn13 = abs(sheet_density_to_cm2(occ[i] - occ17[i])) / 1e13
    print(f"  {b}     {n13:.4f} +/- {dn13:.4f}   "
          f"{freq[i]:.1f} +/- {abs(freq[i] - freq17[i]):.1f}    "
          f"{mass[i]:.3f} +/- {abs(mass[i] - mass17[i]):.3f}")
