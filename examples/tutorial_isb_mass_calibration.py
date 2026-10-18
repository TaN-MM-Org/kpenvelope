"""Tutorial: from a measured intersubband line to the heavy-hole mass
along the c axis, with an error budget (kpenvelope 0.15.0).

A GaN quantum well between AlN barriers absorbs infrared light when a
hole jumps from the top level to a lower one. Which line is seen, and
what does its position tell us? This script

1. lists the levels of the well and the strength of each transition
   from the top level (`level_transitions`, which sums over the two
   states of each Kramers pair, so the result does not depend on how
   the solver picked them),
2. takes a synthetic "measured" line (made below from known values
   with a fixed random seed, as a spectrometer would report it in
   cm^-1),
3. finds which material parameters the line depends on
   (`parameter_sensitivities`, exact derivatives),
4. calibrates the one combination it measures best, A1 + A3 of GaN
   (the heavy-hole mass along c is m0 / |A1 + A3|), by Newton steps
   with those exact derivatives, and
5. gives the error bar of the result: from the line itself, and from
   the other parameters, whose errors here are illustrative choices,
   not published values.

The structure, the offset and every error below are illustrative, not
a real sample. Replace the blocks marked DATA with your own. Runs in a
few seconds and prints:

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

The line measures A1 + A3 of GaN to about 1 %, but what that is worth
depends on the barrier: with these illustrative errors, the A's of
AlN dominate the error bar. `parameter_sensitivities` shows such a
budget before the measurement is made.
"""
import dataclasses

import numpy as np

from kpenvelope import (aln_rinke2008, band_character, dominant_character,
                        energy_from_wavenumber, energy_to_wavenumber,
                        gan_rinke2008, heterostructure_energies,
                        layered_profile, level_transitions,
                        parameter_sensitivities, solve_heterostructure)

GAN, ALN = gan_rinke2008(), aln_rinke2008()

# --- DATA: the structure (3 nm AlN | 3 nm GaN | 3 nm AlN) and the
# shape and size of the band offset, as in README example 6. The
# 0.3 eV offset is illustrative, not a measured GaN/AlN value.
Z = np.linspace(0.0, 9.0, 91)
OFFSET = 0.3


def structure(gan):
    """Per-point parameter sets and band edge, with this GaN set."""
    params, shape = layered_profile(Z, [(3.0, ALN, -1.0), (3.0, gan, 0.0),
                                        (3.0, ALN, -1.0)])
    return params, OFFSET * shape


def with_sum(gan, total):
    """The GaN set with A3 moved so that A1 + A3 = total (A1 kept)."""
    return dataclasses.replace(gan, A3=total - gan.A1)


# ---------------------------------------------------------------- 1
params, edge = structure(GAN)
E, F = solve_heterostructure(Z, params, band_edge=edge, n_states=8)
lev = level_transitions(Z, E, F)
char = band_character(F, Z)
print("1. levels of the cited model (k_t = 0), from the top level")
print("   level  energy (meV)  character  E_top - E (meV)  |z|^2 (nm^2)")
for L, states in enumerate(lev["levels"]):
    print(f"     {L}   {lev['level_energies'][L] * 1e3:10.2f}        "
          f"{dominant_character(char[states[0]])}       "
          f"{lev['transition_energy'][L] * 1e3:8.2f}        "
          f"{lev['z2'][L]:8.4f}")
bright = int(np.argmax(lev["z2"]))
i_top, i_up = int(lev["levels"][0][0]), int(lev["levels"][bright][0])
print("   the line a spectrometer sees (light polarized along c): "
      f"level 0 -> {bright}")

# ---------------------------------------------------------------- 2
# --- DATA: the measured line, as a spectrometer reports it. Here it is
# made from a synthetic sample whose GaN has A1 + A3 = -0.5 (a heavy-
# hole mass of 2.0 m0 along c), with a 0.3 meV measurement error.
true_gan = with_sum(GAN, -0.5)
e_true = heterostructure_energies(Z, *structure(true_gan),
                                  n_states=i_up + 1)
rng = np.random.default_rng(2026)
sigma_line = 0.0003                                     # eV
nu = energy_to_wavenumber(e_true[i_top] - e_true[i_up]
                          + sigma_line * rng.normal())
nu, sigma_nu = round(nu, 1), round(energy_to_wavenumber(sigma_line), 1)
e_meas = energy_from_wavenumber(nu)
sigma_meas = energy_from_wavenumber(sigma_nu)
print(f"2. measured line: {nu:.1f} +/- {sigma_nu:.1f} cm^-1 = "
      f"{e_meas * 1e3:.2f} +/- {sigma_meas * 1e3:.2f} meV")


# ---------------------------------------------------------------- 3
def line_and_derivatives(gan, **kw):
    """The 0 -> bright spacing and its derivatives (materials: AlN,
    GaN, in the order of the stack)."""
    p, b = structure(gan)
    s = parameter_sensitivities(Z, p, band_edge=b, n_states=i_up + 2,
                                **kw)
    e = s["energies"]
    d = s["derivatives"][i_top] - s["derivatives"][i_up]
    d_off = s["d_band_edge_scale"][i_top] - s["d_band_edge_scale"][i_up]
    return e[i_top] - e[i_up], d, d_off, s


line, d, d_off, s = line_and_derivatives(GAN)
names = s["names"]
gi = [j for j, mat in enumerate(s["materials"]) if mat is GAN][0]
ai = 1 - gi
print(f"3. what the 0 -> {bright} line depends on (dE/dparameter, "
      "cited model)")
for name in ("A1", "A3", "delta1", "delta2"):
    k = names.index(name)
    unit = "eV   " if name.startswith("A") else "eV/eV"
    print(f"   GaN {name:<9s} {d[gi, k]:+.4f} {unit}  "
          f"AlN {name:<9s} {d[ai, k]:+.4f} {unit}".rstrip())
print(f"   offset scale  {d_off:+.4f} eV (a 10 % change of the offset "
      f"moves the line {abs(d_off) * 0.1 * 1e3:.2f} meV)")
shown = {"A1", "A3", "delta1", "delta2"}
rest = max(abs(d[j, k]) for j in range(2) for k, n in enumerate(names)
           if n not in shown)        # checked, not assumed
assert rest < 1e-12
print("   every other parameter: 0 (at k_t = 0 the heavy holes feel "
      "only these)")

# ---------------------------------------------------------------- 4
print("4. calibrating A1 + A3 of GaN (Newton steps with the exact "
      "derivative)")
gan = GAN
x = GAN.A1 + GAN.A3
print(f"   start  A1 + A3 = {x:.5f}   line {line * 1e3:.3f} meV")
for step in range(1, 4):
    # A1 and A3 enter the heavy-hole states only as A1 + A3, so the
    # derivative with respect to the sum is the one with respect to A3
    x = x - (line - e_meas) / d[gi, names.index("A3")]
    gan = with_sum(GAN, x)
    line, d, d_off, s = line_and_derivatives(gan)
    print(f"   step {step} A1 + A3 = {x:.5f}   line {line * 1e3:.3f} meV")
dx = d[gi, names.index("A3")]
h = 1e-4
fd = (heterostructure_energies(Z, *structure(with_sum(GAN, x + h)),
                               n_states=i_up + 1)
      - heterostructure_energies(Z, *structure(with_sum(GAN, x - h)),
                                 n_states=i_up + 1)) / (2 * h)
print(f"   dE/d(A1 + A3) at the result: exact {dx:.6f} eV, "
      f"finite difference {fd[i_top] - fd[i_up]:.6f} eV")

# ---------------------------------------------------------------- 5
# --- DATA: the errors of the parameters that are not calibrated here.
# Illustrative choices for the example, not published uncertainties.
err_aln = {n: 0.05 * abs(getattr(ALN, n))
           for n in ("A1", "A2", "A3", "A4", "A5", "A6")}
err_aln.update(delta1=0.005, delta2=0.005, delta3=0.005)
err_gan = dict(delta1=0.005, delta2=0.005, delta3=0.005)
errors = [None, None]
errors[ai], errors[gi] = err_aln, err_gan
line, d, d_off, s = line_and_derivatives(
    gan, sigmas=errors, sigma_band_edge_scale=0.1)
cov = s["covariance"]
sigma_other = np.sqrt(cov[i_top, i_top] + cov[i_up, i_up]
                      - 2 * cov[i_top, i_up])
sx_line = sigma_meas / abs(dx)
sx_other = sigma_other / abs(dx)
m = 1.0 / abs(x)
print("5. result, with illustrative errors: 5 % on A1..A6 of AlN, "
      "5 meV on every splitting, 10 % on the offset")
print(f"   A1 + A3 = {x:.4f} +/- {sx_line:.4f} (line) +/- "
      f"{sx_other:.4f} (other parameters)")
print(f"   m_hh^c = {m:.3f} +/- {m * m * sx_line:.3f} (line) +/- "
      f"{m * m * sx_other:.3f} (other parameters) m0")
parts = [(abs(d[j, k]) * err.get(n, 0.0) / abs(dx), f"{mat} {n}")
         for j, (mat, err) in ((ai, ("AlN", err_aln)), (gi, ("GaN", err_gan)))
         for k, n in enumerate(names)]
parts.append((abs(d_off) * 0.1 / abs(dx), "offset"))
parts.sort(reverse=True)
print("   largest parts of the second error bar (in A1 + A3): "
      + ", ".join(f"{lbl} {v:.4f}" for v, lbl in parts[:3]))
print(f"   the cited set has m0 / |A1 + A3| = "
      f"{1.0 / abs(GAN.A1 + GAN.A3):.3f} m0; the synthetic line was "
      "made with 2.0 m0")
