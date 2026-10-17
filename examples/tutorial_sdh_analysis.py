"""Tutorial: density, cyclotron mass and quantum lifetime from
Shubnikov-de Haas (SdH) data, each with an error bar.

The "measured" numbers are synthetic: they are generated below from
known, illustrative values (not from a real sample) with a fixed random
seed, so the script prints the same output every time and you can see
how well each step recovers the truth. To use it on your own data,
replace the three blocks marked DATA with your measurements.

Run:  python examples/tutorial_sdh_analysis.py  (about one second)

Output (kpenvelope 0.14.0):

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
"""
import numpy as np

from kpenvelope import (fit_sdh_dingle, fit_sdh_mass, lk_thermal_factor,
                        sheet_density_from_sdh)
from kpenvelope.lab import DINGLE_TESLA_PS

rng = np.random.default_rng(20261017)

# illustrative "true" sample, used only to make the synthetic data
M_TRUE, TAU_TRUE = 0.5, 0.15          # m*/m0 and quantum lifetime (ps)


def envelope(T, B, m=M_TRUE, tau=TAU_TRUE):
    """SdH envelope in units of rho_0: 4 R_T exp(-pi / (omega_c tau))."""
    return (4.0 * lk_thermal_factor(T, B, m)
            * np.exp(-DINGLE_TESLA_PS * m / (B * tau)))


# DATA 1: one SdH frequency (tesla) from the Fourier transform of
# rho_xx against 1/B, with its 1-sigma error
F, sigma_F = 150.0, 2.0

# DATA 2: amplitudes at one field and six temperatures, 2 % errors
B_mass = 20.0
T_mass = np.array([2.0, 5.0, 8.0, 11.0, 14.0, 17.0])
a_mass = envelope(T_mass, B_mass)
s_mass = 0.02 * a_mass
a_mass = a_mass + s_mass * rng.standard_normal(T_mass.size)

# DATA 3: amplitudes at one temperature and eight fields, 3 % errors
T_dingle = 2.0
B_dingle = np.linspace(12.0, 26.0, 8)
a_dingle = envelope(T_dingle, B_dingle)
s_dingle = 0.03 * a_dingle
a_dingle = a_dingle + s_dingle * rng.standard_normal(B_dingle.size)

# 1. density (Onsager relation; two spin branches share the contour)
d = sheet_density_from_sdh(F, branches=2, sigma_frequency_T=sigma_F)
print("1. density from the SdH frequency")
print(f"   F = {F:.1f} +/- {sigma_F:.1f} T, spin-degenerate subband "
      "(branches=2)")
print(f"   n = {d['n_sheet_cm2']:.3e} +/- {d['sigma_n_sheet_cm2']:.1e} "
      "cm^-2")

# 2. cyclotron mass (Lifshitz-Kosevich fit)
fm = fit_sdh_mass(T_mass, a_mass, B_mass, sigma_amplitudes=s_mass)
print(f"2. cyclotron mass from the temperature damping at {B_mass:g} T")
print(f"   m* = {fm['mass_ratio']:.3f} +/- {fm['sigma_mass_ratio']:.3f} m0"
      f"   (true {M_TRUE})")
print(f"   chi-square per degree of freedom {fm['chi2_red']:.2f} "
      f"({fm['dof']} degrees of freedom)")

# 3. quantum lifetime (Dingle plot), with the mass and its error
fd = fit_sdh_dingle(B_dingle, a_dingle, T_dingle, fm["mass_ratio"],
                    sigma_amplitudes=s_dingle,
                    sigma_mass_ratio=fm["sigma_mass_ratio"])
print(f"3. quantum lifetime from the field damping at {T_dingle:g} K "
      "(Dingle plot)")
print(f"   tau_q = {fd['tau_q_ps']:.3f} +/- {fd['sigma_tau_q_ps']:.3f} ps"
      f"   (true {TAU_TRUE})")
print(f"     of which {fd['sigma_tau_q_stat_ps']:.3f} ps from the data and "
      f"{fd['sigma_tau_q_mass_ps']:.3f} ps from the mass")
print(f"   Dingle temperature T_D = {fd['dingle_temperature_K']:.2f} +/- "
      f"{fd['sigma_dingle_temperature_K']:.2f} K")
print(f"   A0 = {fd['A0']:.2f} +/- {fd['sigma_A0']:.2f} (4 for a "
      "homogeneous sample, in units of rho_0)")
print(f"   chi-square per degree of freedom {fd['chi2_red']:.2f} "
      f"({fd['dof']} degrees of freedom)")
