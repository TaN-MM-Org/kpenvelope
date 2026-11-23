"""kpenvelope: multiband k.p envelope-function solver for wurtzite heterostructures.

Solves the six-band valence envelope-function problem on a 1D grid,
self-consistently with Poisson's equation, for polarization-induced
two-dimensional hole gases and related wurtzite heterostructures.
Since v0.17.0 it also computes the fixed charge at every interface of a
c-plane nitride stack, coherent or partly relaxed, with the charge of the
misfit dislocations made explicit (`interface_charges`), solves the
electron gases of such stacks (`solve_electrons`), carries error bars
through any calculation (`propagate`), and fits the calibration and
misfit-line charges to measured series (`fit_interface_series`).
Since v0.18.0 it builds InGaN, InAlN and ScAlN from cited tables
(`ingan`, `inaln`, `scaln`), solves drift-diffusion with both carriers,
recombination and contacts (`solve_drift_diffusion`, `iv_curve`,
`cv_curve`), gives the interface phonons of the dielectric continuum
model (`slab_modes`), combines what is known about the misfit-line charge
(`combine_line_charge`), and opens as an app in the browser
(`kpenvelope-app`).
Since v0.19.0 it simulates whole devices in two and three dimensions
(`Device`): current between source, gate and drain with a mobility that
saturates at high field and falls with temperature, deep-level traps that
fill and empty in time (`Device.transient`), and self-heating through a
heat equation down to the heat sink (`ThermalModel`, `solve_heat`).

Methodological basis: T. M. Mahim, A. S. M. Mohsin and M. M. Rahman,
"Origin of the conflicting hole masses in the GaN/AlN two-dimensional
hole gas", Phys. Rev. B 114, L291304 (2026), doi:10.1103/zn44-nmx8,
and the standard wurtzite k.p formulation of Chuang and Chang, Phys.
Rev. B 54, 2491 (1996).
"""
from .params import (WurtziteParameters, aln_rinke2008, demo_single_band,
                     gan_rinke2008)
from .character import (CHARACTER_GROUPS, band_character, character_vs_k,
                        dominant_character)
from .dispersion import (cyclotron_mass, dipole_matrix,
                         dos_from_dispersion, group_velocity, local_mass,
                         oscillator_strengths, spin_splitting,
                         splitting_vs_k, subband_dispersion)
from .hamiltonian import assemble_hamiltonian, HBAR2_OVER_2M0
from .heterostructure import (assemble_heterostructure,
                              heterostructure_energies, layered_profile,
                              solve_heterostructure)
from .solver import solve_subbands, subband_energies
from .selfconsistent import (KB_EV_PER_K, fill_subbands_kgrid,
                             fill_subbands_thermal, solve_self_consistent,
                             solve_self_consistent_hetero,
                             SelfConsistentResult)
from .strain import strain_blocks
from .units import (energy_from_wavenumber, energy_to_wavenumber,
                    sheet_density_from_cm2, sheet_density_to_cm2)
from .lab import (fit_band_offset, fit_sdh_dingle, fit_sdh_mass,
                  lk_thermal_factor,
                  offset_sensitivity, sdh_frequency,
                  sheet_density_from_sdh, sheet_density_from_shift)
from .rashba import (rashba_hamiltonian, rashba_spins,
                     rashba_splitting)
from .isb import (degenerate_levels, depolarization_shift, isb_lineshape,
                  level_transitions, overlap_geometry_integral)
from .sensitivity import PARAMETER_NAMES, parameter_sensitivities
from .velocity import inplane_transitions, velocity_matrix
from .nitride import (AMBACHER2021_SCALN, E_PER_NM2_PER_C_M2,
                      POLARIZATION_SETS, VURGAFTMAN2003, NitrideMaterial,
                      algan, gan, inaln, ingan, polarization_quantum, scaln,
                      with_numbers, zinc_blende_formal_polarization)
from .interfaces import (InterfaceCharge, LineCharge, StackLayer,
                         charge_change_per_relaxation, interface_charges,
                         layer_strains, line_charge_limits,
                         misfit_line_density, strained_polarization,
                         trapped_from_sum_rule)
from .electrons import (ElectronResult, fermi_dirac_half,
                        line_charge_from_mobility_drop,
                        line_charge_mobility_bound, line_scattering_rate,
                        solve_electrons, split_two_carrier, two_carrier_hall)
from .uncertainty import Propagation, propagate
from .fitting import FitResult, fit_interface_series, fit_least_squares
from .driftdiffusion import (Contact, DriftDiffusionResult, Transport,
                             bernoulli, cv_curve, cv_curve_quantum, iv_curve,
                             solve_drift_diffusion, solve_quasi_equilibrium)
from .phonons import (CM1_TO_MEV, GAN_BARKER1973, PolarPhonons,
                      frohlich_constant, single_interface_modes, slab_modes)
from .datafit import MeasurementFit, correlation_warnings, fit_measurements
from .design import (Design, design_error_bars, find_designs, sheet_density,
                     threshold_voltage)
from .twod import (CrossSection, Gate, cv_cross_section, device_grid,
                   solve_cross_section)
from .leakage import (gate_leakage, richardson_constant, richardson_velocity,
                      thermionic_emission_current, tsu_esaki_current,
                      wkb_transmission)
from .device import (IMPACT_GAN, BandToBand, Device, DeviceState, Dielectric, Electrode,
                     Ionization, Mobility, Region, Surface, Trap, TransientResult,
                     TunnelBarrier, Varshni, capture_coefficient, rf_figures, two_port,
                     with_series)
from .mesh import Mesh, MeshDevice, MeshRegion, mesh_from_grid, mesh_from_points
from .process import Anneal, Implant, Process, Structure
from .thermal import (GAN_BULK, SIC_4H_ROOM, ThermalLayer, ThermalMaterial,
                      ThermalModel, boundary_resistance, solve_heat,
                      solve_heat_transient)
from .linecharge import (PUBLISHED_CONSTRAINTS, Constraint,
                         LineChargePosterior, combine_line_charge,
                         line_charge_error_from_density, plan_holography,
                         plan_sum_rule)

__version__ = "0.33.0"
__all__ = [
    "WurtziteParameters", "demo_single_band", "gan_rinke2008",
    "aln_rinke2008", "assemble_hamiltonian", "solve_subbands",
    "subband_dispersion", "local_mass", "spin_splitting",
    "splitting_vs_k", "group_velocity", "dos_from_dispersion",
    "dipole_matrix", "oscillator_strengths",
    "depolarization_shift", "isb_lineshape", "overlap_geometry_integral",
    "strain_blocks", "fill_subbands_kgrid", "fill_subbands_thermal",
    "KB_EV_PER_K", "sheet_density_from_cm2", "sheet_density_to_cm2",
    "solve_self_consistent_hetero",
    "assemble_heterostructure", "layered_profile", "solve_heterostructure",
    "band_character", "character_vs_k", "dominant_character",
    "CHARACTER_GROUPS",
    "solve_self_consistent", "SelfConsistentResult", "HBAR2_OVER_2M0",
    "fit_band_offset", "offset_sensitivity",
    "sheet_density_from_shift",
    "sheet_density_from_sdh", "sdh_frequency", "fit_sdh_mass",
    "lk_thermal_factor", "energy_from_wavenumber", "energy_to_wavenumber",
    "rashba_hamiltonian", "rashba_splitting", "rashba_spins",
    "fit_sdh_dingle", "cyclotron_mass", "subband_energies",
    "heterostructure_energies", "degenerate_levels", "level_transitions",
    "parameter_sensitivities", "PARAMETER_NAMES",
    "velocity_matrix", "inplane_transitions",
    "NitrideMaterial", "algan", "POLARIZATION_SETS", "E_PER_NM2_PER_C_M2",
    "ingan", "inaln", "scaln", "gan", "with_numbers", "VURGAFTMAN2003",
    "AMBACHER2021_SCALN",
    "polarization_quantum", "zinc_blende_formal_polarization",
    "StackLayer", "LineCharge", "InterfaceCharge", "interface_charges",
    "layer_strains", "line_charge_limits", "misfit_line_density",
    "strained_polarization", "charge_change_per_relaxation",
    "trapped_from_sum_rule",
    "solve_electrons", "ElectronResult", "fermi_dirac_half",
    "two_carrier_hall", "split_two_carrier", "line_scattering_rate",
    "line_charge_mobility_bound", "line_charge_from_mobility_drop",
    "propagate", "Propagation",
    "fit_least_squares", "FitResult", "fit_interface_series",
    "PUBLISHED_CONSTRAINTS", "Constraint", "LineChargePosterior",
    "combine_line_charge", "plan_holography", "plan_sum_rule",
    "line_charge_error_from_density",
    "solve_drift_diffusion", "DriftDiffusionResult", "Transport", "Contact",
    "iv_curve", "cv_curve", "bernoulli", "solve_quasi_equilibrium",
    "cv_curve_quantum",
    "PolarPhonons", "GAN_BARKER1973", "CM1_TO_MEV", "single_interface_modes",
    "slab_modes", "frohlich_constant",
    "fit_measurements", "MeasurementFit", "correlation_warnings",
    "find_designs", "Design", "design_error_bars", "sheet_density",
    "threshold_voltage",
    "gate_leakage", "richardson_constant", "richardson_velocity",
    "thermionic_emission_current", "tsu_esaki_current", "wkb_transmission",
    "solve_cross_section", "cv_cross_section", "Gate", "CrossSection",
    "device_grid",
    "Device", "DeviceState", "Electrode", "Mobility", "Surface", "Trap",
    "TransientResult", "capture_coefficient",
    "ThermalMaterial", "ThermalLayer", "ThermalModel", "GAN_BULK", "SIC_4H_ROOM",
    "boundary_resistance", "solve_heat",
    "Dielectric", "Region", "Ionization", "IMPACT_GAN", "with_series", "two_port",
    "rf_figures", "solve_heat_transient",
    "BandToBand", "TunnelBarrier", "Mesh", "MeshDevice", "MeshRegion", "mesh_from_grid",
    "mesh_from_points", "Implant", "Anneal", "Process", "Varshni", "Structure",
]
