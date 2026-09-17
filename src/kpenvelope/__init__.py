"""kpenvelope: multiband k.p envelope-function solver for wurtzite heterostructures.

Solves the six-band valence envelope-function problem on a 1D grid,
self-consistently with Poisson's equation, for polarization-induced
two-dimensional hole gases and related wurtzite heterostructures.

Methodological basis: T. M. Mahim, A. S. M. Mohsin and M. M. Rahman,
"Origin of the conflicting hole masses in the GaN/AlN two-dimensional
hole gas" (under review), and the standard wurtzite k.p formulation of
Chuang and Chang, Phys. Rev. B 54, 2491 (1996).
"""
from .params import (WurtziteParameters, aln_rinke2008, demo_single_band,
                     gan_rinke2008)
from .character import (CHARACTER_GROUPS, band_character, character_vs_k,
                        dominant_character)
from .dispersion import (dipole_matrix, dos_from_dispersion,
                         group_velocity, local_mass, oscillator_strengths,
                         spin_splitting, splitting_vs_k, subband_dispersion)
from .hamiltonian import assemble_hamiltonian, HBAR2_OVER_2M0
from .heterostructure import (assemble_heterostructure, layered_profile,
                              solve_heterostructure)
from .solver import solve_subbands
from .selfconsistent import (KB_EV_PER_K, fill_subbands_kgrid,
                             fill_subbands_thermal, solve_self_consistent,
                             solve_self_consistent_hetero,
                             SelfConsistentResult)
from .strain import strain_blocks
from .units import sheet_density_from_cm2, sheet_density_to_cm2
from .lab import (fit_band_offset, offset_sensitivity,
                  sheet_density_from_shift)
from .isb import (depolarization_shift, isb_lineshape,
                  overlap_geometry_integral)

__version__ = "0.10.0"
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
]
