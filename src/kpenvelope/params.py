"""Band-structure parameter sets.

This package ships NO validated material constants. The A parameters and
crystal-field/spin-orbit splittings of GaN, AlN and their alloys differ
between first-principles parameterizations, and the choice materially
affects computed masses. Populate WurtziteParameters from the literature
(commonly used sets: Vurgaftman and Meyer, J. Appl. Phys. 94, 3675 (2003);
Rinke et al., Phys. Rev. B 77, 075202 (2008)) and record the source in the
mandatory `reference` field.

demo_single_band() returns a degenerate, decoupled parameter set used by
the test-suite; it is numerically convenient and physically meaningless.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class WurtziteParameters:
    """Six-band wurtzite valence parameters (Chuang-Chang convention).

    A1..A6 : dimensionless valence-band effective-mass parameters.
    delta1 : crystal-field splitting (eV).
    delta2, delta3 : spin-orbit parameters (eV); for the common isotropic
        approximation delta2 = delta3 = Delta_so / 3.
    eps_r : static relative permittivity (used by the Poisson solver).
    reference : provenance of the numbers. Required, on purpose.
    """

    A1: float
    A2: float
    A3: float
    A4: float
    A5: float
    A6: float
    delta1: float
    delta2: float
    delta3: float
    eps_r: float
    reference: str


def demo_single_band(A: float = -2.0) -> WurtziteParameters:
    """A decoupled, NON-PHYSICAL parameter set for tests.

    With A1 = A2 = A and every other coupling zero, the six-band
    Hamiltonian reduces to six identical parabolic bands with
    E = A * (hbar^2/2m0) * k^2, which has closed-form well solutions.
    """
    return WurtziteParameters(
        A1=A, A2=A, A3=0.0, A4=0.0, A5=0.0, A6=0.0,
        delta1=0.0, delta2=0.0, delta3=0.0,
        eps_r=10.0,
        reference="synthetic single-band demonstration values; not physical",
    )
