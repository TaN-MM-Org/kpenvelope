"""Band-structure parameter sets.

This module ships two CITED parameter sets, gan_rinke2008() and
aln_rinke2008(), plus a synthetic set for the test-suite. Every number in
the cited sets is traceable to the source given in its `reference` field,
and the test-suite checks the GaN set against closed-form quasi-cubic
results (zone-center splittings of 5.20 and 21.80 meV; asymptotic in-plane
masses m0/|A2+A4-A5| and m0/|A2+A4+A5|).

The general warning stands: A parameters and splittings differ between
first-principles parameterizations, and the choice materially affects
computed masses. For any other material or parameterization, populate
WurtziteParameters from the literature (e.g. Vurgaftman and Meyer,
J. Appl. Phys. 94, 3675 (2003)) and record the source in the mandatory
`reference` field.

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
        May be NaN for a set intended only as a barrier material; the
        self-consistent solver refuses to run on a NaN permittivity.
    reference : provenance of the numbers. Required, on purpose.
    D1..D6 : optional Bir-Pikus deformation potentials (eV) for the
        strain terms; None by default, and a strained assembly on a set
        without them is refused. No values are shipped, on purpose:
        supply them with a citation (e.g. Vurgaftman and Meyer,
        J. Appl. Phys. 94, 3675 (2003)).
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
    D1: float = None
    D2: float = None
    D3: float = None
    D4: float = None
    D5: float = None
    D6: float = None


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


def gan_rinke2008() -> WurtziteParameters:
    """GaN valence parameters of Rinke et al., Phys. Rev. B 77, 075202 (2008).

    A1..A6 and the crystal-field splitting Delta_CR = 10 meV and spin-orbit
    splitting Delta_SO = 17 meV are the consistent GW-based set of Rinke
    et al., as tabulated in Extended Data Table 1 of Chang et al., Nature
    Electronics 9, 346 (2026), and as used in the self-consistent analysis
    of Mahim, Mohsin and Rahman (gan-2dhg-masses-lifetimes). The isotropic
    quasi-cubic convention delta2 = delta3 = Delta_SO / 3 is applied.

    With these numbers the k = 0 splittings of the six-band Hamiltonian
    are 5.20 and 21.80 meV (against accepted experimental values near
    5 to 6 and 22 meV), and the asymptotic in-plane masses are
    m0/|A2+A4-A5| = 1.89 m0 and m0/|A2+A4+A5| = 0.18 m0; the test-suite
    checks both from the assembled Hamiltonian.

    eps_r = 10.4 is the static dielectric constant of GaN for fields along
    the c axis (the growth axis of a c-plane heterostructure), from the
    infrared oscillator fit of Barker and Ilegems, Phys. Rev. B 7, 743
    (1973); it is the value used by the analysis of Mahim et al.
    """
    return WurtziteParameters(
        A1=-5.947, A2=-0.528, A3=5.414,
        A4=-2.512, A5=-2.510, A6=-3.202,
        delta1=0.010,
        delta2=0.017 / 3.0,
        delta3=0.017 / 3.0,
        eps_r=10.4,
        reference=(
            "A1-A6, Delta_CR = 0.010 eV, Delta_SO = 0.017 eV: P. Rinke "
            "et al., Phys. Rev. B 77, 075202 (2008), as tabulated in "
            "Extended Data Table 1 of C. F. C. Chang et al., Nature "
            "Electronics 9, 346 (2026) and used in T. M. Mahim, "
            "A. S. M. Mohsin and M. M. Rahman (under review; code: "
            "github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes, "
            "doi:10.5281/zenodo.21798022). delta2 = delta3 = Delta_SO/3. "
            "eps_r = 10.4 (E parallel c): A. S. Barker and M. Ilegems, "
            "Phys. Rev. B 7, 743 (1973)."
        ),
    )


def aln_rinke2008() -> WurtziteParameters:
    """AlN valence parameters, intended as a BARRIER material.

    A1..A6 and Delta_CR = -295 meV are from Rinke et al., Phys. Rev. B 77,
    075202 (2008), the same source from which the GaN set descends. Rinke
    et al. do not tabulate the spin-orbit splitting of AlN; Delta_SO =
    22 meV is the midpoint of the values computed by de Carvalho et al.,
    Appl. Phys. Lett. 97, 232101 (2010) (21.7 meV parallel, 23.5 meV
    perpendicular), following the treatment of Mahim, Mohsin and Rahman,
    who verify that changing it from 22 to 19 meV changes nothing at four
    figures in their barrier-sensitive results.

    eps_r is deliberately NaN: this set is meant for barrier regions,
    where the states are evanescent and the permittivity never enters the
    supported calculations. The self-consistent solver refuses a NaN
    permittivity rather than running with an unvetted number; supply your
    own cited value if you need an AlN Poisson solve.
    """
    return WurtziteParameters(
        A1=-3.991, A2=-0.311, A3=3.671,
        A4=-1.147, A5=-1.329, A6=-1.952,
        delta1=-0.295,
        delta2=0.022 / 3.0,
        delta3=0.022 / 3.0,
        eps_r=float("nan"),
        reference=(
            "A1-A6, Delta_CR = -0.295 eV: P. Rinke et al., Phys. Rev. B "
            "77, 075202 (2008). Delta_SO = 0.022 eV: midpoint of L. C. "
            "de Carvalho et al., Appl. Phys. Lett. 97, 232101 (2010). "
            "delta2 = delta3 = Delta_SO/3. As used in T. M. Mahim, "
            "A. S. M. Mohsin and M. M. Rahman (under review; "
            "doi:10.5281/zenodo.21798022). eps_r intentionally NaN: "
            "barrier-only set, no vetted permittivity shipped."
        ),
    )
