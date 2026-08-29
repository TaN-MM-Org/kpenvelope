"""Subband character analysis: how much HH, LH and CH is in each state.

A wurtzite valence subband is a six-component envelope spinor, and the
question "is this subband heavy-hole-like?" has a quantitative answer:
the fraction of its norm carried by each basis group. In the
Chuang-Chang basis used by `hamiltonian.assemble_hamiltonian` (Chuang
and Chang, Phys. Rev. B 54, 2491 (1996)), components (1, 4) are the two
spin partners of the heavy hole (HH), components (2, 5) of the light
hole (LH), and components (3, 6) of the crystal-field split-off hole
(CH); the diagonal energies of the assembled Hamiltonian confirm the
assignment (delta1 + delta2 for HH, delta1 - delta2 for LH, 0 for CH at
zone center, before the A-parameter terms).

Band mixing is the physics behind the "one subband, many masses"
problem this package exists to make explicit: a state that is pure HH
at the zone center picks up LH and CH weight as the in-plane momentum
grows, and its local mass moves with that composition.

Exact facts the test suite asserts, rather than states:

* Character fractions of normalized states sum to 1 to machine
  precision.
* At kt = 0 the six-band Hamiltonian couples only LH and CH (through
  delta3); every kt = 0 eigenstate therefore has HH fraction exactly 1
  or exactly 0, while LH and CH mix.
* At finite kt the top subband of the GaN parameter set is no longer
  pure HH: the off-diagonal A5, A6 terms mix the groups.
"""
from __future__ import annotations

import numpy as np

#: index groups of the six-component spinor, Chuang-Chang ordering
CHARACTER_GROUPS = {"HH": (0, 3), "LH": (1, 4), "CH": (2, 5)}
CHARACTER_LABELS = ("HH", "LH", "CH")


def band_character(envelopes, z):
    """Character fractions (HH, LH, CH) of envelope states.

    envelopes : (n_states, 6, N) complex array as returned by
        `solve_subbands` (states normalized to unit total norm), or a
        single (6, N) state.
    z : the uniform grid the envelopes live on (nm).

    Returns an (n_states, 3) float array of fractions in the order
    (HH, LH, CH), each row summing to the squared norm of the state
    (1 for `solve_subbands` output). Fractions are basis populations,
    not projections onto bulk eigenstates; at finite kt the two notions
    differ, and the basis population is the convention stated here.
    """
    F = np.asarray(envelopes, dtype=complex)
    single = F.ndim == 2
    if single:
        F = F[None]
    if F.ndim != 3 or F.shape[1] != 6:
        raise ValueError("envelopes must have shape (n_states, 6, N) "
                         "or (6, N)")
    z = np.asarray(z, dtype=float)
    if z.size != F.shape[2]:
        raise ValueError("z grid length does not match the envelopes")
    dz = z[1] - z[0]
    dens = (np.abs(F) ** 2).sum(axis=2) * dz            # (n_states, 6)
    out = np.empty((F.shape[0], 3))
    for k, label in enumerate(CHARACTER_LABELS):
        i, j = CHARACTER_GROUPS[label]
        out[:, k] = dens[:, i] + dens[:, j]
    return out[0] if single else out


def dominant_character(fractions):
    """Label each state by its dominant basis group.

    fractions : (n_states, 3) or (3,) array from `band_character`.

    Returns a list of labels (or a single label) from ("HH", "LH",
    "CH"). A dominant label is a summary, not a substitute for the
    fractions: a state at 40/35/25 is "HH" here but is really a mixture,
    and the fractions say so.
    """
    fr = np.atleast_2d(np.asarray(fractions, dtype=float))
    if fr.shape[1] != 3:
        raise ValueError("fractions must have 3 columns (HH, LH, CH)")
    labels = [CHARACTER_LABELS[int(k)] for k in np.argmax(fr, axis=1)]
    return labels[0] if np.asarray(fractions).ndim == 1 else labels


def character_vs_k(p, z, kts, theta=0.0, potential=None, n_states=4):
    """Character fractions of each subband along an in-plane path.

    Same call signature spirit as `subband_dispersion`; solves at each
    kt and returns an array of shape (len(kts), n_states, 3) with the
    (HH, LH, CH) fractions of each state, rows ordered exactly as
    `solve_subbands` orders energies (descending, hole ground state
    first).

    Caveat, stated rather than hidden: states are ordered by energy at
    each kt independently, so at an anticrossing the composition of row
    j can change abruptly because the branches swap order, not because
    any physical state changed abruptly.
    """
    from .solver import solve_subbands

    kts = np.asarray(kts, dtype=float)
    if np.any(kts < 0.0):
        raise ValueError("kts must be non-negative magnitudes")
    ct, st = np.cos(theta), np.sin(theta)
    out = np.empty((kts.size, n_states, 3))
    for i, kt in enumerate(kts):
        _, envs = solve_subbands(p, z, kx=kt * ct, ky=kt * st,
                                 potential=potential, n_states=n_states)
        out[i] = band_character(envs, z)
    return out
