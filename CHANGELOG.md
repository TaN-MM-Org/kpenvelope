# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists.

## v0.6.0 - 2026-09-05

- Bir-Pikus strain terms (`strain_blocks`, `strain=` on every
  assembly): validated by the exact structural identity with the
  kinetic template and closed-form k = 0 eigenvalues; deformation
  potentials must be supplied with a citation (none shipped).
- Full k-grid non-parabolic filling (`fill_subbands_kgrid`), agreeing
  with the closed-form parabolic filler on a parabolic model and
  refusing an undersized k-window.
- Spin-splitting analysis (`spin_splitting`, `splitting_vs_k`):
  exact zeros under Kramers/inversion symmetry, Rashba-type splitting
  under asymmetric confinement.
- Finite-barrier self-consistent loop
  (`solve_self_consistent_hetero`), exactly reproducing the hard-wall
  loop on a uniform stack.
- Transport groundwork: `group_velocity` and `dos_from_dispersion`
  with parabolic closed-form anchors; mechanism lifetimes stay
  designed out (no uncited scattering parameters).
- CI now tests Python 3.9, 3.11, 3.12 and 3.13.

## v0.5.0

- Finite barriers: position-dependent materials and band edges with
  the symmetrized Ben Daniel-Duke assembly; finite-square-well and
  decay-constant closed-form anchors.

## v0.4.0 and earlier

- Six-band wurtzite envelope solver, self-consistent Poisson loop,
  cited GaN/AlN parameter sets, dispersion and band-character
  utilities. See the GitHub releases for the per-version anchors.
