# kpenvelope

[![PyPI](https://img.shields.io/pypi/v/kpenvelope)](https://pypi.org/project/kpenvelope/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015269-blue)](https://doi.org/10.5281/zenodo.22015269) [![tests](https://github.com/TaN-MM-Org/kpenvelope/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/kpenvelope/actions)

kpenvelope is a free Python package for nitride semiconductors (GaN,
AlN, InN and their alloys). It started as a tool for one question: what
energy levels do holes have when they are trapped in a thin GaN layer?
It has since grown to cover the whole path from a stack of layers to a
working transistor.

Every material number in it comes with a written source, and every
model is checked against an exact answer or an independent calculation
in tests you can run yourself. When an input cannot give a result you
can trust, it stops and tells you why instead of printing a number that
only looks right.

## What it can do

**Layers and charges**

- Energy levels, wavefunctions and masses of holes trapped in thin
  layers, with the hole gas's own charge included.
- Light absorption between those levels, and what a measured
  magnetoresistance (Shubnikov-de Haas) curve says about density and
  mass.
- The fixed charge at each interface of a nitride stack, also when a
  layer has partly relaxed.
- The electron density of a transistor layer, with error bars, fits to
  measured C-V curves and Hall data, and a search for the layer design
  that gives the density and switch-off voltage you want.

**Whole devices, in 2D and 3D**

- Current in transistors and diodes, traps that fill and empty over
  time, self-heating down to the heat sink (with the heat that carriers
  carry across contacts and junctions, if you want it), and the speed
  limits f_T and f_max.
- Breakdown (avalanche, a current limit, or thermal runaway), tunneling
  through barriers and between bands (including tunnel diodes), hot
  electrons, and electrons hopping over band steps.
- Electrons and holes in a channel as quantum states (their allowed
  energies and where they sit), solved together with the current, also
  on triangle meshes; quantum transport of the channel electrons or
  holes by non-equilibrium Green's functions (tunneling, reflection,
  electrons changing subband, and scattering if you want it), and the
  ballistic limit of a transistor's current.
- Scattering lifetimes and mobility of a 2D gas from its own
  wavefunctions, with electrons moving between its levels: acoustic and
  polar optical phonons (by an exact solution of the Boltzmann
  equation, with the interface vibrations of a heterostructure if you
  want them), alloy disorder, interface roughness and charged impurities,
  screened by all the levels together, with the material numbers you
  give (also in the app).
- Leakage and band gaps that change with temperature, so a leaky
  device can heat itself into runaway.
- Shaped devices: films, recesses, field plates, and curved or slanted
  shapes on triangle meshes, with a built-in generator of quality
  meshes that can also refine where the solution changes fast.
- Process steps that build the shape and the doping: deposition and
  etching (also with the shadowing of narrow openings, particles that
  bounce, and etching driven by ions and a reactive gas together, with
  the fluxes from a model of the plasma's chemistry),
  oxidation (also under mask edges, with the oxide's flow and stress),
  implants (with ranges from a Monte Carlo simulation of the ions) and
  anneals.

**An app** that runs many of these calculations in your web browser,
with no code: charges, electrons and their mobility, error bars, data
fits, transistors with heat, runaway, quantum channels and the quantum
current of short channels, tunnel diodes, and process steps with
simulated implants, plasma etching and oxide stress.

## Install

```
pip install "kpenvelope[device]"
```

It needs Python 3.9 or newer and NumPy. The device part also needs
SciPy, which the `[device]` option installs. Then start the app with:

```
kpenvelope-app
```

It runs on your own computer only. You build the layers on the left and
read the results on the right, and each result starts with a sentence
that says what it means.

## A first look

Units: energies in eV, lengths in nm, densities in cm^-3 or nm^-3 (the
docs say which), currents per millimeter of device width (A/mm).

**Energy levels of holes in a 5 nm GaN layer**

```python
import numpy as np
from kpenvelope import gan_rinke2008, solve_subbands, band_character

p = gan_rinke2008()                  # cited GaN parameter set
z = np.linspace(0.0, 5.0, 51)        # a 5 nm layer, grid in nm

energies, envelopes = solve_subbands(p, z, n_states=4)
frac = band_character(envelopes, z)  # columns: HH, LH, CH
for e, (hh, lh, ch) in zip(energies, frac):
    print(f"E = {e*1e3:8.2f} meV   HH {hh:.3f}  LH {lh:.3f}  CH {ch:.3f}")
```

```
E =     8.26 meV   HH 1.000  LH 0.000  CH 0.000
E =     8.26 meV   HH 1.000  LH 0.000  CH 0.000
E =    -2.28 meV   HH 0.000  LH 0.990  CH 0.010
E =    -2.28 meV   HH 0.000  LH 0.990  CH 0.010
```

The levels come in pairs with the same energy. The top pair is pure
heavy hole (HH); the next is mostly light hole (LH).

**The charge at an AlGaN/GaN interface**

```python
from kpenvelope import StackLayer, algan, interface_charges

gan, bar = algan(0.0), algan(0.22)                 # PBEsol polarization set
for lc in ("bare", "neutral", "electron-counting"):
    stack = [StackLayer(gan, 2000.0), StackLayer(bar, 103.0, relaxation=0.78)]
    c = interface_charges(stack, line_charge=lc)[0]
    print(f"{lc:18s} Lambda = {c.line_charge:+.3f} e/nm   sigma = {c.sigma_cm2:.3e} cm^-2")
c0 = interface_charges([StackLayer(gan, 2000.0), StackLayer(bar, 103.0)])[0]
print(f"coherent barrier                        sigma = {c0.sigma_cm2:.3e} cm^-2")
print(f"mismatch {c.eps_c:.5f}, misfit lines {c.line_density * 1e7:.3e} cm/cm^2")
```

```
bare               Lambda = -2.645 e/nm   sigma = 3.564e+12 cm^-2
neutral            Lambda = +0.000 e/nm   sigma = 1.051e+13 cm^-2
electron-counting  Lambda = +0.085 e/nm   sigma = 1.073e+13 cm^-2
coherent barrier                        sigma = 1.432e+13 cm^-2
mismatch 0.00534, misfit lines 2.626e+05 cm/cm^2
```

When the barrier relaxes, the charge drops. How far it drops depends on
the charge carried by the dislocation lines that let it relax, which no
standard recipe decides, so the package asks you to choose and shows
the range.

**The current of a GaN transistor**

```python
from kpenvelope import StackLayer, algan
from kpenvelope.device import Device, Electrode, Mobility, Surface

layers = [StackLayer(algan(0.0), 1000.0, name="buffer"),
          StackLayer(algan(0.25), 20.0, name="barrier")]
mob = Mobility(1704.0, "Alpert et al., Rev. Sci. Instrum. 91 (2020)",
               saturation_velocity=1.0e7, temperature_exponent=2.35)
contacts = [Electrode("source", "ohmic", 0, 500, implant_depth=40, implant_donors=1e20),
            Electrode("gate", "schottky", 1500, 1750, barrier=1.0),
            Electrode("drain", "ohmic", 3000, 3500, implant_depth=40, implant_donors=1e20)]
dev = Device(layers, 3500.0, contacts, mob, Surface("pinned", barrier=1.0),
             dx_fine=10, dx_max=100, dz_fine=1.0, dz_max=100)
for vg in (0.0, -2.0, -4.0):
    st = dev.solve({"gate": vg, "drain": 5.0})
    print(f"gate {vg:+.0f} V, drain 5 V: {st.current['drain']:.3f} A/mm")
```

```
gate +0 V, drain 5 V: 1.046 A/mm
gate -2 V, drain 5 V: 0.852 A/mm
gate -4 V, drain 5 V: 0.339 A/mm
```

The mobility comes from a cited measurement; the top speed of 1e7 cm/s
is a round number for the example. Add `thermal=` to let the current
heat the device, and the current drops as it warms (example 28 in the
docs).

## Learn more

The details live in the `docs` folder:

- [Examples](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/examples.md):
  73 worked examples, each with the output it prints, and three
  tutorials that go from measured data to a result with error bars.
- [Reference](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/reference.md):
  every function and option, the units, a glossary of the terms, the
  cited material numbers, and the cases where the package refuses to
  answer.
- [How it is checked](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/checks.md):
  what each test compares against and how closely, and the corrections
  made in past versions.
- [Limits](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/limits.md):
  what the models leave out, in detail.
- [Changes](https://github.com/TaN-MM-Org/kpenvelope/blob/main/CHANGELOG.md)
  from version to version.

## How it is checked

More than 300 automated tests run on every change, on Python 3.9 to
3.14 and with the oldest NumPy and SciPy the package allows. Most of
them compare against an exact formula, a symmetry, or the same quantity
worked out a second, different way. Two examples: a 2D device and the
same device on a triangle mesh must give the same current to ten
digits, and the leakage that drives a device into thermal runaway must
match the simple heat-balance model at the voltage where runaway starts.
The full list is in [How it is checked](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/checks.md).

## What it does not do

The main gaps, in plain terms:

- Quantum transport is one band per subband and one carrier at a time.
  It runs on rectangular grids and on meshes, and carriers can change
  subband with or without scattering, but the scattering is one simple
  model (Buttiker probes with one scattering time), not phonons. Without
  scattering, long
  access regions, high drain voltages or small dips along the channel
  can keep it from settling.
- With velocity saturation, the hot-electron model does not turn into
  drift-diffusion as its relaxation time goes to zero. That is how such
  models behave; the attempts to change it are described on the limits
  page, with the numbers.
- Process steps stay simple in other ways. The plasma model is
  averaged over the chamber, for positive ions only, with your rates
  and wall factors; growing oxide flows but its shape
  follows a simple volume rule and there is no elasticity, the Monte
  Carlo implant ranges use a textbook electronic stopping that can be off
  by tens of percent for light ions, diffusion constants come from you,
  and the shapes are built from square cells.
- The built-in quality mesh generator is 2D and isotropic: thin layers
  need many small triangles. It can refine where the solution curves,
  but you run the loop of meshing and solving yourself.
- Tunneling between bands follows straight paths in one direction, not
  paths that bend with the field. Models that depend on each other, such
  as tunneling and heat, are solved by taking turns (one to five rounds
  per voltage in the tests).
- The heat carried by electrons, and their thermal diffusion, are by
  default in a simple form that comes out too small where the electrons
  are very dense; the Fermi-Dirac form is an option, still for one
  energy-independent scattering time.
- The polar optical phonons of the scattering lifetimes are bulk ones,
  or those of one interface between two materials (isotropic, one
  oscillator each). Screening by all the levels together is an option
  (static, and only among the levels you pass).
- No material numbers without a source: where a number is missing, you
  supply it with its source.
- There is no paid support. Questions and bug reports go to the
  [issue tracker](https://github.com/TaN-MM-Org/kpenvelope/issues).

The full list is in [Limits](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/limits.md).

## Citing

If kpenvelope helps your work, please cite it with the concept DOI
[10.5281/zenodo.22015269](https://doi.org/10.5281/zenodo.22015269),
which always points to the latest version ([CITATION.cff](https://github.com/TaN-MM-Org/kpenvelope/blob/main/CITATION.cff)
has the details). The method behind the hole calculations is described
in

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Origin of the
> conflicting hole masses in the GaN/AlN two-dimensional hole gas",
> Phys. Rev. B 114, L291304 (2026), https://doi.org/10.1103/zn44-nmx8

and follows S. L. Chuang and C. S. Chang, Phys. Rev. B 54, 2491 (1996).
The other sources each part builds on are listed in the
[Reference](https://github.com/TaN-MM-Org/kpenvelope/blob/main/docs/reference.md#where-it-comes-from).

## Who makes it

Written and maintained by Tanvir Mahmud Mahim (Department of Electrical
and Electronic Engineering, BRAC University), who reviews every change.
Contributions are welcome: see
[CONTRIBUTING.md](https://github.com/TaN-MM-Org/kpenvelope/blob/main/CONTRIBUTING.md). A
change to the physics comes with a test, and a new number comes with
its source.

Licensed under Apache-2.0.
