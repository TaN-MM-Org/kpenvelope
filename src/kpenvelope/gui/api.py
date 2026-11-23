"""The calculations behind the kpenvelope app, as plain functions of a
JSON-like description of the sample (no web code here, so every one is
tested directly).

A sample ("spec") is a dict:

    {"parameter_set": "pbesol-mahim2026",
     "temperature": 300,
     "layers": [   # bottom first
        {"material": "GaN", "x": 0, "thickness": 1000, "relaxation": 0,
         "doping": "n", "doping_cm3": 1e16},
        {"material": "AlGaN", "x": 0.25, "thickness": 25, "relaxation": 0}],
     "line_charge": {"mode": "measured"},          # only for relaxed layers
     "surface_barrier": 1.0,                        # eV, electron tab
     "band_numbers": {"ScAlN": {"eg":..., "ec":..., "me":..., "source": "..."}}}

Every function returns a dict that the page shows; errors come back as
{"error": "<plain sentence>"} from `handle`.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..driftdiffusion import (Contact, Transport, cv_curve, cv_curve_quantum,
                              iv_curve)
from ..electrons import KB as KB_EV, solve_electrons
from ..interfaces import StackLayer, interface_charges
from ..linecharge import (PUBLISHED_CONSTRAINTS, combine_line_charge,
                          plan_holography)
from ..nitride import algan, gan, inaln, ingan, scaln, with_numbers
from ..uncertainty import propagate

__all__ = ["handle", "options", "build_layers", "run_charges",
           "run_electrons", "run_device", "run_uncertainty",
           "run_line_charge", "run_plan", "python_code"]

#: plain-language names of the parameter sets, and what each one covers
SETS = {
    "pbesol-mahim2026": {
        "label": "Mahim 2026 (PBEsol calculations)",
        "materials": ["GaN", "AlGaN"],
        "about": "Calculated polarization numbers for GaN and AlN, with "
                 "every input file archived. Used in the misfit-dislocation "
                 "paper."},
    "hse-dreyer2016": {
        "label": "Dreyer et al. 2016 (HSE calculations)",
        "materials": ["GaN", "AlGaN", "InGaN", "InAlN"],
        "about": "Widely used modern calculated numbers for GaN, AlN and "
                 "InN (Phys. Rev. X 6, 021038)."},
    "vurgaftman2003": {
        "label": "Vurgaftman and Meyer 2003 (recommended values)",
        "materials": ["GaN", "AlGaN", "InGaN", "InAlN"],
        "about": "The classic review's recommended numbers for GaN, AlN "
                 "and InN (J. Appl. Phys. 94, 3675)."},
    "ambacher2021": {
        "label": "Ambacher et al. 2021 (ScAlN)",
        "materials": ["GaN", "ScAlN"],
        "about": "Composition fits for ScAlN up to 50 percent Sc (J. Appl. "
                 "Phys. 130, 045102). GaN can only be the bottom layer in "
                 "this set. ScAlN has no band numbers here: you enter them."},
}

MATERIALS = {
    "GaN": {"label": "GaN", "x_name": None, "x_max": 0.0},
    "AlGaN": {"label": "AlGaN", "x_name": "aluminum", "x_max": 1.0},
    "InGaN": {"label": "InGaN", "x_name": "indium", "x_max": 1.0},
    "InAlN": {"label": "InAlN", "x_name": "indium", "x_max": 1.0},
    "ScAlN": {"label": "ScAlN", "x_name": "scandium", "x_max": 0.5},
}

LINE_MODES = {
    "measured": "Measured value: -0.84 +- 0.53 charges per nm (fit of the "
                "Shen et al. series; measured on Al0.22Ga0.78N on GaN)",
    "none": "No charge on the dislocation lines",
    "electron-counting": "Electron counting (lines nearly neutral)",
    "bare": "Bare line ends (the largest possible charge)",
    "custom": "My own value",
}


def options():
    """What the page needs to build its menus."""
    return {"sets": SETS, "materials": MATERIALS, "line_modes": LINE_MODES,
            "constraints": {k: {"name": c.name, "kind": c.kind,
                                "value": c.value, "sigma": c.sigma,
                                "applies_to": c.applies_to,
                                "assumptions": c.assumptions}
                            for k, c in PUBLISHED_CONSTRAINTS.items()}}


# ----------------------------------------------------------------- helpers
class SpecError(ValueError):
    """A problem with the sample description, said in plain words."""


def _num(d, key, default=None, lo=None, hi=None, what=None):
    v = d.get(key, default)
    what = what or key
    what = what[0].upper() + what[1:]
    if v is None or v == "":
        raise SpecError(f"Please fill in {what[0].lower() + what[1:]}.")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise SpecError(f"{what} must be a number.") from None
    if not math.isfinite(v):
        raise SpecError(f"{what} must be a finite number.")
    if lo is not None and v < lo:
        raise SpecError(f"{what} must be at least {lo:g}.")
    if hi is not None and v > hi:
        raise SpecError(f"{what} must be at most {hi:g}.")
    return v


def _material(layer, pset, T, band_numbers):
    name = layer.get("material")
    if name not in MATERIALS:
        raise SpecError(f"Unknown material {name!r}.")
    if name not in SETS[pset]["materials"]:
        raise SpecError(f"The number set '{SETS[pset]['label']}' has no "
                        f"{name}. Pick another set, or another material.")
    x = 0.0
    if name != "GaN":
        x = _num(layer, "x", 0.0, 0.0, MATERIALS[name]["x_max"],
                 f"the {MATERIALS[name]['x_name']} fraction of {name}")
    if name == "GaN":
        m = gan(pset, T)
    elif name == "AlGaN":
        m = algan(x, T, polarization=pset)
    elif name == "InGaN":
        m = ingan(x, T, polarization=pset)
    elif name == "InAlN":
        m = inaln(x, T, polarization=pset)
    else:
        m = scaln(x)
        bn = (band_numbers or {}).get("ScAlN")
        if bn:
            src = str(bn.get("source", "")).strip()
            if not src:
                raise SpecError("Please say where your ScAlN band numbers "
                                "come from (the 'source' box).")
            kw = {}
            for k, what in (("eg", "the ScAlN band gap"),
                            ("ec", "the ScAlN conduction-band offset"),
                            ("me", "the ScAlN electron mass")):
                if bn.get(k) not in (None, ""):
                    kw[k] = _num(bn, k, what=what)
            if kw:
                m = with_numbers(m, src, **kw)
    return m


def build_layers(spec):
    """StackLayer list (bottom first) from a sample description."""
    pset = spec.get("parameter_set", "pbesol-mahim2026")
    if pset not in SETS:
        raise SpecError(f"Unknown number set {pset!r}.")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0, "the temperature (K)")
    layers = spec.get("layers") or []
    if len(layers) < 1:
        raise SpecError("Add at least one layer.")
    out = []
    for i, L in enumerate(layers):
        where = f"layer {i + 1} ({L.get('material', '?')})"
        m = _material(L, pset, T, spec.get("band_numbers"))
        t = _num(L, "thickness", None, 1e-3, 1e6, f"the thickness of {where} (nm)")
        r = _num(L, "relaxation", 0.0, 0.0, 100.0,
                 f"the relaxation of {where} (percent)") / 100.0
        if i == 0 and r != 0.0:
            raise SpecError("The bottom layer is the substrate or buffer: "
                            "leave its relaxation at 0.")
        dop = L.get("doping", "none")
        conc = 0.0
        if dop in ("n", "p"):
            conc = _num(L, "doping_cm3", 0.0, 0.0, 1e22,
                        f"the doping of {where} (per cm3)") * 1e-21
        elif dop not in ("none", None, ""):
            raise SpecError(f"Doping must be none, n or p ({where}).")
        depth = L.get("doping_depth")
        depth = None if depth in (None, "") else _num(
            L, "doping_depth", lo=1e-4, what=f"the dopant depth of {where} (eV)")
        kw = {}
        if dop == "n":
            kw = dict(donors=conc, donor_depth=depth)
        elif dop == "p":
            kw = dict(acceptors=conc, acceptor_depth=depth)
        out.append(StackLayer(m, t, relaxation=r, **kw))
    return out


def _line_charge(spec):
    lc = spec.get("line_charge") or {"mode": "measured"}
    mode = lc.get("mode", "measured")
    if mode == "measured":
        c = PUBLISHED_CONSTRAINTS["shen-density-pbesol"]
        return c.value, c.sigma
    if mode in ("none", "neutral"):
        return "neutral", 0.0
    if mode in ("bare", "electron-counting"):
        return mode, 0.0
    if mode == "custom":
        return (_num(lc, "value", what="your line charge (charges per nm)"),
                _num(lc, "sigma", 0.0, 0.0, what="its error"))
    raise SpecError(f"Unknown line-charge choice {mode!r}.")


def _relaxed(layers):
    return any(L.relaxation > 0 for L in layers)


def _charges(layers, spec):
    lam, _ = _line_charge(spec) if _relaxed(layers) else (None, 0.0)
    ref = spec.get("reference", "H")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ic = interface_charges(layers, line_charge=lam, reference=ref)
    return ic, [str(x.message) for x in w]


# ----------------------------------------------------------------- actions
def run_charges(spec):
    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    rows = []
    for c in ic:
        below, above = layers[c.index].label, layers[c.index + 1].label
        s = c.sigma_cm2
        rows.append({
            "between": f"{below} (below) and {above} (above)",
            "depth_nm": c.z, "charge_cm2": s,
            "meaning": ("attracts electrons" if s > 0 else
                        "attracts holes" if s < 0 else "no fixed charge"),
            "mismatch_percent": 100 * c.eps_c,
            "relaxation_percent": 100 * c.relaxation,
            "from_lines_cm2": c.lines * 6.241509074460763e14,
            "coherent_cm2": c.coherent * 6.241509074460763e14,
        })
    _, lam_sigma = _line_charge(spec) if _relaxed(layers) else (None, 0.0)
    return {"interfaces": rows, "warnings": warn,
            "line_charge_sigma": lam_sigma,
            "layers": [{"label": L.label, "thickness": L.thickness,
                        "a_nm": L.material.a, "reference": L.material.reference}
                       for L in layers]}


def _window(layers, spec):
    tot = sum(L.thickness for L in layers)
    depth = spec.get("quantum_depth")
    if depth in (None, ""):
        # the top 120 nm, or the whole stack if thinner
        depth = min(tot, 120.0)
    depth = _num({"d": depth}, "d", lo=1.0, what="the quantum region depth (nm)")
    return (max(0.0, tot - depth), tot)


def run_electrons(spec):
    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    phi = _num(spec, "surface_barrier", None, -2.0, 10.0,
               "the surface barrier (eV)")
    dz = _num(spec, "dz", 0.2, 0.02, 5.0, "the grid step (nm)")
    win = _window(layers, spec)
    if (win[1] - win[0]) / dz > 2500:
        raise SpecError("The quantum region is too large for this grid step; "
                        "make it thinner or the step larger.")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    r = solve_electrons(layers, phi_B=phi, temperature_K=T, interfaces=ic,
                        dz=dz, quantum_window=win, classical_outside=True)
    step = max(1, len(r.z) // 1500)
    return {"z": r.z[::step].tolist(), "ec": r.ec[::step].tolist(),
            "n_cm3": (r.n[::step] * 1e21).tolist(),
            "sheet_cm2": r.sheet_quantum_cm2,
            "sheet_classical_cm2": r.sheet_classical_cm2,
            "subbands_eV": [float(e) for e in r.subband_energies[:6]],
            "converged": r.converged, "warnings": warn,
            "window": list(win),
            "interfaces": [c.z for c in ic]}


def run_mobility(spec):
    """Mobility of the electron layer of the Electrons tab (new in 0.29.0):
    the subbands of its exact (quantum) region, scattered within and between
    them (`kpenvelope.scattering.Subbands`) by the mechanisms ticked in
    spec["scattering"]: "polar" (GaN's polar optical phonons, the
    package's cited Barker-Ilegems set; only for a GaN channel),
    "interface_phonons" {lo_meV, eps_static, eps_high, source} (since
    0.31.0, in place of "polar": the polar phonons of the heterostructure,
    GaN on its side of the boundary next to the electrons, your numbers on
    the other, and the interface modes between them; with "all_layers"
    true, since 0.34.0, every layer and boundary of the exact region, and
    optional numbers spacer_lo_meV, spacer_eps_static, spacer_eps_high,
    spacer_source for the layer just past the boundary next to the
    electrons; since 0.35.0 the air above the top surface when the exact
    region reaches it, and an optional second vibration of the barrier,
    lo2_meV and to2_meV, for alloys with two), "donors"
    (the ionized donors of the stack, no new numbers), "roughness" {rms_nm,
    corr_nm, source} at the boundary next to the electrons, "acoustic"
    {deformation_eV, density, velocity, source}. Numbers you type need
    their source. The donors and the roughness are screened by the
    dielectric matrix of all the levels (screening="full", since 0.30.0)."""
    from ..device import column_states
    from ..phonons import GAN_BARKER1973
    from ..scattering import Subbands
    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    phi = _num(spec, "surface_barrier", None, -2.0, 10.0, "the surface barrier (eV)")
    dz = _num(spec, "dz", 0.2, 0.02, 5.0, "the grid step (nm)")
    win = _window(layers, spec)
    if (win[1] - win[0]) / dz > 2500:
        raise SpecError("The quantum region is too large for this grid step; "
                        "make it thinner or the step larger.")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    sc = spec.get("scattering") or {}
    r = solve_electrons(layers, phi_B=phi, temperature_K=T, interfaces=ic,
                        dz=dz, quantum_window=win, classical_outside=True)
    if not r.sheet_quantum_cm2 > 1e9:
        raise SpecError("There are almost no free electrons in the exact region, "
                        "so there is no electron layer to find a mobility for.")
    inw = (r.z >= win[0] - 1e-9) & (r.z <= win[1] + 1e-9)
    z = r.z[inw]
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    lay = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(layers) - 1)
    mass = np.array([layers[i].material.me for i in lay])
    hz = z[1] - z[0]
    E, p2, w, psi = column_states(z, r.ec[inw], mass, 8, z[0] - hz, z[-1] + hz,
                                  e_max=12 * KB_EV * T, signed=True)
    if E.size == 0:
        raise SpecError("No electron level lies near the filling level.")
    # the levels that hold at least 0.5 percent of the electrons (at most five)
    occ = np.logaddexp(0.0, -E / (KB_EV * T))
    keep = np.flatnonzero(occ >= 0.005 * occ.sum())[:5]
    E, p2, psi = E[keep], p2[:, keep], psi[:, keep]
    peak = int(np.argmax(p2[:, 0]))
    ch = layers[lay[peak]]
    sb = Subbands(z, psi, E, ch.material.me, ch.material.eps_r, fermi=0.0,
                  temperature_K=T, weights=w, screening="full")
    mechs, names = [], []
    iface = sc.get("interface_phonons")
    if sc.get("polar") or iface:
        if not (ch.material.name == "GaN" or ch.material.name.startswith("GaN")
                or getattr(ch.material, "name", "") == algan(0.0).name):
            raise SpecError("Polar optical phonons need the phonon numbers of the "
                            "electrons' layer; the app has them only for GaN.")
        ph = GAN_BARKER1973
        gan_ph = (ph.w_LO_perp / 1000.0, ph.eps0_perp, ph.eps_inf_perp)
        ref = "GaN, " + ph.reference.split(";")[0]
    if iface:
        # the vibrations of the heterostructure: the GaN side, the other
        # side of the boundary next to the electrons (your numbers), and
        # the interface modes between them
        src = str(iface.get("source", "")).strip()
        if not src:
            raise SpecError("Please say where the barrier's vibration numbers come from.")
        other = (_num(iface, "lo_meV", lo=1.0, hi=500.0,
                      what="the barrier's LO phonon energy (meV)") / 1000.0,
                 _num(iface, "eps_static", lo=1.0, hi=100.0,
                      what="the barrier's static permittivity"),
                 _num(iface, "eps_high", lo=1.0, hi=100.0,
                      what="the barrier's high-frequency permittivity"))
        if not other[1] > other[2]:
            raise SpecError("The barrier's static permittivity must be larger than its "
                            "high-frequency one.")
        two_mode = False
        if any(str(iface.get(k, "")).strip() for k in ("lo2_meV", "to2_meV")):
            # an alloy barrier with two vibrations (since 0.35.0): the second
            # LO and TO as given, the first TO from Lyddane-Sachs-Teller
            from ..scattering import _mat
            L2 = _num(iface, "lo2_meV", lo=1.0, hi=500.0,
                      what="the barrier's second LO phonon energy (meV)") / 1000.0
            T2 = _num(iface, "to2_meV", lo=1.0, hi=500.0,
                      what="the barrier's second TO phonon energy (meV)") / 1000.0
            L1, es_, ei_ = other
            T1 = math.sqrt(ei_ * L1 * L1 * L2 * L2 / (es_ * T2 * T2))
            other = {"eps_high": ei_, "lo_eV": [L1, L2], "to_eV": [T1, T2]}
            try:
                _mat(other)
            except ValueError:
                raise SpecError(
                    "The barrier's two vibrations do not fit together: with the "
                    f"permittivities given, the first TO comes out at {1000 * T1:.1f} meV, "
                    "and each TO must lie below its LO, the first pair below the "
                    "second.") from None
            two_mode = True
        zb = np.array([e_ for e_ in edges[1:-1] if win[0] < e_ < win[1]])
        if zb.size == 0:
            raise SpecError("No layer boundary lies in the exact region.")
        zi = float(zb[np.argmin(np.abs(zb - z[peak]))])
        below_is_gan = z[peak] < zi
        spacer = None
        if any(str(iface.get(k, "")).strip() for k in
               ("spacer_lo_meV", "spacer_eps_static", "spacer_eps_high", "spacer_source")):
            src2 = str(iface.get("spacer_source", "")).strip()
            if not src2:
                raise SpecError("Please say where the numbers of the layer next to the "
                                "electrons come from.")
            spacer = (_num(iface, "spacer_lo_meV", lo=1.0, hi=500.0,
                           what="the LO phonon energy of the layer next to the "
                                "electrons (meV)") / 1000.0,
                      _num(iface, "spacer_eps_static", lo=1.0, hi=100.0,
                           what="the static permittivity of the layer next to the electrons"),
                      _num(iface, "spacer_eps_high", lo=1.0, hi=100.0,
                           what="the high-frequency permittivity of the layer next to "
                                "the electrons"))
            if not spacer[1] > spacer[2]:
                raise SpecError("The static permittivity of the layer next to the "
                                "electrons must be larger than its high-frequency one.")
            if not iface.get("all_layers"):
                raise SpecError("Numbers for the layer next to the electrons need "
                                "'every layer of the exact region' ticked.")
        modes, layered, far, far_src = None, False, other, src
        if two_mode and not iface.get("all_layers"):
            # one boundary, but interface_phonons has one vibration per side
            pair = [gan_ph, other] if below_is_gan else [other, gan_ph]
            modes, layered = [sb.layered_phonons([zi], pair)], True
            names.append(f"vibrations of the GaN side, the barrier side and the boundary "
                         f"({ref}; {src})")
        if iface.get("all_layers"):
            # every layer of the exact region (since 0.34.0): GaN layers get
            # GaN's numbers, the layer just past the boundary next to the
            # electrons gets the spacer's numbers if given, every other layer
            # the barrier's; neighbors with the same numbers are merged
            j = int(np.argmin(np.abs(edges - zi)))
            k_next = j if below_is_gan else j - 1
            seg = [win[0]] + list(zb) + [win[1]]
            ifs, mats = [], []
            for a, b in zip(seg[:-1], seg[1:]):
                k = int(np.clip(np.searchsorted(edges, 0.5 * (a + b), side="right") - 1,
                                0, len(layers) - 1))
                nm_ = layers[k].material.name
                if nm_ == "GaN" or nm_.startswith("GaN") or nm_ == algan(0.0).name:
                    m_ = gan_ph
                elif spacer is not None and k == k_next:
                    m_ = spacer
                else:
                    m_ = other
                if mats and m_ == mats[-1]:
                    continue
                if mats:
                    ifs.append(a)
                mats.append(m_)
            if spacer is not None and spacer not in mats:
                raise SpecError("The layer next to the electrons is GaN and has GaN's "
                                "numbers; leave the numbers for that layer empty.")
            if len(mats) == 2 and spacer is not None:
                far, far_src = spacer, src2      # one boundary, the spacer beyond it
            # the air above the top surface (since 0.35.0), when the exact
            # region reaches it
            if win[1] >= edges[-1] - 1e-9:
                ifs.append(float(edges[-1]))
                mats.append(1.0)
            plain = len(mats) == 2 and 1.0 not in mats and not two_mode
            if len(mats) >= 2 and not plain:
                modes, layered = [sb.layered_phonons(ifs, mats)], True
                parts = [ref, src] + ([src2] if spacer is not None else [])
                names.append("vibrations of every layer and boundary of the exact region ("
                             + "; ".join(parts) + ")")
        if modes is None:
            modes = sb.interface_phonons(zi, gan_ph if below_is_gan else far,
                                         far if below_is_gan else gan_ph)
        for m in modes:
            mechs.append(m)
            if layered:
                continue
            if m.name.startswith("half-space"):
                side_gan = (m.name.endswith("below")) == below_is_gan
                names.append(f"vibrations of the GaN side ({ref})" if side_gan
                             else f"vibrations of the barrier side ({far_src})")
            else:
                names.append(f"vibrations of the interface itself at {m.name.split()[1]} meV")
    elif sc.get("polar"):
        mechs.append(sb.polar_optical(*gan_ph))
        names.append("polar optical phonons (" + ref + ")")
    if sc.get("donors"):
        prof = r.donors_ionized[inw] * 1e21
        if np.any(prof > 0):
            mechs.append(sb.impurities(profile_cm3=prof))
            names.append("ionized donors in the layers")
    rough = sc.get("roughness")
    if rough:
        src = str(rough.get("source", "")).strip()
        if not src:
            raise SpecError("Please say where your roughness numbers come from.")
        rms = _num(rough, "rms_nm", lo=0.0, hi=5.0, what="the roughness height (nm)")
        corr = _num(rough, "corr_nm", lo=0.01, hi=100.0,
                    what="the roughness correlation length (nm)")
        # the layer boundary nearest the electrons' peak, and its band step
        zb = np.array([e_ for e_ in edges[1:-1] if win[0] < e_ < win[1]])
        if zb.size == 0:
            raise SpecError("No layer boundary lies in the exact region.")
        zi = float(zb[np.argmin(np.abs(zb - z[peak]))])
        k = int(np.searchsorted(edges, zi))
        step = abs(layers[k].material.ec - layers[k - 1].material.ec)
        mechs.append(sb.roughness(zi, step, rms, corr))
        names.append(f"interface roughness ({src})")
    ac = sc.get("acoustic")
    if ac:
        src = str(ac.get("source", "")).strip()
        if not src:
            raise SpecError("Please say where your acoustic phonon numbers come from.")
        mechs.append(sb.acoustic(_num(ac, "deformation_eV", lo=0.0, hi=100.0,
                                      what="the deformation potential (eV)"),
                                 _num(ac, "density", lo=100.0, hi=30000.0,
                                      what="the mass density (kg/m3)"),
                                 _num(ac, "velocity", lo=100.0, hi=30000.0,
                                      what="the sound velocity (m/s)")))
        names.append(f"acoustic phonons ({src})")
    if not mechs:
        raise SpecError("Tick at least one way the electrons scatter.")
    res = sb.mobility(*mechs)
    lt = sb.lifetimes(*mechs)
    return {"mobility_cm2Vs": res["mobility_cm2Vs"],
            "each": [{"name": nm, "mobility_cm2Vs": m} for nm, m in zip(names, res["each"])],
            "subbands": [{"energy_eV": float(e), "sheet_cm2": float(n_),
                          "mobility_cm2Vs": float(m), "quantum_lifetime_s": q}
                         for e, n_, m, q in zip(E, res["sheets_cm2"], res["subbands"],
                                                lt["quantum_s"])],
            "sheet_cm2": float(res["sheets_cm2"].sum()),
            "sheet_all_cm2": float(r.sheet_quantum_cm2), "warnings": warn,
            "channel": ch.label, "interface": bool(iface)}


def _transport(spec, n_layers):
    t = spec.get("transport") or {}
    src = str(t.get("source", "")).strip()
    if not src:
        raise SpecError("Please say where your transport numbers come from "
                        "(the 'source' box).")
    tr = Transport(
        _num(t, "mobility_n", what="the electron mobility (cm2/Vs)", lo=1e-6),
        _num(t, "mobility_p", what="the hole mobility (cm2/Vs)", lo=1e-6),
        _num(t, "hole_mass", what="the hole mass (in electron masses)", lo=1e-3),
        reference=src,
        tau_n=(math.inf if t.get("lifetime") in (None, "") else
               _num(t, "lifetime", lo=1e-18, what="the lifetime (s)")),
        tau_p=(math.inf if t.get("lifetime") in (None, "") else
               _num(t, "lifetime", lo=1e-18, what="the lifetime (s)")))
    return [tr] * n_layers


def _voltages(dev):
    v0 = _num(dev, "v_start", what="the start voltage")
    v1 = _num(dev, "v_stop", what="the stop voltage")
    st = _num(dev, "v_step", what="the voltage step", lo=1e-4)
    n = int(math.floor(abs(v1 - v0) / st + 1e-9)) + 1
    if n > 201:
        raise SpecError("That is more than 200 voltage points; use a larger step.")
    s = 1.0 if v1 >= v0 else -1.0
    return [v0 + s * k * st for k in range(n)]


def run_device(spec):
    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    dev = spec.get("device") or {}
    kind = dev.get("contact", "metal")
    if kind == "metal":
        top = Contact("schottky", barrier=_num(dev, "barrier", what="the metal barrier (eV)"))
    else:
        top = Contact("ohmic")
    V = _voltages(dev)
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    dz = _num(dev, "dz", 0.5, 0.05, 20.0, "the grid step (nm)")
    kw = dict(temperature_K=T, interfaces=ic, dz=dz)
    mode = dev.get("mode", "cv")
    if mode == "cv":
        bottom = (Contact("zero-field") if dev.get("bottom", "thick") == "thick"
                  else Contact("ohmic"))
        hm = None
        if any(L.acceptors > 0 for L in layers):
            t = spec.get("transport") or {}
            if not str(t.get("source", "")).strip():
                raise SpecError("Your sample has p-type layers, so holes are "
                                "needed: enter the hole mass and its source.")
            hm = _num(t, "hole_mass", what="the hole mass (in electron masses)",
                      lo=1e-3)
        tot = sum(L.thickness for L in layers)
        lo = max(0.0, tot - min(tot, 150.0))
        if dev.get("quantum"):
            if top.kind != "schottky" or bottom.kind != "zero-field":
                raise SpecError("Quantum C-V works with a metal gate on top and "
                                "a thick base at the bottom.")
            if hm is not None:
                raise SpecError("Quantum C-V has no holes, so it cannot treat "
                                "p-type layers; untick it.")
            if len(V) > 41:
                raise SpecError("Quantum C-V is slow: use at most 41 voltage "
                                "points (a larger step).")
            V, C, Q, res = cv_curve_quantum(
                layers, V, barrier=top.barrier, temperature_K=T, interfaces=ic,
                quantum_window=_window(layers, spec))
            sheets = []
            for r in res:
                m = (r.z >= lo - 1e-9)
                zz, nn = r.z[m], r.n[m]
                sheets.append(float(np.sum(0.5 * (nn[1:] + nn[:-1]) * np.diff(zz)) * 1e14))
            return {"mode": "cv", "v": list(map(float, V)),
                    "c_nF_cm2": (C * 1e9).tolist(), "sheet_cm2": sheets,
                    "quantum": True, "warnings": warn}
        V, C, Q, res = cv_curve(layers, V, top=top, bottom=bottom,
                                hole_mass=hm, **kw)
        return {"mode": "cv", "v": list(map(float, V)),
                "c_nF_cm2": (C * 1e9).tolist(),
                "sheet_cm2": [r.sheet(lo, tot) for r in res],
                "warnings": warn}
    if mode == "leak":
        from ..leakage import gate_leakage
        if top.kind != "schottky":
            raise SpecError("Gate leakage needs a metal gate on top.")
        if any(L.acceptors > 0 for L in layers):
            raise SpecError("Gate leakage is worked out for n-type stacks; "
                            "this one has p-type layers.")
        V, J, res = gate_leakage(layers, top.barrier, V, temperature_K=T,
                                 interfaces=ic, dz=0.2)
        return {"mode": "leak", "v": list(map(float, V)), "j_A_cm2": J.tolist(),
                "warnings": warn}
    if mode == "iv":
        tr = _transport(spec, len(layers))
        if top.kind == "schottky" and dev.get("thermionic", True):
            top = Contact("schottky", barrier=top.barrier, thermionic=True)
        if dev.get("btbt"):
            # the vertical stack as a device with nonlocal band-to-band
            # tunneling (tunnel junctions, Zener current; new in 0.23.0)
            if len(V) > 61:
                raise SpecError("With band-to-band tunneling use at most 61 voltage "
                                "points (a larger step).")
            d = _vertical_device(spec, layers, ic, T, dz, top, True)
            J, prev = [], None
            for v in V:
                prev = d.solve({"top": float(v)}, guess=prev)
                J.append(-prev.current["top"] / (20e-7 * 0.1))
            return {"mode": "iv", "v": list(map(float, V)), "j_A_cm2": J, "btbt": True,
                    "sources": [TRANSISTOR_SOURCES["btbt"]], "warnings": warn}
        V, J, res = iv_curve(layers, tr, V, top=top, **kw)
        return {"mode": "iv", "v": list(map(float, V)), "j_A_cm2": J.tolist(),
                "warnings": warn}
    raise SpecError("Choose capacitance (C-V), current (I-V) or gate leakage.")


def run_uncertainty(spec):
    """Error bar of the electron sheet density from the errors you give."""
    unc = spec.get("uncertainty") or {}
    base = build_layers(spec)
    _charges(base, spec)
    lam, lam_sigma = _line_charge(spec) if _relaxed(base) else (None, 0.0)
    values, sigmas = {}, {}
    sb = _num(spec, "surface_barrier", None, -2, 10, "the surface barrier (eV)")
    values["barrier"] = sb
    sigmas["barrier"] = _num(unc, "barrier", 0.0, 0.0, what="the error of the barrier")
    numeric_lam = isinstance(lam, float)
    if numeric_lam:
        values["line"] = lam
        sigmas["line"] = lam_sigma
    errs = unc.get("layers") or []
    for i, L in enumerate(spec["layers"]):
        e = errs[i] if i < len(errs) and isinstance(errs[i], dict) else {}
        if L.get("material") != "GaN":
            values[f"x{i}"] = float(L.get("x", 0.0))
            sigmas[f"x{i}"] = _num(e, "x", 0.0, 0.0, what=f"the error of x in layer {i + 1}")
        values[f"t{i}"] = float(L["thickness"])
        sigmas[f"t{i}"] = _num(e, "thickness", 0.0, 0.0,
                               what=f"the error of the thickness of layer {i + 1}")
    if not any(s > 0 for s in sigmas.values()):
        raise SpecError("Give at least one error bar (in the boxes above).")
    dz = _num(spec, "dz", 0.2, 0.02, 5.0, "the grid step (nm)")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)

    def sheet(**v):
        s = dict(spec)
        s["layers"] = [dict(L) for L in spec["layers"]]
        for i, L in enumerate(s["layers"]):
            if f"x{i}" in v:
                L["x"] = min(max(v[f"x{i}"], 0.0), MATERIALS[L["material"]]["x_max"])
            L["thickness"] = max(v[f"t{i}"], 1e-3)
        layers = build_layers(s)
        line = v["line"] if numeric_lam else lam
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = interface_charges(layers, line_charge=line,
                                   reference=spec.get("reference", "H"))
        r = solve_electrons(layers, phi_B=v["barrier"], temperature_K=T,
                            interfaces=ic, dz=dz,
                            quantum_window=_window(layers, s),
                            classical_outside=True)
        return r.sheet_quantum_cm2

    method = unc.get("method", "linear")
    if method == "linear":
        p = propagate(sheet, values, sigmas, method="linear", step_fraction=0.1)
        names = {"barrier": "surface barrier", "line": "dislocation-line charge"}
        budget = []
        for k, share in sorted(p.budget.items(), key=lambda kv: -kv[1]):
            if k.startswith("x"):
                lab = f"composition of layer {int(k[1:]) + 1}"
            elif k.startswith("t"):
                lab = f"thickness of layer {int(k[1:]) + 1}"
            else:
                lab = names[k]
            budget.append({"input": lab, "share": share})
        return {"method": "linear", "value_cm2": float(p.value),
                "sigma_cm2": float(p.sigma), "budget": budget}
    n = int(_num(unc, "samples", 100, 20, 2000, "the number of samples"))
    p = propagate(sheet, values, sigmas, method="monte-carlo", n_samples=n,
                  seed=int(unc.get("seed", 0)))
    return {"method": "monte-carlo", "value_cm2": float(p.value),
            "sigma_cm2": float(p.sigma),
            "percentiles": {str(k): v for k, v in p.percentiles.items()},
            "samples": p.samples.tolist()}


def run_line_charge(spec):
    names = spec.get("constraints") or ["shen-density-pbesol", "shen-mobility"]
    post = combine_line_charge(names)
    step = max(1, len(post.grid) // 400)
    return {"mean": post.mean, "sd": post.sd, "median": post.median,
            "interval68": list(post.interval68),
            "interval95": list(post.interval95), "removed": post.removed,
            "grid": post.grid[::step].tolist(),
            "density": post.density[::step].tolist(), "used": post.used}


def run_plan(spec):
    plan = spec.get("plan") or {}
    pset = spec.get("parameter_set", "pbesol-mahim2026")
    film = _material({"material": plan.get("material", "AlGaN"),
                      "x": plan.get("x", 0.25)}, pset, 300.0, None)
    sub = gan(pset)
    eps = (sub.a - film.a) / film.a
    r = _num(plan, "relaxation", 50.0, 0.1, 100.0, "the relaxation (percent)") / 100
    err = _num(plan, "charge_error", 1e-3, 1e-6, 1.0,
               "the charge precision (C/m2)")
    out = {"mismatch_percent": 100 * eps,
           "holography_sigma": plan_holography(film, eps, r, err)}
    if eps > 0:
        out["sum_rule"] = ("For the sum rule you also need the same alloy under "
                           "compression, on a substrate with a smaller lattice.")
    return out


def python_code(spec):
    """A Python script that repeats the electron calculation of the page."""
    pset = spec.get("parameter_set", "pbesol-mahim2026")
    T = spec.get("temperature", 300)
    lines = ["# Made by the kpenvelope app. Every number below is yours or",
             "# from the cited set named in parameter_set.",
             "from kpenvelope import (StackLayer, algan, gan, ingan, inaln, scaln,",
             "                        interface_charges, solve_electrons)",
             "", f"pset = {pset!r}", f"T = {float(T)!r}", "layers = ["]
    for i, L in enumerate(spec.get("layers", [])):
        m = L.get("material")
        x = float(L.get("x", 0) or 0)
        mat = {"GaN": "gan(pset, T)",
               "AlGaN": f"algan({x!r}, T, polarization=pset)",
               "InGaN": f"ingan({x!r}, T, polarization=pset)",
               "InAlN": f"inaln({x!r}, T, polarization=pset)",
               "ScAlN": f"scaln({x!r})"}.get(m, "None")
        args = [mat, repr(float(L.get("thickness", 0)))]
        r = float(L.get("relaxation", 0) or 0) / 100
        if r:
            args.append(f"relaxation={r!r}")
        if L.get("doping") == "n":
            args.append(f"donors={float(L.get('doping_cm3', 0)) * 1e-21!r}")
        if L.get("doping") == "p":
            args.append(f"acceptors={float(L.get('doping_cm3', 0)) * 1e-21!r}")
        lines.append(f"    StackLayer({', '.join(args)}),   # layer {i + 1}")
    lines.append("]")
    layers = None
    try:
        layers = build_layers(spec)
    except Exception:          # the script is still useful to edit
        pass
    lc = "None"
    if layers is not None and _relaxed(layers):
        lam, _ = _line_charge(spec)
        lc = repr(lam)
    lines += [f"ic = interface_charges(layers, line_charge={lc})",
              "for c in ic:",
              "    print(c.index, c.sigma_cm2, 'charges per cm2')"]
    if spec.get("surface_barrier") not in (None, "") and layers is not None:
        win = _window(layers, spec)
        dz = float(spec.get("dz") or 0.2)
        lines += [f"r = solve_electrons(layers, phi_B={float(spec['surface_barrier'])!r}, "
                  "temperature_K=T, interfaces=ic,",
                  f"                    dz={dz!r}, quantum_window={tuple(win)!r},",
                  "                    classical_outside=True)",
                  "print('electron sheet density:', r.sheet_quantum_cm2, 'per cm2')"]
    return {"code": "\n".join(lines) + "\n"}



def _parse_pairs(text):
    """Two numbers per line (comma, tab, semicolon or space separated);
    lines that do not start with two numbers (headers) are skipped."""
    rows = []
    for line in str(text or "").replace(";", " ").replace(",", " ").replace("\t", " ").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if math.isfinite(a) and math.isfinite(b):
            rows.append((a, b))
    return rows


_C_UNITS = {"nF/cm2": 1e-9, "uF/cm2": 1e-6, "F/cm2": 1.0, "F/m2": 1e-4}


def run_fit(spec):
    """Fit the unknowns the user ticks to a measured C-V curve and/or Hall
    density."""
    from ..datafit import fit_measurements

    fit = spec.get("fit") or {}
    base = build_layers(spec)
    n_layers = len(base)
    unknowns = fit.get("unknowns") or {}
    if not unknowns:
        raise SpecError("Tick at least one unknown to fit.")
    cv = None
    pairs = _parse_pairs(fit.get("data"))
    if pairs:
        if len(pairs) < 2:
            raise SpecError("Paste at least two C-V points (voltage and "
                            "capacitance on each line).")
        units = fit.get("units", "nF/cm2")
        if units in ("pF", "nF", "F"):
            area = _num(fit, "area_um2", lo=1e-6, what="the gate area (um2)") * 1e-8
            scale = {"pF": 1e-12, "nF": 1e-9, "F": 1.0}[units] / area
        elif units in _C_UNITS:
            scale = _C_UNITS[units]
        else:
            raise SpecError("Unknown capacitance unit.")
        V = np.array([a for a, _ in pairs])
        C = np.array([b for _, b in pairs]) * scale
        if np.any(np.diff(np.sort(V)) == 0):
            raise SpecError("Two C-V points have the same voltage.")
        sig = fit.get("sigma")
        sig = None if sig in (None, "") else _num(
            fit, "sigma", lo=1e-30, what="the capacitance error") * scale
        cv = (V, C, sig)
    sheet = None
    hall = fit.get("hall") or {}
    if hall.get("value") not in (None, ""):
        hv = _num(hall, "value", lo=1.0, what="the Hall density (per cm2)")
        hs = hall.get("sigma")
        hs = None if hs in (None, "") else _num(hall, "sigma", lo=1e-6,
                                                 what="the Hall density error")
        sheet = (hv, hs, None)
    if cv is None and sheet is None:
        raise SpecError("Paste a C-V curve or enter a Hall density.")
    if cv is not None and sheet is not None and (cv[2] is None or sheet[1] is None):
        raise SpecError("To fit C-V and Hall data together, give the error of "
                        "both (their units differ).")
    lam, _ = _line_charge(spec) if _relaxed(base) else (None, 0.0)
    start, fixed, lengths, bounds = {}, {}, [], {}
    labels = {}
    for k, v0 in unknowns.items():
        val = _num({"v": v0}, "v", what=f"the starting value of {k}")
        if k in ("barrier", "surface", "parasitic", "line"):
            start[k] = val
            if k in ("barrier", "surface"):
                # an energy step cannot be negative or exceed the gap of
                # the top layer (where the step sits)
                eg = base[-1].material.eg
                bounds[k] = (0.0, eg if math.isfinite(eg) else 10.0)
        elif k[:1] in ("x", "t", "q") and k[1:].isdigit() and 0 <= int(k[1:]) < n_layers:
            i = int(k[1:])
            if k[0] == "x":
                mat = spec["layers"][i].get("material")
                if mat == "GaN":
                    raise SpecError("GaN has no fraction to fit.")
                bounds[k] = (0.0, MATERIALS[mat]["x_max"])
            if k[0] == "t":
                bounds[k] = (0.5, 1e6)
                lengths.append(k)
            if k[0] == "q" and i == 0:
                raise SpecError("An extra charge sits at a boundary: pick a "
                                "layer above the bottom one.")
            start[k] = val
        else:
            raise SpecError(f"Cannot fit {k!r}.")
    names = {"barrier": "the metal energy step (eV)",
             "surface": "the free-surface energy step (eV)",
             "parasitic": "the parasitic capacitance (nF/cm2)",
             "line": "the defect-line charge (charges per nm)"}
    for k in start:
        if k[0] in "xtq" and k[1:].isdigit():
            i = int(k[1:])
            lab = f"layer {i + 1} ({spec['layers'][i].get('material')})"
            labels[k] = {"x": f"the fraction of {lab}",
                         "t": f"the thickness of {lab} (nm)",
                         "q": f"an extra charge below {lab} (1e13 per cm2)"}[k[0]]
        else:
            labels[k] = names[k]
    if cv is not None and "barrier" not in start:
        fixed["barrier"] = _num((spec.get("device") or {}), "barrier",
                                what="the metal energy step (in the Voltage sweep tab)")
    if sheet is not None and "surface" not in start:
        fixed["surface"] = _num(spec, "surface_barrier",
                                what="the surface energy step (Electrons tab)")
    if "line" in start and not _relaxed(base):
        raise SpecError("No layer is relaxed, so there is no defect-line "
                        "charge to fit.")

    def build(v):
        s2 = dict(spec)
        s2["layers"] = [dict(L) for L in spec["layers"]]
        extra = {}
        for k, val in v.items():
            if k[:1] in "xtq" and k[1:].isdigit():
                i = int(k[1:])
                if k[0] == "x":
                    s2["layers"][i]["x"] = val
                elif k[0] == "t":
                    s2["layers"][i]["thickness"] = val
                else:
                    extra[i - 1] = val * 0.1        # 1e13 cm^-2 -> e/nm^2
        layers = build_layers(s2)
        line = v.get("line", lam) if _relaxed(layers) else None
        return {"layers": layers, "line_charge": line, "extra_sheet": extra}

    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    method = "quantum" if fit.get("quantum") else "quasi-equilibrium"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_measurements(build, start, cv=cv, sheet=sheet, fixed=fixed,
                               bounds=bounds, method=method, temperature_K=T,
                               dz=0.5, lengths=lengths)
    rows = [{"name": k, "label": labels[k], "value": res.values[k],
             "error": res.errors[k], "start": start[k]} for k in res.names]
    warn = []
    for w in res.warnings:
        for k, lab in labels.items():
            w = w.replace(f"{k} and ", f"{lab} and ").replace(f"and {k} are", f"and {lab} are")
            if w.startswith(k + " ended"):
                w = lab[0].upper() + lab[1:] + w[len(k):]
        warn.append(w)
    out = {"rows": rows, "warnings": warn, "chi2": res.chi2, "dof": res.dof,
           "scaled": res.scaled, "method": method}
    if cv is not None:
        out["v"] = res.cv_voltage.tolist()
        out["c_meas_nF"] = (res.cv_measured * 1e9).tolist()
        out["c_fit_nF"] = (res.cv_fitted * 1e9).tolist()
    if sheet is not None:
        out["hall_meas"] = res.sheet_measured
        out["hall_fit"] = res.sheet_fitted
    return out



def run_design(spec):
    """Search compositions and thicknesses of one layer for the targets."""
    from ..design import design_error_bars, find_designs

    des = spec.get("design") or {}
    base_spec = dict(spec)
    build_layers(base_spec)                      # checks the sample first
    i = int(_num(des, "layer", what="the layer to design"))
    if not (1 <= i < len(spec["layers"])):
        raise SpecError("Pick a layer above the bottom one to design.")
    mat = spec["layers"][i].get("material")
    variables = {}
    if des.get("vary_x"):
        if not MATERIALS[mat]["x_name"]:
            raise SpecError("GaN has no fraction to vary.")
        lo = _num({"v": des["vary_x"][0]}, "v", 0.0, 0.0, MATERIALS[mat]["x_max"], "the lowest fraction")
        hi = _num({"v": des["vary_x"][1]}, "v", 0.0, 0.0, MATERIALS[mat]["x_max"], "the highest fraction")
        if not hi > lo:
            raise SpecError("The highest fraction must be above the lowest.")
        variables["x"] = (lo, hi)
    if des.get("vary_t"):
        lo = _num({"v": des["vary_t"][0]}, "v", 0.5, 0.5, 1e4, "the smallest thickness (nm)")
        hi = _num({"v": des["vary_t"][1]}, "v", 0.5, 0.5, 1e4, "the largest thickness (nm)")
        if not hi > lo:
            raise SpecError("The largest thickness must be above the smallest.")
        variables["t"] = (lo, hi)
    if not variables:
        raise SpecError("Tick what may change: the fraction, the thickness, or both.")
    targets = {}
    for k, what in (("sheet_bare", "the electron density"),
                    ("sheet_gated", "the electron density under the gate"),
                    ("threshold", "the switch-off voltage")):
        t = (des.get("targets") or {}).get(k)
        if t and t[0] not in (None, ""):
            val = _num({"v": t[0]}, "v", what=what)
            tol = _num({"v": t[1]}, "v", lo=1e-12, what=f"the tolerance of {what}")
            targets[k] = (val, tol)
    if not targets:
        raise SpecError("Give at least one target.")
    barrier = surface = None
    if "sheet_gated" in targets or "threshold" in targets:
        barrier = _num(spec.get("device") or {}, "barrier",
                       what="the metal energy step (Voltage sweep tab)")
    if "sheet_bare" in targets:
        surface = _num(spec, "surface_barrier", what="the surface energy step (Electrons tab)")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)

    def build(v):
        s2 = dict(spec)
        s2["layers"] = [dict(L) for L in spec["layers"]]
        if "x" in v:
            s2["layers"][i]["x"] = v["x"]
        if "t" in v:
            s2["layers"][i]["thickness"] = v["t"]
        return build_layers(s2)

    quick = bool(des.get("quick", True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        designs, mp = find_designs(build, variables, targets, barrier=barrier,
                                   surface=surface, n_grid=5 if quick else 7,
                                   refine=2, top=5, temperature_K=T,
                                   return_map=True)
    if not designs:
        raise SpecError("No design in this range gives electrons; widen the range.")

    def row(d):
        return {"values": {k: float(v) for k, v in d.values.items() if k in variables},
                "sheet_bare": d.sheet_bare, "sheet_gated": d.sheet_gated,
                "threshold": d.threshold, "miss": d.miss, "meets": d.meets_targets}

    out = {"designs": [row(d) for d in designs], "layer": i, "material": mat,
           "targets": {k: list(v) for k, v in targets.items()},
           "map": {"axes": mp["axes"],
                   "score": [None if d is None else d.score for d in mp["designs"]]}}
    sig = des.get("sigmas") or {}
    sigmas = {}
    for k, what in (("x", "fraction"), ("t", "thickness"), ("barrier", "metal step"),
                    ("surface", "surface step")):
        if sig.get(k) not in (None, "", 0, "0"):
            val = _num(sig, k, lo=0.0, what=f"the error of the {what}")
            if val > 0 and (k in variables or (k == "barrier" and barrier is not None)
                            or (k == "surface" and surface is not None)):
                sigmas[k] = val
    if sigmas:
        best = dict(designs[0].values)
        vals = {k: best[k] for k in variables}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eb = design_error_bars(build, vals, sigmas, barrier=barrier,
                                   surface=surface, quantities=tuple(targets),
                                   temperature_K=T)
        out["error_bars"] = {k: {"value": float(p.value), "sigma": float(p.sigma),
                                 "budget": p.budget} for k, p in eb.items()}
    return out



def run_cross(spec):
    """2D cross-section: gate, access regions and the electron layer."""
    from ..twod import Gate, solve_cross_section

    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    cs = spec.get("cross") or {}
    lg = _num(cs, "gate_length", lo=5.0, hi=20000.0, what="the gate length (nm)")
    acc = _num(cs, "access", lo=20.0, hi=20000.0, what="the space beside the gate (nm)")
    vg = _num(cs, "voltage", lo=-50.0, hi=10.0, what="the gate voltage (V)")
    barrier = _num(spec.get("device") or {}, "barrier",
                   what="the metal energy step (Voltage sweep tab)")
    surface = _num(spec, "surface_barrier", what="the surface energy step (Electrons tab)")
    if any(L.acceptors > 0 for L in layers):
        raise SpecError("The cross-section is worked out for n-type stacks; "
                        "this one has p-type layers.")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    width = lg + 2 * acc
    gate = Gate(acc, acc + lg, barrier, vg)
    r = solve_cross_section(layers, gate, width, surface, temperature_K=T,
                            interfaces=ic)
    if not r.converged:
        warn.append("The 2D calculation did not fully settle.")
    tot = r.z[-1]
    depth = min(tot, float(cs.get("depth") or 60.0))
    m = r.z >= tot - depth - 1e-9
    zz = tot - r.z[m]
    xi = np.linspace(0, width, min(160, r.x.size * 2))
    zi = np.linspace(0, depth, 90)
    # resample onto a regular picture grid (linear in x, then in z)
    def resample(field):
        a = np.array([np.interp(xi, r.x, field[:, j]) for j in np.flatnonzero(m)]).T
        zs = zz
        order = np.argsort(zs)
        return np.array([np.interp(zi, zs[order], a[k, order]) for k in range(a.shape[0])])
    logn = resample(np.log10(np.maximum(r.n, 1.0)))
    ec = resample(r.ec)
    from ..driftdiffusion import Contact as _C, solve_quasi_equilibrium as _q
    one_d = _q(layers, _C("schottky", barrier=barrier, voltage=vg), _C("zero-field"),
               temperature_K=T, interfaces=ic, dz=0.25)
    return {"x": xi.tolist(), "depth": zi.tolist(), "log10n": logn.tolist(),
            "ec": ec.tolist(), "sheet_x": r.sheet_x.tolist(), "x_nodes": r.x.tolist(),
            "gate": [gate.x0, gate.x1], "sheet_center": float(np.interp(acc + lg / 2, r.x, r.sheet_x)),
            "sheet_1d": one_d.sheet(max(0.0, tot - 150.0), tot),
            "interfaces": [float(tot - c.z) for c in ic if tot - c.z <= depth],
            "warnings": warn}


# ------------------------------------------------- transistor (2D, current)
#: built-in numbers of the transistor tab, each with its source (opened and
#: read when this was written; see the package documentation)
TRANSISTOR_SOURCES = {
    "mobility": "1704 cm²/Vs measured at room temperature, falling as "
                "(T/300 K)^-2.35 (their power-law fit up to 576 °C), for an "
                "AlGaN/GaN electron layer: H. S. Alpert et al., Rev. Sci. "
                "Instrum. 91 (2020), doi:10.1063/1.5139911 (arXiv:1912.00977)",
    "vsat": "saturation speed from the electron density: v = 10^7 / (0.38 + "
            "(n_s / 1.8 x 10^13)^0.45) cm/s, the fit of S. Bajaj et al. "
            "(Appl. Phys. Lett. 107 (2015), doi:10.1063/1.4933181) to their "
            "drifted Fermi-Dirac calculations, which matched their pulsed "
            "measurements",
    "gan_heat": "GaN: 230 W/(m K) at 300 K, falling as (T/300)^-1.43 (Mion "
                "et al. 2006; exponent as quoted by Danilchenko et al. 2007)",
    "sic_heat": "4H-SiC: 370 W/(m K) (Goldberg et al. 2001 via the Ioffe "
                "database), held constant",
    "fe": "iron in GaN: an acceptor 0.5 eV below the conduction band "
          "(Wickramaratne et al., Phys. Rev. B 99, 205202 (2019); Uren et "
          "al., IEEE Trans. Electron Devices 59, 3327 (2012)); capture cross "
          "section 10^-15 cm² and 10^15 per cm³ shallow compensating donors "
          "as in Uren et al.",
    "sin": "silicon nitride: relative permittivity 7 (J. Robertson, Eur. Phys. J. "
           "Appl. Phys. 28, 265 (2004), Table 2)",
    "passivated": "passivated surface: the interface under the passivation keeps the "
                  "charge the bare surface carried (from the 1D solver with your "
                  "surface energy step); any extra charge your film adds is not "
                  "included",
    "no_heat_cap": "heat flow through the passivation is neglected (it is thin and "
                   "conducts heat poorly)",
    "impact": "GaN impact ionization: alpha_n = 2.11e9 exp(-3.689e7/E), alpha_p = "
              "4.39e6 exp(-1.8e7/E) per cm, E in V/cm: D. Ji, B. Ercan and S. "
              "Chowdhury, Appl. Phys. Lett. 115, 073503 (2019), "
              "doi:10.1063/1.5099245, applied in the GaN layers only (no numbers "
              "are built in for AlGaN, which is taken as not ionizing); breakdown "
              "when the ionization integral along the field line through the "
              "highest field in GaN reaches 1",
    "mason": "f_max from Mason's unilateral gain U (S. J. Mason, IRE Trans. Circuit "
             "Theory 1, 20 (1954)); f_T where the current gain |h21| falls to 1",
    "gap_T": "band gap against temperature: the Varshni numbers of I. Vurgaftman and "
             "J. R. Meyer, J. Appl. Phys. 94, 3675 (2003), the ones the layer records "
             "use; the change is shared evenly between the two bands (an assumption, "
             "no source gives the split)",
    "peltier": "Peltier and Thomson heat: the thermodynamic heat of G. K. Wachutka, "
               "IEEE Trans. Comput.-Aided Des. 9, 1141 (1990), with the package's own "
               "thermoelectric power",
    "thermionic": "electrons cross the barrier step by thermionic emission (the form "
                  "of K. Yang, J. R. East and G. I. Haddad, Solid-State Electron. 36, "
                  "321 (1993))",
    "quantum": "electrons in the channel as quantum states: the one-band Schroedinger "
               "equation with each layer's electron mass along every grid column "
               "(kpenvelope's QuantumChannel, checked against its own 1D "
               "Schroedinger-Poisson solver)",
    "btbt": "band-to-band tunneling: Kane's two-band rate (as written in Eq. 13 of "
            "arXiv:2108.09328) on straight paths through the band profile, with the "
            "reduced mass from the electron mass of each layer and your hole mass",
}


def _gap_shifts(spec, layers):
    """Cited gap-temperature numbers for every layer of the stack."""
    from ..nitride import gap_shift
    out = {}
    for L_spec, L in zip(spec.get("layers") or [], layers):
        name = L_spec.get("material")
        if name == "ScAlN":
            raise SpecError("ScAlN has no cited band-gap numbers against temperature; "
                            "untick 'the band gap follows the temperature'.")
        x = 0.0 if name == "GaN" else float(L_spec.get("x") or 0.0)
        out[L.label] = gap_shift(name, x, 0.5)
    return out


def _bajaj_vsat(ns_cm2):
    return 1e7 / (0.38 + (ns_cm2 / 1.8e13) ** 0.45)


def _transistor_device(spec, with_traps=False):
    from ..device import Device, Electrode, Mobility, Surface, Trap, capture_coefficient
    from ..driftdiffusion import Contact as _C, solve_quasi_equilibrium as _q

    layers = build_layers(spec)
    ic, warn = _charges(layers, spec)
    if any(L.acceptors > 0 for L in layers):
        raise SpecError("The transistor tab is worked out for n-type stacks; "
                        "this one has p-type layers.")
    if len(layers) < 2:
        raise SpecError("A transistor needs a barrier layer on top of the base.")
    t = spec.get("transistor") or {}
    ls = _num(t, "contact", 500.0, 50.0, 20000.0, "the contact length (nm)")
    gsg = _num(t, "gap_sg", 1000.0, 20.0, 50000.0, "the source-gate gap (nm)")
    lg = _num(t, "gate_length", 250.0, 10.0, 20000.0, "the gate length (nm)")
    ggd = _num(t, "gap_gd", 1250.0, 20.0, 50000.0, "the gate-drain gap (nm)")
    T = _num(spec, "temperature", 300.0, 1.0, 1500.0)
    gate_barrier = _num(spec.get("device") or {}, "barrier",
                        what="the metal energy step (Voltage sweep tab)")
    surface = _num(spec, "surface_barrier", what="the surface energy step (Electrons tab)")
    top = sum(L.thickness for L in layers)
    implant = top - layers[0].thickness + 20.0
    width = 2 * ls + gsg + lg + ggd
    x_g0 = ls + gsg
    el = [Electrode("source", "ohmic", 0.0, ls, implant_depth=implant, implant_donors=1e20),
          Electrode("gate", "schottky", x_g0, x_g0 + lg, barrier=gate_barrier),
          Electrode("drain", "ohmic", width - ls, width, implant_depth=implant,
                    implant_donors=1e20)]
    rc = _num(t, "contact_res", 0.0, 0.0, 100.0, "the contact resistance (Ω·mm)")
    pas = _num(t, "passivation", 0.0, 0.0, 2000.0, "the passivation thickness (nm)")
    plate = _num(t, "plate", 0.0, 0.0, 50000.0, "the field-plate length (nm)")
    if plate > 0 and pas <= 0:
        raise SpecError("A field plate sits on the passivation: give the passivation "
                        "a thickness.")
    if plate > 0 and x_g0 + lg + plate > width - ls - 20.0:
        raise SpecError("The field plate reaches the drain; make it shorter.")
    cap, extra_src = (), []
    if pas > 0:
        from ..device import Dielectric
        eps_in = t.get("passivation_eps")
        if eps_in in (None, ""):
            eps_p, eps_src = 7.0, TRANSISTOR_SOURCES["sin"]
        else:
            eps_p = _num(t, "passivation_eps", lo=1.0, hi=100.0,
                         what="the passivation's permittivity")
            eps_src = str(t.get("passivation_source") or "").strip() or "yours"
        cap = (Dielectric("passivation", pas, eps_p, eps_src),)
        box = dict(z0=top, z1=top + pas)
        el = [Electrode("source", "ohmic", 0.0, ls, implant_depth=implant,
                        implant_donors=1e20, resistance=rc, **box),
              Electrode("gate", "schottky", x_g0, x_g0 + lg, barrier=gate_barrier, **box),
              Electrode("drain", "ohmic", width - ls, width, implant_depth=implant,
                        implant_donors=1e20, resistance=rc, **box)]
        if plate > 0:
            # one metal with the gate: an L-shaped gate
            el.insert(2, Electrode("gate", "schottky", x_g0, x_g0 + lg + plate,
                                   barrier=gate_barrier, z0=top + pas, z1=top + pas))
        extra_src = [f"passivation: {eps_src}", TRANSISTOR_SOURCES["passivated"]]
    elif rc > 0:
        import dataclasses as _dc
        el = [_dc.replace(e, resistance=rc) if e.kind == "ohmic" else e for e in el]
    mu = _num(t, "mobility", 1704.0, 1.0, 1e5, "the electron mobility (cm²/Vs)")
    gamma = _num(t, "mobility_temperature", 2.35, 0.0, 5.0,
                 "how fast the mobility falls with temperature")
    src = str(t.get("mobility_source") or "").strip()
    if not src:
        if mu == 1704.0 and gamma == 2.35:
            src = TRANSISTOR_SOURCES["mobility"]
        else:
            # the user's own numbers: say so rather than naming a paper that
            # does not hold them
            src = (f"{mu:g} cm²/Vs falling as (T/300 K)^-{gamma:g}: your own numbers "
                   "(no source given)")
    one = _q(layers, _C("schottky", barrier=surface), _C("zero-field"),
             temperature_K=T, interfaces=ic, dz=0.5)
    ns = one.sheet(max(0.0, top - 150.0), top)
    vs_in = t.get("vsat")
    if vs_in in (None, "", "auto"):
        if ns < 1e10:
            raise SpecError("This stack has almost no electrons under a free "
                            "surface, so it is not a transistor channel.")
        vs = _bajaj_vsat(ns)
        vs_src = TRANSISTOR_SOURCES["vsat"]
    else:
        vs = _num(t, "vsat", lo=1e5, hi=1e8, what="the saturation speed (cm/s)")
        vs_src = src
    mob = Mobility(mu, f"{src}; v_sat: {vs_src}", saturation_velocity=vs, beta=1.0,
                   temperature_exponent=gamma)
    traps = []
    if with_traps:
        lag = spec.get("lag") or {}
        dens = _num(lag, "density", 1e17, 1e14, 1e19, "the iron density (per cm³)")
        clean = _num(lag, "clean", 100.0, 0.0, layers[0].thickness - 10.0,
                     "the trap-free channel thickness (nm)")
        cn = capture_coefficient(1e-15, layers[0].material.me, T)
        traps.append(Trap("acceptor", 0.5, "ec", dens, cn, 0.0, TRANSISTOR_SOURCES["fe"],
                          where=layers[0].label,
                          z_range=(0.0, layers[0].thickness - clean)))
        donors = _num(lag, "donors", 1e15, 0.0, 1e18,
                      "the compensating donor density (per cm³)")
        import dataclasses as _dc
        layers[0] = _dc.replace(layers[0], donors=layers[0].donors + donors * 1e-21)
    if pas > 0:
        from ..nitride import QE as _QE
        surf = Surface("states", fixed_charge=float(one.gate_charge / _QE))
    else:
        surf = Surface("pinned", barrier=surface)
    extra = {}
    if t.get("full_temp"):
        extra.update(temperature_model="full", band_gap_temperature=_gap_shifts(spec, layers))
        extra_src.append(TRANSISTOR_SOURCES["gap_T"])
    if t.get("peltier"):
        extra["heat_model"] = "thermoelectric"
        extra_src.append(TRANSISTOR_SOURCES["peltier"])
        if t.get("dense_heat"):
            # the heat each electron carries in its Fermi-Dirac form (right
            # for the dense channel electrons; needs Fermi-Dirac statistics,
            # the device's default)
            extra["thermoelectric"] = "fermi-dirac"
    if t.get("thermionic"):
        extra["heterojunctions"] = "thermionic"
        extra_src.append(TRANSISTOR_SOURCES["thermionic"])
    dz_fine = 1.0
    if t.get("quantum"):
        if t.get("thermionic"):
            raise SpecError("Choose either thermionic emission over the barrier step or "
                            "quantum electrons in the channel: the quantum treatment "
                            "already covers the electrons at that step.")
        from ..device import QuantumChannel
        above = sum(L.thickness for L in layers[1:])
        z_int = top - above
        extra["quantum"] = QuantumChannel((max(z_int - 40.0, 1.0), top - 0.5))
        extra_src.append(TRANSISTOR_SOURCES["quantum"])
        dz_fine = 0.5
    dev = Device(layers, width, el, mob, surf, cap=cap,
                 traps=traps, temperature_K=T, interfaces=ic, dx_fine=10.0,
                 dx_max=150.0, dz_fine=dz_fine, dz_max=100.0, fine_depth=min(60.0, top),
                 **extra)
    return dev, dict(layers=layers, warn=warn, width=width, gate=[x_g0, x_g0 + lg],
                     contacts=[[0.0, ls], [width - ls, width]], ns=ns, vsat=vs,
                     mobility=mu, gamma=gamma, sources=[mob.reference] + extra_src,
                     passivation=pas, plate=[x_g0 + lg, x_g0 + lg + plate] if plate else None,
                     contact_res=rc)


def _map(dev, st, depth=100.0, te=None):
    x, z = dev.axes
    top = dev.top                         # the semiconductor's top (below any passivation)
    depth = min(depth, top)
    m = (z >= top - depth - 1e-9) & (z <= top + 1e-9)
    zz = top - z[m]
    xi = np.linspace(0, x[-1], 180)
    zi = np.linspace(0, depth, 90)

    def resample(field):
        a = np.array([np.interp(xi, x, field[:, j]) for j in np.flatnonzero(m)]).T
        order = np.argsort(zz)
        return np.array([np.interp(zi, zz[order], a[k, order]) for k in range(a.shape[0])])

    out = {"x": xi.tolist(), "depth": zi.tolist(),
           "log10n": resample(np.log10(np.maximum(st.n, 1.0))).tolist(),
           "temp": resample(st.temperature).tolist(),
           "ec": resample(st.ec).tolist()}
    if te is not None:
        out["te"] = resample(te).tolist()
    return out


def _fet_thermal(spec, info):
    """The heat-flow model of the transistor tab (None without heating)."""
    from ..thermal import (GAN_BULK, SIC_4H_ROOM, ThermalLayer, ThermalMaterial,
                           ThermalModel, boundary_resistance)
    t = spec.get("transistor") or {}
    sub = t.get("substrate", "SiC")
    thick = _num(t, "substrate_um", 100.0, 1.0, 2000.0, "the substrate thickness (µm)")
    if sub == "SiC":
        mat = SIC_4H_ROOM
    else:
        k = _num(t, "substrate_k", what="the substrate's thermal conductivity (W/m/K)",
                 lo=0.1, hi=5000.0)
        mat = ThermalMaterial("substrate", k, _num(t, "substrate_alpha", 0.0, 0.0, 3.0,
                                                   "how fast it falls with temperature"),
                              str(t.get("substrate_source") or "yours"))
    sublayers = [ThermalLayer("substrate", thick * 1e3, mat)]
    rb = t.get("boundary")
    if rb not in (None, "", 0, "0"):
        rb = _num(t, "boundary", lo=1e-12, hi=1e-5,
                  what="the boundary resistance (m² K/W)")
        sublayers.append(boundary_resistance(rb, "yours"))
    pitch = _num(t, "pitch_um", 50.0, 1.0, 1000.0, "the finger pitch (µm)")
    ext = max(0.0, (pitch * 1e3 - info["width"]) / 2)
    mats = GAN_BULK
    if info.get("passivation"):
        mats = {L.label: GAN_BULK for L in info["layers"]}
        mats["passivation"] = ThermalMaterial(
            "passivation (heat flow neglected)", 1e-3, 0.0,
            TRANSISTOR_SOURCES["no_heat_cap"], density=1e-3, cp_a=1e-3)
    model = ThermalModel(mats, substrate=tuple(sublayers),
                         sink_temperature_K=_num(spec, "temperature", 300.0),
                         lateral_extension=ext)
    src = [TRANSISTOR_SOURCES["gan_heat"],
           TRANSISTOR_SOURCES["sic_heat"] if sub == "SiC" else "substrate: yours"]
    if info.get("passivation"):
        src.append(TRANSISTOR_SOURCES["no_heat_cap"])
    return model, src


def run_transistor(spec):
    """Drain-current curves of a 2D transistor (source, gate, drain), with
    optional self-heating; gate leakage by tunneling and (with an energy
    relaxation time) the electron temperature at the last point."""
    dev, info = _transistor_device(spec)
    t = spec.get("transistor") or {}
    sweep = t.get("sweep", "drain")
    if sweep not in ("drain", "gate"):
        raise SpecError("Choose to sweep the drain or the gate voltage.")
    v0 = _num(t, "from", 0.0, -60.0, 200.0, "the start voltage")
    v1 = _num(t, "to", 10.0, -60.0, 200.0, "the end voltage")
    step = _num(t, "step", 1.0, 0.01, 50.0, "the voltage step")
    n = int(math.floor(abs(v1 - v0) / step + 1e-9)) + 1
    if n > 81:
        raise SpecError("That is more than 80 points; use a larger step.")
    vals = np.linspace(v0, v0 + math.copysign((n - 1) * step, v1 - v0), n)
    other = _num(t, "fixed", 0.0 if sweep == "drain" else 5.0, -60.0, 200.0,
                 "the fixed voltage")
    fixed = {"gate": other} if sweep == "drain" else {"drain": other}
    tau = t.get("tau_e")
    tau = None if tau in (None, "") else _num(t, "tau_e", lo=1e-4, hi=1e3,
                                              what="the energy relaxation time (ps)")
    thermal, heat_src = (None, [])
    if t.get("heating"):
        thermal, heat_src = _fet_thermal(spec, info)
    coupled = bool(t.get("hot_coupled")) and tau is not None
    if coupled:
        V, I, S = dev.sweep(sweep, vals, bias=fixed, thermal=thermal,
                            energy_relaxation=tau * 1e-12)
    else:
        V, I, S = dev.sweep(sweep, vals, bias=fixed, thermal=thermal)
    last = S[-1]
    te = None
    hot = None
    if tau is not None:
        te = (last.electron_temperature if coupled
              else dev.electron_temperature(last, tau * 1e-12))
        # the hottest place where electrons actually are (at least 1e-3 of
        # the densest point; in nearly empty regions the energy balance
        # gives a few stray electrons huge temperatures that carry nothing)
        where = last.n >= 1e-3 * float(np.max(last.n))
        # in the electron layer: the top layers and 20 nm of the base below
        zc = dev.top - (dev.top - info["layers"][0].thickness + 20.0)
        layer = np.broadcast_to((dev.axes[-1] >= zc).reshape(1, -1), te.shape)
        where = where & layer
        k = np.unravel_index(int(np.argmax(np.where(where, te, -np.inf))), te.shape)
        hot = {"peak": float(te[k]), "x": float(dev.axes[0][k[0]]),
               "depth": float(dev.top - dev.axes[-1][k[1]]), "tau_ps": tau,
               "coupled": coupled}
    leak = None
    try:
        Ig, Jc = dev.gate_tunneling(last)
        leak = {"current": Ig, "peak_density": float(np.max(np.abs(Jc)))}
    except ValueError:
        leak = None
    out = {"voltages": V.tolist(), "sweep": sweep, "fixed": other,
           "drain": I["drain"].tolist(), "gate": I["gate"].tolist(),
           "source": I["source"].tolist(),
           "tmax": [float(s.temperature.max()) for s in S],
           "power": [s.power for s in S], "heating": bool(thermal),
           "map": _map(dev, last, te=None if te is None else np.where(
               last.n >= 1e-3 * float(np.max(last.n)), te, last.temperature)),
           "gate_x": info["gate"], "contacts": info["contacts"],
           "plate_x": info["plate"], "passivation": info["passivation"],
           "contact_res": info["contact_res"],
           "width": info["width"], "ns": info["ns"], "vsat": info["vsat"],
           "mobility": info["mobility"], "sources": info["sources"] + heat_src,
           "leak": leak, "hot": hot, "warnings": info["warn"]}
    if tau is not None:
        out["sources"].append(f"energy relaxation time {tau:g} ps: yours")
    if thermal and max(out["tmax"]) > _TOO_HOT:
        out["warnings"] = out["warnings"] + [_too_hot_note(max(out["tmax"]))]
    return out


def run_quantum(spec):
    """The current at one bias with the channel electrons treated as waves
    (non-equilibrium Green's functions, `Device.negf`), next to the
    drift-diffusion current and the ballistic limit (new in 0.30.0). Only
    for short transistors (source to drain at most 600 nm): longer ones take
    too long here and need scattering to settle."""
    t = dict(spec.get("transistor") or {})
    q = spec.get("quantum_transport") or {}
    span = sum(_num(t, k, d) for k, d in (("gap_sg", 1000.0), ("gate_length", 250.0),
                                          ("gap_gd", 1250.0)))
    if span > 600.0:
        raise SpecError("The quantum current is worked out for short transistors only: "
                        "make the two gaps and the gate together at most 600 nm "
                        f"(they are {span:g} nm now).")
    t["quantum"] = True
    dev, info = _transistor_device(dict(spec, transistor=t))
    vg = _num(q, "gate", 0.0, -20.0, 5.0, "the gate voltage")
    vd = _num(q, "drain", 0.05, 0.0, 2.0, "the drain voltage")
    bias = {"gate": vg, "drain": vd}
    dd = dev.solve(bias)
    kw = {}
    if q.get("scattering"):
        # the scattering time that gives the tab's mobility: tau = mu m / q
        from ..electrons import M0
        from ..nitride import QE as _QE
        m = info["layers"][0].material.me
        kw["scattering_time"] = info["mobility"] * 1e-4 * m * M0 / _QE
    if q.get("mixing"):
        # the lowest four states of every column, joined along the channel
        # (coupled mode space, new in 0.31.0 in the app)
        kw["coupled_modes"] = 4
    # with both bumping and mixing levels every column is joined to every
    # other through the bumps: a coarser step along the channel keeps it to
    # a few minutes
    step = 1.0 if (kw.get("scattering_time") and kw.get("coupled_modes")) else 0.5
    try:
        st = dev.negf(bias, guess=dd, dx=step, **kw)
    except RuntimeError as e:
        raise SpecError("The wave calculation did not settle at this bias. Tick "
                        "'with scattering', or move the gate or drain voltage a little. "
                        f"({e})".split(" (NEGF")[0])
    top, _ = dev.ballistic_current(st)
    return {"negf": st.negf["current"], "drift_diffusion": dd.current["drain"],
            "ballistic": float(top), "gate": vg, "drain": vd,
            "scattering_time_s": kw.get("scattering_time"),
            "mixing": bool(kw.get("coupled_modes")),
            "subbands": kw.get("coupled_modes") or len(st.negf["modes"]),
            "iterations": st.negf["iterations"],
            "sources": info["sources"], "warnings": info["warn"]}


def run_rf(spec):
    """Small-signal speed at one bias: current gain, Mason's gain, f_T and
    f_max (with the contact and gate resistances in series)."""
    from ..device import rf_figures, two_port, with_series
    dev, info = _transistor_device(spec)
    rf = spec.get("rf") or {}
    vg = _num(rf, "gate", -3.0, -60.0, 10.0, "the gate voltage")
    vd = _num(rf, "drain", 10.0, 0.0, 200.0, "the drain voltage")
    rg = _num(rf, "gate_res", 0.0, 0.0, 1e4, "the gate resistance (Ω·mm)")
    st = dev.solve({"gate": vg, "drain": vd})
    f = np.logspace(8.0, 12.5, 37)
    ports, Y = dev.small_signal(st, f)
    R = {e.name: e.resistance for e in dev.electrodes}
    R["gate"] = R.get("gate", 0.0) + rg
    Y2 = two_port(ports, with_series(ports, Y, R))
    fig = rf_figures(f, Y2)
    gm = float(Y2[0, 1, 0].real)
    cgs = float(Y2[0, 0, 0].imag / (2 * math.pi * f[0]))

    def clean(v):
        return None if v is None or not math.isfinite(v) else float(v)
    return {"f": f.tolist(), "h21": np.abs(fig["h21"]).tolist(),
            "U": [clean(u) if u > 0 else None for u in np.real(fig["U"])],
            "fT": clean(fig["fT"]), "fT_method": fig["fT_method"],
            "fmax": clean(fig["fmax"]) if rg > 0 else None, "fmax_method": fig["fmax_method"],
            "fT_extrapolated": clean(fig["fT_extrapolated"]),
            "fmax_extrapolated": clean(fig["fmax_extrapolated"]) if rg > 0 else None,
            "gm": gm, "cg": cgs, "current": st.current["drain"], "gate_res": rg,
            "sources": info["sources"] + [TRANSISTOR_SOURCES["mason"]],
            "warnings": info["warn"]}


def run_breakdown(spec):
    """Avalanche breakdown in the off state: raise the drain voltage until
    the ionization integral along the worst field line reaches 1."""
    import dataclasses as _dc
    from ..device import IMPACT_GAN
    dev, info = _transistor_device(spec)
    gan_layers = tuple(sorted({L.label for L in info["layers"] if L.label == "GaN"}))
    if not gan_layers:
        raise SpecError("Breakdown uses impact ionization numbers measured on GaN; "
                        "this stack has no GaN layer.")
    # the GaN numbers are applied only in GaN (none are built in for AlGaN)
    ion = _dc.replace(IMPACT_GAN, where=gan_layers)
    b = spec.get("breakdown") or {}
    vg = _num(b, "gate", -10.0, -60.0, 10.0, "the gate voltage")
    vmax = _num(b, "to", 300.0, 1.0, 3000.0, "the highest drain voltage")
    step = _num(b, "step", 10.0, 0.1, 500.0, "the voltage step")
    n = int(math.floor(vmax / step + 1e-9))
    if n > 120:
        raise SpecError("That is more than 120 steps; use a larger step.")
    vals = step * np.arange(1, n + 1)
    if b.get("criterion", "avalanche") == "current":
        ic = _num(b, "current_mA", 1.0, 1e-6, 1e4, "the breakdown current (mA/mm)")
        try:
            r = dev.breakdown_by_current("drain", vals, bias={"gate": vg}, current=ic * 1e-3)
        except RuntimeError:
            raise SpecError("The calculation stopped converging before the current "
                            "was reached; try a smaller step.")
        share = r["share"]
        return {"criterion": "current", "current_mA": ic, "vbr": r["voltage"],
                "first_step": r["first_step"],
                "voltages": r["voltages"].tolist(),
                "drain": r["current"]["drain"].tolist(),
                "gate_share": share.get("gate", 0.0), "source_share": share.get("source", 0.0),
                "sources": info["sources"], "warnings": info["warn"]}
    try:
        vbr, V, I, S = dev.breakdown_voltage("drain", vals, bias={"gate": vg},
                                             ionization=ion)
    except ValueError:
        raise SpecError("The field is already high enough to break down at the first "
                        "step; use a smaller step.")
    except RuntimeError:
        raise SpecError("The calculation stopped converging before breakdown; try a "
                        "smaller step, a longer gate-drain gap or a lower top voltage.")
    last = S[-1]
    r = dev.ionization_integral(last, ion)
    start = r["start"]
    warn = list(info["warn"])
    if abs(last.current["drain"]) > 1e-3:
        warn.append(f"The transistor is not off: {last.current['drain']:.3g} A per mm "
                    "still flows at the last step (under the gate or through the base). "
                    "Use a more negative gate voltage; in a real device this leakage, "
                    "not avalanche, may set the breakdown.")
    return {"criterion": "avalanche", "voltages": V.tolist(), "integral": I.tolist(), "vbr": vbr,
            "peak_field": r["peak_field"], "where_x": float(start[0]),
            "where_depth": float(dev.top - start[-1]),
            "off_current": last.current["drain"],
            "sources": info["sources"] + [TRANSISTOR_SOURCES["impact"]],
            "warnings": warn}


def run_pulse(spec):
    """A pulse from cold: switch on at t = 0 and follow the current and the
    hottest temperature while the device warms up."""
    dev, info = _transistor_device(spec)
    p = spec.get("pulse") or {}
    vg = _num(p, "gate", 0.0, -60.0, 10.0, "the gate voltage")
    vd = _num(p, "drain", 10.0, 0.0, 200.0, "the drain voltage")
    length = _num(p, "length", 100.0, 0.01, 1e6, "the pulse length (µs)")
    if (spec.get("transistor") or {}).get("substrate", "SiC") != "SiC":
        raise SpecError("Heating in time needs the substrate's density and specific "
                        "heat; the app has them for silicon carbide only.")
    from ..thermal import GAN_BULK, SIC_4H_ROOM
    model, heat_src = _fet_thermal(spec, info)
    heat_src = heat_src + [GAN_BULK.reference, SIC_4H_ROOM.reference]
    times = np.logspace(-9.0, math.log10(length * 1e-6), 19)
    t_, I, Tmax, _ = dev.pulse({"gate": vg, "drain": vd}, times, model)
    hot = dev.solve({"gate": vg, "drain": vd}, thermal=model)
    return {"times": t_.tolist(), "drain": I["drain"].tolist(), "tmax": Tmax.tolist(),
            "steady": hot.current["drain"], "steady_t": float(hot.temperature.max()),
            "sources": info["sources"] + heat_src, "warnings": info["warn"]}


def run_lag(spec):
    """Drain lag (current collapse) from iron traps in the buffer: hold a
    stress voltage, switch, and follow the current in time."""
    dev, info = _transistor_device(spec, with_traps=True)
    lag = spec.get("lag") or {}
    sg = _num(lag, "stress_gate", 0.0, -60.0, 10.0, "the stress gate voltage")
    sd = _num(lag, "stress_drain", 20.0, 0.0, 200.0, "the stress drain voltage")
    mg = _num(lag, "gate", 0.0, -60.0, 10.0, "the measured gate voltage")
    md = _num(lag, "drain", 5.0, 0.0, 200.0, "the measured drain voltage")
    stress = dev.solve({"gate": sg, "drain": sd})
    final = dev.solve({"gate": mg, "drain": md}, guess=stress)
    times = np.logspace(-9, 1, 21)
    r = dev.transient(stress, {"gate": mg, "drain": md}, times)
    Id = r.current["drain"]
    dip = 1.0 - Id[0] / final.current["drain"]
    rec = None
    if abs(Id[-1] - Id[0]) > 0:
        target = Id[0] + (1 - math.exp(-1)) * (Id[-1] - Id[0])
        k = np.flatnonzero((Id - target) * np.sign(Id[-1] - Id[0]) >= 0)
        if k.size:
            k = int(k[0])
            if k > 0:
                a, b = math.log(times[k - 1]), math.log(times[k])
                fa, fb = Id[k - 1] - target, Id[k] - target
                rec = float(math.exp(a + (b - a) * fa / (fa - fb))) if fa != fb else float(times[k])
            else:
                rec = float(times[0])
    en = 1.0 / dev.emission_times()[0][0]
    return {"times": times.tolist(), "drain": Id.tolist(),
            "steady": final.current["drain"], "stress_current": stress.current["drain"],
            "dip": dip, "recovery": rec, "emission_time": 1.0 / en,
            "occupancy": r.mean_occupancy[0].tolist(), "sources": [TRANSISTOR_SOURCES["fe"]]
            + info["sources"], "warnings": info["warn"]}


_TOO_HOT = 1000.0


def _too_hot_note(T):
    return (f"The hottest point reaches {T:.0f} K. Check that the material numbers "
            "you rely on still hold that hot, and read the result as 'far too hot' "
            "rather than as a prediction.")


def run_runaway(spec):
    """Thermal runaway: raise the drain voltage with self-heating until no
    steady temperature is left (new in 0.23.0)."""
    dev, info = _transistor_device(spec)
    model, heat_src = _fet_thermal(spec, info)
    r = spec.get("runaway") or {}
    vg = _num(r, "gate", 0.0, -60.0, 10.0, "the gate voltage")
    vmax = _num(r, "to", 50.0, 0.5, 2000.0, "the highest drain voltage")
    step = _num(r, "step", 5.0, 0.01, 500.0, "the voltage step")
    n = int(math.floor(vmax / step + 1e-9))
    if n < 1 or n > 80:
        raise SpecError("Use between 1 and 80 steps.")
    stop = _num(r, "stop_K", _TOO_HOT, 310.0, 5000.0, "the temperature to stop at (K)")
    vals = step * np.arange(1, n + 1)
    res = dev.thermal_runaway("drain", vals, model, bias={"gate": vg}, stop_above=stop)
    warn = list(info["warn"])
    if len(res["hottest"]) and float(np.max(res["hottest"])) > _TOO_HOT:
        warn.append(_too_hot_note(float(np.max(res["hottest"]))))
    return {"stable": res["stable"], "runaway": res["runaway"], "too_hot": res["too_hot"],
            "stop_K": stop, "voltages": res["voltages"].tolist(),
            "hottest": res["hottest"].tolist(), "power": res["power"].tolist(), "gate": vg,
            "sources": info["sources"] + heat_src, "warnings": warn}


def _vertical_device(spec, layers, ic, T, dz, top, btbt):
    """The stack as a vertical device (contacts on top and at the bottom) for
    an I-V curve with band-to-band tunneling (new in 0.23.0)."""
    from ..device import BandToBand, Device, Electrode, Mobility, Surface, Trap
    t = spec.get("transport") or {}
    src = str(t.get("source", "")).strip()
    mu_n = _num(t, "mobility_n", what="the electron mobility (cm2/Vs)", lo=1e-6)
    mu_p = _num(t, "mobility_p", what="the hole mobility (cm2/Vs)", lo=1e-6)
    hm = _num(t, "hole_mass", what="the hole mass (in electron masses)", lo=1e-3)
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    n = max(int(math.ceil(edges[-1] / dz)), 10)
    if n > 4000:
        raise SpecError("The grid would have more than 4000 points; use a larger "
                        "grid step.")
    z = np.unique(np.round(np.concatenate([np.linspace(0, edges[-1], n + 1), edges]), 9))
    if top.kind == "schottky":
        el_top = Electrode("top", "schottky", 0, 20, barrier=top.barrier,
                           thermionic=bool(top.thermionic))
    else:
        el_top = Electrode("top", "ohmic", 0, 20)
    traps = []
    if t.get("lifetime") not in (None, ""):
        tau = _num(t, "lifetime", lo=1e-18, what="the lifetime (s)")
        # Shockley-Read-Hall through a midgap level with that lifetime; the
        # level's density is far below any doping, so it holds no charge
        Nt = 1e10
        for lab in sorted({L.label for L in layers}):
            eg = next(L.material.eg for L in layers if L.label == lab)
            traps.append(Trap("acceptor", 0.5 * eg, "ec", Nt, 1.0 / (tau * Nt),
                              1.0 / (tau * Nt), f"lifetime {tau:g} s: {src}", lab))
    kw = {}
    if btbt:
        kw["band_to_band"] = BandToBand(TRANSISTOR_SOURCES["btbt"], model="nonlocal")
    return Device(layers, 20.0, [el_top, Electrode("bottom", "ohmic", 0, 20, face="bottom")],
                  Mobility(mu_n, src), Surface("pinned", barrier=1.0),
                  grid=dict(x=np.array([0.0, 10.0, 20.0]), z=z), temperature_K=T,
                  interfaces=ic, hole_mass=hm, hole_mobility=Mobility(mu_p, src),
                  traps=traps, **kw)


_PROCESS_HELP = ("one step per line, for example: 'deposit material=SiN thickness=50 "
                 "mode=conformal' (or mode=directional, or mode=flux exponent=1 sticking=1), "
                 "'etch SiN=1 time=60 mode=directional from=100 to=900' (or mode=isotropic, "
                 "or mode=flux exponent=1), "
                 "'implant species=donor dose=1e13 range=30 straggle=15 from=100 to=900', "
                 "'anneal time=60 temperature=1300 D0=1e-4 Ea=3', 'oxidize material=Si "
                 "oxide=SiO2 B=2.9 BA=0.021 time=7200 consumed=0.44' (add mode=2d for "
                 "oxidation that reaches under mask edges, and viscosity=1e12 to let the "
                 "oxide flow; stress_volume=0.025 temperature=1273 slows growth where it "
                 "is squeezed; modulus=3e10 makes the oxide springy as well, "
                 "modulus_Si3N4=1e11 lets a mask bend instead of staying rigid, "
                 "viscosity_Si3N4=1e14 lets it flow slowly, "
                 "shape=flow lets the flow decide where the new oxide goes and lifts "
                 "masks that bend or flow), 'implant ranges=simulate ion_Z=5 ion_mass=11 energy=30000 "
                 "target_Z=14 target_mass=28.09 atoms=5e22 species=acceptor dose=1e13' "
                 "(the depth profile from a simulation of the ions), 'plasmaetch "
                 "neutrals=1e18 ions=1e16 stick_Si=0.2 yield_Si=5 atoms_Si=5e22 time=10' "
                 "(ions and a reactive gas together; fluxes=plasma neutral=Cl ion=Cl2+ "
                 "takes the fluxes from the plasma's chemistry)")


_PLASMA_HELP = ("one item per line: 'species Ar mass=39.95', 'species Ar+ mass=39.95 "
                "charge=1 returns=Ar' (what it turns into at the wall; 'returns=Cl2:0.5' "
                "for half a molecule), 'species Cl mass=35.45 wall=0.1 returns=Cl2:0.5' "
                "(wall: the chance it reacts when it hits the wall), 'reaction e + Ar -> "
                "Ar+ + e + e A=2e-14 b=0.5 C=16 energy=15.8 source=...' (k = A Te^b "
                "exp(-C/Te) in m3/s, energy lost by the electron in eV; add 'elastic' for "
                "elastic collisions), 'chamber radius=0.15 length=0.1 power=500 pump=0.05 "
                "h=0.3 gas_temperature=300 h_source=...' (meters, watts, m3/s), 'feed "
                "Ar=20' (sccm); a source runs to the end of its line")


def _parse_plasma(text):
    """The app's plasma chemistry (new in 0.35.0): the lines of _PLASMA_HELP
    into a `GlobalPlasma`, and the sources named."""
    from ..plasma import SCCM, GlobalPlasma, Reaction, Species
    species, reactions, chamber, feed, sources = [], [], None, {}, []
    for k, raw in enumerate(str(text).splitlines(), 1):
        ln = raw.split("#")[0].strip()
        if not ln:
            continue
        src = None
        for key in ("h_source=", "source="):
            if key in ln:
                ln, src = ln.split(key, 1)
                src = src.strip()
                ln = ln.strip()
                if not src:
                    raise SpecError(f"Plasma line {k}: the source is empty.")
                break
        words = ln.split()
        kind = words[0].lower()

        def kv(ws):
            d = {}
            for w in ws:
                if "=" in w:
                    a, b = w.split("=", 1)
                    d[a] = b
            return d

        def num(d, name, default=None, lo=None, hi=None):
            if name not in d:
                if default is None:
                    raise SpecError(f"Plasma line {k} ({kind}) needs {name}=...")
                return default
            try:
                v = float(d[name])
            except ValueError:
                raise SpecError(f"Plasma line {k}: {name} must be a number.") from None
            if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
                raise SpecError(f"Plasma line {k}: {name}={d[name]} is out of range.")
            return v
        try:
            if kind == "species":
                if len(words) < 2 or "=" in words[1]:
                    raise SpecError(f"Plasma line {k}: give the species a name first.")
                d = kv(words[2:])
                ret = {}
                for part in [x for x in d.get("returns", "").split(",") if x]:
                    nm, _, c = part.partition(":")
                    try:
                        ret[nm] = float(c) if c else 1.0
                    except ValueError:
                        raise SpecError(f"Plasma line {k}: returns=name or name:number.") from None
                species.append(Species(words[1], num(d, "mass", lo=1e-3, hi=1e4),
                                       int(num(d, "charge", 0.0, -1.0, 1.0)),
                                       num(d, "wall", 0.0, 0.0, 1.0), ret))
            elif kind == "reaction":
                body = " ".join(w for w in words[1:] if "=" not in w and w != "elastic")
                if "->" not in body:
                    raise SpecError(f"Plasma line {k}: write the reaction as 'a + b -> c + d'.")
                lhs, rhs = body.split("->", 1)
                def side(t):
                    return tuple(x for x in t.split() if x != "+")
                if not src:
                    raise SpecError(f"Plasma line {k}: say where the rate comes from "
                                    "(source=... at the end of the line).")
                d = kv(words[1:])
                reactions.append(Reaction(side(lhs), side(rhs), num(d, "A", lo=0.0), src,
                                          num(d, "b", 0.0), num(d, "C", 0.0),
                                          num(d, "energy", 0.0), "elastic" in words))
                sources.append(src)
            elif kind == "chamber":
                if not src:
                    raise SpecError(f"Plasma line {k}: say where h comes from "
                                    "(h_source=... at the end of the line).")
                d = kv(words[1:])
                chamber = dict(radius_m=num(d, "radius", lo=1e-4, hi=10.0),
                               length_m=num(d, "length", lo=1e-4, hi=10.0),
                               power_W=num(d, "power", lo=1e-6, hi=1e6),
                               pump_m3_s=num(d, "pump", lo=1e-9, hi=1e3),
                               h=num(d, "h", lo=1e-6, hi=1.0), h_source=src,
                               gas_temperature_K=num(d, "gas_temperature", 300.0, 1.0, 5000.0))
                sources.append(src)
            elif kind == "feed":
                for nm, v in kv(words[1:]).items():
                    feed[nm] = num({nm: v}, nm, lo=0.0) * SCCM
            else:
                raise SpecError(f"Plasma line {k}: unknown item '{kind}' ({_PLASMA_HELP}).")
        except SpecError:
            raise
        except ValueError as e:
            raise SpecError(f"Plasma line {k}: {e}") from None
    if not species or not reactions or chamber is None or not feed:
        raise SpecError("The plasma needs species, reactions, a chamber line and a feed "
                        f"({_PLASMA_HELP}).")
    try:
        gp = GlobalPlasma(species, reactions, feed=feed, **chamber)
    except ValueError as e:
        raise SpecError(f"The plasma: {e}") from None
    return gp, sources


def _flux_size(st, k):
    """Line-of-sight fluxes cost the square of the number of surface cells."""
    if st.xc.size > 400 or st.zc.size > 400:
        raise SpecError(f"Step {k}: mode=flux needs at most 400 cells along a side; "
                        "use larger cells.")


def run_process(spec):
    """Build a cross-section from process steps (new in 0.23.0): deposition,
    etching, oxidation, implants and anneals on a grid of square cells."""
    from ..process import Anneal, Implant, Structure
    p = spec.get("process") or {}
    src = str(p.get("source", "")).strip()
    if not src:
        raise SpecError("Please say where your process numbers come from (rates, "
                        "ranges, diffusivities): the 'source' box.")
    W = _num(p, "width", 1000.0, 10.0, 20000.0, "the width (nm)")
    H = _num(p, "height", 400.0, 10.0, 20000.0, "the height (nm)")
    h = _num(p, "cell", 5.0, 0.1, 1000.0, "the cell size (nm)")
    sub = str(p.get("substrate") or "GaN").strip()
    s_top = _num(p, "substrate_top", 300.0, 0.0, H, "the top of the wafer (nm)")
    if W / h > 1200 or H / h > 1200:
        raise SpecError("That is more than 1200 cells along a side; use larger cells.")
    try:
        st = Structure(W, 0.0, H, h, sub, s_top)
    except ValueError as e:
        raise SpecError(f"The width and height must be whole numbers of cells ({e}).")
    lines = [ln.split("#")[0].strip() for ln in str(p.get("recipe") or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise SpecError("Write at least one step: " + _PROCESS_HELP + ".")
    if len(lines) > 40:
        raise SpecError("At most 40 steps.")
    oxide_x = None
    sim_notes = []
    flow_oxide = None
    elastic_solids = set()
    # the plasma's chemistry (new in 0.35.0): solved once if given
    plasma, plasma_res, plasma_src = None, None, []
    if str(p.get("plasma") or "").strip():
        plasma, plasma_src = _parse_plasma(p["plasma"])
        try:
            plasma_res = plasma.solve()
        except RuntimeError as e:
            raise SpecError(f"The plasma: {e}") from None
    for k, ln in enumerate(lines, 1):
        words = ln.split()
        kind, args = words[0].lower(), {}
        for w in words[1:]:
            if "=" not in w:
                raise SpecError(f"Step {k}: '{w}' should be name=value ({_PROCESS_HELP}).")
            a, b = w.split("=", 1)
            args[a] = b

        def num(name, default=None, lo=None, hi=None):
            if name not in args:
                if default is None:
                    raise SpecError(f"Step {k} ({kind}) needs {name}=...")
                return default
            try:
                v = float(args[name])
            except ValueError:
                raise SpecError(f"Step {k}: {name} must be a number.")
            if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
                raise SpecError(f"Step {k}: {name}={args[name]} is out of range.")
            return v
        win = None
        if "from" in args or "to" in args:
            win = (num("from", 0.0, 0.0, W), num("to", W, 0.0, W))
            if win[1] <= win[0]:
                raise SpecError(f"Step {k}: 'to' must be larger than 'from'.")
        before = float(st.donors.sum() + st.acceptors.sum())
        try:
            if kind == "deposit":
                mat = args.get("material")
                if not mat:
                    raise SpecError(f"Step {k}: give material=...")
                mode = args.get("mode", "conformal")
                nd = num("donors", 0.0, 0.0)
                na = num("acceptors", 0.0, 0.0)
                if mode == "flux":
                    _flux_size(st, k)
                st.deposit(mat, num("thickness", lo=1e-6), mode=mode, window=win,
                           semiconductor=args.get("semiconductor", "no") == "yes",
                           donors=nd, acceptors=na,
                           exponent=num("exponent", 1.0, 1.0, 1e4),
                           sticking=num("sticking", 1.0, 1e-6, 1.0))
            elif kind == "etch":
                rates = {a: num(a, lo=0.0) for a in args
                         if a not in ("time", "mode", "from", "to", "exponent", "sticking")}
                if not rates:
                    raise SpecError(f"Step {k}: give each material's rate, like SiN=1 (nm/s).")
                if args.get("mode") == "flux":
                    _flux_size(st, k)
                st.etch(rates, num("time", lo=1e-12), mode=args.get("mode", "isotropic"),
                        window=win, exponent=num("exponent", 1.0, 1.0, 1e4),
                        sticking=num("sticking", 1.0, 1e-6, 1.0))
            elif kind == "implant" and args.get("ranges") == "simulate":
                # the depth profile from a Monte Carlo run of the ions (ZBL
                # potential, Lindhard-Scharff stopping); the target's atoms
                # and atom density are yours
                from .. import bca
                sp = args.get("species", "donor")
                tgt = bca.Target((bca.Element(int(num("target_Z", lo=1, hi=118)),
                                              num("target_mass", lo=0.5, hi=400.0)),),
                                 num("atoms", lo=1e20, hi=1e24), src)
                n_ions = int(num("ions", 1000, 100, 20000))
                rr = bca.simulate(int(num("ion_Z", lo=1, hi=118)), num("ion_mass", lo=0.5, hi=400.0),
                                  num("energy", lo=10.0, hi=1e7), tgt, n_ions=n_ions,
                                  tilt_deg=num("tilt", 0.0, 0.0, 89.0), seed=0)
                st.implant(rr.implant(sp, num("dose", lo=1e-30), 0.0, window=win,
                                      activation=num("activation", 1.0, 0.0, 1.0)))
                sim_notes.append(f"step {k}: simulated range {rr.projected_range:.1f} nm, "
                                 f"spread {rr.straggle:.1f} nm ({n_ions} ions, "
                                 f"{100 * rr.backscattered / rr.n_ions:.1f}% bounced back)")
            elif kind == "implant":
                sp = args.get("species", "donor")
                st.implant(Implant(sp, num("dose", lo=1e-30), num("range", lo=0.0),
                                   num("straggle", lo=1e-6), src, 0.0,
                                   lateral_straggle_nm=num("lateral", 0.0, 0.0) or None,
                                   window=win, activation=num("activation", 1.0, 0.0, 1.0)))
            elif kind == "plasmaetch":
                # ions and a reactive gas together: per material its sticking,
                # ion yield and atom density as stick_X=, yield_X=, atoms_X=
                from ..process import EtchKinetics
                stick = {a[6:]: num(a, lo=0.0, hi=1.0) for a in args if a.startswith("stick_")}
                yld = {a[6:]: num(a, lo=0.0) for a in args if a.startswith("yield_")}
                atoms = {a[6:]: num(a, lo=1e18) for a in args if a.startswith("atoms_")}
                if not stick or set(stick) != set(atoms):
                    raise SpecError(f"Step {k}: give stick_X=, yield_X= and atoms_X= for "
                                    "each material X that the gas etches.")
                if args.get("fluxes") == "plasma":
                    # the fluxes from the plasma's chemistry
                    from ..plasma import fluxes_cm2
                    if plasma_res is None:
                        raise SpecError(f"Step {k}: fluxes=plasma needs the plasma's "
                                        "chemistry (the box below the steps).")
                    try:
                        nf, jf = fluxes_cm2(plasma_res, args.get("neutral", ""),
                                            args.get("ion", ""))
                    except ValueError as e:
                        raise SpecError(f"Step {k}: {e} (give neutral= and ion=).") from None
                    args["neutrals"], args["ions"] = repr(nf), repr(jf)
                elif "fluxes" in args:
                    raise SpecError(f"Step {k}: fluxes=plasma, or give neutrals= and ions=.")
                kin = EtchKinetics(num("neutrals", lo=0.0), num("ions", lo=0.0),
                                   num("sites", 1e15, 1e10, 1e18), stick, yld, atoms, src)
                _flux_size(st, k)
                st.plasma_etch(kin, num("time", lo=1e-12))
            elif kind == "anneal":
                st.anneal(Anneal(num("time", lo=1e-12), num("temperature", lo=1.0),
                                 num("D0", lo=1e-300), num("Ea", lo=0.0), src,
                                 species=args.get("species", "both")))
            elif kind == "oxidize":
                mat, ox = args.get("material", "Si"), args.get("oxide", "SiO2")
                omode = args.get("mode", "column")
                flow = {}
                if "viscosity" in args:
                    if omode != "2d":
                        raise SpecError(f"Step {k}: the oxide's flow needs mode=2d.")
                    flow["viscosity"] = {ox: num("viscosity", lo=1e3, hi=1e25)}
                    # viscosity_X= lets the solid X (a mask) flow too (new
                    # in 0.35.0; with shape=flow it rides on the oxide)
                    for a in args:
                        if a.startswith("viscosity_"):
                            if a[10:] in (ox, mat):
                                raise SpecError(f"Step {k}: viscosity_X= is for other "
                                                "solids, such as a mask.")
                            flow["viscosity"][a[10:]] = num(a, lo=1e3, hi=1e25)
                    flow_oxide = ox
                    if "stress_volume" in args:
                        flow["stress_volume_nm3"] = num("stress_volume", lo=0.0, hi=1.0)
                        flow["temperature_K"] = num("temperature", lo=300.0, hi=2000.0)
                    # elastic too (new in 0.33.0): modulus= for the oxide,
                    # modulus_X= makes the solid X (a mask) elastic
                    mods = {a[8:]: num(a, lo=1e5, hi=1e13) for a in args
                            if a.startswith("modulus_")}
                    if "modulus" in args:
                        mods[ox] = num("modulus", lo=1e5, hi=1e13)
                    if mods:
                        flow["shear_modulus"] = mods
                        elastic_solids |= set(mods) - {ox}
                    # the shape from the flow (new in 0.34.0)
                    if args.get("shape", "volume") not in ("volume", "flow"):
                        raise SpecError(f"Step {k}: shape is volume or flow.")
                    flow["shape"] = args.get("shape", "volume")
                elif any(a == "modulus" or a.startswith(("modulus_", "viscosity_")) or a == "shape"
                         for a in args):
                    raise SpecError(f"Step {k}: modulus=, viscosity_X= and shape= need "
                                    "viscosity= (and mode=2d).")
                oxide_x = st.oxidize(mat, ox, num("B", lo=1e-300), num("BA", lo=1e-300),
                                     num("time", lo=1e-12), num("consumed", 0.44, 1e-6, 0.999999),
                                     src, mode=omode, **flow)
                if omode == "2d":
                    oxide_x = None           # cell counts, not the closed form
            else:
                raise SpecError(f"Step {k}: unknown step '{kind}' ({_PROCESS_HELP}).")
            if kind == "implant":
                # how much of the dose stayed in the semiconductor of the
                # picture (a mask takes some; ions that go deeper than the
                # bottom of the picture are lost)
                width = (win[1] - win[0]) if win else W
                want = num("dose") * num("activation", 1.0) * width / W
                got = (float(st.donors.sum() + st.acceptors.sum()) - before) * (h * 1e-7) ** 2 / (W * 1e-7)
                if want > 0 and got < 0.9 * want:
                    sim_notes.append(f"step {k}: only {100 * got / want:.0f}% of the dose "
                                     "stayed in the semiconductor of the picture (the rest "
                                     "went into a mask or past its bottom)")
        except SpecError:
            raise
        except ValueError as e:
            raise SpecError(f"Step {k} ({kind}): {e}")
    mats = sorted({m for m in st.mat.ravel().tolist() if m is not None})
    code = {m: i for i, m in enumerate(mats)}
    # at most 240 x 120 pixels in the picture
    fx = max(1, int(math.ceil(st.xc.size / 240)))
    fz = max(1, int(math.ceil(st.zc.size / 120)))
    pic = [[(-1 if st.mat[i, k] is None else code[st.mat[i, k]])
            for k in range(0, st.zc.size, fz)] for i in range(0, st.xc.size, fx)]
    dop = np.log10(np.maximum(st.donors + st.acceptors, 1.0))
    dmap = [[float(dop[i, k]) for k in range(0, st.zc.size, fz)]
            for i in range(0, st.xc.size, fx)]
    xm = W / 2
    return {"materials": mats, "picture": pic, "doping_log10": dmap,
            "width": W, "height": H,
            "areas": {m: st.area(m) for m in mats},
            "thickness_center": {m: st.thickness(m, xm) for m in mats},
            "donor_dose": float(st.donors.sum() * (h * 1e-7) ** 2 / (W * 1e-7)),
            "acceptor_dose": float(st.acceptors.sum() * (h * 1e-7) ** 2 / (W * 1e-7)),
            "oxide_exact": None if oxide_x is None else float(np.max(oxide_x)),
            "steps": len(lines), "sources": [src] + list(dict.fromkeys(plasma_src)),
            "warnings": [],
            "plasma": None if plasma_res is None else _plasma_summary(plasma_res),
            "notes": sim_notes,
            "stress_GPa": _oxide_stress(st, flow_oxide),
            "solid_stress_GPa": _solid_stress(st, elastic_solids)}


def _plasma_summary(r):
    """the plasma's state for the app: per cm^2 and cm^3"""
    return {"Te_eV": r["Te_eV"], "ne_cm3": r["ne"] * 1e-6, "pressure_Pa": r["pressure_Pa"],
            "densities_cm3": {k: v * 1e-6 for k, v in r["densities"].items()},
            "ion_flux_cm2": {k: v * 1e-4 for k, v in r["ion_flux"].items()},
            "neutral_flux_cm2": {k: v * 1e-4 for k, v in r["neutral_flux"].items()},
            "ion_energy_eV": r["ion_energy_eV"]}


def _solid_stress(st, names):
    """the range of sigma_xx (GPa) in each elastic solid, or None"""
    S = getattr(st, "oxide_stress", None)
    if S is None or not names:
        return None
    out = {}
    for m in sorted(names):
        v = S["sxx"][st.mat == m]
        v = v[np.isfinite(v)]
        if v.size:
            out[m] = [float(v.min()) / 1e9, float(v.max()) / 1e9]
    return out or None


def _oxide_stress(st, ox):
    """the most compressive stress in the oxide that flowed (GPa), or None"""
    S = getattr(st, "oxide_stress", None)
    if S is None or ox is None:
        return None
    m = st.mat == ox
    if not m.any():
        return None
    return float(min(np.nanmin(S["sxx"][m]), np.nanmin(S["szz"][m]))) / 1e9


ACTIONS = {"options": lambda spec: options(), "charges": run_charges,
           "electrons": run_electrons, "device": run_device,
           "uncertainty": run_uncertainty, "line_charge": run_line_charge,
           "plan": run_plan, "code": python_code, "fit": run_fit,
           "design": run_design, "cross": run_cross,
           "transistor": run_transistor, "lag": run_lag, "rf": run_rf,
           "breakdown": run_breakdown, "pulse": run_pulse, "runaway": run_runaway,
           "process": run_process, "mobility": run_mobility, "quantum": run_quantum}


def _plain(e):
    s = str(e)
    if "(NaN)" in s and "ScAlN" in s.replace("Sc0", "ScAlN"):
        return ("ScAlN has no band gap, band offset or electron mass in its "
                "source. Enter your own values (and where they come from) in "
                "the ScAlN box to see electrons or device curves.")
    if "(NaN)" in s:
        return ("A layer is missing a number this calculation needs (" +
                s.split("has no ")[-1].split(" (NaN)")[0] + ").")
    if "refuses a relaxed" in s or "depends on the charge of its misfit" in s:
        return ("A layer is relaxed, so its charge depends on the charge of "
                "the dislocation lines. Choose a line-charge option.")
    return s


def handle(action, spec):
    """Run one action; errors come back as {"error": message}."""
    if action not in ACTIONS:
        return {"error": f"unknown action {action!r}"}
    try:
        out = ACTIONS[action](spec or {})
    except SpecError as e:
        return {"error": str(e)}
    except (ValueError, RuntimeError, ArithmeticError) as e:
        return {"error": _plain(e)}
    return _clean(out)


def _clean(o):
    """Make the result JSON-safe (no NaN or infinity, no NumPy types)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o

