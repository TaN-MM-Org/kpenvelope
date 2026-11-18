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
from ..electrons import solve_electrons
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
    if mode == "iv":
        tr = _transport(spec, len(layers))
        V, J, res = iv_curve(layers, tr, V, top=top, **kw)
        return {"mode": "iv", "v": list(map(float, V)), "j_A_cm2": J.tolist(),
                "warnings": warn}
    raise SpecError("Choose capacitance (C-V) or current (I-V).")


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


ACTIONS = {"options": lambda spec: options(), "charges": run_charges,
           "electrons": run_electrons, "device": run_device,
           "uncertainty": run_uncertainty, "line_charge": run_line_charge,
           "plan": run_plan, "code": python_code}


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

