"""Process steps that set a device's doping: ion implantation through a
mask window and annealing that lets the dopants diffuse (new in 0.21.0).

This is a small process model, not a process simulator. It covers what a
device simulation most often needs from the process: where the dopants
end up. Every number comes from you with its source, because the package
ships no implant ranges or diffusivities:

- `Implant`: the as-implanted profile is a Gaussian in depth (projected
  range R_p, straggle dR_p), the usual two-moment description of an
  implant. Through a window x0 < x < x1 the lateral spread is a Gaussian of
  width dR_l convolved with the window, which gives the error-function edge
  0.5 [erf((x - x0) / (sqrt 2 dR_l)) - erf((x - x1) / (sqrt 2 dR_l))].
  Take R_p, dR_p, dR_l from a range calculation (for example SRIM) or a
  measured SIMS profile for your ion, energy and material.
- `Anneal`: Fick's law dC/dt = D d^2C/dx^2 (in 2D) with D = D_0 exp(-E_a /
  kT), a constant diffusivity (no concentration or Fermi-level
  dependence, no clustering or segregation), no flux through any face.
  It is solved with finite volumes on a tensor grid, exactly in time.
- `activation`: the fraction of the implanted ions that end up as active
  donors or acceptors.

`Process.doping` returns the function `Device(doping=...)` takes.

Since 0.22.0, `Structure` also builds the shape of a cross-section from
process steps, on a grid of square cells (a cell-based, or voxel,
geometry; surfaces follow the cell edges, so refine the cell size where a
curved surface matters):

- `deposit`: conformal (every exposed surface grows by the thickness:
  the cells of gas within that distance of the solid, by an exact
  Euclidean distance transform of the cell centers) or directional (each
  column grows on its top, as in evaporation).
- `etch`: isotropic (the front moves into every material at its own rate,
  the arrival time from a fast-marching solution of |grad T| = 1/rate;
  a rate of 0 is a mask) or directional (each column is etched from the
  top at the rates of the materials it meets).
- `oxidize`: Deal-Grove growth, x^2 + A x = B (t + tau), with tau from the
  oxide already there (B. E. Deal and A. S. Grove, J. Appl. Phys. 36,
  3770 (1965)), column by column where the oxidizable material is exposed
  or under its own oxide; a fraction of the oxide thickness you give comes
  from the consumed material (0.44 for silicon). One-dimensional in each
  column: no lateral oxidation under a mask edge (bird's beak), no
  stress, no dopant segregation (dopants in consumed cells are dropped).
- `implant`: the Gaussian profile measured down from the local surface of
  each column (the same range in every material it passes), stored where
  the material is a semiconductor; `anneal`: Fick diffusion inside the
  semiconductor cells only (no flux into oxide, mask or gas), exact in
  time.
- `mesh` and `doping` hand the result to `mesh.MeshDevice`;
  `surface_nodes` picks contact nodes on an exposed surface.

B, B/A, rates, ranges, diffusivities and the consumed fraction all come
from you with their source.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

_KB_EV = 8.617333262e-5          # eV/K (CODATA 2018)

__all__ = ["Implant", "Anneal", "Process", "Structure"]


@dataclasses.dataclass(frozen=True)
class Implant:
    """An ion implantation.

    species : "donor" or "acceptor". dose_cm2 : ions per cm^2.
    projected_range_nm, straggle_nm : R_p and dR_p of the depth profile
        (nm), with `reference`.
    lateral_straggle_nm : dR_l (default dR_p).
    window : (x0, x1) of the mask opening (nm), None for no mask.
    surface_z : z of the surface the ions enter (nm, z up the stack, as in
        `Device`); the depth below it is surface_z - z.
    activation : the fraction that becomes electrically active (0..1).
    """

    species: str
    dose_cm2: float
    projected_range_nm: float
    straggle_nm: float
    reference: str
    surface_z: float
    lateral_straggle_nm: float = None
    window: tuple = None
    activation: float = 1.0

    def __post_init__(self):
        if self.species not in ("donor", "acceptor"):
            raise ValueError("Implant.species must be 'donor' or 'acceptor'")
        for f in ("dose_cm2", "straggle_nm"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"Implant.{f} must be > 0")
        if not math.isfinite(self.projected_range_nm):
            raise ValueError("Implant.projected_range_nm must be finite")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Implant needs a reference for its range and straggle")
        if not math.isfinite(self.surface_z):
            raise ValueError("Implant.surface_z must be finite")
        if self.lateral_straggle_nm is not None and not (
                math.isfinite(self.lateral_straggle_nm) and self.lateral_straggle_nm > 0):
            raise ValueError("Implant.lateral_straggle_nm must be > 0")
        if self.window is not None:
            w = tuple(float(v) for v in self.window)
            if len(w) != 2 or not w[1] > w[0]:
                raise ValueError("Implant.window must be (x0, x1) with x1 > x0")
            object.__setattr__(self, "window", w)
        if not (0.0 <= self.activation <= 1.0):
            raise ValueError("Implant.activation must be between 0 and 1")

    def concentration(self, x, z):
        """As-implanted active concentration (cm^-3) at (x, z) in nm. Above
        the surface it is zero."""
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        depth = self.surface_z - z
        s = self.straggle_nm
        vert = (self.dose_cm2 / (math.sqrt(2 * math.pi) * s * 1e-7)
                * np.exp(-0.5 * ((depth - self.projected_range_nm) / s) ** 2))
        vert = np.where(depth >= 0, vert, 0.0)
        if self.window is None:
            lat = np.ones_like(x)
        else:
            from math import erf
            sl = self.lateral_straggle_nm or s
            x0, x1 = self.window
            verf = np.vectorize(erf)
            lat = 0.5 * (verf((x - x0) / (math.sqrt(2) * sl))
                         - verf((x - x1) / (math.sqrt(2) * sl)))
        return self.activation * vert * lat


@dataclasses.dataclass(frozen=True)
class Anneal:
    """A diffusion anneal: time_s at temperature_K with D = D0 exp(-Ea/kT)
    (D0 in cm^2/s, Ea in eV, with `reference`), for the donors, the
    acceptors or both."""

    time_s: float
    temperature_K: float
    D0_cm2_s: float
    Ea_eV: float
    reference: str
    species: str = "both"

    def __post_init__(self):
        for f in ("time_s", "temperature_K", "D0_cm2_s"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"Anneal.{f} must be > 0")
        if not (math.isfinite(self.Ea_eV) and self.Ea_eV >= 0):
            raise ValueError("Anneal.Ea_eV must be >= 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Anneal needs a reference for D0 and Ea")
        if self.species not in ("donor", "acceptor", "both"):
            raise ValueError("Anneal.species must be 'donor', 'acceptor' or 'both'")

    @property
    def diffusivity_nm2_s(self):
        return self.D0_cm2_s * math.exp(-self.Ea_eV / (_KB_EV * self.temperature_K)) * 1e14


class Process:
    """A sequence of implants and anneals on the region x in [0, width],
    z in [z_bottom, z_top] (nm), worked out on a tensor grid of nx by nz
    points (the anneal's grid; the result is interpolated onto the
    device). Steps run in the order given."""

    def __init__(self, steps, width: float, z_bottom: float, z_top: float,
                 nx: int = 201, nz: int = 401):
        steps = list(steps)
        for s in steps:
            if not isinstance(s, (Implant, Anneal)):
                raise ValueError("steps must be Implant or Anneal objects")
        if not (width > 0 and z_top > z_bottom):
            raise ValueError("give width > 0 and z_top > z_bottom")
        self.steps = steps
        self.x = np.linspace(0.0, float(width), int(nx))
        self.z = np.linspace(float(z_bottom), float(z_top), int(nz))
        self.donors, self.acceptors = self._run()

    def _run(self):
        X, Z = np.meshgrid(self.x, self.z, indexing="ij")
        Nd = np.zeros(X.shape)
        Na = np.zeros(X.shape)
        for s in self.steps:
            if isinstance(s, Implant):
                c = s.concentration(X, Z)
                if s.species == "donor":
                    Nd = Nd + c
                else:
                    Na = Na + c
            else:
                D = s.diffusivity_nm2_s
                if s.species in ("donor", "both"):
                    Nd = self._diffuse(Nd, D, s.time_s)
                if s.species in ("acceptor", "both"):
                    Na = self._diffuse(Na, D, s.time_s)
        return Nd, Na

    def _diffuse(self, C, D, t):
        """dC/dt = D lap C with no flux through the faces: finite volumes on
        the tensor grid, integrated exactly in time. The operator splits
        into commuting x and z parts, so C(t) = E_x C(0) E_z^T with
        E = expm(D t V^-1 L) of each axis (the total is conserved to
        rounding)."""
        from scipy.linalg import expm

        def prop(a):
            n = a.size
            h = np.diff(a)
            vol = np.zeros(n)
            vol[:-1] += 0.5 * h
            vol[1:] += 0.5 * h
            L = np.zeros((n, n))
            idx = np.arange(n - 1)
            L[idx, idx] -= 1.0 / h
            L[idx + 1, idx + 1] -= 1.0 / h
            L[idx, idx + 1] += 1.0 / h
            L[idx + 1, idx] += 1.0 / h
            return expm(D * t * (L / vol[:, None]))
        return prop(self.x) @ C @ prop(self.z).T

    def doping(self):
        """The function (x, z) -> (N_d, N_a) in cm^-3 for `Device(doping=)`:
        the result interpolated linearly, zero outside the process grid."""
        from scipy.interpolate import RegularGridInterpolator
        fd = RegularGridInterpolator((self.x, self.z), self.donors, bounds_error=False,
                                     fill_value=0.0)
        fa = RegularGridInterpolator((self.x, self.z), self.acceptors, bounds_error=False,
                                     fill_value=0.0)

        def f(x, z, *rest):
            if rest:                       # 3D: (x, y, z), uniform along y
                z = rest[-1]
            pt = np.array([[x, z]])
            return float(fd(pt)[0]), float(fa(pt)[0])
        return f

    def sheet(self, species: str = "donor"):
        """Dose per cm^2 at each x (the integral over z)."""
        C = self.donors if species == "donor" else self.acceptors
        trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return trap(C, self.z, axis=1) * 1e-7


# ----------------------------------------------------- topography (0.22.0)
class Structure:
    """A cross-section built by process steps (new in 0.22.0): square cells
    of size h (nm) over x in [0, width], z in [z_bottom, z_top]; each cell
    holds a material label or None (gas). It starts with the substrate
    material up to substrate_top.

    semiconductors : the labels whose cells keep implanted dopants (and in
    which they diffuse).
    """

    def __init__(self, width: float, z_bottom: float, z_top: float, h: float,
                 substrate: str, substrate_top: float, semiconductors=None):
        if not (width > 0 and z_top > z_bottom and h > 0):
            raise ValueError("give width > 0, z_top > z_bottom and h > 0")
        nx = int(round(width / h))
        nz = int(round((z_top - z_bottom) / h))
        if abs(nx * h - width) > 1e-9 * width or abs(nz * h - (z_top - z_bottom)) > 1e-9 * (
                z_top - z_bottom):
            raise ValueError("width and z_top - z_bottom must be whole numbers of cells")
        self.h = float(h)
        self.x = np.linspace(0.0, float(width), nx + 1)               # cell edges
        self.z = np.linspace(float(z_bottom), float(z_top), nz + 1)
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        self.zc = 0.5 * (self.z[1:] + self.z[:-1])
        self.mat = np.full((nx, nz), None, dtype=object)
        self.mat[:, self.zc < substrate_top] = substrate
        self.semis = set([substrate] if semiconductors is None else semiconductors)
        self.donors = np.zeros((nx, nz))           # cm^-3
        self.acceptors = np.zeros((nx, nz))
        self.log = []

    # ---------------------------------------------------------- helpers
    @property
    def solid(self):
        return self.mat != None  # noqa: E711

    def _gas_open(self):
        """Gas cells connected to the top of the domain."""
        from scipy import ndimage
        gas = ~self.solid
        lab, n = ndimage.label(gas)
        top = set(lab[:, -1].tolist()) - {0}
        return np.isin(lab, list(top))

    def _window(self, window):
        if window is None:
            return np.ones(self.xc.size, bool)
        x0, x1 = window
        return (self.xc > x0) & (self.xc < x1)

    def top_surface(self):
        """z of the top of the highest solid cell in every column (NaN where
        a column is empty)."""
        sol = self.solid
        out = np.full(self.xc.size, np.nan)
        for i in range(self.xc.size):
            k = np.flatnonzero(sol[i])
            if k.size:
                out[i] = self.z[k[-1] + 1]
        return out

    def thickness(self, material, x):
        """Total thickness (nm) of `material` in the column at x."""
        i = int(np.clip(np.searchsorted(self.x, x) - 1, 0, self.xc.size - 1))
        return float(np.sum(self.mat[i] == material) * self.h)

    def area(self, material):
        """Cross-section area of a material (nm^2)."""
        return float(np.sum(self.mat == material) * self.h * self.h)

    # ---------------------------------------------------------- steps
    def deposit(self, material: str, thickness: float, mode: str = "conformal",
                window=None, semiconductor: bool = False, donors: float = 0.0,
                acceptors: float = 0.0):
        """Grow `material` by `thickness` nm on the exposed surface:
        conformal (every gas cell connected to the top whose center lies
        within thickness + h/2 of a solid cell center, the Euclidean
        distance on the cell grid; on a flat surface exactly thickness/h
        cells) or directional (on top of each column, within window
        (x0, x1) if given). Epitaxial growth of a semiconductor: give
        semiconductor=True and its doping as grown (donors, acceptors in
        cm^-3)."""
        if not thickness > 0:
            raise ValueError("thickness must be > 0")
        if donors < 0 or acceptors < 0:
            raise ValueError("doping must be >= 0")
        if semiconductor or donors > 0 or acceptors > 0:
            self.semis.add(material)
        before = self.solid.copy()
        if mode == "conformal":
            from scipy import ndimage
            sol = self.solid
            if not sol.any():
                raise ValueError("nothing to deposit on")
            d = ndimage.distance_transform_edt(~sol, sampling=self.h)
            fill = self._gas_open() & (d <= thickness + 0.5 * self.h + 1e-9 * self.h)
            fill &= self._window(window)[:, None]
            self.mat[fill] = material
        elif mode == "directional":
            win = self._window(window)
            top = self.top_surface()
            for i in np.flatnonzero(win):
                base = top[i] if np.isfinite(top[i]) else self.z[0]
                sel = (self.zc > base) & (self.zc < base + thickness)
                self.mat[i, sel] = material
        else:
            raise ValueError("mode must be 'conformal' or 'directional'")
        new = self.solid & ~before
        self.donors[new] = donors
        self.acceptors[new] = acceptors
        self.log.append(("deposit", material, thickness, mode))

    def etch(self, rates: dict, time: float, mode: str = "isotropic", window=None):
        """Remove material: rates {label: nm/s} (a label not listed, or a
        rate of 0, is not etched: a mask), for `time` seconds. isotropic:
        the front starts on every surface the gas touches (within window if
        given) and moves into each material at its rate; a cell goes when
        the front passes its center (fast marching).
        directional: every column in window is etched straight down from
        its top."""
        if not time > 0:
            raise ValueError("time must be > 0")
        rate = np.zeros(self.mat.shape)
        for lab, r in rates.items():
            if not (math.isfinite(r) and r >= 0):
                raise ValueError("etch rates must be >= 0 (nm/s)")
            rate[self.mat == lab] = r
        win = self._window(window)
        if mode == "directional":
            for i in np.flatnonzero(win):
                t_used = 0.0
                for k in range(self.zc.size - 1, -1, -1):
                    if self.mat[i, k] is None:
                        continue
                    r = rate[i, k]
                    if r <= 0:
                        break
                    if t_used + 0.5 * self.h / r <= time + 1e-12 * time:
                        t_used += self.h / r
                        self.mat[i, k] = None
                        self.donors[i, k] = self.acceptors[i, k] = 0.0
                    else:
                        break
        elif mode == "isotropic":
            T = self._arrival(rate, win)
            go = (T <= time * (1 + 1e-12)) & self.solid
            self.mat[go] = None
            self.donors[go] = 0.0
            self.acceptors[go] = 0.0
        else:
            raise ValueError("mode must be 'isotropic' or 'directional'")
        self.log.append(("etch", dict(rates), time, mode))

    def _arrival(self, rate, win):
        """Arrival time of the etch front at every cell center: fast
        marching (second-order upwind where it can, J. A. Sethian, Level Set
        Methods and Fast Marching Methods, 1999) of |grad T| = 1 / rate from the
        surfaces of the open gas, started half a cell from each exposed
        cell (the surface lies on the cell edge)."""
        import heapq
        nx, nz = rate.shape
        h = self.h
        T = np.full((nx, nz), np.inf)
        known = np.zeros((nx, nz), bool)
        gas = self._gas_open() & win[:, None]
        sol = self.solid
        etchable = sol & (rate > 0) & win[:, None]
        exposed = np.zeros((nx, nz), bool)
        exposed[:-1] |= gas[1:]
        exposed[1:] |= gas[:-1]
        exposed[:, :-1] |= gas[:, 1:]
        exposed[:, 1:] |= gas[:, :-1]
        exposed &= etchable
        heap = []
        for i, k in zip(*np.nonzero(exposed)):
            T[i, k] = 0.5 * h / rate[i, k]
            heap.append((T[i, k], int(i), int(k)))
        heapq.heapify(heap)
        ok = sol & (rate > 0)
        while heap:
            t, i, k = heapq.heappop(heap)
            if known[i, k] or t > T[i, k]:
                continue
            known[i, k] = True
            for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, k + dk
                if not (0 <= a < nx and 0 <= b < nz) or known[a, b] or not ok[a, b]:
                    continue
                f = h / rate[a, b]
                # second-order upwind where two known points line up on the
                # upwind side, first order where only one does
                terms = []
                for (d1, e1) in (((-1, 0), (1, 0)), ((0, -1), (0, 1))):
                    best = None
                    for (u, v) in (d1, e1):
                        a1, b1 = a + u, b + v
                        if not (0 <= a1 < nx and 0 <= b1 < nz and known[a1, b1]):
                            continue
                        t1 = T[a1, b1]
                        a2, b2 = a1 + u, b1 + v
                        if (0 <= a2 < nx and 0 <= b2 < nz and known[a2, b2]
                                and T[a2, b2] <= t1):
                            cand = (2.25, (4.0 * t1 - T[a2, b2]) / 3.0, t1)
                        else:
                            cand = (1.0, t1, t1)
                        if best is None or cand[2] < best[2]:
                            best = cand
                    if best is not None:
                        terms.append(best)
                tn = math.inf
                if terms:
                    sa = sum(t_[0] for t_ in terms)
                    sac = sum(t_[0] * t_[1] for t_ in terms)
                    sacc = sum(t_[0] * t_[1] * t_[1] for t_ in terms)
                    disc = sac * sac - sa * (sacc - f * f)
                    if disc >= 0:
                        tn = (sac + math.sqrt(disc)) / sa
                    if not (tn >= max(t_[2] for t_ in terms) - 1e-15) or disc < 0:
                        tn = min(t_[1] + f / math.sqrt(t_[0]) for t_ in terms)
                if tn < T[a, b]:
                    T[a, b] = tn
                    heapq.heappush(heap, (tn, a, b))
        return T

    def oxidize(self, material: str, oxide: str, B_nm2_s: float, B_over_A_nm_s: float,
                time: float, consumed_fraction: float, reference: str):
        """Thermal oxidation by the Deal-Grove law (see the module notes):
        in every column whose top is `material`, or `oxide` lying directly
        on `material`, the oxide grows from its present thickness x0 to x
        with x^2 + A x = B (t + tau), tau = (x0^2 + A x0) / B, A = B / (B/A).
        consumed_fraction of the new oxide comes from the material below
        (its surface moves down), the rest raises the oxide's top. B in
        nm^2/s, B/A in nm/s (for example B = C1 exp(-E1/kT)), with their
        source. Returns the oxide thickness of every column (nm, exact,
        before the cells round it)."""
        if not (B_nm2_s > 0 and B_over_A_nm_s > 0 and time > 0):
            raise ValueError("B, B/A and time must be > 0")
        if not (0 < consumed_fraction < 1):
            raise ValueError("consumed_fraction must be between 0 and 1")
        if not (isinstance(reference, str) and reference.strip()):
            raise ValueError("oxidize needs a reference for B, B/A and the consumed fraction")
        A = B_nm2_s / B_over_A_nm_s
        h = self.h
        out = np.zeros(self.xc.size)
        if not hasattr(self, "_ox_exact"):
            self._ox_exact = {}
        track = self._ox_exact.setdefault(oxide, np.zeros(self.xc.size))
        for i in range(self.xc.size):
            col = self.mat[i]
            ks = np.flatnonzero(col != None)  # noqa: E711
            if ks.size == 0:
                continue
            k = int(ks[-1])
            x0 = 0.0
            while k >= 0 and col[k] == oxide:
                x0 += h
                k -= 1
            if k < 0 or col[k] != material:
                continue
            # the oxide's exact thickness from an earlier oxidation, while the
            # cells still agree with it (the cells round it to h)
            if abs(track[i] - x0) <= h + 1e-9 and track[i] > 0:
                x0 = track[i]
            # the top of the material and of the oxide (cell edges)
            m_top = self.z[k + 1]
            tau = (x0 * x0 + A * x0) / B_nm2_s
            x = 0.5 * A * (math.sqrt(1.0 + 4.0 * B_nm2_s * (time + tau) / (A * A)) - 1.0)
            out[i] = x
            dx = x - x0
            track[i] = x
            x0c = float(np.sum(col[k + 1:] == oxide) * h)
            new_m_top = m_top - consumed_fraction * dx
            new_o_top = m_top + x0c + (1.0 - consumed_fraction) * dx
            if new_o_top > self.z[-1] + 1e-9:
                raise ValueError("the oxide grows out of the top of the domain; raise z_top")
            for kk in range(self.zc.size):
                zc = self.zc[kk]
                if new_m_top <= zc < m_top:
                    self.mat[i, kk] = oxide
                    self.donors[i, kk] = self.acceptors[i, kk] = 0.0
                elif m_top <= zc < new_o_top:
                    self.mat[i, kk] = oxide
        self.log.append(("oxidize", material, oxide, B_nm2_s, B_over_A_nm_s, time,
                         consumed_fraction, reference))
        return out

    def implant(self, imp: "Implant"):
        """An `Implant` into the structure: its depth is measured from the
        top surface of each column (not imp.surface_z), the lateral spread
        and window as for a flat surface. Ions that stop in a cell that is
        not a semiconductor are dropped (taken by the mask or oxide)."""
        if not isinstance(imp, Implant):
            raise ValueError("implant takes an Implant")
        top = self.top_surface()
        X, Z = np.meshgrid(self.xc, self.zc, indexing="ij")
        surf = np.where(np.isfinite(top), top, self.z[0])[:, None]
        shifted = dataclasses.replace(imp, surface_z=0.0)
        c = shifted.concentration(X, Z - surf)
        keep = np.isin(self.mat, list(self.semis))
        c = np.where(keep, c, 0.0)
        if imp.species == "donor":
            self.donors += c
        else:
            self.acceptors += c
        self.log.append(("implant", imp))

    def anneal(self, an: "Anneal"):
        """An `Anneal`: dC/dt = D lap C inside the semiconductor cells, no
        flux through any face to another material or gas, exact in time
        (the matrix exponential acting on the profile)."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import expm_multiply
        if not isinstance(an, Anneal):
            raise ValueError("anneal takes an Anneal")
        semi = np.isin(self.mat, list(self.semis))
        idx = -np.ones(semi.shape, int)
        idx[semi] = np.arange(int(semi.sum()))
        n = int(semi.sum())
        if n == 0:
            return
        rows, cols, vals = [], [], []
        for di, dk in ((1, 0), (0, 1)):
            a = semi[:semi.shape[0] - di, :semi.shape[1] - dk]
            b = semi[di:, dk:]
            both = a & b
            ia = idx[:semi.shape[0] - di, :semi.shape[1] - dk][both]
            ib = idx[di:, dk:][both]
            rows += [ia, ib, ia, ib]
            cols += [ib, ia, ia, ib]
            one = np.ones(ia.size)
            vals += [one, one, -one, -one]
        Lap = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n, n)) / (self.h * self.h)
        M = (an.diffusivity_nm2_s * an.time_s) * Lap
        for name in (("donors",) if an.species == "donor" else ("acceptors",) if
                     an.species == "acceptor" else ("donors", "acceptors")):
            C = getattr(self, name)
            v = C[semi]
            if np.any(v > 0):
                C[semi] = expm_multiply(M, v)
        self.log.append(("anneal", an))

    # ---------------------------------------------------------- to a device
    def mesh(self):
        """The cells as a `mesh.Mesh` (every cell two right triangles, gas
        left out), regions named by material."""
        from .mesh import mesh_from_grid

        def region_of(xx, zz):
            i = int(np.clip(np.searchsorted(self.x, xx) - 1, 0, self.xc.size - 1))
            k = int(np.clip(np.searchsorted(self.z, zz) - 1, 0, self.zc.size - 1))
            return self.mat[i, k]
        return mesh_from_grid(self.x, self.z, region_of)

    def doping(self):
        """(x, z) -> (N_d, N_a) in cm^-3 at a node: the mean over the
        semiconductor cells that touch it (for `MeshDevice(doping=)`)."""
        semi = np.isin(self.mat, list(self.semis))

        def f(x, z):
            i = int(round((x - self.x[0]) / self.h))
            k = int(round((z - self.z[0]) / self.h))
            nd = na = w = 0.0
            for a in (i - 1, i):
                for b in (k - 1, k):
                    if 0 <= a < self.xc.size and 0 <= b < self.zc.size and semi[a, b]:
                        nd += self.donors[a, b]
                        na += self.acceptors[a, b]
                        w += 1.0
            return (nd / w, na / w) if w else (0.0, 0.0)
        return f

    def surface_nodes(self, mesh, material: str, x0: float, x1: float):
        """Mesh nodes on the exposed top of `material` (a cell of it with gas,
        or the top of the domain, above) with x0 <= x <= x1: contacts."""
        P = mesh.points
        out = []
        for n_, (xx, zz) in enumerate(P):
            if not (x0 - 1e-9 <= xx <= x1 + 1e-9):
                continue
            i = int(round((xx - self.x[0]) / self.h))
            k = int(round((zz - self.z[0]) / self.h))
            below = [self.mat[a, k - 1] for a in (i - 1, i)
                     if 0 <= a < self.xc.size and 0 <= k - 1 < self.zc.size]
            above = [self.mat[a, k] for a in (i - 1, i)
                     if 0 <= a < self.xc.size and 0 <= k < self.zc.size]
            if material in below and all(v is None for v in above):
                out.append(n_)
        return np.array(out, int)
