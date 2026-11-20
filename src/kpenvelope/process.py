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
- Since 0.24.0, `deposit` and `etch` also take mode="flux": every exposed
  face receives the particles that reach it from above in a straight
  line (`fluxes`: an in-plane angular distribution cos^n, n = 1 for a gas
  at rest and large n for an ion beam, exact shadowing by the cell
  geometry, no re-emission), so a narrow trench etches or fills more
  slowly at the bottom than a wide one; and `oxidize` takes mode="2d":
  the oxidant diffuses through the oxide in two dimensions (Deal and
  Grove's B and B/A as the diffusion and reaction), so oxide grows under
  a mask edge where a pad oxide lets the oxidant in (the bird's beak).
  The new oxide's extra volume lifts its column (no stress or viscous
  flow).
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
_BEAM_CACHE = {}


def _beam_tables(n):
    """Cumulative flux G_side(phi) of the in-plane distribution
    cos^n(phi) / C_n (flux through a horizontal plane) onto a face facing
    up (side 0), +x (1) or -x (2): G(b) - G(a) is the flux from the
    directions a..b (radians from the vertical)."""
    if n in _BEAM_CACHE:
        return _BEAM_CACHE[n]
    from math import lgamma, sqrt
    phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 40001)
    c = np.cos(phi)
    Cn = sqrt(math.pi) * math.exp(lgamma(0.5 * (n + 1)) - lgamma(0.5 * n + 1))
    base = np.where(c > 0, c ** (n - 1), 0.0) / Cn          # p(phi) / cos(phi)
    out = {}
    for side, w in ((0, c), (1, np.sin(phi)), (2, -np.sin(phi))):
        f = base * np.maximum(w, 0.0)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(phi))])
        out[side] = (lambda x, cum=cum: float(np.interp(x, phi, cum)))
    _BEAM_CACHE[n] = out
    return out


def _gaps(lo, hi, a, b):
    """The parts of [a, b] not covered by the intervals [lo, hi]."""
    if lo.size == 0:
        return [(a, b)]
    o = np.argsort(lo)
    lo, hi = lo[o], hi[o]
    run = np.maximum.accumulate(hi)
    out = []
    if lo[0] > a:
        out.append((a, min(lo[0], b)))
    gap = lo[1:] > run[:-1]
    for s_, e_ in zip(run[:-1][gap], lo[1:][gap]):
        u, v = max(s_, a), min(e_, b)
        if v > u:
            out.append((u, v))
    if run[-1] < b:
        out.append((max(run[-1], a), b))
    return out


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
                acceptors: float = 0.0, exponent: float = 1.0):
        """Grow `material` by `thickness` nm on the exposed surface:
        conformal (every gas cell connected to the top whose center lies
        within thickness + h/2 of a solid cell center, the Euclidean
        distance on the cell grid; on a flat surface exactly thickness/h
        cells) or directional (on top of each column, within window
        (x0, x1) if given) or flux (new in 0.24.0: the film grows where the
        particles from above land, see `fluxes`, with the in-plane angular
        distribution cos^exponent; thickness is what grows on a flat, open
        surface, so a trench bottom or wall that sees less of the sky gets
        less, and overhangs shadow what lies below them). Epitaxial growth
        of a semiconductor: give semiconductor=True and its doping as grown
        (donors, acceptors in cm^-3)."""
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
        elif mode == "flux":
            if window is not None:
                raise ValueError("mode='flux' covers the whole surface (no window)")
            self._flux_step([(float(exponent), 1.0)], float(thickness),
                            deposit_material=material)
        else:
            raise ValueError("mode must be 'conformal', 'directional' or 'flux'")
        new = self.solid & ~before
        self.donors[new] = donors
        self.acceptors[new] = acceptors
        self.log.append(("deposit", material, thickness, mode))

    def etch(self, rates: dict, time: float, mode: str = "isotropic", window=None,
             exponent: float = 1.0, beams=None):
        """Remove material: rates {label: nm/s} (a label not listed, or a
        rate of 0, is not etched: a mask), for `time` seconds. isotropic:
        the front starts on every surface the gas touches (within window if
        given) and moves into each material at its rate; a cell goes when
        the front passes its center (fast marching).
        directional: every column in window is etched straight down from
        its top.
        flux (new in 0.24.0): the etch rate of every exposed cell is its
        rate on a flat, open surface times the flux of particles that reach
        it (`fluxes`, angular distribution cos^exponent), so narrow trenches
        etch more slowly at the bottom (aspect-ratio dependent etching), a
        beam etches straight down and a gas at rest also etches the walls
        it sees. beams: several kinds of particle at once, a list of
        (exponent, rates), for example a neutral gas (exponent 1) and an ion
        beam (exponent 100), whose rates add. Masks are the materials with
        no rate."""
        if not time > 0:
            raise ValueError("time must be > 0")
        if mode == "flux":
            if window is not None:
                raise ValueError("mode='flux' covers the whole surface (no window)")
            bl = [(float(exponent), dict(rates))] if beams is None else [
                (float(n), dict(r)) for (n, r) in beams]
            for (n, r) in bl:
                for v in r.values():
                    if not (math.isfinite(v) and v >= 0):
                        raise ValueError("etch rates must be >= 0 (nm/s)")
            self._flux_step(bl, float(time))
            self.log.append(("etch", dict(rates), time, mode))
            return
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
            raise ValueError("mode must be 'isotropic', 'directional' or 'flux'")
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

    # ------------------------------------------- line-of-sight fluxes (0.24.0)
    def _faces(self):
        """The faces between solid cells and the open gas that particles
        from above can reach: (i, k, side) with side 0 = top, 1 = right
        (+x), 2 = left (-x), and their centers. Downward faces (overhangs)
        see no particle from above and are left out."""
        sol = self.solid
        gas = self._gas_open()
        nx, nz = sol.shape
        out = []
        top = sol.copy()
        top[:, :-1] &= gas[:, 1:]
        for i, k in zip(*np.nonzero(top)):
            out.append((i, k, 0, self.xc[i], self.z[k + 1]))
        right = np.zeros_like(sol)
        right[:-1] = sol[:-1] & gas[1:]
        for i, k in zip(*np.nonzero(right)):
            out.append((i, k, 1, self.x[i + 1], self.zc[k]))
        left = np.zeros_like(sol)
        left[1:] = sol[1:] & gas[:-1]
        for i, k in zip(*np.nonzero(left)):
            out.append((i, k, 2, self.x[i], self.zc[k]))
        return out

    def fluxes(self, exponent: float = 1.0):
        """The flux of particles from above onto every exposed face,
        relative to a flat, open surface (new in 0.24.0).

        The particles arrive with the in-plane angular distribution
        p(phi) ~ cos^n(phi) of their flux through a horizontal plane (phi
        from the vertical; n = exponent). n = 1 is a gas at rest above the
        wafer (cosine law in 3D, integrated along the out-of-plane
        direction: dGamma = (1/2) cos(phi - phi_n) dphi on a face with
        normal angle phi_n, the crossed-strings view factor of 2D
        geometries); a large n is a narrow ion beam. Each face receives the
        directions that reach the sky without crossing a solid cell (exact
        for the cell geometry: every solid cell's shadow is the angular
        range of its corners), with mirror images of the cross-section on
        both sides. Particles stick where they land (no re-emission).

        Returns a list of (i, k, side, flux), side 0 = top, 1 = +x, 2 = -x.
        """
        if not (math.isfinite(exponent) and exponent >= 1.0):
            raise ValueError("exponent must be >= 1 (1: a gas at rest, large: a beam)")
        faces = self._faces()
        if not faces:
            return []
        G = _beam_tables(float(exponent))
        sol = self.solid
        gas = self._gas_open()
        nb = np.zeros_like(sol)
        nb[:-1] |= gas[1:]
        nb[1:] |= gas[:-1]
        nb[:, :-1] |= gas[:, 1:]
        nb[:, 1:] |= gas[:, :-1]
        ii, kk = np.nonzero(sol & nb)                       # surface cells block
        x0c, x1c = self.x[ii], self.x[ii + 1]
        z0c, z1c = self.z[kk], self.z[kk + 1]
        # the cross-section and its mirror images on both sides
        X0 = np.concatenate([x0c, 2 * self.x[0] - x1c, 2 * self.x[-1] - x1c])
        X1 = np.concatenate([x1c, 2 * self.x[0] - x0c, 2 * self.x[-1] - x0c])
        Z0 = np.tile(z0c, 3)
        Z1 = np.tile(z1c, 3)
        own = np.concatenate([np.stack([ii, kk], 1), -np.ones((2 * ii.size, 2), int)])
        rng = {0: (-0.5 * math.pi, 0.5 * math.pi), 1: (0.0, 0.5 * math.pi),
               2: (-0.5 * math.pi, 0.0)}
        out = []
        tol = 1e-9 * self.h
        for (i, k, side, px, pz) in faces:
            sel = (Z1 > pz + tol) & ~((own[:, 0] == i) & (own[:, 1] == k))
            dxs = np.stack([X0[sel] - px, X1[sel] - px, X0[sel] - px, X1[sel] - px], 1)
            dzs = np.stack([Z0[sel] - pz, Z0[sel] - pz, Z1[sel] - pz, Z1[sel] - pz], 1)
            th = np.arctan2(dxs, dzs)
            lo = np.clip(th.min(1), -0.5 * math.pi, 0.5 * math.pi)
            hi = np.clip(th.max(1), -0.5 * math.pi, 0.5 * math.pi)
            a, b = rng[side]
            keep = (hi > a) & (lo < b) & (hi > lo)
            lo, hi = lo[keep], hi[keep]
            vis = _gaps(lo, hi, a, b)
            f = 0.0
            for (u, v) in vis:
                f += G[side](v) - G[side](u)
            out.append((int(i), int(k), int(side), max(f, 0.0)))
        return out

    def _flux_step(self, beams, time, deposit_material=None):
        """Advance etching (deposit_material None) or deposition for `time`
        with the face fluxes of `beams` [(exponent, {material: rate})] or,
        for deposition, [(exponent, rate)]. Each cell collects removed (or
        added) area; at h^2 it changes and the rest goes on into the next
        cell along its fastest face. The fluxes are recomputed after every
        step, and no surface moves more than half a cell in one step."""
        h = self.h
        nx, nz = self.mat.shape
        acc = np.zeros((nx, nz))
        normal = {0: (0, 1), 1: (1, 0), 2: (-1, 0)}
        etchable = set() if deposit_material is not None else {
            m for (_, r) in beams for m, v in r.items() if v > 0}
        t = 0.0
        steps = 0
        while t < time * (1 - 1e-12):
            rate = np.zeros((nx, nz))
            best = np.zeros((nx, nz))
            bdir = np.zeros((nx, nz), int)
            for (n, rates) in beams:
                for (i, k, side, f) in self.fluxes(n):
                    if f <= 0:
                        continue
                    if deposit_material is None:
                        r = rates.get(self.mat[i, k], 0.0) * f * h
                        a_, b_ = i, k
                    else:
                        di, dk = normal[side]
                        a_, b_ = i + di, k + dk
                        if not (0 <= a_ < nx and 0 <= b_ < nz):
                            raise ValueError("the film grows out of the domain; raise z_top")
                        r = rates * f * h
                    if r > 0:
                        rate[a_, b_] += r
                        if r > best[a_, b_]:
                            best[a_, b_] = r
                            bdir[a_, b_] = side
            if not np.any(rate > 0):
                break
            dt = min(time - t, 0.5 * h * h / float(rate.max()))
            acc += rate * dt
            want = (lambda c: c is not None) if deposit_material is None else (lambda c: c is None)
            while True:
                done = (acc >= h * h * (1 - 1e-12))
                if not done.any():
                    break
                for i, k in zip(*np.nonzero(done)):
                    e = acc[i, k] - h * h
                    acc[i, k] = 0.0
                    if not want(self.mat[i, k]):
                        continue
                    if deposit_material is None:
                        self.mat[i, k] = None
                        self.donors[i, k] = self.acceptors[i, k] = 0.0
                        di, dk = normal[bdir[i, k]]
                        a_, b_ = i - di, k - dk          # deeper into the solid
                    else:
                        self.mat[i, k] = deposit_material
                        di, dk = normal[bdir[i, k]]
                        a_, b_ = i + di, k + dk          # further into the gas
                    if 0 <= a_ < nx and 0 <= b_ < nz and want(self.mat[a_, b_]) and (
                            deposit_material is not None or self.mat[a_, b_] in etchable):
                        acc[a_, b_] += e
                        bdir[a_, b_] = bdir[i, k]
            t += dt
            steps += 1
            if steps > 100000:
                raise RuntimeError("the flux step did not finish; use larger cells")
        # a cell more than half done at the end goes, as with the other modes
        # (the front has passed its center)
        half = acc >= 0.5 * h * h
        if deposit_material is None:
            half &= self.solid
            self.mat[half] = None
            self.donors[half] = self.acceptors[half] = 0.0
        else:
            half &= ~self.solid
            self.mat[half] = deposit_material
        return steps

    def oxidize(self, material: str, oxide: str, B_nm2_s: float, B_over_A_nm_s: float,
                time: float, consumed_fraction: float, reference: str,
                mode: str = "column"):
        """Thermal oxidation by the Deal-Grove law (see the module notes):
        in every column whose top is `material`, or `oxide` lying directly
        on `material`, the oxide grows from its present thickness x0 to x
        with x^2 + A x = B (t + tau), tau = (x0^2 + A x0) / B, A = B / (B/A).
        consumed_fraction of the new oxide comes from the material below
        (its surface moves down), the rest raises the oxide's top. B in
        nm^2/s, B/A in nm/s (for example B = C1 exp(-E1/kT)), with their
        source. Returns the oxide thickness of every column (nm, exact,
        before the cells round it).

        mode="2d" (new in 0.24.0): the oxidant diffuses through the oxide in
        two dimensions, so oxide also grows sideways and under the edge of a
        mask that lies on oxide (see `_oxidize_2d`); then the returned
        thicknesses are cell counts (nm)."""
        if not (B_nm2_s > 0 and B_over_A_nm_s > 0 and time > 0):
            raise ValueError("B, B/A and time must be > 0")
        if not (0 < consumed_fraction < 1):
            raise ValueError("consumed_fraction must be between 0 and 1")
        if not (isinstance(reference, str) and reference.strip()):
            raise ValueError("oxidize needs a reference for B, B/A and the consumed fraction")
        if mode == "2d":
            return self._oxidize_2d(material, oxide, B_nm2_s, B_over_A_nm_s, time,
                                    consumed_fraction, reference)
        if mode != "column":
            raise ValueError("mode must be 'column' or '2d'")
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

    def _oxidant(self, material, oxide, A, with_dirs=False, extra_thick=None):
        """Steady oxidant in the oxide (new in 0.24.0), normalized to its
        value at the gas surface: div grad C = 0 in the oxide cells, C = 1
        on faces to the open gas (and the top of the domain), no flux into
        masks or through the sides and bottom, and at the oxidizable
        material the reaction flux k C with k / D = 2 / A (Deal and Grove:
        A = 2 D / k). Returns (i, k, c): the material cells behind every
        reacting face and the normalized oxidant at that face (c = 1 where
        the material is bare under the gas). extra_thick (4, nx, nz): oxide
        not yet in whole cells in front of each face (nm), added to its
        diffusion path."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        h = self.h
        ox = self.mat == oxide
        mt = self.mat == material
        gas = self._gas_open()
        nx, nz = ox.shape

        def pad(m, top=False):
            p_ = np.zeros((nx + 2, nz + 2), bool)
            p_[1:-1, 1:-1] = m
            if top:
                p_[1:-1, -1] = True          # above the domain: open gas
            return p_
        oxp, mtp, gasp = pad(ox), pad(mt), pad(gas, True)
        idx = -np.ones((nx + 2, nz + 2), int)
        n = int(ox.sum())
        idx[oxp] = np.arange(n)
        ext = np.zeros((4, nx, nz)) if extra_thick is None else extra_thick
        diag = np.zeros(n)
        rhs = np.zeros(n)
        rows, cols = [], []
        li, lk, lc_row, ld, lx = [], [], [], [], []   # cell, oxide row (-1: bare), side, extra
        core = (slice(1, -1), slice(1, -1))
        for q, (di, dk) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            nb = (slice(1 + di, nx + 1 + di), slice(1 + dk, nz + 1 + dk))
            o_ox = oxp[core] & oxp[nb]
            rows.append(idx[core][o_ox])
            cols.append(idx[nb][o_ox])
            o_g = oxp[core] & gasp[nb]
            np.add.at(diag, idx[core][o_g], 2.0)
            np.add.at(rhs, idx[core][o_g], 2.0)
            o_m = oxp[core] & mtp[nb]
            I, K = np.nonzero(o_m)
            # seen from the material cell, the oxide lies in direction -d
            qq = (1, 0, 3, 2)[q]
            dx_ = ext[qq, I + di, K + dk]
            # oxide cell center -> reacting face: half a cell and the oxide
            # not yet in cells by diffusion, then the reaction (D = 1, k = 2/A)
            np.add.at(diag, idx[core][o_m], 2.0 * h / (A + h + 2.0 * dx_))
            li.append(I + di)
            lk.append(K + dk)
            lc_row.append(idx[core][o_m])
            ld.append(np.full(I.size, qq))
            lx.append(dx_)
            m_g = mtp[core] & gasp[nb]
            I, K = np.nonzero(m_g)
            li.append(I)
            lk.append(K)
            lc_row.append(-np.ones(I.size, int))
            ld.append(np.full(I.size, q))
            lx.append(ext[q, I, K])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        if n:
            np.add.at(diag, rows, 1.0)
            M = sp.csr_matrix((np.concatenate([diag, -np.ones(rows.size)]),
                               (np.concatenate([np.arange(n), rows]),
                                np.concatenate([np.arange(n), cols]))), shape=(n, n))
            C = spsolve(M.tocsc(), rhs) if np.any(rhs) else np.zeros(n)
        else:
            C = np.zeros(0)
        li, lk, r = np.concatenate(li), np.concatenate(lk), np.concatenate(lc_row)
        x_ = np.concatenate(lx)
        # interface value: the reaction flux k C_i equals the diffusive flux
        # from the oxide cell center (or from the gas, for bare material)
        Cr = C[np.maximum(r, 0)] if n else np.zeros(r.size)
        c = np.where(r >= 0, Cr * A / (A + h + 2.0 * x_), A / (A + 2.0 * x_))
        if with_dirs:
            return li, lk, c, np.concatenate(ld)
        return li, lk, c

    def _oxidize_2d(self, material, oxide, B, BA, time, cf, reference):
        """Two-dimensional oxidation (new in 0.24.0): the oxidant diffuses
        through the oxide in two dimensions (`_oxidant`), each face where
        the material meets oxide or gas grows oxide at (B/A) c, c the
        normalized oxidant there, consuming cf of the new oxide's volume
        from the material cell behind the face. The rest of the new oxide's
        volume goes in front of the face (`_push_oxide`): above a face that
        looks up, the column moves up (a mask on top is lifted); in front
        of a face that looks sideways, it fills the nearest open gas. So
        oxide grows under the edge of a mask wherever oxidant reaches it
        through oxide (a pad oxide), as in LOCOS. Far
        from any edge this is Deal-Grove: in one dimension
        c = 1 / (1 + 2 x / A) and dx/dt = B / (A + 2 x)."""
        A = B / BA
        h = self.h
        nx, nz = self.mat.shape
        acc = np.zeros((nx, nz))            # material area consumed (nm^2)
        lift = np.zeros((4, nx, nz))        # oxide area still to put in front of each face
        t = 0.0
        steps = 0
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while t < time * (1 - 1e-12):
            # the oxide grown in front of each face but not yet in whole
            # cells: in place of the material consumed so far in the cell, and
            # the new volume still to be put in
            ext = (acc[None, :, :] / h) * np.ones((4, 1, 1)) + lift / h
            ext = np.where((self.mat == material)[None], ext, 0.0)
            li, lk, c, dl = self._oxidant(material, oxide, A, with_dirs=True,
                                          extra_thick=ext)
            if li.size == 0:
                break
            v = BA * c                      # oxide growth speed at each face, nm/s
            rate = np.zeros((nx, nz))       # material area consumed per s
            np.add.at(rate, (li, lk), cf * v * h)
            extra = np.zeros((4, nx, nz))   # oxide area to put in front, per s
            np.add.at(extra, (dl, li, lk), (1.0 - cf) * v * h)
            # the face that consumes a cell fastest sets where it continues
            best = np.full((nx, nz), -1.0)
            bdir = np.zeros((nx, nz), int)
            for q in range(4):
                m = dl == q
                cand = np.zeros((nx, nz))
                np.add.at(cand, (li[m], lk[m]), v[m])
                upd = cand > best
                best[upd] = cand[upd]
                bdir[upd] = q
            # each step moves no interface by more than half a cell
            dt = min(time - t, 0.5 * h * h / float(rate.max()))
            acc += rate * dt
            lift += extra * dt
            while True:
                done = (rate > 0) & (acc >= h * h * (1 - 1e-12)) & (self.mat == material)
                if not done.any():
                    break
                for i, k in zip(*np.nonzero(done)):
                    e = acc[i, k] - h * h
                    self.mat[i, k] = oxide
                    self.donors[i, k] = self.acceptors[i, k] = 0.0
                    acc[i, k] = 0.0
                    # the rest goes on into the next material cell, away from
                    # the face (the oxide side)
                    di, dk = dirs[bdir[i, k]]
                    a2, b2 = i - di, k - dk
                    if 0 <= a2 < nx and 0 <= b2 < nz and self.mat[a2, b2] == material:
                        acc[a2, b2] += e
                        rate[a2, b2] = max(rate[a2, b2], 1e-300)
                        # so does the new oxide still to be put in
                        lift[:, a2, b2] += lift[:, i, k]
                        lift[:, i, k] = 0.0
            for q, i, k in zip(*np.nonzero(lift >= h * h * (1 - 1e-12))):
                while lift[q, i, k] >= h * h * (1 - 1e-12):
                    self._push_oxide(i, k, dirs[q], oxide)
                    lift[q, i, k] -= h * h
            t += dt
            steps += 1
            if steps > 200000:
                raise RuntimeError("the oxidation did not finish; use larger cells")
        # a cell more than half consumed at the end turns to oxide, and half a
        # cell of new volume or more is put in (the front passed the center)
        half = (acc >= 0.5 * h * h) & (self.mat == material)
        self.mat[half] = oxide
        self.donors[half] = self.acceptors[half] = 0.0
        for q, i, k in zip(*np.nonzero(lift >= 0.5 * h * h)):
            self._push_oxide(i, k, dirs[q], oxide)
        self.log.append(("oxidize", material, oxide, B, BA, time, cf, reference, "2d"))
        top = np.zeros(nx)
        for i in range(nx):
            top[i] = float(np.sum(self.mat[i] == oxide) * h)
        return top

    def _push_oxide(self, i, k, d, oxide):
        """One cell of new oxide in front of the face of cell (i, k) facing
        direction d. A face looking up or down: the cell in front gets the
        new oxide and everything above it in its column moves up one cell,
        up to the first gas cell (a mask on top is lifted). A face looking
        sideways: the new oxide fills the open gas cell nearest to the
        cell in front (the oxide swells out through its free surface)."""
        from scipy import ndimage
        nx, nz = self.mat.shape
        di, dk = d
        a0, b0 = i + di, k + dk
        if not (0 <= a0 < nx and 0 <= b0 < nz):
            a0, b0 = i, k
        if dk == 0:
            gas = self._gas_open()
            if gas.any():
                _, (ia, ib) = ndimage.distance_transform_edt(~gas, return_indices=True)
                g0, g1 = int(ia[a0, b0]), int(ib[a0, b0])
                self.mat[g0, g1] = oxide
                self.donors[g0, g1] = self.acceptors[g0, g1] = 0.0
                return
        # up the column from (a0, b0)
        b = b0
        while b < nz and self.mat[a0, b] is not None:
            b += 1
        if b >= nz:
            raise ValueError("the oxide lifts material out of the domain; raise z_top")
        for arr in (self.mat, self.donors, self.acceptors):
            arr[a0, b0 + 1:b + 1] = arr[a0, b0:b].copy()
        self.mat[a0, b0] = oxide
        self.donors[a0, b0] = self.acceptors[a0, b0] = 0.0

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
