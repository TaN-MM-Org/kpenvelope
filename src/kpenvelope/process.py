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
    depth_profile : (depths in nm, density per nm) to use instead of the
        Gaussian (new in 0.26.0; `bca.RangeResult.implant` makes it from a
        Monte Carlo run); normalized to the dose.
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
    depth_profile: tuple = None

    def __post_init__(self):
        if self.depth_profile is not None:
            dz, f = (np.asarray(v, float) for v in self.depth_profile)
            if not (dz.ndim == 1 and dz.shape == f.shape and dz.size >= 2
                    and np.all(np.diff(dz) > 0) and np.all(f >= 0) and np.any(f > 0)):
                raise ValueError("Implant.depth_profile must be (depths in nm, increasing; "
                                 "density per nm >= 0)")
            trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            object.__setattr__(self, "depth_profile", (dz, f / trap(f, dz)))
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
        if self.depth_profile is not None:
            dz, f = self.depth_profile
            vert = self.dose_cm2 * 1e7 * np.interp(depth, dz, f, left=0.0, right=0.0)
        else:
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


@dataclasses.dataclass(frozen=True)
class EtchKinetics:
    """Ion-enhanced etching by a reactive gas and an ion flux (new in
    0.26.0), for `Structure.plasma_etch`.

    On every exposed face, a share theta of the adsorption sites is
    covered by the reactive neutrals. Neutrals stick to free sites with
    probability s0 (the rest leave by the cosine law and land elsewhere);
    ions remove ion_yield atoms from covered sites, the covered sites also
    leave thermally at thermal_rate, and ions sputter sputter_yield atoms
    from bare ones. In steady state

        s0 G_n (1 - theta) = (Y G_i + k n_s) theta,
        etch flux = Y G_i theta + k n_s theta + Y_sp G_i (1 - theta),

    with G_n and G_i the neutral and ion fluxes that reach the face (from
    neutral_flux and ion_flux on a flat, open surface, through the
    geometry: `fluxes` with the exponents, and for neutrals also what the
    other faces re-emit, `view_factors`), n_s = site density, and the
    recession speed the etch flux over the material's atom density. Every
    number is yours, with its reference: neutral_flux, ion_flux (cm^-2
    s^-1), site_density (cm^-2), and per material sticking (s0), ion_yield,
    sputter_yield, thermal_rate (1/s) and atoms_cm3. A material without a
    sticking entry takes no neutrals (they leave it again) and is not
    etched. The gas-phase chemistry of the plasma that sets the fluxes is
    not modeled.
    """

    neutral_flux: float
    ion_flux: float
    site_density: float
    sticking: dict
    ion_yield: dict
    atoms_cm3: dict
    reference: str
    sputter_yield: dict = None
    thermal_rate: dict = None
    neutral_exponent: float = 1.0
    ion_exponent: float = 100.0

    def __post_init__(self):
        for f in ("neutral_flux", "ion_flux"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"EtchKinetics.{f} must be >= 0")
        if not (math.isfinite(self.site_density) and self.site_density > 0):
            raise ValueError("EtchKinetics.site_density must be > 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("EtchKinetics needs a reference")
        for f in ("sticking", "ion_yield", "atoms_cm3", "sputter_yield", "thermal_rate"):
            d = getattr(self, f)
            if d is None:
                object.__setattr__(self, f, {})
                continue
            for k_, v in d.items():
                if not (math.isfinite(v) and v >= 0):
                    raise ValueError(f"EtchKinetics.{f}[{k_!r}] must be >= 0")
        for k_, v in self.sticking.items():
            if v > 1:
                raise ValueError("sticking must be in [0, 1]")
        for k_ in self.sticking:
            if not self.atoms_cm3.get(k_, 0) > 0:
                raise ValueError(f"EtchKinetics.atoms_cm3 needs {k_!r} (> 0)")
        if not (self.neutral_exponent >= 1 and self.ion_exponent >= 1):
            raise ValueError("exponents must be >= 1 (see Structure.fluxes)")

    def flat(self, material):
        """(coverage, recession speed in nm/s) on a flat, open surface."""
        return self.rate(material, self.neutral_flux, self.ion_flux)

    def rate(self, material, Gn, Gi):
        """(coverage, recession speed nm/s) for neutral and ion fluxes
        (cm^-2 s^-1) arriving at a face of `material` (arrays allowed)."""
        s0 = self.sticking.get(material, 0.0)
        if s0 <= 0:
            z = np.zeros_like(np.asarray(Gn, float))
            return z, z
        Y = self.ion_yield.get(material, 0.0)
        Ysp = self.sputter_yield.get(material, 0.0)
        k = self.thermal_rate.get(material, 0.0)
        Gn = np.asarray(Gn, float)
        Gi = np.asarray(Gi, float)
        a = s0 * Gn
        b = Y * Gi + k * self.site_density
        tot = a + b
        th = np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), 0.0)
        R = Y * Gi * th + k * self.site_density * th + Ysp * Gi * (1.0 - th)
        return th, R / self.atoms_cm3[material] * 1e7


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
                acceptors: float = 0.0, exponent: float = 1.0, sticking: float = 1.0):
        """Grow `material` by `thickness` nm on the exposed surface:
        conformal (every gas cell connected to the top whose center lies
        within thickness + h/2 of a solid cell center, the Euclidean
        distance on the cell grid; on a flat surface exactly thickness/h
        cells) or directional (on top of each column, within window
        (x0, x1) if given) or flux (new in 0.24.0: the film grows where the
        particles from above land, see `fluxes`, with the in-plane angular
        distribution cos^exponent; thickness is what grows on a flat, open
        surface, so a trench bottom or wall that sees less of the sky gets
        less, and overhangs shadow what lies below them; sticking < 1 (new
        in 0.25.0) re-emits the rest by the cosine law, which makes films in
        narrow openings more even). Epitaxial growth
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
            if not 0.0 < sticking <= 1.0:
                raise ValueError("sticking must be in (0, 1]")
            self._flux_step([(float(exponent), 1.0)], float(thickness),
                            deposit_material=material, sticking=float(sticking))
        else:
            raise ValueError("mode must be 'conformal', 'directional' or 'flux'")
        new = self.solid & ~before
        self.donors[new] = donors
        self.acceptors[new] = acceptors
        self.log.append(("deposit", material, thickness, mode))

    def etch(self, rates: dict, time: float, mode: str = "isotropic", window=None,
             exponent: float = 1.0, beams=None, sticking: float = 1.0):
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
        no rate. sticking (new in 0.25.0): the share of the particles that
        react where they land; the rest leave again by the cosine law
        (`view_factors`) and land elsewhere, so with low sticking narrow
        openings etch more evenly. The rates are those of a flat, open
        surface."""
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
            if not 0.0 < sticking <= 1.0:
                raise ValueError("sticking must be in (0, 1]")
            self._flux_step(bl, float(time), sticking=float(sticking))
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

    def _all_faces(self):
        """Every face between a solid cell and the open gas: (i, k, side),
        side 0 = up, 1 = +x, 2 = -x, 3 = down, with its center and normal."""
        sol = self.solid
        gas = self._gas_open()
        nx, nz = sol.shape
        out = []
        for side, (di, dk) in enumerate(((0, 1), (1, 0), (-1, 0), (0, -1))):
            nb = np.zeros_like(sol)
            if di == 1:
                nb[:-1] = gas[1:]
            elif di == -1:
                nb[1:] = gas[:-1]
            elif dk == 1:
                nb[:, :-1] = gas[:, 1:]
                nb[:, -1] = True                        # above the domain
            else:
                nb[:, 1:] = gas[:, :-1]
            for i, k in zip(*np.nonzero(sol & nb)):
                cx = self.xc[i] + 0.5 * di * self.h
                cz = self.zc[k] + 0.5 * dk * self.h
                out.append((int(i), int(k), side, cx, cz, di, dk))
        return out

    def view_factors(self, n_bins: int = 1440, points: int = 4):
        """Diffuse (cosine-law) view factors between the exposed faces (new in
        0.25.0): F[a, b] is the share of what face a re-emits that reaches
        face b first, the rest leaving through the top. In the plane of the
        cross-section a cosine emitter sends (1/2) cos(alpha) d alpha into
        the angle alpha from its normal (the 2D crossed-strings law). Each
        face's half-plane is cut into n_bins directions; along each, the
        nearest solid cell (by its center) takes the bin, and the ray enters
        it through the face it meets first. The cross-section repeats as its
        mirror image at both sides. F is averaged over `points` points along
        each emitting face, which makes it reciprocal (F[a, b] = F[b, a] for
        faces of equal length) to the accuracy of that average. Returns
        (faces, F)."""
        faces = self._all_faces()
        nf = len(faces)
        index = {(f[0], f[1], f[2]): q for q, f in enumerate(faces)}
        sol = self.solid
        gas = self._gas_open()
        nb = np.zeros_like(sol)
        nb[:-1] |= gas[1:]
        nb[1:] |= gas[:-1]
        nb[:, :-1] |= gas[:, 1:]
        nb[:, 1:] |= gas[:, :-1]
        nb[:, -1] |= sol[:, -1]
        ii, kk = np.nonzero(sol & nb)
        x0c, x1c, z0c, z1c = self.x[ii], self.x[ii + 1], self.z[kk], self.z[kk + 1]
        X0 = np.concatenate([x0c, 2 * self.x[0] - x1c, 2 * self.x[-1] - x1c])
        X1 = np.concatenate([x1c, 2 * self.x[0] - x0c, 2 * self.x[-1] - x0c])
        Z0, Z1 = np.tile(z0c, 3), np.tile(z1c, 3)
        img = np.repeat([0, 1, 2], ii.size)                # 1, 2: mirror images
        ci, ck = np.tile(ii, 3), np.tile(kk, 3)
        edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_bins + 1)
        wbin = 0.5 * (np.sin(edges[1:]) - np.sin(edges[:-1]))      # cosine law
        mid = 0.5 * (edges[1:] + edges[:-1])
        F = np.zeros((nf, nf))
        offs = ((np.arange(points) + 0.5) / points - 0.5) * self.h
        samples = [(a, f, o) for a, f in enumerate(faces) for o in offs]
        for (a, (i, k, side, px0, pz0, nxn, nzn), o) in samples:
            tx, tz = nzn, -nxn                               # tangent
            px, pz = px0 + o * tx, pz0 + o * tz             # a point along the face
            own = (img == 0) & (ci == i) & (ck == k)
            dx = np.stack([X0 - px, X1 - px, X0 - px, X1 - px], 1)
            dz = np.stack([Z0 - pz, Z0 - pz, Z1 - pz, Z1 - pz], 1)
            dn = dx * nxn + dz * nzn
            dt = dx * tx + dz * tz
            front = (dn.max(1) > 1e-9 * self.h) & ~own
            if not front.any():
                continue
            al = np.arctan2(dt[front], np.maximum(dn[front], 0.0))
            lo = np.clip(al.min(1), -0.5 * math.pi, 0.5 * math.pi)
            hi = np.clip(al.max(1), -0.5 * math.pi, 0.5 * math.pi)
            b0 = np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, n_bins - 1)
            b1 = np.clip(np.searchsorted(edges, hi, side="left") - 1, 0, n_bins - 1)
            cnt = np.maximum(b1 - b0 + 1, 0)
            cell = np.repeat(np.arange(lo.size), cnt)
            if cell.size == 0:
                continue
            bins = np.repeat(b0, cnt) + (np.arange(cell.size) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            fidx = np.flatnonzero(front)[cell]
            # the ray of each bin's center direction, and where it enters the
            # cell (slab method); pairs it misses are dropped
            ang = mid[bins]
            ux = np.cos(ang) * nxn + np.sin(ang) * tx
            uz = np.cos(ang) * nzn + np.sin(ang) * tz
            with np.errstate(divide="ignore", invalid="ignore"):
                txa = (X0[fidx] - px) / ux
                txb = (X1[fidx] - px) / ux
                tza = (Z0[fidx] - pz) / uz
                tzb = (Z1[fidx] - pz) / uz
            tx_in = np.where(np.abs(ux) > 1e-15, np.minimum(txa, txb), -np.inf)
            tx_out = np.where(np.abs(ux) > 1e-15, np.maximum(txa, txb), np.inf)
            tz_in = np.where(np.abs(uz) > 1e-15, np.minimum(tza, tzb), -np.inf)
            tz_out = np.where(np.abs(uz) > 1e-15, np.maximum(tza, tzb), np.inf)
            # a ray parallel to a slab must lie inside it
            okx = (np.abs(ux) > 1e-15) | ((px > X0[fidx]) & (px < X1[fidx]))
            okz = (np.abs(uz) > 1e-15) | ((pz > Z0[fidx]) & (pz < Z1[fidx]))
            t_in = np.maximum(tx_in, tz_in)
            t_out = np.minimum(tx_out, tz_out)
            hit = okx & okz & (t_out > t_in) & (t_out > 1e-12 * self.h)
            cell, bins, fidx = cell[hit], bins[hit], fidx[hit]
            t_in, ux, uz = t_in[hit], ux[hit], uz[hit]
            tx_in, tz_in = tx_in[hit], tz_in[hit]
            if cell.size == 0:
                continue
            order = np.lexsort((t_in, bins))
            first = order[np.r_[True, bins[order][1:] != bins[order][:-1]]]
            bins, fidx = bins[first], fidx[first]
            ux, uz, tx_in, tz_in = ux[first], uz[first], tx_in[first], tz_in[first]
            # the entered face: the slab crossed last on the way in
            sx = np.where(ux > 0, 2, 1)                     # moving +x enters the -x face
            sz = np.where(uz > 0, 3, 0)                     # moving up enters the down face
            first_side = np.where(tx_in > tz_in, sx, sz)
            other_side = np.where(tx_in > tz_in, sz, sx)
            mirror = img[fidx] > 0
            swap = {1: 2, 2: 1, 0: 0, 3: 3}
            hi_, hk_ = ci[fidx], ck[fidx]
            for q in range(bins.size):
                for sd in (int(first_side[q]), int(other_side[q])):
                    if mirror[q]:
                        sd = swap[sd]
                    b = index.get((int(hi_[q]), int(hk_[q]), sd))
                    if b is not None:
                        F[a, b] += wbin[bins[q]] / points
                        break
        return faces, F

    def _radiosity(self, faces, F, direct, sticking):
        """Total arriving flux on every face when a share 1 - s of what
        arrives is re-emitted by the cosine law: Gamma = direct + F^T ((1 - s)
        Gamma)."""
        keep = 1.0 - np.broadcast_to(np.asarray(sticking, float), (len(faces),))
        A = np.eye(len(faces)) - F.T * keep[None, :]
        return np.linalg.solve(A, direct)

    def _face_fluxes(self, n, sticking):
        """(i, k, side, flux) for every exposed face: the direct flux of
        `fluxes`, and with sticking < 1 also what the faces re-emit."""
        if sticking >= 1.0:
            return self.fluxes(n)
        faces, F = self.view_factors()
        direct = {(i, k, sd): f for (i, k, sd, f) in self.fluxes(n)}
        d = np.array([direct.get((f[0], f[1], f[2]), 0.0) for f in faces])
        G = self._radiosity(faces, F, d, sticking)
        return [(f[0], f[1], f[2], max(float(g), 0.0)) for f, g in zip(faces, G)]

    def _flux_step(self, beams, time, deposit_material=None, sticking=1.0, face_rates=None):
        """Advance etching (deposit_material None) or deposition for `time`
        with the face fluxes of `beams` [(exponent, {material: rate})] or,
        for deposition, [(exponent, rate)]. Each cell collects removed (or
        added) area; at h^2 it changes and the rest goes on into the next
        cell along its fastest face. The fluxes are recomputed after every
        step, and no surface moves more than half a cell in one step."""
        h = self.h
        nx, nz = self.mat.shape
        acc = np.zeros((nx, nz))
        normal = {0: (0, 1), 1: (1, 0), 2: (-1, 0), 3: (0, -1)}
        if face_rates is not None:
            etchable = set(face_rates.etchable)
        else:
            etchable = set() if deposit_material is not None else {
                m for (_, r) in beams for m, v in r.items() if v > 0}
        t = 0.0
        steps = 0
        while t < time * (1 - 1e-12):
            rate = np.zeros((nx, nz))
            best = np.zeros((nx, nz))
            bdir = np.zeros((nx, nz), int)
            if face_rates is not None:
                for (i, k, side, v) in face_rates():
                    r = v * h
                    if r > 0:
                        rate[i, k] += r
                        if r > best[i, k]:
                            best[i, k] = r
                            bdir[i, k] = side
            for (n, rates) in ([] if face_rates is not None else beams):
                for (i, k, side, f) in self._face_fluxes(n, sticking):
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

    def kinetic_rates(self, kin: "EtchKinetics", tol: float = 1e-10, max_iter: int = 500):
        """For the present shape: every exposed face (i, k, side) with its
        neutral and ion fluxes (cm^-2 s^-1), coverage and recession speed
        (nm/s) under `kin`. The neutrals' coverage and their re-emission
        depend on each other (a covered face re-emits more); they are
        iterated until the fluxes change by less than tol (relative)."""
        faces, F = self.view_factors()
        if not faces:
            return []
        dn = {(i, k, sd): f for (i, k, sd, f) in self.fluxes(kin.neutral_exponent)}
        di = {(i, k, sd): f for (i, k, sd, f) in self.fluxes(kin.ion_exponent)}
        key = [(f[0], f[1], f[2]) for f in faces]
        Dn = np.array([dn.get(q, 0.0) for q in key]) * kin.neutral_flux
        Gi = np.array([di.get(q, 0.0) for q in key]) * kin.ion_flux
        mats = [self.mat[f[0], f[1]] for f in faces]
        s0 = np.array([kin.sticking.get(m, 0.0) for m in mats])
        theta = np.zeros(len(faces))
        Gn = Dn.copy()
        for _ in range(max_iter):
            Gn_new = self._radiosity(faces, F, Dn, s0 * (1.0 - theta))
            th_new = np.zeros(len(faces))
            for m in set(mats):
                sel = np.array([q == m for q in mats])
                th_new[sel] = kin.rate(m, Gn_new[sel], Gi[sel])[0]
            ch = float(np.max(np.abs(Gn_new - Gn))) / max(float(np.max(np.abs(Gn_new))), 1e-300)
            Gn, theta = Gn_new, th_new
            if ch < tol:
                break
        else:
            raise RuntimeError("the coverage and the re-emitted neutrals did not settle")
        v = np.zeros(len(faces))
        for m in set(mats):
            sel = np.array([q == m for q in mats])
            v[sel] = kin.rate(m, Gn[sel], Gi[sel])[1]
        return [dict(face=f, neutral=float(a), ion=float(b), coverage=float(c),
                     speed=float(d)) for f, a, b, c, d in zip(key, Gn, Gi, theta, v)]

    def plasma_etch(self, kin: "EtchKinetics", time: float):
        """Etch for `time` seconds with ion-enhanced surface kinetics (new in
        0.26.0, see `EtchKinetics` and `kinetic_rates`): each exposed cell
        moves in at the speed of its faces, recomputed as the shape
        changes (as in the flux modes of `etch`)."""
        if not isinstance(kin, EtchKinetics):
            raise ValueError("plasma_etch needs an EtchKinetics")
        if not time > 0:
            raise ValueError("time must be > 0")

        def face_rates():
            return [(q["face"][0], q["face"][1], q["face"][2], q["speed"])
                    for q in self.kinetic_rates(kin) if q["speed"] > 0]
        face_rates.etchable = [m for m, v in kin.sticking.items() if v > 0]
        self._flux_step([], float(time), face_rates=face_rates)
        self.log.append(("plasma_etch", kin.reference, time, "kinetics"))

    def stokes(self, viscosity: dict, source=None, gas_ratio: float = 1e-4,
               shear_modulus: dict = None, dt: float = None, history: dict = None):
        """Slow viscous flow of the cross-section (new in 0.26.0), as of
        oxide during oxidation: eta lap(u) - grad(p) = 0 and div(u) = q on a
        staggered grid (velocities on the cell faces, pressure in the
        cells). viscosity : {material: eta} (Pa s, yours with its source);
        a material not in it is rigid (it does not move). The gas is a fluid
        gas_ratio times the lowest viscosity given (a free surface in the
        limit gas_ratio -> 0), open at the top of the domain; the sides are
        mirror planes and the bottom holds still. source : q per cell (1/s),
        the rate at which new volume appears there (default none).
        Returns dict(u (nx+1, nz) and w (nx, nz+1) in nm/s, p (nx, nz) in Pa,
        and the stresses sxx, szz, sxz at the cell centers, Pa: tension
        positive).

        shear_modulus = {material: G in Pa} (new in 0.33.0) with dt (s) and
        history (the "tau" of the previous call, or None to start from no
        stress) makes those materials viscoelastic (Maxwell): the stress
        tau = eta (grad u + grad u^T) relaxes with the time eta / G,
        tau + (eta / G) d tau/dt = eta (grad u + grad u^T), taken one step
        dt at a time (backward Euler: tau_new = eta' (grad u + grad u^T) +
        chi tau_old, eta' = eta G dt / (eta + G dt), chi = eta / (eta + G
        dt)). A material with a modulus but no viscosity is an elastic
        solid (eta' = G dt, chi = 1): it deforms and keeps its stress
        instead of being rigid. The result has "tau" (txx, tzz in the cells,
        txz at the corners, Pa) for the next step."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        nx, nz = self.mat.shape
        h = self.h * 1e-9                           # m
        viscosity = dict(viscosity or {})
        shear = dict(shear_modulus or {})
        if not (viscosity or shear) or any(not (v > 0 and math.isfinite(v))
                                           for v in viscosity.values()):
            raise ValueError("viscosity must give each flowing material a value > 0 (Pa s)")
        if any(not (v > 0 and math.isfinite(v)) for v in shear.values()):
            raise ValueError("shear_modulus must give each material a value > 0 (Pa)")
        if shear and not (dt is not None and dt > 0 and math.isfinite(dt)):
            raise ValueError("a shear modulus needs the time step dt > 0 (s)")
        if not (0 < gas_ratio < 1):
            raise ValueError("gas_ratio must be between 0 and 1")
        # the effective viscosity of each material over the step and the
        # share of its old stress that it keeps
        eff, keep = {}, {}
        for m in set(viscosity) | set(shear):
            ev, G = viscosity.get(m, math.inf), shear.get(m, math.inf)
            if math.isinf(G):
                eff[m], keep[m] = ev, 0.0
            elif math.isinf(ev):
                eff[m], keep[m] = G * dt, 1.0
            else:
                eff[m], keep[m] = ev * G * dt / (ev + G * dt), ev / (ev + G * dt)
        eta = np.zeros((nx, nz))            # effective viscosity over the step
        etav = np.zeros((nx, nz))           # the viscosity itself (inf: elastic)
        chi = np.zeros((nx, nz))
        solid = np.zeros((nx, nz), bool)    # elastic, no viscosity
        rigid = np.zeros((nx, nz), bool)
        etag = gas_ratio * (min(viscosity.values()) if viscosity else min(eff.values()))
        for i in range(nx):
            for k in range(nz):
                m = self.mat[i, k]
                if m is None:
                    eta[i, k] = etav[i, k] = etag
                elif m in eff:
                    eta[i, k] = eff[m]
                    etav[i, k] = viscosity.get(m, math.inf)
                    chi[i, k] = keep[m]
                    solid[i, k] = m not in viscosity
                else:
                    rigid[i, k] = True
        if history is None or not shear:
            Txx = np.zeros((nx, nz))
            Tzz = np.zeros((nx, nz))
            Txz = np.zeros((nx + 1, nz + 1))
        else:
            Txx = np.nan_to_num(np.asarray(history["txx"], float))
            Tzz = np.nan_to_num(np.asarray(history["tzz"], float))
            Txz = np.nan_to_num(np.asarray(history["txz"], float))
            if Txx.shape != (nx, nz) or Txz.shape != (nx + 1, nz + 1):
                raise ValueError("history does not match the grid")
        # the old stress each cell keeps (zero where nothing is kept)
        Hxx, Hzz = chi * Txx, chi * Tzz
        q = np.zeros((nx, nz)) if source is None else np.asarray(source, float)
        if q.shape != (nx, nz):
            raise ValueError("source must have one value per cell")
        if np.any(q[rigid] != 0):
            raise ValueError("no source in a rigid cell")
        nu, nw = (nx + 1) * nz, nx * (nz + 1)
        n_all = nu + nw + nx * nz

        def U(i, k):
            return i * nz + k

        def W(i, k):
            return nu + i * (nz + 1) + k

        def P(i, k):
            return nu + nw + i * nz + k

        def etac(i, k):                       # cell viscosity, mirrored at the sides
            i = min(max(i, 0), nx - 1)
            return eta[i, k]

        def cmean(i, k, arr):
            """mean of a cell array over the cells around the corner (x_i, z_k)"""
            vals = []
            for a in (i - 1, i):
                for b in (k - 1, k):
                    if 0 <= b < nz:
                        aa = min(max(a, 0), nx - 1)
                        vals.append(arr[aa, b])
            return float(np.mean(vals)) if vals else 0.0

        def cany(i, k, arr):
            for a in (i - 1, i):
                for b in (k - 1, k):
                    if 0 <= b < nz and arr[min(max(a, 0), nx - 1), b]:
                        return True
            return False

        def etan(i, k, arr=None):
            """effective viscosity at the corner (x_i, z_k): the mean of the
            cells' viscosities times one minus the mean share of old stress
            kept there, so that a steady flow ends on the viscous stress of
            the mean viscosity; next to an elastic solid, the mean of the
            effective viscosities"""
            if arr is not None:
                return cmean(i, k, arr)
            if not np.any(chi > 0):
                return cmean(i, k, eta)
            if cany(i, k, solid):
                return cmean(i, k, eta)
            return cmean(i, k, etav) * (1.0 - cmean(i, k, chi))
        # the old shear stress each corner keeps
        Hxz = np.zeros((nx + 1, nz + 1))
        if np.any(chi > 0):
            for i in range(nx + 1):
                for k in range(nz + 1):
                    Hxz[i, k] = etan(i, k, chi) * Txz[i, k]
        rows, cols, vals = [], [], []
        rhs = np.zeros(n_all)

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)
        # fixed velocities: walls, and every face touching a rigid cell
        ufix = np.zeros((nx + 1, nz), bool)
        ufix[0, :] = ufix[nx, :] = True
        ufix[:-1][rigid] = True
        ufix[1:][rigid] = True
        wfix = np.zeros((nx, nz + 1), bool)
        wfix[:, 0] = True
        wfix[:, :-1][rigid] = True
        wfix[:, 1:][rigid] = True
        inv = 1.0 / (h * h)
        # x momentum
        for i in range(nx + 1):
            for k in range(nz):
                r = U(i, k)
                if ufix[i, k]:
                    add(r, r, 1.0)
                    continue
                # normal stress 2 eta du/dx across the cells i-1, i
                for (cc, sgn) in ((i, 1.0), (i - 1, -1.0)):
                    e2 = 2.0 * eta[cc, k] * inv
                    # d/dx of (u_{cc+1} - u_cc) for cc = i: +; cc = i-1: -
                    if sgn > 0:
                        add(r, U(i + 1, k), e2)
                        add(r, U(i, k), -e2)
                    else:
                        add(r, U(i, k), -e2)
                        add(r, U(i - 1, k), e2)
                # shear at the corners above (k+1) and below (k)
                for (kc, sgn) in ((k + 1, 1.0), (k, -1.0)):
                    if kc == nz:                      # open top: no shear
                        continue
                    en = etan(i, kc) * inv
                    if kc == 0:                       # still bottom: u = 0 there
                        add(r, U(i, k), -2.0 * en)
                        continue
                    ka, kb = (k + 1, k) if sgn > 0 else (k, k - 1)
                    add(r, U(i, ka), sgn * en)
                    add(r, U(i, kb), -sgn * en)
                    add(r, W(i, kc), sgn * en)
                    add(r, W(i - 1, kc), -sgn * en)
                # pressure
                add(r, P(i, k), -1.0 / h)
                add(r, P(i - 1, k), 1.0 / h)
                # the old stress the cells and corners keep (a known force)
                rhs[r] -= (Hxx[i, k] - Hxx[i - 1, k]) / h
                top_ = Hxz[i, k + 1] if k + 1 < nz else 0.0
                rhs[r] -= (top_ - Hxz[i, k]) / h
        # z momentum
        for i in range(nx):
            for k in range(nz + 1):
                r = W(i, k)
                if wfix[i, k]:
                    add(r, r, 1.0)
                    continue
                if k == nz:
                    # open top: -p + 2 eta dw/dz = 0 in the top cell
                    e2 = 2.0 * eta[i, nz - 1] / h
                    add(r, W(i, nz), e2)
                    add(r, W(i, nz - 1), -e2)
                    add(r, P(i, nz - 1), -1.0)
                    rhs[r] -= Hzz[i, nz - 1]
                    continue
                for (cc, sgn) in ((k, 1.0), (k - 1, -1.0)):
                    e2 = 2.0 * eta[i, cc] * inv
                    if sgn > 0:
                        add(r, W(i, k + 1), e2)
                        add(r, W(i, k), -e2)
                    else:
                        add(r, W(i, k), -e2)
                        add(r, W(i, k - 1), e2)
                for (ic, sgn) in ((i + 1, 1.0), (i, -1.0)):
                    if ic == 0 or ic == nx:           # mirror sides: no shear
                        continue
                    en = etan(ic, k) * inv
                    ia, ib = (i + 1, i) if sgn > 0 else (i, i - 1)
                    add(r, W(ia, k), sgn * en)
                    add(r, W(ib, k), -sgn * en)
                    add(r, U(ic, k), sgn * en)
                    add(r, U(ic, k - 1), -sgn * en)
                add(r, P(i, k), -1.0 / h)
                add(r, P(i, k - 1), 1.0 / h)
                rhs[r] -= (Hzz[i, k] - Hzz[i, k - 1]) / h
                right_ = Hxz[i + 1, k] if i + 1 < nx else 0.0
                left_ = Hxz[i, k] if i > 0 else 0.0
                rhs[r] -= (right_ - left_) / h
        # continuity (scaled by eta / h to match the momentum rows)
        for i in range(nx):
            for k in range(nz):
                r = P(i, k)
                if rigid[i, k]:
                    add(r, r, 1.0)
                    continue
                sc = eta[i, k] / h
                add(r, U(i + 1, k), sc / h)
                add(r, U(i, k), -sc / h)
                add(r, W(i, k + 1), sc / h)
                add(r, W(i, k), -sc / h)
                rhs[r] = sc * q[i, k]
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n_all, n_all))
        # scale: pressure unknowns by h / eta_ref, every row to order one
        eref = max(eff.values())
        cs = np.ones(n_all)
        cs[nu + nw:] = eref / h
        A = A @ sp.diags(cs)
        rs = np.asarray(abs(A).max(axis=1).todense()).ravel()
        rs[rs == 0] = 1.0
        A = sp.diags(1.0 / rs) @ A
        x = spl.spsolve(A.tocsc(), rhs / rs) * cs
        if not np.all(np.isfinite(x)):
            raise RuntimeError("the flow could not be solved")
        u = x[:nu].reshape(nx + 1, nz)
        w = x[nu:nu + nw].reshape(nx, nz + 1)
        p = x[nu + nw:].reshape(nx, nz)
        dudx = (u[1:] - u[:-1]) / h
        dwdz = (w[:, 1:] - w[:, :-1]) / h
        txx = 2.0 * eta * dudx + Hxx
        tzz = 2.0 * eta * dwdz + Hzz
        sxx = -p + txx
        szz = -p + tzz
        # shear at the corners, then the mean of the four around each cell
        sc_ = np.zeros((nx + 1, nz + 1))
        for i in range(1, nx):
            for k in range(1, nz):
                sc_[i, k] = etan(i, k) * ((u[i, k] - u[i, k - 1]) / h
                                          + (w[i, k] - w[i - 1, k]) / h) + Hxz[i, k]
        sxz = 0.25 * (sc_[:-1, :-1] + sc_[1:, :-1] + sc_[:-1, 1:] + sc_[1:, 1:])
        # for the next step, also the shear at the still bottom (u = 0 there,
        # half a cell below the first velocity), which the momentum rows use
        hist_xz = sc_.copy()
        for i in range(1, nx):
            hist_xz[i, 0] = etan(i, 0) * 2.0 * u[i, 0] / h + Hxz[i, 0]
        p = np.where(rigid, np.nan, p)
        nanr = np.where(rigid, np.nan, 1.0)
        return dict(u=u * 1e9, w=w * 1e9, p=p, sxx=sxx * nanr, szz=szz * nanr,
                    sxz=sxz * nanr, rigid=rigid,
                    tau=dict(txx=np.where(rigid, 0.0, txx), tzz=np.where(rigid, 0.0, tzz),
                             txz=hist_xz))

    def oxidize(self, material: str, oxide: str, B_nm2_s: float, B_over_A_nm_s: float,
                time: float, consumed_fraction: float, reference: str,
                mode: str = "column", viscosity: dict = None, stress_volume_nm3: float = None,
                temperature_K: float = None, flow_every: int = 1,
                shear_modulus: dict = None, shape: str = "volume"):
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
        thicknesses are cell counts (nm).

        With mode="2d", viscosity={material: eta in Pa s} (new in 0.26.0)
        solves the slow viscous flow that the new oxide volume drives
        (`stokes`; materials not listed are rigid), every flow_every steps,
        and keeps it as `oxide_stress` (velocities, pressure, stresses).
        stress_volume_nm3 (with temperature_K) makes the reaction rate at
        each face follow the stress normal to it, k = k0 exp(sigma_nn V / kT)
        (compression slows it: D.-B. Kao, J. P. McVittie, W. D. Nix and
        K. C. Saraswat, IEEE Trans. Electron Devices 35, 25 (1988)); V is
        yours, with its source. The shape still follows the volume rule
        above (column lift, nearest gas); the flow sets the stress.

        shear_modulus={material: G in Pa} (new in 0.33.0, with viscosity)
        makes the flow viscoelastic (Maxwell; see `stokes`): the stress
        builds up over the relaxation time eta / G instead of at once, and
        a material with a modulus but no viscosity (a nitride mask, say) is
        an elastic solid that bends and holds stress instead of being
        rigid. The stress is carried with the cells (new oxide starts
        without stress).

        shape="flow" (new in 0.34.0, with mode="2d" and viscosity): the new
        oxide's volume is carried by the computed flow instead of the
        volume rule: the open gas cells next to the oxide fill with what
        flows out to them through each face (u . n times the face, every
        step), and turn to oxide when full, so the free surface swells
        where the oxide flows out. Where new oxide forms against a mask or
        the gas directly, it goes in as before (a mask's column is lifted:
        cells of solids do not move with the flow)."""
        if not (B_nm2_s > 0 and B_over_A_nm_s > 0 and time > 0):
            raise ValueError("B, B/A and time must be > 0")
        if not (0 < consumed_fraction < 1):
            raise ValueError("consumed_fraction must be between 0 and 1")
        if not (isinstance(reference, str) and reference.strip()):
            raise ValueError("oxidize needs a reference for B, B/A and the consumed fraction")
        if stress_volume_nm3 is not None and (viscosity is None or temperature_K is None):
            raise ValueError("stress_volume_nm3 needs viscosity and temperature_K")
        if shear_modulus is not None and viscosity is None:
            raise ValueError("shear_modulus needs viscosity (the oxide's flow)")
        if shape not in ("volume", "flow"):
            raise ValueError('shape must be "volume" or "flow"')
        if shape == "flow" and (viscosity is None or mode != "2d"):
            raise ValueError('shape="flow" needs mode="2d" and viscosity')
        if mode == "2d":
            return self._oxidize_2d(material, oxide, B_nm2_s, B_over_A_nm_s, time,
                                    consumed_fraction, reference, viscosity=viscosity,
                                    stress_volume=stress_volume_nm3,
                                    temperature=temperature_K, flow_every=int(flow_every),
                                    shear_modulus=shear_modulus, shape=shape)
        if viscosity is not None:
            raise ValueError("viscous flow and stress need mode='2d'")
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

    def _oxidant(self, material, oxide, A, with_dirs=False, extra_thick=None, kfac=None):
        """Steady oxidant in the oxide (new in 0.24.0), normalized to its
        value at the gas surface: div grad C = 0 in the oxide cells, C = 1
        on faces to the open gas (and the top of the domain), no flux into
        masks or through the sides and bottom, and at the oxidizable
        material the reaction flux k C with k / D = 2 / A (Deal and Grove:
        A = 2 D / k). Returns (i, k, c): the material cells behind every
        reacting face and the normalized oxidant at that face (c = 1 where
        the material is bare under the gas). extra_thick (4, nx, nz): oxide
        not yet in whole cells in front of each face (nm), added to its
        diffusion path. kfac (4, nx, nz): a factor on the reaction rate k of
        each face (as ext; default 1), so A becomes A / kfac there."""
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
        li, lk, lc_row, ld, lx, lA = [], [], [], [], [], []   # cell, oxide row, side, extra, A
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
            Af = A if kfac is None else A / kfac[qq, I + di, K + dk]
            # oxide cell center -> reacting face: half a cell and the oxide
            # not yet in cells by diffusion, then the reaction (D = 1, k = 2/A)
            np.add.at(diag, idx[core][o_m], 2.0 * h / (Af + h + 2.0 * dx_))
            li.append(I + di)
            lk.append(K + dk)
            lc_row.append(idx[core][o_m])
            ld.append(np.full(I.size, qq))
            lx.append(dx_)
            lA.append(np.broadcast_to(np.asarray(Af, float), (I.size,)))
            m_g = mtp[core] & gasp[nb]
            I, K = np.nonzero(m_g)
            li.append(I)
            lk.append(K)
            lc_row.append(-np.ones(I.size, int))
            ld.append(np.full(I.size, q))
            lx.append(ext[q, I, K])
            lA.append(np.broadcast_to(np.asarray(
                A if kfac is None else A / kfac[q, I, K], float), (I.size,)))
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
        Aa = np.concatenate(lA)
        # interface value: the reaction flux k C_i equals the diffusive flux
        # from the oxide cell center (or from the gas, for bare material)
        Cr = C[np.maximum(r, 0)] if n else np.zeros(r.size)
        c = np.where(r >= 0, Cr * Aa / (Aa + h + 2.0 * x_), Aa / (Aa + 2.0 * x_))
        if with_dirs:
            return li, lk, c, np.concatenate(ld)
        return li, lk, c

    def _oxidize_2d(self, material, oxide, B, BA, time, cf, reference, viscosity=None,
                    stress_volume=None, temperature=None, flow_every=1, shear_modulus=None,
                    shape="volume"):
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
        kfac = None
        q_prev = None
        kT = None if temperature is None else 1.380649e-23 * float(temperature)
        # viscoelastic memory (new in 0.33.0): the stress of the last flow
        # solve, kept with the cells, and the time it was solved at
        self._vehist = None
        t_flow = 0.0
        flow_shape = shape == "flow"
        # shape="flow" (new in 0.34.0): the open gas cells next to the
        # oxide fill with the oxide that the flow carries out to them
        fill = np.zeros((nx, nz))
        vel = None
        # and (new in 0.35.0) a mask that flows or is elastic rides on the
        # oxide below it: push collects the oxide that flows up into the
        # bottom cell of such a solid; a whole cell of it lifts the solid's
        # column by one cell and puts oxide under it
        movable = (set(viscosity or ()) | set(shear_modulus or ())) - {oxide, material}
        push = np.zeros((nx, nz))
        made = 0.0                          # oxide area made so far (nm^2)
        ox0 = float((self.mat == oxide).sum()) * h * h

        def flow(src, step=None):
            nonlocal t_flow
            if shear_modulus:
                # backward Euler over the time since the last solve, or (the
                # flowing shape, which solves at the start of a step) over
                # the step ahead
                d_t = max(t - t_flow, 1e-300) if step is None else step
                r_ = self.stokes(viscosity, source=src, shear_modulus=shear_modulus,
                                 dt=d_t, history=self._vehist)
                self._vehist = {k_: v_.copy() for k_, v_ in r_["tau"].items()}
                t_flow = t if step is None else t + step
                return r_
            return self.stokes(viscosity, source=src)
        while t < time * (1 - 1e-12):
            # the oxide grown in front of each face but not yet in whole
            # cells: in place of the material consumed so far in the cell, and
            # the new volume still to be put in
            ext = (acc[None, :, :] / h) * np.ones((4, 1, 1)) + lift / h
            ext = np.where((self.mat == material)[None], ext, 0.0)
            solved = False
            if (viscosity is not None and q_prev is not None and steps % flow_every == 0
                    and not flow_shape):
                # the viscous flow of the oxide that the new volume drives, and
                # its stress normal to each reacting face (new in 0.26.0)
                fluid = np.array([[m_ in viscosity for m_ in row] for row in self.mat], bool)
                self.oxide_stress = flow(np.where(fluid, q_prev, 0.0))
                solved = True
            if flow_shape and q_prev is not None:
                solved = True            # the stress of the last step's flow
            if solved:
                if stress_volume is not None:
                    kfac = np.ones((4, nx, nz))
                    S = self.oxide_stress
                    for qd, (di, dk) in enumerate(dirs):
                        # the normal stress in the oxide just beyond the
                        # cell where the new volume appears (that cell's own
                        # expansion is not a stress on the face: Kao et al.
                        # take the oxide's stress at the interface, the new
                        # oxide entering there as a velocity)
                        sn = S["sxx"] if dk == 0 else S["szz"]
                        fr = np.zeros((nx, nz))
                        for step_ in (2, 1):
                            sh_i, sh_k = step_ * di, step_ * dk
                            a0, a1 = max(0, -sh_i), nx - max(0, sh_i)
                            b0, b1 = max(0, -sh_k), nz - max(0, sh_k)
                            cand = np.full((nx, nz), np.nan)
                            cand[a0:a1, b0:b1] = sn[a0 + sh_i:a1 + sh_i, b0 + sh_k:b1 + sh_k]
                            put = (fr == 0.0) & np.isfinite(cand)
                            if step_ == 2:
                                # only where both cells in front are oxide
                                c1 = np.full((nx, nz), np.nan)
                                a0b, a1b = max(0, -di), nx - max(0, di)
                                b0b, b1b = max(0, -dk), nz - max(0, dk)
                                c1[a0b:a1b, b0b:b1b] = sn[a0b + di:a1b + di, b0b + dk:b1b + dk]
                                put &= np.isfinite(c1)
                            fr = np.where(put, cand, fr)
                        fr = np.where(np.isfinite(fr), fr, 0.0)
                        # tension speeds the reaction, compression slows it
                        kfac[qd] = np.exp(np.clip(fr * stress_volume * 1e-27 / kT, -50, 50))
            li, lk, c, dl = self._oxidant(material, oxide, A, with_dirs=True,
                                          extra_thick=ext, kfac=kfac)
            if li.size == 0:
                break
            kf = 1.0 if kfac is None else kfac[dl, li, lk]
            v = BA * kf * c                 # oxide growth speed at each face, nm/s
            if viscosity is not None:
                # new volume per s in the oxide cell in front of each face
                qn = np.zeros((nx, nz))
                for qd, (di, dk) in enumerate(dirs):
                    m = dl == qd
                    fi, fk = li[m] + di, lk[m] + dk
                    ok = (fi >= 0) & (fi < nx) & (fk >= 0) & (fk < nz)
                    fi, fk, vv = fi[ok], fk[ok], v[m][ok]
                    isox = np.array([self.mat[a_, b_] == oxide for a_, b_ in zip(fi, fk)],
                                    bool)
                    np.add.at(qn, (fi[isox], fk[isox]), (1.0 - cf) * vv[isox] / h)
                q_prev = qn
            rate = np.zeros((nx, nz))       # material area consumed per s
            np.add.at(rate, (li, lk), cf * v * h)
            extra = np.zeros((4, nx, nz))   # oxide area to put in front, per s
            if flow_shape:
                # the new volume that appears in oxide goes with the flow;
                # in front of a face looking at gas or a mask it goes in as
                # before (the gas cell fills, the mask's column is lifted)
                front_ox = np.zeros(li.size, bool)
                for qd, (di, dk) in enumerate(dirs):
                    m = np.flatnonzero(dl == qd)
                    fi, fk = li[m] + di, lk[m] + dk
                    ok = (fi >= 0) & (fi < nx) & (fk >= 0) & (fk < nz)
                    front_ox[m[ok]] = [self.mat[a_, b_] == oxide
                                       for a_, b_ in zip(fi[ok], fk[ok])]
                keep_ = ~front_ox
                np.add.at(extra, (dl[keep_], li[keep_], lk[keep_]),
                          (1.0 - cf) * v[keep_] * h)
            else:
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
            if flow_shape and viscosity is not None and qn.any():
                if steps % flow_every == 0 or vel is None:
                    fluid = np.array([[m_ in viscosity for m_ in row] for row in self.mat],
                                     bool)
                    src_ = np.where(fluid, qn, 0.0)
                    hist0 = self._vehist
                    for _ in range(4):
                        self._vehist = hist0
                        self.oxide_stress = flow(src_, dt)
                        vel = (self.oxide_stress["u"], self.oxide_stress["w"])
                        vmax = float(max(np.abs(vel[0]).max(), np.abs(vel[1]).max()))
                        # nor does the flow move oxide by more than half a
                        # cell; a springy oxide is solved again over the
                        # shorter step
                        if not (vmax > 0 and 0.5 * h / vmax < dt * (1 - 1e-9)):
                            break
                        dt = 0.5 * h / vmax
                        if not shear_modulus:
                            break
            if flow_shape and vel is not None:
                # the flow carries oxide out through its faces to the open
                # gas, which fills (area per s through each face, nm^2/s)
                u_, w_ = vel
                vmax = float(max(np.abs(u_).max(), np.abs(w_).max()))
                if vmax > 0:
                    dt = min(dt, 0.5 * h / vmax)
                isox = self.mat == oxide
                gas = self._gas_open()
                fx = u_ * h * dt                      # (nx+1, nz) through x faces
                fz = w_ * h * dt                      # (nx, nz+1) through z faces
                # x faces between (i-1, k) and (i, k); what flows back from
                # the gas into the oxide is taken off the gas cell's fill
                # (since 0.35.0), so that the oxide's volume is kept exactly
                out_r = isox[:-1] & gas[1:]
                fill[1:][out_r] += fx[1:-1][out_r]
                out_l = isox[1:] & gas[:-1]
                fill[:-1][out_l] -= fx[1:-1][out_l]
                out_u = isox[:, :-1] & gas[:, 1:]
                fill[:, 1:][out_u] += fz[:, 1:-1][out_u]
                out_d = isox[:, 1:] & gas[:, :-1]
                fill[:, :-1][out_d] -= fz[:, 1:-1][out_d]
                while True:
                    full_ = (fill >= h * h * (1 - 1e-12)) & gas
                    if not full_.any():
                        break
                    for i, k in zip(*np.nonzero(full_)):
                        self.mat[i, k] = oxide
                        self.donors[i, k] = self.acceptors[i, k] = 0.0
                        self._hist_new_cell(i, k)
                        rest = fill[i, k] - h * h
                        fill[i, k] = 0.0
                        # what is left goes on to the open gas next to it,
                        # where this cell's oxide flows out (else upward)
                        best_, bv = None, -1.0
                        for (di, dk), sp_ in (((1, 0), u_[min(i + 1, nx), k]),
                                              ((-1, 0), -u_[i, k]),
                                              ((0, 1), w_[i, min(k + 1, nz)]),
                                              ((0, -1), -w_[i, k])):
                            a2, b2 = i + di, k + dk
                            if 0 <= a2 < nx and 0 <= b2 < nz and self.mat[a2, b2] is None \
                                    and sp_ > bv:
                                best_, bv = (a2, b2), sp_
                        if best_ is not None:
                            fill[best_] += rest
                    gas = self._gas_open()
                if movable:
                    # the oxide and the solids as they were when the flow
                    # was taken (the fill above has not moved any solid)
                    mov = np.isin(self.mat, list(movable))
                    # the oxide that flows into a movable solid, through any
                    # of its faces (signed: a solid moving into the oxide
                    # takes it back), goes to the bottom cell of that solid
                    # in its column, which only moves up
                    into = np.zeros((nx, nz))
                    m_ = isox[:-1] & mov[1:]
                    into[1:][m_] += fx[1:-1][m_]
                    m_ = isox[1:] & mov[:-1]
                    into[:-1][m_] -= fx[1:-1][m_]
                    m_ = isox[:, :-1] & mov[:, 1:]
                    into[:, 1:][m_] += fz[:, 1:-1][m_]
                    m_ = isox[:, 1:] & mov[:, :-1]
                    into[:, :-1][m_] -= fz[:, 1:-1][m_]
                    for i, k in zip(*np.nonzero(into)):
                        kb = k
                        while kb > 0 and mov[i, kb - 1]:
                            kb -= 1
                        push[i, kb] += into[i, k]
                    for i, k in zip(*np.nonzero(push >= h * h * (1 - 1e-12))):
                        while (push[i, k] >= h * h * (1 - 1e-12) and k > 0
                               and self.mat[i, k - 1] == oxide):
                            # oxide in at (i, k), the column above moves up
                            # (with what is still to come for its cells; a
                            # gas cell's fill moves on to the gas above)
                            b = k
                            while b < nz and self.mat[i, b] is not None:
                                b += 1
                            if b >= nz:
                                raise ValueError("the oxide lifts material out of the "
                                                 "domain; raise z_top")
                            if b + 1 < nz:
                                fill[i, b + 1] += fill[i, b]
                                fill[i, b] = 0.0
                            rest = push[i, k] - h * h
                            push[i, k + 1:b + 1] = push[i, k:b].copy()
                            push[i, k] = 0.0
                            self._push_oxide(i, k - 1, (0, 1), oxide)
                            k += 1
                            push[i, k] = rest
            acc += rate * dt
            lift += extra * dt
            made += float(v.sum()) * h * dt
            while True:
                done = (rate > 0) & (acc >= h * h * (1 - 1e-12)) & (self.mat == material)
                if not done.any():
                    break
                for i, k in zip(*np.nonzero(done)):
                    e = acc[i, k] - h * h
                    self.mat[i, k] = oxide
                    self._hist_new_cell(i, k)
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
            if flow_shape:
                self._carry_parts(fill, push, movable)
            t += dt
            steps += 1
            if steps > 200000:
                raise RuntimeError("the oxidation did not finish; use larger cells")
        # the oxide's balance (new in 0.35.0): what was made, what is in
        # whole cells, and what was still in parts of cells at the end
        part = float(acc[self.mat == material].sum() + lift.sum() + push.sum()
                     + fill[self.mat == None].sum())  # noqa: E711
        whole = float((self.mat == oxide).sum()) * h * h - ox0
        # a cell more than half consumed at the end turns to oxide, and half a
        # cell of new volume or more is put in (the front passed the center)
        half = (acc >= 0.5 * h * h) & (self.mat == material)
        for i, k in zip(*np.nonzero(half)):
            self._hist_new_cell(i, k)
        self.mat[half] = oxide
        self.donors[half] = self.acceptors[half] = 0.0
        for q, i, k in zip(*np.nonzero(lift >= 0.5 * h * h)):
            self._push_oxide(i, k, dirs[q], oxide)
        if flow_shape:
            # so do the gas cells the flow has filled half way or more, and
            # the masks it has pushed up half a cell or more
            half_g = (fill >= 0.5 * h * h) & (self.mat == None)  # noqa: E711
            for i, k in zip(*np.nonzero(half_g)):
                self.mat[i, k] = oxide
                self._hist_new_cell(i, k)
            for i, k in zip(*np.nonzero(push >= 0.5 * h * h)):
                if self.mat[i, k] in movable and k > 0 and self.mat[i, k - 1] == oxide:
                    self._push_oxide(i, k - 1, (0, 1), oxide)
        if viscosity is not None and q_prev is not None and not (flow_shape and shear_modulus):
            # the flow and stress of the final shape (with the last growth;
            # a springy oxide shaped by its flow keeps the stress of its last
            # step, which already reaches the end)
            fluid = np.array([[m_ in viscosity for m_ in row] for row in self.mat], bool)
            self.oxide_stress = flow(np.where(fluid, q_prev, 0.0))
        self.oxidation_balance = dict(
            made_nm2=made, whole_cells_nm2=whole, parts_nm2=part,
            rounding_nm2=float((self.mat == oxide).sum()) * h * h - ox0 - made)
        self.log.append(("oxidize", material, oxide, B, BA, time, cf, reference, "2d"))
        top = np.zeros(nx)
        for i in range(nx):
            top[i] = float(np.sum(self.mat[i] == oxide) * h)
        return top

    def _carry_parts(self, fill, push, movable):
        """shape="flow": a part of a cell's oxide that sits on a cell which
        has meanwhile changed (a gas cell taken by oxide put in by the
        simple rule, or lifted into by a mask; a solid's cell that moved)
        goes on to where it belongs, so that no oxide is lost: a gas cell's
        fill to the nearest open gas cell, a solid's push to the bottom of
        the movable solid above it in its column, else to the open gas"""
        from scipy import ndimage
        nx, nz = self.mat.shape
        gas = self._gas_open()
        mov = np.isin(self.mat, list(movable)) if movable else np.zeros_like(gas)
        bad_f = (fill != 0.0) & ~gas
        bad_p = (push != 0.0) & ~mov
        if not (bad_f.any() or bad_p.any()):
            return
        for i, k in zip(*np.nonzero(bad_p)):
            b = k + 1
            while b < nz and not mov[i, b] and self.mat[i, b] is not None:
                b += 1
            if b < nz and mov[i, b]:
                push[i, b] += push[i, k]
            else:
                fill[i, k] += push[i, k]
                bad_f[i, k] = fill[i, k] != 0.0 and not gas[i, k]
            push[i, k] = 0.0
        if bad_f.any() and gas.any():
            # shared equally among the open gas cells nearest to it (all of
            # them, so that a symmetric shape stays symmetric)
            dist = ndimage.distance_transform_edt(~gas)
            for i, k in zip(*np.nonzero(bad_f)):
                r = float(dist[i, k])
                R = int(math.ceil(r)) + 1
                a0, a1, b0, b1 = max(0, i - R), min(nx, i + R + 1), max(0, k - R), min(nz, k + R + 1)
                ii, kk = np.nonzero(gas[a0:a1, b0:b1])
                d2 = (ii + a0 - i) ** 2 + (kk + b0 - k) ** 2
                near = d2 <= r * r * (1 + 1e-12) + 1e-12
                share = fill[i, k] / int(near.sum())
                for a_, b_ in zip(ii[near] + a0, kk[near] + b0):
                    fill[a_, b_] += share
                fill[i, k] = 0.0

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
                self._hist_new_cell(g0, g1)
                return
        # up the column from (a0, b0)
        b = b0
        while b < nz and self.mat[a0, b] is not None:
            b += 1
        if b >= nz:
            raise ValueError("the oxide lifts material out of the domain; raise z_top")
        for arr in (self.mat, self.donors, self.acceptors):
            arr[a0, b0 + 1:b + 1] = arr[a0, b0:b].copy()
        H = getattr(self, "_vehist", None)
        if H is not None:
            # the lifted cells take their stress with them
            for key in ("txx", "tzz"):
                H[key][a0, b0 + 1:b + 1] = H[key][a0, b0:b].copy()
            for col in (a0, a0 + 1):
                H["txz"][col, b0 + 1:b + 2] = H["txz"][col, b0:b + 1].copy()
        self.mat[a0, b0] = oxide
        self.donors[a0, b0] = self.acceptors[a0, b0] = 0.0
        self._hist_new_cell(a0, b0)

    def _hist_new_cell(self, i, k):
        """new oxide starts without stress (viscoelastic memory, 0.33.0)"""
        H = getattr(self, "_vehist", None)
        if H is None:
            return
        H["txx"][i, k] = H["tzz"][i, k] = 0.0
        H["txz"][i:i + 2, k:k + 2] = 0.0

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
