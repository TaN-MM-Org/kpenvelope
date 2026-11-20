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

`Process.doping` returns the function `Device(doping=...)` takes. Etching
and deposition are `Region` and `Dielectric` in the device itself.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

_KB_EV = 8.617333262e-5          # eV/K (CODATA 2018)

__all__ = ["Implant", "Anneal", "Process"]


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
