"""Implant ranges by binary-collision Monte Carlo (new in 0.26.0).

Ions enter an amorphous target and slow down in a sequence of binary
collisions with its atoms, losing energy to the electrons continuously in
between (the method of TRIM: J. P. Biersack and L. G. Haggmark, Nucl.
Instrum. Methods 174, 257 (1980)):

* Nuclear collisions: the ZBL universal potential
  V(r) = Z1 Z2 e^2 / r phi(r / a), phi(x) = 0.18175 e^-3.1998x
  + 0.50986 e^-0.94229x + 0.28022 e^-0.4029x + 0.02817 e^-0.20162x,
  a = 0.8854 a0 / (Z1^0.23 + Z2^0.23) (J. F. Ziegler, J. P. Biersack and
  U. Littmark, The Stopping and Range of Ions in Matter, Pergamon, 1985).
  The scattering angle in the center-of-mass frame is the classical
  integral theta = pi - 2 p int_r0^inf dr / (r^2 sqrt(1 - V/E_c - p^2/r^2)),
  computed here by Gauss-Legendre quadrature (64 points, after the change
  of variable that removes the turning-point singularity) on a table of
  reduced energy and impact parameter, and interpolated.
* Electronic stopping: Lindhard and Scharff (Phys. Rev. 124, 128 (1961)),
  S_e = 8 pi e^2 a0 Z1^(7/6) Z2 / (Z1^(2/3) + Z2^(2/3))^(3/2) v / v0
  (v0 = e^2 / hbar, the Bohr velocity), meant for ion speeds below about
  Z1^(2/3) v0; `electronic_scale` multiplies it (your correction, with its
  source), or give your own S_e(E).
* Flight between collisions: one atomic spacing N^(-1/3); the impact
  parameter p = sqrt(U / (pi N^(2/3))) with U uniform in (0, 1), and the
  target atom drawn by the composition.

The target recoils are not followed (no damage cascade, no sputtering),
and the target stays amorphous (no channeling). The result gives the final
positions of the ions and the moments of their distribution.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

_E2 = 1.439964547          # e^2 / (4 pi eps0), eV nm
_A0 = 0.0529177210903      # Bohr radius, nm
_V0 = 2.18769126364e6      # Bohr velocity, m/s
_U = 1.66053906660e-27     # atomic mass unit, kg
_QE = 1.602176634e-19

_ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_ZBL_D = np.array([3.1998, 0.94229, 0.4029, 0.20162])


def zbl_screening(x):
    """phi(x) of the ZBL universal potential (x = r / a)."""
    x = np.asarray(x, float)
    return np.sum(_ZBL_C[:, None] * np.exp(-_ZBL_D[:, None] * x.ravel()[None, :]),
                  axis=0).reshape(x.shape)


def zbl_length(Z1, Z2):
    """The ZBL screening length a (nm)."""
    return 0.8854 * _A0 / (Z1 ** 0.23 + Z2 ** 0.23)


_GL_S, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_S = 0.5 * (_GL_S + 1.0)
_GL_W = 0.5 * _GL_W


def reduced_angle(eps, b, screening=zbl_screening):
    """Center-of-mass scattering angle theta (rad) at reduced energy eps
    (= a E_c / (Z1 Z2 e^2)) and reduced impact parameter b (= p / a), for
    the potential phi(x) / x (phi = 1: Coulomb, theta = 2 atan(1 / (2 eps
    b))). Arrays broadcast."""
    eps, b = np.broadcast_arrays(np.asarray(eps, float), np.asarray(b, float))
    shape = eps.shape
    eps, b = eps.ravel(), b.ravel()
    # distance of closest approach: g(x) = 1 - phi(x) / (x eps) - b^2 / x^2 = 0,
    # between b and b + 1 / eps (the Coulomb root is the largest possible)
    lo = np.maximum(b, 1e-12)
    hi = b + 1.0 / eps + 1e-12

    def g(x):
        return 1.0 - screening(x) / (x * eps) - (b / x) ** 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pos = g(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    x0 = 0.5 * (lo + hi)
    # theta = pi - 2 (b / x0) int_0^1 du / sqrt(G(u)), u = x0 / x, u = 1 - s^2
    s = _GL_S[None, :]
    u = 1.0 - s * s
    xx = x0[:, None] / u
    G = 1.0 - screening(xx) / (xx * eps[:, None]) - (b[:, None] * u / x0[:, None]) ** 2
    integ = 2.0 * s / np.sqrt(np.maximum(G, 1e-300))
    I = (integ * _GL_W[None, :]).sum(1)
    th = math.pi - 2.0 * (b / x0) * I
    return np.clip(th, 0.0, math.pi).reshape(shape)


class _AngleTable:
    """sin^2(theta / 2) on a grid of log eps and log b, interpolated
    bilinearly in those logs (and in log sin^2)."""

    def __init__(self, n_eps=241, n_b=241):
        self.le = np.linspace(math.log(1e-5), math.log(1e5), n_eps)
        self.lb = np.linspace(math.log(1e-4), math.log(1e3), n_b)
        E, B = np.meshgrid(np.exp(self.le), np.exp(self.lb), indexing="ij")
        th = reduced_angle(E, B)
        self.ls = np.log(np.maximum(np.sin(0.5 * th) ** 2, 1e-300))

    def __call__(self, eps, b):
        le = np.clip(np.log(eps), self.le[0], self.le[-1])
        lb = np.log(np.maximum(b, 1e-300))
        hb = lb > self.lb[-1]
        lbc = np.clip(lb, self.lb[0], self.lb[-1])
        fe = (le - self.le[0]) / (self.le[1] - self.le[0])
        fb = (lbc - self.lb[0]) / (self.lb[1] - self.lb[0])
        i = np.clip(fe.astype(int), 0, self.le.size - 2)
        j = np.clip(fb.astype(int), 0, self.lb.size - 2)
        te, tb = fe - i, fb - j
        L = self.ls
        v = ((1 - te) * (1 - tb) * L[i, j] + te * (1 - tb) * L[i + 1, j]
             + (1 - te) * tb * L[i, j + 1] + te * tb * L[i + 1, j + 1])
        s2 = np.exp(v)
        # beyond the table in b the deflection is negligible
        return np.where(hb, 0.0, np.minimum(s2, 1.0))


_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        _TABLE = _AngleTable()
    return _TABLE


@dataclasses.dataclass(frozen=True)
class Element:
    """A target element: atomic number Z, mass (u) and atomic fraction."""

    Z: int
    mass: float
    fraction: float = 1.0


@dataclasses.dataclass(frozen=True)
class Target:
    """An amorphous target: elements and the atom density (cm^-3), with
    the source of the density."""

    elements: tuple
    atoms_cm3: float
    reference: str

    def __post_init__(self):
        el = tuple(self.elements)
        if not el or not all(isinstance(e, Element) and e.Z >= 1 and e.mass > 0
                             and e.fraction > 0 for e in el):
            raise ValueError("Target.elements must be Element objects")
        if not (self.atoms_cm3 > 0 and math.isfinite(self.atoms_cm3)):
            raise ValueError("Target.atoms_cm3 must be > 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Target needs a reference for its atom density")
        object.__setattr__(self, "elements", el)


def electronic_stopping(E_eV, Z1, M1, Z2):
    """Lindhard-Scharff electronic stopping cross section (eV nm^2) of an
    ion (Z1, mass M1 in u) at energy E (eV) on an atom Z2."""
    v = np.sqrt(2.0 * np.asarray(E_eV, float) * _QE / (M1 * _U))
    zs = Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0)
    return 8.0 * math.pi * _E2 * _A0 * Z1 ** (7.0 / 6.0) * Z2 / zs ** 1.5 * v / _V0


def nuclear_stopping(E_eV, Z1, M1, Z2, M2, n_p=400, screening=zbl_screening,
                     b_max=200.0):
    """Nuclear stopping cross section (eV nm^2), S_n = int T(p) 2 pi p dp
    up to p = b_max a, from the scattering angle of the ZBL potential
    (direct quadrature; another screening function phi(x) may be given)."""
    E = np.atleast_1d(np.asarray(E_eV, float))
    a = zbl_length(Z1, Z2)
    Ec = E * M2 / (M1 + M2)
    eps = a * Ec / (Z1 * Z2 * _E2)
    gam = 4.0 * M1 * M2 / (M1 + M2) ** 2
    out = np.empty(E.size)
    # p from 0 to many screening lengths, on a log grid in b (plus 0)
    t = np.linspace(math.log(1e-5), math.log(b_max), n_p)
    b = np.exp(t)
    for k in range(E.size):
        th = reduced_angle(np.full(b.size, eps[k]), b, screening=screening)
        T = gam * E[k] * np.sin(0.5 * th) ** 2
        f = T * 2.0 * math.pi * (b * a) ** 2           # T 2 pi p dp, dp = p dt
        out[k] = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    return out if np.ndim(E_eV) else float(out[0])


def nuclear_stopping_fit(E_eV, Z1, M1, Z2, M2):
    """The universal fit to the ZBL nuclear stopping (reduced
    s_n(eps) = ln(1 + 1.1383 eps) / (2 (eps + 0.01312 eps^0.21226 +
    0.19593 eps^0.5)), ln(eps) / (2 eps) above eps = 30, as coded in
    Geant4's G4hZiegler1985Nuclear after Ziegler, Biersack and Littmark
    1985), in eV nm^2: for comparison only."""
    E = np.asarray(E_eV, float)
    a = zbl_length(Z1, Z2)
    eps = a * E * M2 / ((M1 + M2) * Z1 * Z2 * _E2)
    sn = np.where(eps <= 30.0,
                  np.log1p(1.1383 * eps) / (2.0 * (eps + 0.01312 * eps ** 0.21226
                                                   + 0.19593 * np.sqrt(eps))),
                  np.log(np.maximum(eps, 1e-300)) / (2.0 * np.maximum(eps, 1e-300)))
    # S_n = (pi a^2 gamma E / eps) s_n
    gam = 4.0 * M1 * M2 / (M1 + M2) ** 2
    return math.pi * a * a * gam * E / eps * sn


@dataclasses.dataclass
class RangeResult:
    """Final positions of the ions (nm): depth below the surface along its
    normal and the lateral offsets (y and x), the ions that came back out
    (backscattered), and the moments of the depth distribution."""

    depth: np.ndarray
    lateral_x: np.ndarray
    lateral_y: np.ndarray
    backscattered: int
    path: np.ndarray
    n_ions: int

    @property
    def projected_range(self):
        return float(np.mean(self.depth))

    @property
    def straggle(self):
        return float(np.std(self.depth))

    @property
    def lateral_straggle(self):
        return float(np.sqrt(0.5 * (np.var(self.lateral_x) + np.var(self.lateral_y))))

    @property
    def skewness(self):
        d = self.depth - self.depth.mean()
        return float(np.mean(d ** 3) / np.std(self.depth) ** 3)

    @property
    def kurtosis(self):
        d = self.depth - self.depth.mean()
        return float(np.mean(d ** 4) / np.var(self.depth) ** 2)

    def histogram(self, bins=60):
        """(depth bin centers in nm, fraction of the ions per nm)."""
        h, e = np.histogram(self.depth, bins=bins)
        return 0.5 * (e[1:] + e[:-1]), h / (self.n_ions * np.diff(e))

    def implant(self, species, dose_cm2, surface_z, window=None, activation=1.0,
                lateral=True):
        """A `process.Implant` with these ranges: its depth profile is this
        distribution (the histogram, `depth_profile`), the lateral straggle
        the simulated one; the dose counts only the ions that stayed in."""
        from .process import Implant
        x, f = self.histogram()
        kept = 1.0 - self.backscattered / self.n_ions
        return Implant(species, dose_cm2 * kept, self.projected_range, self.straggle,
                       "kpenvelope.bca Monte Carlo (ZBL potential, Lindhard-Scharff "
                       "electronic stopping)", surface_z,
                       lateral_straggle_nm=self.lateral_straggle if lateral else None,
                       window=window, activation=activation,
                       depth_profile=(x, f / max(kept, 1e-300)))


def simulate(Z1: int, M1: float, energy_eV: float, target: Target, n_ions: int = 2000,
             tilt_deg: float = 0.0, seed: int = 0, electronic_scale: float = 1.0,
             electronic=None, nuclear: bool = True, cutoff_eV: float = None,
             max_steps: int = 200000):
    """Slow n_ions ions (atomic number Z1, mass M1 in u, energy in eV) in
    the target (the surface is z = 0, depth positive), entering at tilt_deg
    from the normal. electronic : None (Lindhard-Scharff times
    electronic_scale) or a function S_e(E_eV, element) -> eV nm^2.
    nuclear=False turns the collisions off (a check). The ion stops below
    cutoff_eV (default 5 eV)."""
    if not (Z1 >= 1 and M1 > 0 and energy_eV > 0 and n_ions >= 1):
        raise ValueError("Z1 >= 1, M1 > 0, energy > 0 and n_ions >= 1 needed")
    if not (electronic_scale >= 0):
        raise ValueError("electronic_scale must be >= 0")
    rng = np.random.default_rng(seed)
    N = target.atoms_cm3 * 1e-21                   # nm^-3
    L = N ** (-1.0 / 3.0)
    pmax2 = 1.0 / (math.pi * N ** (2.0 / 3.0))
    els = target.elements
    frac = np.array([e.fraction for e in els], float)
    frac = frac / frac.sum()
    cum = np.cumsum(frac)
    a_k = np.array([zbl_length(Z1, e.Z) for e in els])
    M2 = np.array([e.mass for e in els])
    Z2 = np.array([e.Z for e in els], float)
    gam = 4.0 * M1 * M2 / (M1 + M2) ** 2
    epsf = a_k * M2 / ((M1 + M2) * Z1 * Z2 * _E2)       # eps = epsf * E
    cutoff = 5.0 if cutoff_eV is None else float(cutoff_eV)

    def se_mix(E):
        if electronic is None:
            s = sum(f * electronic_stopping(E, Z1, M1, e.Z) for f, e in zip(frac, els))
            return electronic_scale * s
        return sum(f * np.asarray(electronic(E, e), float) for f, e in zip(frac, els))
    tab = _table()
    t = math.radians(tilt_deg)
    pos = np.zeros((n_ions, 3))
    dirv = np.tile(np.array([math.sin(t), 0.0, math.cos(t)]), (n_ions, 1))
    E = np.full(n_ions, float(energy_eV))
    path = np.zeros(n_ions)
    alive = np.ones(n_ions, bool)
    back = np.zeros(n_ions, bool)
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        Ei = E[idx]
        # flight, with the electronic loss along it
        step = np.full(idx.size, L)
        dE = N * se_mix(Ei) * step
        # do not lose more than there is: the last flight is shortened
        short = dE >= Ei - cutoff
        if np.any(short):
            step = np.where(short, step * np.clip((Ei - cutoff) / np.maximum(dE, 1e-300),
                                                  0.0, 1.0), step)
            dE = np.where(short, np.maximum(Ei - cutoff, 0.0), dE)
        pos[idx] += dirv[idx] * step[:, None]
        path[idx] += step
        Ei = Ei - dE
        out = pos[idx, 2] < 0.0
        if nuclear:
            k = np.searchsorted(cum, rng.random(idx.size))
            k = np.minimum(k, len(els) - 1)
            p = np.sqrt(rng.random(idx.size) * pmax2)
            eps = epsf[k] * Ei
            s2 = tab(eps, p / a_k[k])
            T = gam[k] * Ei * s2
            th = 2.0 * np.arcsin(np.sqrt(s2))
            # lab deflection of the ion
            mr = M1 / M2[k]
            psi = np.arctan2(np.sin(th), np.cos(th) + mr)
            Ei = Ei - T
            phi = rng.random(idx.size) * 2.0 * math.pi
            d = dirv[idx]
            # rotate d by psi about a random axis perpendicular to it
            ref = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]),
                           np.array([[1.0, 0.0, 0.0]]))
            e1 = np.cross(d, ref)
            e1 /= np.linalg.norm(e1, axis=1)[:, None]
            e2 = np.cross(d, e1)
            cp, sp = np.cos(psi)[:, None], np.sin(psi)[:, None]
            nd = cp * d + sp * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
            dirv[idx] = nd / np.linalg.norm(nd, axis=1)[:, None]
        E[idx] = Ei
        stop = (Ei <= cutoff) | out
        back[idx[out]] = True
        alive[idx[stop]] = False
    else:
        raise RuntimeError("some ions did not stop; raise max_steps")
    keep = ~back
    return RangeResult(depth=pos[keep, 2], lateral_x=pos[keep, 0] - 0.0,
                       lateral_y=pos[keep, 1], backscattered=int(back.sum()),
                       path=path[keep], n_ions=n_ions)
