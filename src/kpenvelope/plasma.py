"""The chemistry of a low-pressure plasma, averaged over its volume (new in
0.31.0): the densities of every species and the electron temperature of a
discharge from the reactions you give, and the ion and neutral fluxes it
sends to the wafer, for `process.EtchKinetics`.

The model is a global (volume-averaged) model of a plasma (with negative
ions too since 0.33.0) in a cylinder of radius R and length L (volume V, wall area A). For every
heavy species s, in steady state,

    0 = sum_r nu_rs R_r + Q_s / V - n_s S / V - (wall loss)_s
        + (wall gain)_s,

with R_r = k_r(T_e) times the product of the reactants' densities
(electrons included), Q_s the feed (particles per second), S the pumping
speed (m^3/s). Electrons follow from quasi-neutrality, n_e = sum of the
positive ions. Losses to the wall:

* a positive ion of mass M leaves at the Bohm speed u_B = sqrt(e T_e / M)
  from the sheath edge, where its density is h n_s: a loss of
  h n_s u_B A / V. The edge-to-center ratio h is yours (with its source):
  it depends on the pressure and the shape, and the published forms are
  fits that the package has not checked;
* a neutral with wall reaction probability gamma leaves at the kinetic
  rate gamma n_s vbar / 4 A / V, vbar = sqrt(8 k T_g / (pi M)) (the flux
  of a Maxwellian gas through a plane): the low-pressure limit, where a
  particle crosses the chamber without collisions. At higher pressure
  diffusion to the wall slows the loss and this overstates it.
* a negative ion is held inside by the sheath (the wall is negative with
  respect to the plasma), so it is lost only in the gas (detachment,
  recombination with positive ions) and by the pump. Electrons follow from
  quasi-neutrality, n_e = sum n_+ - sum n_-. With negative ions the
  density profile and the Bohm speed at the edge change; here u_B stays
  sqrt(e T_e / M) and h is yours, so h must carry that change (its
  published forms for electronegative plasmas are fits as well).

What a particle lost at the wall turns into (an ion comes back as its
neutral, two atoms as a molecule) is given with each species.

The electrons' energy balance:

    P / V = e sum_r R_r eps_r + e sum_el R_el 3 (m_e / M) T_e
            + e (A / V) sum_ions h n_i u_B,i (2 T_e + eps_wall,i),

with eps_r the energy an electron loses in reaction r (its threshold), the
elastic collisions' share 3 (m_e / M) T_e, and at the wall 2 T_e carried
by every electron (a Maxwellian flux) and, by every ion, the energy it
gains in front of a floating wall: T_e / 2 in the presheath (to reach
u_B) and the sheath drop (T_e / 2) ln(M / (2 pi m_e)), at which the
electron and ion fluxes are equal. These are textbook results of kinetic
theory (e.g. M. A. Lieberman and A. J. Lichtenberg, Principles of Plasma
Discharges and Materials Processing, 2nd ed., Wiley, 2005, chapters 2, 6
and 10); every rate coefficient, threshold, sticking probability and the
h factor are yours. Negative ions, the electron energy distribution
(the rates are for a Maxwellian), the gas heating and the spatial
profiles are not modeled.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["Species", "Reaction", "GlobalPlasma", "SCCM", "fluxes_cm2"]

QE = 1.602176634e-19            # C (exact)
KB = 1.380649e-23               # J/K (exact)
ME = 9.1093837015e-31           # kg (CODATA 2018)
AMU = 1.66053906660e-27         # kg (CODATA 2018)
#: one standard cubic centimeter per minute in particles per second, at
#: 273.15 K and 101.325 kPa (Loschmidt's number 2.686780111e25 m^-3, CODATA
#: 2018); some flow controllers use 20 C instead (multiply by 293.15/273.15)
SCCM = 2.686780111e25 * 1e-6 / 60.0


@dataclasses.dataclass(frozen=True)
class Species:
    """A heavy species: name, mass (atomic mass units), charge (0 for a
    neutral, 1 for a positive ion, -1 for a negative ion: since 0.33.0;
    negative ions never reach the wall), wall: the probability that a neutral
    reacts at the wall when it hits it (0 to 1; ions always do), and
    wall_products: what each particle lost at the wall comes back as
    ({name: number}, e.g. {"Ar": 1} for Ar+, {"Cl2": 0.5} for Cl)."""

    name: str
    mass_amu: float
    charge: int = 0
    wall: float = 0.0
    wall_products: dict = None

    def __post_init__(self):
        if not (isinstance(self.name, str) and self.name and self.name != "e"):
            raise ValueError("a species needs a name (not 'e')")
        if not (math.isfinite(self.mass_amu) and self.mass_amu > 0):
            raise ValueError("mass_amu must be > 0")
        if self.charge not in (-1, 0, 1):
            raise ValueError("charge must be -1, 0 or 1")
        if not (0.0 <= self.wall <= 1.0):
            raise ValueError("wall must be between 0 and 1")
        if self.wall_products is None:
            object.__setattr__(self, "wall_products", {})


@dataclasses.dataclass(frozen=True)
class Reaction:
    """One reaction. reactants and products are tuples of species names,
    with "e" for an electron (an electron-impact reaction has one "e" among
    its reactants). The rate coefficient (m^3/s for two reactants) is
    k = A T_e^b exp(-C / T_e) for electron impact (T_e in eV), or k = A for
    reactions between heavy species. energy_eV is the energy an electron
    loses in it (its threshold; 0 for heavy-species reactions);
    elastic=True for elastic electron collisions (reactants and products
    the same), which take 3 (m_e / M) T_e each. reference: the source of
    the numbers (required)."""

    reactants: tuple
    products: tuple
    A: float
    reference: str
    b: float = 0.0
    C: float = 0.0
    energy_eV: float = 0.0
    elastic: bool = False

    def __post_init__(self):
        object.__setattr__(self, "reactants", tuple(self.reactants))
        object.__setattr__(self, "products", tuple(self.products))
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("a Reaction needs a reference")
        if not (math.isfinite(self.A) and self.A >= 0 and math.isfinite(self.b)
                and math.isfinite(self.C) and math.isfinite(self.energy_eV)):
            raise ValueError("the rate numbers must be finite, A >= 0")
        if self.reactants.count("e") > 1:
            raise ValueError("at most one electron among the reactants")
        if self.elastic and sorted(self.reactants) != sorted(self.products):
            raise ValueError("an elastic collision has the same reactants and products")

    @property
    def electron_impact(self):
        return "e" in self.reactants

    def k(self, Te):
        if self.electron_impact:
            return self.A * Te ** self.b * math.exp(-self.C / Te)
        return self.A


class GlobalPlasma:
    """A volume-averaged plasma (see the module docstring).

    species : list of `Species`; reactions : list of `Reaction`;
    radius_m, length_m : the cylinder; power_W : the power the electrons
    absorb; feed : {name: particles per second} (`SCCM` converts);
    pump_m3_s : the pumping speed; h : the edge-to-center density ratio of
    the ions (with h_source, its reference); gas_temperature_K.

    `solve()` returns the steady state."""

    def __init__(self, species, reactions, radius_m, length_m, power_W, feed, pump_m3_s,
                 h, h_source, gas_temperature_K=300.0):
        self.sp = list(species)
        names = [s.name for s in self.sp]
        if len(set(names)) != len(names):
            raise ValueError("two species share a name")
        self.idx = {n: i for i, n in enumerate(names)}
        self.rx = list(reactions)
        for r in self.rx:
            for n in r.reactants + r.products:
                if n != "e" and n not in self.idx:
                    raise ValueError(f"unknown species {n!r} in a reaction")
            q = lambda side: sum(-1 if n == "e" else self.sp[self.idx[n]].charge
                                 for n in side)
            if q(r.reactants) != q(r.products):
                raise ValueError(f"a reaction does not keep the charge: {r.reactants} -> "
                                 f"{r.products}")
            m = lambda side: sum(0.0 if n == "e" else self.sp[self.idx[n]].mass_amu
                                 for n in side)
            if abs(m(r.reactants) - m(r.products)) > 1e-6 * max(m(r.reactants), 1.0):
                raise ValueError(f"a reaction does not keep the mass: {r.reactants} -> "
                                 f"{r.products}")
        for s in self.sp:
            for n in s.wall_products:
                if n not in self.idx:
                    raise ValueError(f"unknown wall product {n!r}")
            if s.charge == 1 and not s.wall_products:
                raise ValueError(f"say what the ion {s.name} becomes at the wall")
            back = sum(self.sp[self.idx[n]].mass_amu * c for n, c in s.wall_products.items())
            if s.wall_products and abs(back - s.mass_amu) > 1e-6 * s.mass_amu:
                raise ValueError(f"the wall products of {s.name} do not keep its mass")
        if not any(s.charge == 1 for s in self.sp):
            raise ValueError("there must be at least one positive ion")
        for v, nm in ((radius_m, "radius_m"), (length_m, "length_m"), (power_W, "power_W"),
                      (pump_m3_s, "pump_m3_s"), (gas_temperature_K, "gas_temperature_K")):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{nm} must be > 0")
        if not (0.0 < h <= 1.0):
            raise ValueError("h must be in (0, 1]")
        if not (isinstance(h_source, str) and h_source.strip()):
            raise ValueError("give the source of h")
        self.R, self.L = float(radius_m), float(length_m)
        self.V = math.pi * self.R ** 2 * self.L
        self.A = 2 * math.pi * self.R ** 2 + 2 * math.pi * self.R * self.L
        self.P, self.S, self.h, self.Tg = float(power_W), float(pump_m3_s), float(h), \
            float(gas_temperature_K)
        self.feed = np.zeros(len(self.sp))
        for n, q in dict(feed).items():
            if n not in self.idx:
                raise ValueError(f"unknown species {n!r} in the feed")
            if not (math.isfinite(q) and q >= 0):
                raise ValueError("feeds must be >= 0")
            self.feed[self.idx[n]] = q
        if not self.feed.sum() > 0:
            raise ValueError("feed some gas")
        self.M = np.array([s.mass_amu * AMU for s in self.sp])
        self.ion = np.array([s.charge == 1 for s in self.sp])
        self.neg = np.array([s.charge == -1 for s in self.sp])
        self.heavy_neutral = np.array([s.charge == 0 for s in self.sp])
        self.vbar = np.sqrt(8 * KB * self.Tg / (math.pi * self.M))
        self.gam = np.array([0.0 if s.charge else s.wall for s in self.sp])
        # where the wall sends what it takes: W[j, i] particles of j per
        # particle of i lost
        n = len(self.sp)
        self.W = np.zeros((n, n))
        for i, s in enumerate(self.sp):
            for nm, c in s.wall_products.items():
                self.W[self.idx[nm], i] += c

    # ------------------------------------------------------------ pieces
    def _ne(self, n):
        """electrons from quasi-neutrality: positive ions less negative ions"""
        return float(n[self.ion].sum() - n[self.neg].sum())

    def _rates(self, n, Te, ne=None):
        """R_r for every reaction (m^-3 s^-1); n the heavy densities, ne
        the electrons (default: from quasi-neutrality)."""
        ne = self._ne(n) if ne is None else ne
        out = np.empty(len(self.rx))
        for k, r in enumerate(self.rx):
            v = r.k(Te)
            for nm in r.reactants:
                v *= ne if nm == "e" else n[self.idx[nm]]
            out[k] = v
        return out

    def _wall_rates(self, n, Te):
        """loss frequency (1/s) of every heavy species at the wall"""
        uB = np.sqrt(QE * Te / self.M)
        # negative ions stay inside: the sheath at the wall turns them back
        nu = np.where(self.ion, self.h * uB,
                      np.where(self.neg, 0.0, self.gam * self.vbar / 4.0)) * self.A / self.V
        return nu, uB

    def residual(self, n, Te):
        """dn/dt for every species and d(3/2 n_e T_e)/dt (eV m^-3 s^-1),
        the electrons from quasi-neutrality"""
        dn, _, dE = self._balance(n, self._ne(n), Te)
        return dn, dE

    def _balance(self, n, ne, Te):
        """dn/dt of the heavy species, of the electrons (made and lost in
        reactions, pumped, and leaving with the positive ions at the wall)
        and d(3/2 n_e T_e)/dt, for given electrons ne"""
        R = self._rates(n, Te, ne)
        dn = self.feed / self.V - n * self.S / self.V
        for k, r in enumerate(self.rx):
            for nm in r.reactants:
                if nm != "e":
                    dn[self.idx[nm]] -= R[k]
            for nm in r.products:
                if nm != "e":
                    dn[self.idx[nm]] += R[k]
        nu, uB = self._wall_rates(n, Te)
        lost = nu * n
        dn += -lost + self.W @ lost
        # the electrons' energy
        heat = self.P / (QE * self.V)
        coll = 0.0
        for k, r in enumerate(self.rx):
            if not r.electron_impact:
                continue
            if r.elastic:
                heavy = [nm for nm in r.reactants if nm != "e"][0]
                coll += R[k] * 3.0 * ME / self.M[self.idx[heavy]] * Te
            else:
                coll += R[k] * r.energy_eV
        wall_e = 0.0
        for i in np.flatnonzero(self.ion):
            eps_i = 0.5 * Te + 0.5 * Te * math.log(self.M[i] / (2 * math.pi * ME))
            wall_e += lost[i] * (2.0 * Te + eps_i)
        dE = heat - coll - wall_e
        made = np.array([r.products.count("e") - r.reactants.count("e") for r in self.rx])
        dne = float(made @ R) - ne * self.S / self.V - float(lost[self.ion].sum())
        return dn, dne, dE

    # ------------------------------------------------------------ solve
    def _start_Te(self, n0):
        """T_e at which the electrons the gas makes (with the feed's
        neutral densities) balance those it loses to attachment, the wall
        and the pump: the start of the time integration"""
        from scipy.optimize import brentq
        Mion = float(self.M[self.ion].min())

        def bal(Te):
            nu = 0.0
            for r in self.rx:
                if r.reactants.count("e") != 1 or r.elastic:
                    continue
                heavy = [nm for nm in r.reactants if nm != "e"]
                if any(self.sp[self.idx[h_]].charge for h_ in heavy):
                    continue
                v = r.k(Te) * (r.products.count("e") - 1)
                for h_ in heavy:
                    v *= n0[self.idx[h_]]
                nu += v
            return nu - self.h * math.sqrt(QE * Te / Mion) * self.A / self.V - self.S / self.V
        try:
            return brentq(bal, 0.3, 100.0)
        except ValueError:
            return 3.0

    def solve(self, Te_guess=None, t_end=10.0, rtol=1e-10, guess=None):
        """The steady state: time integration from a start (the feed alone,
        a small ion seed) until it settles, then Newton's method on the
        steady equations (in the logarithms of the densities and of T_e).
        Returns dict(Te_eV, densities (m^-3, by name), ne, pressure_Pa,
        ion_flux (m^-2 s^-1 at the wall, by ion), neutral_flux (by
        neutral), ion_energy_eV (by ion, at a floating wall), residual).
        guess: an earlier result (of a nearby set of conditions) to start
        Newton's method from directly; the time integration is used when
        that does not settle."""
        from scipy.integrate import solve_ivp
        from scipy.optimize import root
        n0 = np.maximum(self.feed / self.S, 1e10)
        n0 = np.where(self.ion, 1e-3 * n0.max() * 1e-3, n0)
        n0 = np.where(self.neg, 1e-3 * n0[self.ion].sum(), n0)
        ns = len(self.sp)
        # the electrons are an unknown of their own while the plasma
        # settles (so they stay positive however many negative ions form);
        # the charge they and the ions carry is kept, so the start's
        # neutrality holds, and the last step imposes it exactly
        if Te_guess is None:
            Te_guess = self._start_Te(n0)
        y0 = np.concatenate([np.log(n0), [math.log(self._ne(n0))], [math.log(Te_guess)]])

        def f(t, y):
            n = np.exp(np.clip(y[:ns], -200, 200))
            ne = float(np.exp(np.clip(y[ns], -200, 200)))
            Te = float(np.exp(np.clip(y[ns + 1], -10, 10)))
            dn, dne, dE = self._balance(n, ne, Te)
            # d(3/2 n_e T_e)/dt = dE  ->  dT_e/dt from dn_e/dt
            dTe = (dE - 1.5 * Te * dne) / (1.5 * ne)
            return np.concatenate([dn / n, [dne / ne, dTe / Te]])
        def integrate():
            sol = solve_ivp(f, (0.0, t_end), y0, method="BDF", rtol=1e-8, atol=1e-10)
            return sol.y[:, -1]

        def g(y):
            n = np.exp(y[:ns])
            ne = float(np.exp(y[ns]))
            Te = float(np.exp(y[ns + 1]))
            dn, _, dE = self._balance(n, ne, Te)
            scale_n = self.feed.sum() / self.V + n * self.S / self.V + 1e-300
            return np.concatenate([dn / np.maximum(scale_n, n * 1e-3),
                                   [(ne - self._ne(n)) / ne],
                                   [dE / max(self.P / (QE * self.V), 1e-300)]])
        def polish(y):
            with np.errstate(all="ignore"):
                try:
                    r = root(g, y, method="hybr", tol=1e-14)
                    gy, gr = np.max(np.abs(g(y))), np.max(np.abs(g(r.x)))
                except (ZeroDivisionError, FloatingPointError, ValueError):
                    return y, math.inf
            if not np.isfinite(gr) and not np.isfinite(gy):
                return y, math.inf
            return (r.x, float(gr)) if gr < gy or not np.isfinite(gy) else (y, float(gy))
        res = math.inf
        if guess is not None:
            d = guess["densities"]
            yg = np.concatenate([np.log([max(d[s_.name], 1e-30) for s_ in self.sp]),
                                 [math.log(guess["ne"])], [math.log(guess["Te_eV"])]])
            y, res = polish(yg)
        if not res < 1e-6:
            y, res = polish(integrate())
        if not res < 1e-6:
            raise RuntimeError(f"the plasma did not settle (relative residual {res:.1e}); "
                               "start from a nearby solution with guess=")
        y = np.concatenate([y[:ns], y[ns + 1:]])
        n = np.exp(y[:ns])
        Te = float(np.exp(y[ns]))
        nu, uB = self._wall_rates(n, Te)
        flux = nu * n * self.V / self.A                           # per m^2 of wall
        return dict(
            Te_eV=Te, ne=self._ne(n), residual=res,
            densities={s.name: float(n[i]) for i, s in enumerate(self.sp)},
            pressure_Pa=float(n[self.heavy_neutral].sum() * KB * self.Tg),
            ion_flux={s.name: float(self.h * n[i] * uB[i])
                      for i, s in enumerate(self.sp) if s.charge == 1},
            neutral_flux={s.name: float(n[i] * self.vbar[i] / 4.0)
                          for i, s in enumerate(self.sp) if not s.charge},
            wall_loss={s.name: float(flux[i]) for i, s in enumerate(self.sp)},
            ion_energy_eV={s.name: float(0.5 * Te + 0.5 * Te * math.log(
                self.M[i] / (2 * math.pi * ME))) for i, s in enumerate(self.sp)
                if s.charge == 1})


def fluxes_cm2(result, neutral, ion):
    """(neutral_flux, ion_flux) in cm^-2 s^-1 for `process.EtchKinetics`:
    the flux of one neutral species through a plane (n vbar / 4) and the
    Bohm flux of one ion (h n u_B), from a `GlobalPlasma.solve` result."""
    if neutral not in result["neutral_flux"]:
        raise ValueError(f"no neutral {neutral!r} in the result")
    if ion not in result["ion_flux"]:
        raise ValueError(f"no ion {ion!r} in the result")
    return result["neutral_flux"][neutral] * 1e-4, result["ion_flux"][ion] * 1e-4
