"""One-dimensional drift-diffusion for c-plane nitride stacks: electrons,
holes, recombination, ohmic and Schottky contacts, the polarization
charges of `interface_charges`, current-voltage and capacitance-voltage
curves (new in v0.18.0).

Unknowns. On a grid along z (the stack's [0001] axis, bottom first, as in
the rest of the package) the solver finds the electrostatic potential psi
(V) and the quasi-Fermi levels F_n and F_p (eV) at every node. With
E_c = E_c0 - psi and E_v = E_c - E_g,

    n = N_c F((F_n - E_c) / kT),   p = N_v F((E_v - F_p) / kT),

where F is exp (statistics="boltzmann") or the Fermi-Dirac integral
F_1/2 (statistics="fermi", the default). N_c uses the electron mass of the
material record (as `solve_electrons` does); N_v uses the hole
density-of-states mass you give in `Transport`.

Equations (steady state):

    d/dz (eps0 eps_r dpsi/dz) = -q (p - n + N_d+ - N_a-) - (sheet charges),
    dGamma_n/dz = G - R,    dGamma_p/dz = G - R,

with particle fluxes Gamma_n = -mu_n n dF_n/dz and Gamma_p = mu_p p
dF_p/dz (F in eV). They are discretized by finite volumes with the
Scharfetter-Gummel flux (D. L. Scharfetter and H. K. Gummel, IEEE Trans.
Electron Devices 16, 64 (1969)) written for the variable w = F_n/kT - ln n
(and v = F_p/kT + ln p for holes). For Boltzmann statistics this is the
classic scheme; for Fermi-Dirac statistics it is the same flux with the
band edge corrected by the degeneracy factor, which keeps the current
exactly zero when the quasi-Fermi level is flat and reduces to
Gamma = -mu n dF/dz in the continuum limit. Band offsets enter through E_c0
and E_g of each layer, so an abrupt offset is a step between two nodes.

Recombination: Shockley-Read-Hall with lifetimes tau_n, tau_p and a trap
level, radiative B n p, Auger (C_n n + C_p p); each acts on
n p - n_eq p_eq = n p (1 - exp(-(F_n - F_p)/kT)), so it is exactly zero in
equilibrium. Optional generation G(z). Dopants ionize completely unless a
depth is given (donor degeneracy 2 as in `solve_electrons`; acceptor
degeneracy `acceptor_degeneracy`, 4 by default as in textbooks for cubic
semiconductors; wurtzite's top valence band is only spin degenerate, so
pass 2 if your source does).

Contacts (`Contact`): "ohmic" fixes F_n = F_p = -V and the charge-neutral
potential; "schottky" fixes F_n = F_p = -V and E_c - F = barrier at the
contact (infinite surface recombination velocity: the carrier densities
take their equilibrium values against the metal Fermi level). V is the
applied voltage in volts; a positive V lowers the contact's Fermi level.

Solution. The equilibrium potential is found first (Newton on Poisson
alone); the bias is then ramped in steps of at most `max_bias_step` with a
fully coupled Newton method on (psi, F_n, F_p). The Jacobian is built by
finite differences in nine colored sweeps (exact sparsity, any physics
model) and solved as a banded system (SciPy's banded LAPACK solver when
SciPy is installed, otherwise a NumPy banded LU with partial pivoting).

Limits, stated plainly: drift-diffusion only (no thermionic emission or
tunneling across abrupt offsets: an abrupt offset passes current like a
very thin graded layer, which overestimates transport over tall barriers),
no quantization (use `solve_electrons` for subbands), constant mobilities
(no field dependence or velocity saturation), no lattice heating, no
transient or small-signal AC (the capacitance is quasi-static, dQ/dV), and
every transport number (mobilities, lifetimes, B, C, hole mass) comes from
you with a reference, because there is no single accepted set.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .electrons import EPS0, HBAR, KB, M0, _thomas, fermi_dirac_half
from .interfaces import InterfaceCharge, StackLayer, interface_charges
from .nitride import QE

__all__ = ["Transport", "Contact", "DriftDiffusionResult",
           "solve_drift_diffusion", "solve_quasi_equilibrium", "iv_curve",
           "cv_curve", "cv_curve_quantum", "bernoulli"]

#: q / eps0 in V nm, for densities in nm^-3 and lengths in nm
_CP = QE / EPS0 * 1e9
_A_CM2 = QE * 1e14          # e nm^-2 s^-1 -> A cm^-2
_JAC_STEP = 1e-7            # finite-difference step of the Jacobian (V, eV)


@dataclasses.dataclass(frozen=True)
class Transport:
    """Transport numbers of one layer. Every one comes from your source.

    mobility_n, mobility_p : cm^2 / (V s), > 0.
    hole_dos_mass : valence-band density-of-states mass (m0), > 0.
    reference : where the numbers come from. Required, on purpose.
    tau_n, tau_p : Shockley-Read-Hall lifetimes (s); both infinite (no SRH)
        or both finite.
    trap_level : SRH trap energy above midgap (eV).
    radiative : B (cm^3 / s), >= 0.
    auger_n, auger_p : C_n, C_p (cm^6 / s), >= 0.
    """

    mobility_n: float
    mobility_p: float
    hole_dos_mass: float
    reference: str
    tau_n: float = math.inf
    tau_p: float = math.inf
    trap_level: float = 0.0
    radiative: float = 0.0
    auger_n: float = 0.0
    auger_p: float = 0.0

    def __post_init__(self):
        for f in ("mobility_n", "mobility_p", "hole_dos_mass"):
            v = getattr(self, f)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"Transport.{f} must be finite and > 0")
        if not (isinstance(self.reference, str) and self.reference.strip()):
            raise ValueError("Transport needs a reference for its numbers")
        tn, tp = self.tau_n, self.tau_p
        if not (tn > 0 and tp > 0):
            raise ValueError("lifetimes must be > 0 (math.inf for no SRH)")
        if math.isinf(tn) != math.isinf(tp):
            raise ValueError("give both SRH lifetimes, or neither")
        if not math.isfinite(self.trap_level):
            raise ValueError("trap_level must be finite (eV)")
        for f in ("radiative", "auger_n", "auger_p"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"Transport.{f} must be finite and >= 0")


@dataclasses.dataclass(frozen=True)
class Contact:
    """A contact at the bottom or the top of the stack.

    kind : "ohmic", "schottky", or "zero-field" (bottom only, and only
        for `solve_quasi_equilibrium`: a thick buffer with no electric
        field at its bottom, as in `solve_electrons`).
    barrier : Schottky barrier, E_c - E_F(metal) at the contact (eV).
    voltage : applied voltage (V).
    thermionic : for a Schottky contact in `solve_drift_diffusion`, take
        the carriers across the metal by thermionic emission (flux
        v_R (n - n_0), Richardson velocity v_R = sqrt(kT / 2 pi m); C. R.
        Crowell and S. M. Sze, Solid-State Electron. 9, 1035 (1966))
        instead of holding them at their equilibrium densities (new in
        v0.18.0). The forward current then follows the thermionic-
        emission law when the bulk is not limiting (checked in the tests).
    """

    kind: str = "ohmic"
    barrier: float = None
    voltage: float = 0.0
    thermionic: bool = False

    def __post_init__(self):
        if self.kind not in ("ohmic", "schottky", "zero-field"):
            raise ValueError("Contact.kind must be 'ohmic', 'schottky' or "
                             "'zero-field'")
        if self.kind == "schottky":
            if self.barrier is None or not math.isfinite(self.barrier):
                raise ValueError("a Schottky contact needs a finite barrier (eV)")
        elif self.barrier is not None:
            raise ValueError(f"a {self.kind} contact takes no barrier")
        if not math.isfinite(self.voltage):
            raise ValueError("Contact.voltage must be finite (V)")
        if self.thermionic and self.kind != "schottky":
            raise ValueError("thermionic emission needs a Schottky contact")


@dataclasses.dataclass
class DriftDiffusionResult:
    """Steady state (lengths nm, energies eV, densities cm^-3).

    z : nodes. psi : potential (V). ec, ev : band edges. efn, efp :
    quasi-Fermi levels. n, p : carriers. donors_ionized,
    acceptors_ionized. recombination, generation : cm^-3 s^-1.
    z_mid : cell centers; jn, jp, j : current densities on the cells
    (A/cm^2, positive along +z). current_density : the terminal
    current (at an ohmic contact); current_spread : the largest deviation
    of j from it over the cells where a carrier is above the density floor
    (a check of the solution, A/cm^2).
    gate_charge : charge on the top contact (C/cm^2).
    interfaces : the `InterfaceCharge` list used.
    converged, newton_iterations, last_step : of the final Newton solve.
    top, bottom : the contacts. temperature_K, statistics.
    """

    z: np.ndarray
    psi: np.ndarray
    ec: np.ndarray
    ev: np.ndarray
    efn: np.ndarray
    efp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    donors_ionized: np.ndarray
    acceptors_ionized: np.ndarray
    recombination: np.ndarray
    generation: np.ndarray
    z_mid: np.ndarray
    jn: np.ndarray
    jp: np.ndarray
    j: np.ndarray
    current_density: float
    current_spread: float
    gate_charge: float
    interfaces: list
    converged: bool
    newton_iterations: int
    last_step: float
    top: Contact
    bottom: Contact
    temperature_K: float
    statistics: str
    _state: tuple = dataclasses.field(default=None, repr=False)

    def sheet(self, z_min=None, z_max=None, carrier="n"):
        """Carriers per cm^2 between z_min and z_max (nm), trapezoid rule."""
        lo = self.z[0] if z_min is None else z_min
        hi = self.z[-1] if z_max is None else z_max
        m = (self.z >= lo - 1e-9) & (self.z <= hi + 1e-9)
        c = {"n": self.n, "p": self.p}[carrier]
        zz, cc = self.z[m], c[m]
        return float(np.sum(0.5 * (cc[1:] + cc[:-1]) * np.diff(zz)) * 1e-7)


def bernoulli(x):
    """B(x) = x / (exp(x) - 1), with B(0) = 1, stable for all x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs * xs / 12.0
    xl = x[~small]
    with np.errstate(over="ignore"):
        out[~small] = np.where(xl > 700.0, xl * np.exp(-xl),
                               xl / np.expm1(np.minimum(xl, 700.0)))
    return out


# ------------------------------------------------------- banded linear solve
def _solve_banded_numpy(l, u, ab, b):
    """Banded LU with partial pivoting; ab in LAPACK 'diagonal-ordered'
    form, ab[u + i - j, j] = a[i, j]."""
    n = ab.shape[1]
    W = 2 * l + u + 1
    R = np.zeros((n, W))          # R[i, c] = a[i, i + c - l]
    jj = np.arange(n)
    for d in range(l + u + 1):
        i = jj + d - u
        ok = (i >= 0) & (i < n)
        R[i[ok], u - d + l] = ab[d, jj[ok]]
    x = np.array(b, dtype=float)
    for k in range(n):
        rows = np.arange(k, min(n, k + l + 1))
        cols = np.arange(k, min(n, k + l + u + 1))
        vals = R[rows, k - rows + l]
        p = int(rows[np.argmax(np.abs(vals))])
        if p != k:
            ck, cp = cols - k + l, cols - p + l
            tmp = R[k, ck].copy()
            R[k, ck] = R[p, cp]
            R[p, cp] = tmp
            x[k], x[p] = x[p], x[k]
        piv = R[k, l]
        if piv == 0.0:
            raise np.linalg.LinAlgError("singular Jacobian")
        rr = rows[1:]
        if rr.size:
            f = R[rr, k - rr + l] / piv
            R[rr[:, None], cols[None, :] - rr[:, None] + l] -= \
                f[:, None] * R[k, cols - k + l][None, :]
            x[rr] -= f * x[k]
    for k in range(n - 1, -1, -1):
        cols = np.arange(k + 1, min(n, k + l + u + 1))
        x[k] = (x[k] - R[k, cols - k + l] @ x[cols]) / R[k, l]
    return x


def _solve_banded(l, u, ab, b, use_scipy=True):
    if use_scipy:
        try:
            from scipy.linalg import solve_banded
        except ImportError:          # pragma: no cover - depends on install
            use_scipy = False
    if use_scipy:
        return solve_banded((l, u), ab, b)
    return _solve_banded_numpy(l, u, ab, b)


# ------------------------------------------------------------------ problem
def _need(v, what, label):
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"layer {label!r} has no {what} (NaN); the "
                         "drift-diffusion solver needs a cited value")
    return float(v)


class _Problem:
    def __init__(self, layers, transport, T, dz, statistics, interfaces,
                 extra, generation, g_acc, hole_mass=None):
        self.T = T
        self.kT = KB * T
        self.fermi = statistics == "fermi"
        self.g_acc = g_acc
        edges = np.cumsum([0.0] + [L.thickness for L in layers])
        Ltot = edges[-1]
        N = int(round(Ltot / dz)) + 1
        if N < 5:
            raise ValueError("the grid needs at least 5 nodes; lower dz")
        z = np.linspace(0.0, Ltot, N)
        h = z[1] - z[0]
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0,
                      len(layers) - 1)
        mid = 0.5 * (z[1:] + z[:-1])
        cidx = np.clip(np.searchsorted(edges, mid, side="right") - 1, 0,
                       len(layers) - 1)
        eps = [_need(L.material.eps_r, "eps_r", L.label) for L in layers]
        ec = [_need(L.material.ec, "conduction-band edge ec", L.label)
              for L in layers]
        eg = [_need(L.material.eg, "band gap eg", L.label) for L in layers]
        me = [_need(L.material.me, "electron mass me", L.label) for L in layers]
        self.z, self.h, self.N = z, h, N
        self.Ec0 = np.array([ec[i] for i in idx])
        self.Eg = np.array([eg[i] for i in idx])
        pref = (KB * T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
        self.Nc = pref * (np.array([me[i] for i in idx]) * M0) ** 1.5
        if transport is not None:
            tr = list(transport)
            hm = [t.hole_dos_mass for t in tr]
        else:
            tr = None
            hm = hole_mass
        self.holes = hm is not None
        self.Nv = (pref * (np.array([hm[i] for i in idx]) * M0) ** 1.5
                   if self.holes else np.zeros(N))
        self.eps_c = np.array([eps[i] for i in cidx])
        if tr is None:              # quasi-equilibrium: no transport needed
            tr = [Transport(1.0, 1.0, 1.0, reference="unused")] * len(layers)
        self.Dn = np.array([tr[i].mobility_n for i in cidx]) * 1e14 * self.kT
        self.Dp = np.array([tr[i].mobility_p for i in cidx]) * 1e14 * self.kT
        self.ND = np.array([layers[i].donors for i in idx], float)
        self.Ed = np.array([np.nan if layers[i].donor_depth is None
                            else layers[i].donor_depth for i in idx])
        self.NA = np.array([layers[i].acceptors for i in idx], float)
        self.Ea = np.array([np.nan if layers[i].acceptor_depth is None
                            else layers[i].acceptor_depth for i in idx])
        self.taun = np.array([tr[i].tau_n for i in idx])
        self.taup = np.array([tr[i].tau_p for i in idx])
        self.srh = np.isfinite(self.taun)
        self.Et = np.array([tr[i].trap_level for i in idx])
        self.B = np.array([tr[i].radiative for i in idx]) * 1e21
        self.Cn = np.array([tr[i].auger_n for i in idx]) * 1e42
        self.Cp = np.array([tr[i].auger_p for i in idx]) * 1e42
        vol = np.full(N, h)
        vol[0] = vol[-1] = 0.5 * h
        self.vol = vol
        sheet = np.zeros(N)
        for c in interfaces:
            s = c.sigma_nm2 + extra.get(c.index, 0.0)
            k = int(round(edges[c.index + 1] / h))
            sheet[min(max(k, 0), N - 1)] += s
        self.sheet = sheet
        if generation is None:
            G = np.zeros(N)
        elif callable(generation):
            G = np.array([float(generation(zz)) for zz in z]) * 1e-21
        else:
            G = np.broadcast_to(np.asarray(generation, float), (N,)) * 1e-21
        if not np.all(np.isfinite(G)):
            raise ValueError("generation must be finite (cm^-3 s^-1)")
        self.G = G

    # occupation: F and ln F
    def occ(self, eta):
        eta = np.clip(eta, -700.0, 700.0)
        if not self.fermi:
            return np.exp(eta), eta
        F, _ = fermi_dirac_half(eta)
        F = np.asarray(F, float)
        lnF = np.where(eta < -40.0, eta, np.log(np.maximum(F, 1e-300)))
        return F, lnF

    def state(self, psi, fn, fp):
        kT = self.kT
        Ec = self.Ec0 - psi
        Ev = Ec - self.Eg
        Fn_, lFn = self.occ((fn - Ec) / kT)
        Fp_, lFp = self.occ((Ev - fp) / kT)
        n = self.Nc * Fn_
        p = self.Nv * Fp_
        lnn = np.log(self.Nc) + lFn
        if self.holes:
            lnp = np.log(self.Nv) + lFp
        else:
            lnp = np.full_like(lFp, -745.0)
        ndp = self.ND.copy()
        m = np.isfinite(self.Ed) & (self.ND > 0)
        if m.any():
            x = np.clip((fn[m] - Ec[m] + self.Ed[m]) / kT, -700, 700)
            ndp[m] = self.ND[m] / (1.0 + 2.0 * np.exp(x))
        nam = self.NA.copy()
        m = np.isfinite(self.Ea) & (self.NA > 0)
        if m.any():
            y = np.clip((Ev[m] + self.Ea[m] - fp[m]) / kT, -700, 700)
            nam[m] = self.NA[m] / (1.0 + self.g_acc * np.exp(y))
        return Ec, Ev, n, p, lnn, lnp, ndp, nam

    def recomb(self, Ec, Ev, n, p, lnn, lnp, fn, fp):
        kT = self.kT
        eq = np.exp(np.clip(lnn + lnp - (fn - fp) / kT, -745.0, 700.0))
        num = n * p - eq
        R = (self.B + self.Cn * n + self.Cp * p) * num
        if self.srh.any():
            s = self.srh
            Et = 0.5 * (Ec[s] + Ev[s]) + self.Et[s]
            n1 = self.Nc[s] * np.exp(np.clip((Et - Ec[s]) / kT, -745, 700))
            p1 = self.Nv[s] * np.exp(np.clip((Ev[s] - Et) / kT, -745, 700))
            R[s] += num[s] / (self.taup[s] * (n[s] + n1)
                              + self.taun[s] * (p[s] + p1))
        return R

    def fluxes(self, fn, fp, n, p, lnn, lnp):
        kT, h = self.kT, self.h
        w = fn / kT - lnn
        v = fp / kT + lnp
        dw = w[1:] - w[:-1]
        dv = v[1:] - v[:-1]
        # Scharfetter-Gummel, n_i B(dw) - n_{i+1} B(-dw), rewritten exactly
        # as -n_i B(dw) expm1(dF_n/kT): the flux is then proportional to the
        # quasi-Fermi difference itself, which keeps small currents accurate
        # where the drift and diffusion parts nearly cancel.
        Gn = -self.Dn / h * n[:-1] * bernoulli(dw) * np.expm1(
            np.clip((fn[1:] - fn[:-1]) / kT, -700.0, 700.0))
        Gp = -self.Dp / h * p[:-1] * bernoulli(-dv) * np.expm1(
            np.clip(-(fp[1:] - fp[:-1]) / kT, -700.0, 700.0))
        return Gn, Gp

    def rho(self, n, p, ndp, nam):
        return p - n + ndp - nam

    def residual(self, x, bc):
        psi, fn, fp = x[0::3], x[1::3], x[2::3]
        Ec, Ev, n, p, lnn, lnp, ndp, nam = self.state(psi, fn, fp)
        R = self.recomb(Ec, Ev, n, p, lnn, lnp, fn, fp)
        Gn, Gp = self.fluxes(fn, fp, n, p, lnn, lnp)
        h = self.h
        flux = self.eps_c * (psi[1:] - psi[:-1]) / h
        F = np.empty_like(x)
        rp = np.zeros(self.N)
        rp[1:-1] = flux[1:] - flux[:-1]
        rp += _CP * (self.vol * self.rho(n, p, ndp, nam) + self.sheet)
        rn = np.zeros(self.N)
        rn[1:-1] = Gn[1:] - Gn[:-1]
        rn -= self.vol * (self.G - R)
        rq = np.zeros(self.N)
        rq[1:-1] = Gp[1:] - Gp[:-1]
        rq -= self.vol * (self.G - R)
        for k, (pb, fb) in ((0, bc[0]), (self.N - 1, bc[1])):
            rp[k] = psi[k] - pb
            rn[k] = fn[k] - fb
            rq[k] = fp[k] - fb
        te = getattr(self, "te", (None, None))
        for side, k in ((0, 0), (1, self.N - 1)):
            if te[side] is None:
                continue
            ef = bc[side][1]
            kT = self.kT
            n0 = self.Nc[k] * self.occ(np.array([-te[side] / kT]))[0][0]
            p0 = (self.Nv[k] * self.occ(np.array([(te[side] - self.Eg[k]) / kT]))[0][0]
                  if self.holes else 0.0)
            del ef
            outn = self.vRn[k] * (n[k] - n0)      # flux into the metal
            outp = self.vRp[k] * (p[k] - p0)
            src = self.vol[k] * (self.G[k] - R[k])
            if side == 1:
                rn[k] = outn - Gn[-1] - src
                rq[k] = outp - Gp[-1] - src
            else:
                rn[k] = Gn[0] + outn - src
                rq[k] = Gp[0] + outp - src
        F[0::3], F[1::3], F[2::3] = rp, rn, rq
        return F

    # contacts ------------------------------------------------------------
    def neutral_psi(self, k, ef):
        """psi at node k that makes the charge zero with F_n = F_p = ef.
        The charge depends on psi + ef only, so it is solved once per node
        (at ef = 0, by bisection) and shifted."""
        cache = self.__dict__.setdefault("_neutral", {})
        if k not in cache:
            kT = self.kT

            def g(ps):
                Ec = self.Ec0[k] - ps
                Ev = Ec - self.Eg[k]
                n = self.Nc[k] * self.occ(np.array([-Ec / kT]))[0][0]
                p = self.Nv[k] * self.occ(np.array([Ev / kT]))[0][0]
                nd = self.ND[k]
                if math.isfinite(self.Ed[k]) and nd > 0:
                    nd = nd / (1.0 + 2.0 * math.exp(
                        min(700.0, (-Ec + self.Ed[k]) / kT)))
                na = self.NA[k]
                if math.isfinite(self.Ea[k]) and na > 0:
                    na = na / (1.0 + self.g_acc * math.exp(
                        min(700.0, (Ev + self.Ea[k]) / kT)))
                return p - n + nd - na

            lo = self.Ec0[k] - self.Eg[k] - 2.0
            hi = self.Ec0[k] + 2.0
            if not (g(lo) > 0 > g(hi)):
                raise ValueError(
                    "the layer at an ohmic contact has no charge-neutral level "
                    "(undoped, with holes left out). Dope it, include holes, "
                    "or use Contact('zero-field') at the bottom")
            for _ in range(200):
                m = 0.5 * (lo + hi)
                if g(m) > 0:
                    lo = m
                else:
                    hi = m
            cache[k] = 0.5 * (lo + hi)
        return cache[k] - ef

    def bc(self, bottom, top):
        out = []
        for k, c in ((0, bottom), (self.N - 1, top)):
            ef = -c.voltage
            if c.kind == "ohmic":
                out.append((self.neutral_psi(k, ef), ef))
            elif c.kind == "schottky":
                out.append((self.Ec0[k] - ef - c.barrier, ef))
            else:                       # zero-field: psi is free there
                out.append((None, ef))
        return out

    # equilibrium Poisson ---------------------------------------------------
    def equilibrium(self, bc, tol=1e-11, max_iter=400, psi0=None, ef=0.0):
        """Poisson with F_n = F_p = ef everywhere (equilibrium when the
        contacts are at the same voltage; the quasi-equilibrium of a C-V
        sweep otherwise). bc[0][0] None means zero field at the bottom."""
        N, h, kT = self.N, self.h, self.kT
        zero = np.full(N, float(ef))
        free = bc[0][0] is None
        if psi0 is not None:
            psi = np.array(psi0, dtype=float)
        else:
            p0 = bc[1][0] if free else bc[0][0]
            psi = np.linspace(p0, bc[1][0], N)
        psi[-1] = bc[1][0]
        if not free:
            psi[0] = bc[0][0]
        ec = self.eps_c / h
        for _ in range(max_iter):
            Ec, Ev, n, p, lnn, lnp, ndp, nam = self.state(psi, zero, zero)
            if self.fermi:
                _, dFn = fermi_dirac_half(np.clip((ef - Ec) / kT, -700, 700))
                _, dFp = fermi_dirac_half(np.clip((Ev - ef) / kT, -700, 700))
                dn = self.Nc * np.asarray(dFn) / kT
                dp = -self.Nv * np.asarray(dFp) / kT
            else:
                dn = n / kT
                dp = -p / kT
            dnd = np.zeros(N)
            m = np.isfinite(self.Ed) & (self.ND > 0)
            if m.any():
                x = np.exp(np.clip((ef - Ec[m] + self.Ed[m]) / kT, -700, 700))
                dnd[m] = -self.ND[m] * 2 * x / (1 + 2 * x) ** 2 / kT
            dna = np.zeros(N)
            m = np.isfinite(self.Ea) & (self.NA > 0)
            if m.any():
                y = np.exp(np.clip((Ev[m] + self.Ea[m] - ef) / kT, -700, 700))
                dna[m] = self.NA[m] * self.g_acc * y / (1 + self.g_acc * y) ** 2 / kT
            drho = dp - dn + dnd - dna
            r = np.zeros(N)
            flux = ec * (psi[1:] - psi[:-1])
            r[1:-1] = flux[1:] - flux[:-1]
            r += _CP * (self.vol * self.rho(n, p, ndp, nam) + self.sheet)
            r[-1] = psi[-1] - bc[1][0]
            diag = np.zeros(N)
            diag[1:-1] = -(ec[1:] + ec[:-1])
            diag += _CP * self.vol * drho
            lo = ec.copy()
            up = ec.copy()
            diag[-1] = 1.0
            lo[-1] = 0.0
            if free:
                r[0] += ec[0] * (psi[1] - psi[0])
                diag[0] += -ec[0]
            else:
                r[0] = psi[0] - bc[0][0]
                diag[0] = 1.0
                up[0] = 0.0
            step = _thomas(lo, diag, up, -r)
            step = np.clip(step, -0.2, 0.2)
            psi = psi + step
            if np.max(np.abs(step)) < tol:
                break
        else:
            raise RuntimeError("the equilibrium Poisson solve did not converge")
        x = np.zeros(3 * N)
        x[0::3] = psi
        x[1::3] = ef
        x[2::3] = ef
        return x

    # coupled Newton ----------------------------------------------------------
    def jacobian(self, x, bc, F0, use_scipy):
        n3 = x.size
        ab = np.zeros((11, n3))
        hstep = _JAC_STEP
        nodes = np.arange(self.N)
        for col in range(3):
            for var in range(3):
                pert = nodes[nodes % 3 == col]
                xx = x.copy()
                cols = 3 * pert + var
                xx[cols] += hstep
                dF = (self.residual(xx, bc) - F0) / hstep
                rows = 3 * (pert[:, None] - 1) + np.arange(9)[None, :]
                cc = np.broadcast_to(cols[:, None], rows.shape)
                ok = (rows >= 0) & (rows < n3)
                ab[5 + rows[ok] - cc[ok], cc[ok]] = dF[rows[ok]]
        return ab

    def newton(self, x, bc, tol, max_iter, use_scipy, max_step=0.2):
        last = math.inf
        for it in range(1, max_iter + 1):
            F = self.residual(x, bc)
            ab = self.jacobian(x, bc, F, use_scipy)
            n3 = x.size
            s = np.zeros(n3)
            jj = np.arange(n3)
            for d in range(11):
                i = jj + d - 5
                ok = (i >= 0) & (i < n3)
                s[i[ok]] = np.maximum(s[i[ok]], np.abs(ab[d, jj[ok]]))
            s[s == 0.0] = 1.0
            for d in range(11):
                i = jj + d - 5
                ok = (i >= 0) & (i < n3)
                ab[d, jj[ok]] /= s[i[ok]]
            dx = _solve_banded(5, 5, ab, -F / s, use_scipy)
            if not np.all(np.isfinite(dx)):
                return x, False, it, math.inf
            # Size of the update where it matters: a quasi-Fermi level is
            # weighted by n / (n + floor), so where a carrier is far below
            # the floor (and cannot change any charge or current that
            # matters) its quasi-Fermi level does not hold up convergence.
            _, _, n, p, _, _, _, _ = self.state(x[0::3], x[1::3], x[2::3])
            wn = n / (n + self.floor)
            wp = p / (p + self.floor)
            big = float(max(np.max(np.abs(dx[0::3])),
                            np.max(np.abs(dx[1::3]) * wn),
                            np.max(np.abs(dx[2::3]) * wp)))
            if big > max_step:
                dx *= max_step / big
            dx = np.clip(dx, -max_step, max_step)
            x = x + dx
            last = big
            if big < tol:
                return x, True, it, last
        return x, False, max_iter, last


def _result(P, x, bc, interfaces, top, bottom, conv, it, last, stats):
    psi, fn, fp = x[0::3], x[1::3], x[2::3]
    Ec, Ev, n, p, lnn, lnp, ndp, nam = P.state(psi, fn, fp)
    R = P.recomb(Ec, Ev, n, p, lnn, lnp, fn, fp)
    Gn, Gp = P.fluxes(fn, fp, n, p, lnn, lnp)
    jn = -Gn * _A_CM2
    jp = Gp * _A_CM2
    j = jn + jp
    # terminal current: at an ohmic contact (both carriers are at their
    # equilibrium densities there), else the median over the cells
    if bottom.kind == "ohmic":
        jm = float(j[0])
    elif top.kind == "ohmic":
        jm = float(j[-1])
    else:
        jm = float(np.median(j))
    ok = np.maximum(np.minimum(n[1:], n[:-1]),
                    np.minimum(p[1:], p[:-1])) > P.floor
    rho = P.rho(n, p, ndp, nam)
    D_top = (-P.eps_c[-1] * (psi[-1] - psi[-2]) / (P.h * _CP)
             + 0.5 * P.h * rho[-1] + P.sheet[-1])
    return DriftDiffusionResult(
        z=P.z, psi=psi, ec=Ec, ev=Ev, efn=fn, efp=fp, n=n * 1e21, p=p * 1e21,
        donors_ionized=ndp * 1e21, acceptors_ionized=nam * 1e21,
        recombination=R * 1e21, generation=P.G * 1e21,
        z_mid=0.5 * (P.z[1:] + P.z[:-1]), jn=jn, jp=jp, j=j,
        current_density=jm,
        current_spread=float(np.max(np.abs(j[ok] - jm))) if ok.any() else 0.0,
        gate_charge=float(-D_top * QE * 1e14), interfaces=list(interfaces),
        converged=conv, newton_iterations=it, last_step=last, top=top,
        bottom=bottom, temperature_K=P.T, statistics=stats,
        _state=(x.copy(), bottom.voltage, top.voltage))


def _prepare(layers, top, bottom, temperature_K, line_charge, reference,
             extra_sheet, interfaces, dz, statistics, acceptor_degeneracy):
    for L in layers:
        if not isinstance(L, StackLayer):
            raise ValueError("layers must be StackLayer objects, bottom first")
    if not (isinstance(top, Contact) and isinstance(bottom, Contact)):
        raise ValueError("top and bottom must be Contact objects")
    if top.kind == "zero-field":
        raise ValueError("the top contact cannot be 'zero-field'")
    if statistics not in ("fermi", "boltzmann"):
        raise ValueError("statistics must be 'fermi' or 'boltzmann'")
    if not (math.isfinite(temperature_K) and temperature_K > 0):
        raise ValueError("temperature_K must be finite and positive")
    if not (math.isfinite(dz) and dz > 0):
        raise ValueError("dz must be finite and positive (nm)")
    if not (math.isfinite(acceptor_degeneracy) and acceptor_degeneracy > 0):
        raise ValueError("acceptor_degeneracy must be > 0")
    if interfaces is None:
        interfaces = interface_charges(layers, line_charge=line_charge,
                                       reference=reference)
    if len(interfaces) != len(layers) - 1 or not all(
            isinstance(c, InterfaceCharge) for c in interfaces):
        raise ValueError("interfaces must be the interface_charges list of "
                         "these layers")
    extra = dict(extra_sheet or {})
    for k, v in extra.items():
        if not (isinstance(k, int) and 0 <= k < len(layers) - 1):
            raise ValueError(f"extra_sheet key {k!r} is not an interface index")
        if not math.isfinite(v):
            raise ValueError("extra_sheet values must be finite (e/nm^2)")
    return interfaces, extra


def _check_guess(guess, P):
    if not (isinstance(guess, DriftDiffusionResult) and guess._state is not None
            and guess._state[0].size == 3 * P.N):
        raise ValueError("guess must be a result of the same stack and grid")
    x, vb0, vt0 = guess._state
    return x.copy(), vb0, vt0


def solve_drift_diffusion(layers, transport, top: Contact, bottom: Contact = None,
                          temperature_K: float = 300.0, line_charge=None,
                          reference: str = "H", extra_sheet: dict = None,
                          interfaces=None, dz: float = 0.5,
                          statistics: str = "fermi", generation=None,
                          acceptor_degeneracy: float = 4.0,
                          max_bias_step: float = 0.05, tol: float = 1e-12,
                          max_newton: int = 80, guess=None,
                          use_scipy: bool = True,
                          density_floor_cm3: float = 1.0) -> DriftDiffusionResult:
    """Steady-state drift-diffusion solution of a c-plane nitride stack.

    layers : `StackLayer` list, bottom first (donors, acceptors and their
        depths are read from them; the materials need eg, ec, me, eps_r).
    transport : one `Transport` for all layers, or one per layer.
    top, bottom : `Contact`s (bottom defaults to an ohmic contact at 0 V).
    line_charge, reference, extra_sheet, interfaces : as in
        `solve_electrons` (the polarization charges of `interface_charges`).
    dz : grid step (nm). statistics : "fermi" or "boltzmann".
    generation : None, a number, an array on the grid, or a callable of z
        (nm), in cm^-3 s^-1.
    max_bias_step : largest voltage step (V) of the bias ramp from
        equilibrium (or from `guess`, an earlier result of the same stack).
    tol : Newton stops when the largest update is below tol (V or eV).
    use_scipy : use SciPy's banded solver when SciPy is installed.
    density_floor_cm3 : carrier density (cm^-3) below which a quasi-Fermi
        level no longer holds up convergence (it can change no charge or
        current worth resolving: 1 cm^-3 carries at most about 1e-13 A/cm^2
        by diffusion); also the floor for the current check.

    A vertical current needs a conducting path between the contacts. A
    region with almost no free carriers between them (an undoped,
    insulating buffer under an electron gas, say) leaves the quasi-Fermi
    level of what lies beyond it undefined, and Newton then stalls; for the
    capacitance of such a stack use `solve_quasi_equilibrium` (the default
    of `cv_curve`).
    """
    bottom = bottom or Contact("ohmic")
    interfaces, extra = _prepare(layers, top, bottom, temperature_K,
                                 line_charge, reference, extra_sheet,
                                 interfaces, dz, statistics,
                                 acceptor_degeneracy)
    if bottom.kind == "zero-field":
        raise ValueError("'zero-field' is for solve_quasi_equilibrium; a "
                         "current needs a real contact at the bottom")
    if isinstance(transport, Transport):
        transport = [transport] * len(layers)
    transport = list(transport)
    if len(transport) != len(layers) or not all(
            isinstance(t, Transport) for t in transport):
        raise ValueError("give one Transport, or one per layer")
    if not (math.isfinite(max_bias_step) and max_bias_step > 0):
        raise ValueError("max_bias_step must be > 0 (V)")
    if not (math.isfinite(density_floor_cm3) and density_floor_cm3 > 0):
        raise ValueError("density_floor_cm3 must be > 0")
    P = _Problem(layers, transport, float(temperature_K), dz, statistics,
                 interfaces, extra, generation, float(acceptor_degeneracy))
    P.floor = float(density_floor_cm3) * 1e-21
    if bottom.thermionic or top.thermionic:
        kTj = KB * float(temperature_K) * QE
        idx_m = P.Nc / ((KB * float(temperature_K) * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27)
        me_node = idx_m ** (2.0 / 3.0) / M0             # mass back from N_c
        mh_node = (P.Nv / ((KB * float(temperature_K) * QE / (2 * np.pi * HBAR ** 2)) ** 1.5
                           * 2.0 * 1e-27)) ** (2.0 / 3.0) / M0
        P.vRn = np.sqrt(kTj / (2 * np.pi * me_node * M0)) * 1e9          # nm/s
        P.vRp = np.sqrt(kTj / (2 * np.pi * np.maximum(mh_node, 1e-30) * M0)) * 1e9
        P.te = (bottom.barrier if bottom.thermionic else None,
                top.barrier if top.thermionic else None)
    if guess is not None:
        x, vb0, vt0 = _check_guess(guess, P)
    else:
        c0 = (Contact("ohmic") if bottom.kind == "ohmic"
              else Contact("schottky", bottom.barrier))
        c1 = (Contact("ohmic") if top.kind == "ohmic"
              else Contact("schottky", top.barrier))
        x = P.equilibrium(P.bc(c0, c1))
        vb0 = vt0 = 0.0
    vb1, vt1 = bottom.voltage, top.voltage
    span = max(abs(vb1 - vb0), abs(vt1 - vt0))
    nsteps = max(1, int(math.ceil(span / max_bias_step - 1e-12)))
    t = 0.0
    dt = 1.0 / nsteps
    conv, it, last = False, 0, math.inf
    tries = 0
    while t < 1.0 - 1e-15:
        tn = min(1.0, t + dt)
        cb = dataclasses.replace(bottom, voltage=vb0 + tn * (vb1 - vb0))
        ct = dataclasses.replace(top, voltage=vt0 + tn * (vt1 - vt0))
        bc = P.bc(cb, ct)
        xn, conv, it, last = P.newton(x, bc, tol, max_newton, use_scipy)
        if conv:
            x, t = xn, tn
            tries = 0
            dt = min(dt * 1.5, 1.0 / nsteps)
        else:
            dt *= 0.5
            tries += 1
            if tries > 12:
                raise RuntimeError(
                    "drift-diffusion did not converge on the bias ramp. If a "
                    "region between the contacts has almost no free carriers "
                    "(an undoped buffer, say), a vertical current is not well "
                    "defined there: use solve_quasi_equilibrium for C-V. "
                    "Otherwise try a smaller dz or max_bias_step.")
    return _result(P, x, P.bc(bottom, top), interfaces, top, bottom, conv,
                   it, last, statistics)


def solve_quasi_equilibrium(layers, top: Contact, bottom: Contact = None,
                            temperature_K: float = 300.0, line_charge=None,
                            reference: str = "H", extra_sheet: dict = None,
                            interfaces=None, dz: float = 0.5,
                            statistics: str = "fermi", hole_mass=None,
                            acceptor_degeneracy: float = 4.0, guess=None,
                            tol: float = 1e-11) -> DriftDiffusionResult:
    """The semiconductor in equilibrium with its bottom contact while the
    top contact is biased: the standard model of a quasi-static C-V sweep.

    The electrons (and holes, if included) keep one Fermi level, that of
    the bottom contact (in a transistor, the source and drain hold the
    electron gas); only the top metal moves, by its voltage relative to the
    bottom one. No current flows. This is Poisson's equation alone, so it is
    robust for any stack, insulating buffers included.

    bottom : "ohmic" (default; the bottom layer must have a neutral level) or
        "zero-field" (a thick buffer with no field at its bottom, as in
        `solve_electrons`).
    hole_mass : None (default) leaves holes out, or a hole density-of-states
        mass (m0) for every layer (one number or one per layer). In a
        wide-gap nitride a C-V sweep is too fast for holes to be generated,
        so a depleted surface stays depleted (deep depletion); leaving holes
        out is the model for that. Layers with acceptors need holes.
    Other arguments as in `solve_drift_diffusion`.
    """
    bottom = bottom or Contact("ohmic")
    interfaces, extra = _prepare(layers, top, bottom, temperature_K,
                                 line_charge, reference, extra_sheet,
                                 interfaces, dz, statistics,
                                 acceptor_degeneracy)
    hm = None
    if hole_mass is not None:
        hm = ([float(hole_mass)] * len(layers)
              if isinstance(hole_mass, (int, float)) else
              [float(v) for v in hole_mass])
        if len(hm) != len(layers) or not all(math.isfinite(v) and v > 0
                                             for v in hm):
            raise ValueError("hole_mass must be > 0, one number or one per layer")
    elif any(L.acceptors > 0 for L in layers):
        raise ValueError("a layer has acceptors, so holes are needed: give "
                         "hole_mass (cited)")
    P = _Problem(layers, None, float(temperature_K), dz, statistics,
                 interfaces, extra, None, float(acceptor_degeneracy),
                 hole_mass=hm)
    P.floor = 1e-21
    # only the voltage difference matters: the semiconductor sits at the
    # bottom contact's Fermi level
    ef = -bottom.voltage
    ct = dataclasses.replace(top, voltage=top.voltage)
    bc = P.bc(bottom, ct)
    psi0 = None
    if guess is not None:
        psi0 = _check_guess(guess, P)[0][0::3]
    x = P.equilibrium(bc, tol=tol, psi0=psi0, ef=ef)
    return _result(P, x, bc, interfaces, top, bottom, True, 0, 0.0, statistics)


def iv_curve(layers, transport, voltages, top: Contact, bottom: Contact = None,
             sweep: str = "top", **kwargs):
    """Current density (A/cm^2) at each voltage, warm-started point to point.

    voltages : applied to the top contact (sweep="top") or the bottom one.
    kwargs : passed to `solve_drift_diffusion`.
    Returns (voltages array, current array, list of results).
    """
    bottom = bottom or Contact("ohmic")
    if sweep not in ("top", "bottom"):
        raise ValueError("sweep must be 'top' or 'bottom'")
    res, prev = [], None
    for v in voltages:
        t = dataclasses.replace(top, voltage=float(v)) if sweep == "top" else top
        b = dataclasses.replace(bottom, voltage=float(v)) if sweep == "bottom" else bottom
        r = solve_drift_diffusion(layers, transport, t, b, guess=prev, **kwargs)
        res.append(r)
        prev = r
    V = np.asarray(voltages, float)
    return V, np.array([r.current_density for r in res]), res


def cv_curve(layers, voltages, top: Contact, bottom: Contact = None,
             dv: float = 0.005, transport=None, method: str = "quasi-equilibrium",
             **kwargs):
    """Quasi-static capacitance of the top contact, C = dQ/dV (F/cm^2), by a
    central difference of +-dv/2 around each voltage of the top contact.

    method : "quasi-equilibrium" (default; `solve_quasi_equilibrium`, which
        needs no transport numbers) or "drift-diffusion" (the full steady
        state with leakage currents; needs `transport` and a conducting
        path between the contacts).
    kwargs : passed to the solver.
    Returns (voltages, capacitance, gate charge (C/cm^2), results).
    """
    bottom = bottom or Contact("ohmic")
    if not (math.isfinite(dv) and dv > 0):
        raise ValueError("dv must be > 0 (V)")
    if method == "quasi-equilibrium":
        def solve(t, guess):
            return solve_quasi_equilibrium(layers, t, bottom, guess=guess, **kwargs)
    elif method == "drift-diffusion":
        if transport is None:
            raise ValueError("method='drift-diffusion' needs transport numbers")

        def solve(t, guess):
            return solve_drift_diffusion(layers, transport, t, bottom,
                                         guess=guess, **kwargs)
    else:
        raise ValueError("method must be 'quasi-equilibrium' or 'drift-diffusion'")
    C, Q, res, prev = [], [], [], None
    for v in voltages:
        r = solve(dataclasses.replace(top, voltage=float(v)), prev)
        ra = solve(dataclasses.replace(top, voltage=v + dv / 2), r)
        rb = solve(dataclasses.replace(top, voltage=v - dv / 2), r)
        C.append((ra.gate_charge - rb.gate_charge) / dv)
        Q.append(r.gate_charge)
        res.append(r)
        prev = r
    return np.asarray(voltages, float), np.array(C), np.array(Q), res


def cv_curve_quantum(layers, voltages, barrier: float,
                     temperature_K: float = 300.0, line_charge=None,
                     reference: str = "H", interfaces=None,
                     extra_sheet: dict = None, dz: float = 0.2,
                     quantum_window=None, dv: float = 0.01, **solver_kwargs):
    """Quasi-static C-V with the electrons treated by quantum mechanics.

    Each gate voltage V is one `solve_electrons` run with the surface step
    barrier - V (the metal's Fermi level is V below that of the electrons)
    and zero field at the bottom (a thick base). The charge on the metal is
    minus all the charge in the stack, Q = -q [sum of the fixed sheets +
    int (N_d+ - n) dz], and C = dQ/dV by a central difference of +-dv/2.
    Quantum electrons sit at a different depth and fill their states
    differently from classical ones, so C differs from that of `cv_curve`
    by a few percent (for a 25 nm Al0.25Ga0.75N barrier on GaN, between
    -0.6 and +1.7 percent from -5 to -1 V); that difference is the quantum
    correction. It is slower: three electron solves per voltage.

    quantum_window : as in `solve_electrons` (default: the top 120 nm, or
        the whole stack if thinner). Electrons below it are classical.
    Returns (voltages, capacitance (F/cm^2), gate charge (C/cm^2), results).
    """
    from .electrons import solve_electrons

    if not (math.isfinite(barrier)):
        raise ValueError("barrier must be finite (eV)")
    if not (math.isfinite(dv) and dv > 0):
        raise ValueError("dv must be > 0 (V)")
    if interfaces is None:
        interfaces = interface_charges(layers, line_charge=line_charge,
                                       reference=reference)
    tot = sum(L.thickness for L in layers)
    win = quantum_window or (max(0.0, tot - min(tot, 120.0)), tot)
    extra = dict(extra_sheet or {})
    fixed = sum(c.sigma_nm2 for c in interfaces) + sum(extra.values())
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    def charge(v):
        r = solve_electrons(layers, phi_B=barrier - v, temperature_K=temperature_K,
                            interfaces=interfaces, extra_sheet=extra, dz=dz,
                            quantum_window=win, classical_outside=True,
                            **solver_kwargs)
        if not r.converged:
            raise RuntimeError(f"the electron solver did not converge at {v} V")
        net = fixed + float(trap(r.donors_ionized - r.n, r.z))      # e/nm^2
        return -net * QE * 1e14, r                                    # C/cm^2

    C, Q, res = [], [], []
    for v in voltages:
        q0, r = charge(float(v))
        qa, _ = charge(float(v) + dv / 2)
        qb, _ = charge(float(v) - dv / 2)
        C.append((qa - qb) / dv)
        Q.append(q0)
        res.append(r)
    return np.asarray(voltages, float), np.array(C), np.array(Q), res
