"""Electron gases at c-plane nitride interfaces: a one-dimensional
Schroedinger-Poisson solver driven by the interface charges of
`kpenvelope.interfaces` (new in v0.17.0).

The holes of a GaN/AlN hole gas need the six-band model of the rest of the
package. Electrons near the conduction-band minimum of GaN and AlGaN do not:
a single band with the mass along c is the standard description, and it is
what this module solves.

Geometry. z runs along [0001] from the bottom of the stack (z = 0, inside
the bottom layer) to the free surface (z = sum of the thicknesses). Layers
are `StackLayer` objects, bottom first; the fixed charge of every interface
comes from `interface_charges`, so relaxed layers and their misfit-line
charge enter exactly as there. Extra fixed sheets (for example a
calibration charge) can be added at any interface.

Boundary conditions:

* top surface: the conduction-band edge sits phi_B above the Fermi level
  (surface Fermi-level pinning; the surface-state charge adjusts freely);
* bottom: zero electric field (a thick, neutral buffer).

The Fermi level is E_F = 0. Inside the quantum window the electrons fill
the eigenstates of -hbar^2/2 d/dz (1/m) d/dz + E_c (BenDaniel-Duke form,
finite differences) with two-dimensional Fermi-Dirac occupation and the
state-averaged in-plane mass. Outside it (only with classical_outside=True)
they form a classical three-dimensional Fermi gas,
n = N_c F_1/2((E_F - E_c)/kT), N_c = 2 (m kT / 2 pi hbar^2)^(3/2). Donors are
fully ionized unless the layer gives a donor depth E_d; then
N_d+ = N_d / (1 + 2 exp((E_F - E_c + E_d)/kT)).

Self-consistency follows the predictor-corrector scheme of A. Trellakis,
A. T. Galick, A. Pacelli and U. Ravaioli, J. Appl. Phys. 81, 7880 (1997): with
the eigenstates held fixed, the Poisson equation is solved by Newton
iteration with each subband shifted by the local change of the potential;
the eigenstates are then recomputed, until the potential changes by less
than `tol` (V).

This is a NumPy-only re-implementation of src/nmp/sp1d.py of
T. M. Mahim, nitride-misfit-polarization v1.3.1 (doi:10.5281/zenodo.24315036);
the tests check it against that code's archived numbers for the
Al0.22Ga0.78N/GaN series of Shen et al., Appl. Phys. Lett. 76, 2746 (2000).

Also here: the two-carrier Hall relations, to separate a parallel channel
from the electron sheet, and the scattering of the sheet by charged misfit
lines, which bounds the line charge from a measured mobility.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .interfaces import InterfaceCharge, StackLayer, interface_charges
from .nitride import E_PER_NM2_PER_C_M2, QE

__all__ = ["solve_electrons", "ElectronResult", "fermi_dirac_half",
           "two_carrier_hall", "split_two_carrier",
           "line_scattering_rate", "line_charge_mobility_bound",
           "line_charge_from_mobility_drop"]

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
EPS0 = 8.8541878128e-12
KB = 8.617333262e-5          # eV/K


# ----------------------------------------------------------------- F_1/2
_TAB = None


_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _fd_tables():
    """F_1/2 and F_-1/2 (normalized, 2/sqrt(pi) prefactor) on -40..40, step
    0.02, by the trapezoid rule in t = sqrt(x) (smooth, decaying integrand:
    the rule converges faster than any power of the step). Returns the
    grid, log F_1/2 and its derivative F_-1/2 / F_1/2."""
    eta = np.linspace(-40.0, 40.0, 4001)
    F = np.empty_like(eta)
    Fm = np.empty_like(eta)
    c = 2.0 / math.sqrt(math.pi)
    for j, e in enumerate(eta):
        t = np.linspace(0.0, math.sqrt(max(e, 0.0) + 70.0), 6001)
        ex = np.exp(np.minimum(t * t - e, 700.0))
        occ = 1.0 / (1.0 + ex)
        F[j] = c * _trapezoid(2.0 * t * t * occ, t)
        Fm[j] = c * _trapezoid(2.0 * t * t * ex * occ * occ, t)
    return eta, np.log(F), Fm / F


def fermi_dirac_half(eta):
    """F_1/2(eta) = (2/sqrt(pi)) int_0^inf sqrt(x)/(1 + exp(x - eta)) dx and
    its derivative F_-1/2(eta), for scalars or arrays.

    Tabulated on -40 <= eta <= 40 and interpolated with cubic Hermite
    polynomials in log F (relative error below 1e-8); exp(eta) below the
    table and the degenerate limit (4/(3 sqrt(pi))) eta^1.5 above it.
    """
    global _TAB
    if _TAB is None:
        _TAB = _fd_tables()
    E, LF, DLF = _TAB
    eta = np.asarray(eta, dtype=float)
    e = np.clip(eta, E[0], E[-1])
    h = E[1] - E[0]
    i = np.clip(((e - E[0]) / h).astype(int), 0, len(E) - 2)
    s = (e - E[i]) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    lf = h00 * LF[i] + h10 * h * DLF[i] + h01 * LF[i + 1] + h11 * h * DLF[i + 1]
    d00 = (6 * s**2 - 6 * s) / h
    d10 = 3 * s**2 - 4 * s + 1
    d01 = (-6 * s**2 + 6 * s) / h
    d11 = 3 * s**2 - 2 * s
    dlf = d00 * LF[i] + d10 * DLF[i] + d01 * LF[i + 1] + d11 * DLF[i + 1]
    F = np.exp(lf)
    dF = F * dlf
    lo = eta < E[0]
    hi = eta > E[-1]
    ex = np.exp(np.minimum(eta, E[0]))
    F = np.where(lo, ex, F)
    dF = np.where(lo, ex, dF)
    F = np.where(hi, 4.0 / (3.0 * math.sqrt(math.pi)) * np.abs(eta) ** 1.5, F)
    dF = np.where(hi, 2.0 / math.sqrt(math.pi) * np.abs(eta) ** 0.5, dF)
    if F.ndim == 0:
        return float(F), float(dF)
    return F, dF


def _thomas(a, b, c, d):
    """Solve a tridiagonal system: sub-diagonal a (n-1), diagonal b (n),
    super-diagonal c (n-1), right side d (n)."""
    n = len(b)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    bb = b[0]
    cp[0] = c[0] / bb
    dp[0] = d[0] / bb
    for k in range(1, n):
        m = b[k] - a[k - 1] * cp[k - 1]
        if k < n - 1:
            cp[k] = c[k] / m
        dp[k] = (d[k] - a[k - 1] * dp[k - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


@dataclasses.dataclass
class ElectronResult:
    """Self-consistent electron solution (lengths nm, energies eV, E_F = 0).

    z, ec : grid and conduction-band edge.
    n, n_quantum, n_classical, donors_ionized : volume densities (nm^-3).
    sheet_quantum, sheet_classical : sheet densities (nm^-2) of the quantum
        electrons (the 2DEG, inside the window) and of the classical ones
        (for example the undepleted part of a doped barrier).
    subband_energies : eigenvalues inside the window (eV, relative to E_F).
    interfaces : the `InterfaceCharge` list used; extra_sheet : the extra
        fixed sheets added (e per nm^2, by interface index).
    converged, iterations, last_change : of the outer loop (V).
    """

    z: np.ndarray
    ec: np.ndarray
    n: np.ndarray
    n_quantum: np.ndarray
    n_classical: np.ndarray
    donors_ionized: np.ndarray
    sheet_quantum: float
    sheet_classical: float
    subband_energies: np.ndarray
    interfaces: list
    extra_sheet: dict
    converged: bool
    iterations: int
    last_change: float

    @property
    def sheet_quantum_cm2(self):
        return self.sheet_quantum * 1e14

    @property
    def sheet_classical_cm2(self):
        return self.sheet_classical * 1e14


def _need(v, what, label):
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"layer {label!r} has no {what} (NaN); the electron "
                         "solver needs a cited value for it")
    return v


def solve_electrons(layers, phi_B: float, temperature_K: float = 300.0,
                    line_charge=None, reference: str = "H",
                    extra_sheet: dict = None, interfaces=None,
                    dz: float = 0.1, quantum_window=None, n_states: int = 40,
                    classical_outside: bool = False, tol: float = 1e-6,
                    max_outer: int = 200) -> ElectronResult:
    """Self-consistent electrons in a c-plane nitride stack.

    layers : `StackLayer` list, bottom first (the materials need ec, me and
        eps_r: `algan` provides them).
    phi_B : surface barrier (eV), E_c at the surface minus E_F.
    line_charge, reference : passed to `interface_charges` (a relaxed layer
        needs line_charge). Or pass precomputed `interfaces`.
    extra_sheet : {interface index: extra fixed charge in e per nm^2}, added
        to the polarization charge there (for example a calibration charge).
    dz : grid step (nm). quantum_window : (z_min, z_max) in nm of the
        quantum region (default: the whole stack). n_states : number of
        eigenstates kept. classical_outside : classical electrons outside
        the window.
    """
    for L in layers:
        if not isinstance(L, StackLayer):
            raise ValueError("layers must be StackLayer objects, bottom first")
        if L.acceptors > 0.0:
            raise ValueError(f"layer {L.label!r} has acceptors; this solver "
                             "has no holes. Use solve_drift_diffusion, which "
                             "has both carriers.")
    if not (math.isfinite(phi_B)):
        raise ValueError("phi_B must be finite (eV)")
    if not (math.isfinite(temperature_K) and temperature_K > 0.0):
        raise ValueError("temperature_K must be finite and positive")
    if not (math.isfinite(dz) and dz > 0.0):
        raise ValueError("dz must be finite and positive")
    if not (isinstance(n_states, int) and n_states >= 1):
        raise ValueError("n_states must be an integer >= 1")
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
    T = float(temperature_K)
    eps_r = [_need(L.material.eps_r, "eps_r", L.label) for L in layers]
    ec_l = [_need(L.material.ec, "conduction-band edge ec", L.label) for L in layers]
    me_l = [_need(L.material.me, "electron mass me", L.label) for L in layers]

    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    Ltot = edges[-1]
    n = int(round(Ltot / dz)) + 1
    z = np.linspace(0.0, Ltot, n)
    dzn = z[1] - z[0]
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(layers) - 1)
    eps = np.array([eps_r[i] for i in idx]) * EPS0
    Ec0 = np.array([ec_l[i] for i in idx])
    me = np.array([me_l[i] for i in idx])
    Nd = np.array([layers[i].donors for i in idx]) * 1e27            # m^-3
    Ed = np.array([np.nan if layers[i].donor_depth is None else layers[i].donor_depth
                   for i in idx])
    incomplete = ~np.isnan(Ed)
    Nc = 2.0 * (me * M0 * KB * T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5
    sheet = np.zeros(n)
    for c in interfaces:
        s = c.sigma + extra.get(c.index, 0.0) / E_PER_NM2_PER_C_M2
        sheet[int(round(edges[c.index + 1] / dzn))] += s
    qw = quantum_window or (0.0, Ltot)
    if not (len(qw) == 2 and qw[0] < qw[1]):
        raise ValueError("quantum_window must be (z_min, z_max) with z_min < z_max")
    inq = (z >= qw[0] - 1e-9) & (z <= qw[1] + 1e-9)
    kT = KB * T
    dzm = dzn * 1e-9

    # Poisson operator -d/dz(eps d/dz): zero field at the bottom (half cell),
    # Dirichlet at the top.
    em = 0.5 * (eps[1:] + eps[:-1]) / dzm ** 2
    diag = np.zeros(n)
    diag[:-1] += em
    diag[1:] += em
    off = -em.copy()
    A_lo = off.copy()          # sub-diagonal (row k+1, col k)
    A_up = off.copy()          # super-diagonal (row k, col k+1)
    diag[-1] = 1.0
    A_lo[-1] = 0.0
    half = np.ones(n)
    half[0] = 0.5

    def matvec(phi):
        y = diag * phi
        y[:-1] += A_up * phi[1:]
        y[1:] += A_lo * phi[:-1]
        return y

    def classical(phi):
        nn = np.zeros(n)
        dn = np.zeros(n)
        if classical_outside:
            out = ~inq
            eta = -(Ec0[out] - phi[out]) / kT
            F, dF = fermi_dirac_half(eta)
            nn[out] = Nc[out] * F
            dn[out] = Nc[out] * dF / kT
        ndp = Nd.copy()
        dndp = np.zeros(n)
        m = incomplete & (Nd > 0)
        if m.any():
            x = (-(Ec0[m] - phi[m]) + Ed[m]) / kT
            ex = np.exp(np.clip(x, -700, 700))
            ndp[m] = Nd[m] / (1 + 2 * ex)
            dndp[m] = -Nd[m] * 2 * ex / (1 + 2 * ex) ** 2 / kT
        return nn, dn, ndp, dndp

    def subbands(U):
        sel = inq
        if sel.sum() < 3:
            return np.zeros(0), np.zeros((sel.sum(), 0)), np.zeros(0)
        Us = U[sel]
        m = me[sel] * M0
        mh = 0.5 * (m[1:] + m[:-1])
        t = HBAR ** 2 / (2 * dzm ** 2) / QE
        dg = Us + t * (np.r_[1 / mh, 1 / mh[-1]] + np.r_[1 / mh[0], 1 / mh])
        od = -t / mh
        H = np.diag(dg) + np.diag(od, 1) + np.diag(od, -1)
        w, v = np.linalg.eigh(H)
        k = min(n_states, len(dg) - 1)
        w, v = w[:k], v[:, :k]
        v = v / np.sqrt(dzm)
        md = (m[:, None] * v ** 2).sum(0) / (v ** 2).sum(0)
        g = md / (np.pi * HBAR ** 2) * QE
        return w, v ** 2, g

    def carriers(states, dphi):
        w, v2, g = states
        d = dphi[inq]
        nn = np.zeros(n)
        dn = np.zeros(n)
        for j in range(len(w)):
            x = (w[j] - d) / kT
            nn[inq] += g[j] * kT * np.logaddexp(0.0, -x) * v2[:, j]
            dn[inq] += g[j] * (0.5 * (1 - np.tanh(0.5 * x))) * v2[:, j]
        return nn, dn

    sheet_rho = sheet / dzm
    phi_top = Ec0[-1] - phi_B

    def fixed(phi):
        _, _, ndp, dndp = classical(phi)
        return sheet_rho + QE * ndp, QE * dndp

    rho0, _ = fixed(np.zeros(n))
    b = rho0 * half
    b[-1] = phi_top
    phi = _thomas(A_lo, diag, A_up, b)
    hist = []
    for _ in range(max_outer):
        se = subbands(Ec0 - phi)
        phi_k = phi.copy()
        for _inner in range(200):
            dphi = phi - phi_k
            nq, dnq = carriers(se, dphi)
            nc, dnc, _, _ = classical(phi)
            rf, drf = fixed(phi)
            rho = (rf - QE * (nq + nc)) * half
            R = matvec(phi) - rho
            R[-1] = phi[-1] - phi_top
            jd = (QE * (dnq + dnc) - drf) * half
            dg = diag.copy()
            dg[:-1] += jd[:-1]
            step = _thomas(A_lo, dg, A_up, -R)
            step = np.clip(step, -0.2, 0.2)
            phi = phi + step
            if np.max(np.abs(step)) < 1e-9:
                break
        d = float(np.max(np.abs(phi - phi_k)))
        hist.append(d)
        if d < tol:
            break
    se = subbands(Ec0 - phi)
    zero = np.zeros(n)
    nq, _ = carriers(se, zero)
    nc, _, ndp, _ = classical(phi)
    return ElectronResult(
        z=z, ec=Ec0 - phi, n=(nq + nc) * 1e-27, n_quantum=nq * 1e-27,
        n_classical=nc * 1e-27, donors_ionized=ndp * 1e-27,
        sheet_quantum=float(nq.sum() * dzm * 1e-18),
        sheet_classical=float(nc.sum() * dzm * 1e-18),
        subband_energies=se[0], interfaces=list(interfaces), extra_sheet=extra,
        converged=bool(hist and hist[-1] < tol), iterations=len(hist),
        last_change=hist[-1] if hist else 0.0)


# ------------------------------------------------------------ Hall relations
def two_carrier_hall(n1, mu1, n2, mu2):
    """Single-field Hall density and mobility of two parallel channels:
    n_H = (n1 mu1 + n2 mu2)^2 / (n1 mu1^2 + n2 mu2^2),
    mu_H = (n1 mu1^2 + n2 mu2^2) / (n1 mu1 + n2 mu2). Any consistent units."""
    s = n1 * mu1 + n2 * mu2
    s2 = n1 * mu1 ** 2 + n2 * mu2 ** 2
    if not s > 0:
        raise ValueError("the two channels carry no current")
    return s * s / s2, s2 / s


def split_two_carrier(n_hall, mu_hall, n2, mu2):
    """The channel (n1, mu1) behind a measured Hall pair, given the parallel
    channel (n2, mu2): the exact inverse of `two_carrier_hall`. Refuses a
    parallel channel that alone would carry more than the measurement."""
    A = n_hall * mu_hall - n2 * mu2
    B = n_hall * mu_hall ** 2 - n2 * mu2 ** 2
    if not (A > 0 and B > 0):
        raise ValueError("the given parallel channel carries more than the "
                         "measured pair allows")
    return A * A / B, B / A


# --------------------------------------------------- scattering by the lines
def _fang_howard_b(ns_m2, m_rel, eps_r, ndepl_m2=0.0):
    m = m_rel * M0
    eps = eps_r * EPS0
    return (33 * m * QE ** 2 * (ndepl_m2 + 11 * ns_m2 / 32)
            / (8 * eps * HBAR ** 2)) ** (1 / 3)


def line_scattering_rate(sheet_nm2, line_charge_e_per_nm, line_density_nm,
                         mass, eps_r, depletion_nm2=0.0, n_theta=2000):
    """Transport scattering rate (1/s) of a 2D electron gas by randomly
    oriented charged lines lying in the interface plane.

    A line of charge lambda per length has the in-plane transform
    lambda e / (2 eps q); for lines of total length L_A per area the
    orientation average gives W(q) = L_A (lambda e/(2 eps q))^2 (2/q) F(q)^2,
    F(q) = (b/(b+q))^3 (Fang-Howard), screened in the Thomas-Fermi
    approximation with the Fang-Howard screening factor
    G(q) = (8b^3 + 9b^2 q + 3 b q^2)/(8 (b+q)^3) (T. Ando, A. B. Fowler and
    F. Stern, Rev. Mod. Phys. 54, 437 (1982)); the kernel of T. M. Mahim,
    A. S. M. Mohsin and M. M. Rahman, Phys. Rev. B 114, L291304 (2026).
    """
    for v, w in ((sheet_nm2, "sheet_nm2"), (line_density_nm, "line_density_nm"),
                 (mass, "mass"), (eps_r, "eps_r")):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{w} must be finite and positive")
    ns = sheet_nm2 * 1e18
    lam = line_charge_e_per_nm * QE * 1e9
    LA = line_density_nm * 1e9
    m = mass * M0
    eps = eps_r * EPS0
    kF = math.sqrt(2 * math.pi * ns)
    b = _fang_howard_b(ns, mass, eps_r, depletion_nm2 * 1e18)
    th = (np.arange(n_theta) + 0.5) * (2 * math.pi / n_theta)
    q = 2 * kF * np.sin(th / 2)
    F = (b / (b + q)) ** 3
    G = (8 * b ** 3 + 9 * b ** 2 * q + 3 * b * q ** 2) / (8 * (b + q) ** 3)
    qtf = m * QE ** 2 / (2 * math.pi * eps * HBAR ** 2)
    epsq = 1 + qtf / q * G
    W = LA * (lam * QE / (2 * eps * q)) ** 2 * (2 / q) * F ** 2
    integrand = W / epsq ** 2 * (1 - np.cos(th))
    return float(m / (2 * math.pi * HBAR ** 3) * integrand.sum()
                 * (2 * math.pi / n_theta))


def line_charge_mobility_bound(sheet_nm2, mobility_cm2Vs, line_density_nm,
                               mass, eps_r):
    """Largest |Lambda| (e/nm) the measured mobility allows if every
    scattering event came from the lines."""
    if not (math.isfinite(mobility_cm2Vs) and mobility_cm2Vs > 0.0):
        raise ValueError("mobility_cm2Vs must be finite and positive")
    total = QE / (mass * M0) / (mobility_cm2Vs * 1e-4)
    unit = line_scattering_rate(sheet_nm2, 1.0, line_density_nm, mass, eps_r)
    return math.sqrt(total / unit)


def line_charge_from_mobility_drop(mu_reference_cm2Vs, mu_relaxed_cm2Vs,
                                   sheet_nm2, line_density_nm, mass, eps_r):
    """|Lambda| (e/nm) whose scattering, added to that of a reference sample
    (Matthiessen's rule), lowers the mobility to the relaxed sample's."""
    if not (mu_reference_cm2Vs > mu_relaxed_cm2Vs > 0.0):
        raise ValueError("need mu_reference > mu_relaxed > 0")
    extra = QE / (mass * M0) * (1 / (mu_relaxed_cm2Vs * 1e-4)
                                - 1 / (mu_reference_cm2Vs * 1e-4))
    unit = line_scattering_rate(sheet_nm2, 1.0, line_density_nm, mass, eps_r)
    return math.sqrt(extra / unit)
