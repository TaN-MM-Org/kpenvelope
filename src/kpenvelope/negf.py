"""Non-equilibrium Green's functions for electrons along a channel (new in
0.26.0).

The building block is a one-dimensional effective-mass chain, the
finite-difference form of

    -(hbar^2 / 2) d/dx (1 / m(x)) d/dx psi + U(x) psi = E psi

on equally spaced sites x_j = x_0 + j a (the mass on each link is the mean
of its two sites), joined at both ends to semi-infinite leads that continue
the end sites' potential and mass. Its retarded Green's function

    G(E) = [E - H - Sigma_L(E) - Sigma_R(E) - Sigma_B(E)]^-1

with the exact lead self-energy Sigma = -t exp(i k a) (t = hbar^2 / (2 m
a^2), E - U = 2 t (1 - cos k a)) gives the transmission T = Gamma_L
Gamma_R |G_N1|^2 and the spectral function of each contact,
A_c = G Gamma_c G^+ (S. Datta, Electronic Transport in Mesoscopic
Systems, Cambridge University Press, 1995). Only the needed parts of G are
formed, by the left- and right-connected recursions of a tridiagonal
matrix (O(N) per energy, vectorized over the energies).

In a 2D device (x, z) that is uniform along its width y, every subband of
the channel is such a chain (the uncoupled mode-space approach of
R. Venugopal, Z. Ren, S. Datta, M. S. Lundstrom and D. Jovanovic,
J. Appl. Phys. 92, 3730 (2002), doi:10.1063/1.1503165), and the free
motion along y is summed in closed form: an electron with energy E along x
is filled from contact c with

    N_c(E) = sqrt(m kT / (2 pi hbar^2)) F_-1/2((F_c - E) / kT)   per unit width,

so that per unit width (spin included)

    I = (2 q / h) int T(E) [N_S(E) - N_D(E)] dE,
    n_j = (2 / 2 pi a) int sum_c A_c,jj(E) N_c(E) dE    (per unit area).

Over a flat potential with T = 1 above the top this is exactly the
top-of-the-barrier current of `Device.ballistic_current`.

Scattering is added, if wanted, with elastic Buttiker probes (M. Buttiker,
Phys. Rev. B 33, 3020 (1986)): every site is joined to a reservoir with
the energy-independent self-energy -i eta / 2 (eta = hbar / tau, tau a
momentum and phase relaxation time), whose filling at every energy is set
so that no current flows into it; the energy of an electron is kept. A
chain with probes is a resistor: a long uniform one carries the Drude
conductance q^2 n tau / m (checked in the tests).
"""

from __future__ import annotations

import math

import numpy as np

from .electrons import HBAR, M0, fermi_dirac_half
from .nitride import QE

_H2M = HBAR ** 2 / (2.0 * M0 * QE) * 1e18            # hbar^2 / 2 m0, eV nm^2
_G0 = 2.0 * QE * QE / (2.0 * math.pi * HBAR)        # 2 q^2 / h, A/V


def lead_self_energy(E, U, t):
    """Self-energy (eV, complex) and broadening Gamma = -2 Im Sigma of a
    semi-infinite chain with site energy U + 2 t and coupling t, at the
    energies E (array). Inside the band U <= E <= U + 4 t it carries waves
    out; outside it is real (evanescent)."""
    E = np.asarray(E, float)
    x = 1.0 - (E - U) / (2.0 * t)
    inband = np.abs(x) <= 1.0
    root = np.sqrt(np.abs(1.0 - x * x))
    s = np.where(inband, x + 1j * root, x - np.sign(x) * root)
    sig = -t * s
    return sig, -2.0 * sig.imag


def link_couplings(mass, a):
    """t on every link (eV) for site masses (m0) and spacing a (nm)."""
    m = np.asarray(mass, float)
    ml = 0.5 * (m[1:] + m[:-1])
    return _H2M / (ml * a * a)


class Chain:
    """A 1D effective-mass chain between two leads (see the module text).

    U : site potentials (eV), mass : site masses (m0), a : spacing (nm),
    eta : None, a number or one per site (eV), the broadening of the
    Buttiker probes (eta = hbar / tau). The leads continue the end sites.
    """

    def __init__(self, U, mass, a, eta=None):
        U = np.asarray(U, float)
        m = np.broadcast_to(np.asarray(mass, float), U.shape).astype(float)
        if U.ndim != 1 or U.size < 2:
            raise ValueError("a chain needs at least two sites")
        if not (a > 0 and np.all(m > 0)):
            raise ValueError("spacing and masses must be positive")
        self.U, self.m, self.a = U, m, float(a)
        self.N = U.size
        self.tl = link_couplings(m, a)
        self.tL = _H2M / (m[0] * a * a)
        self.tR = _H2M / (m[-1] * a * a)
        tpad = np.concatenate([[self.tL], self.tl, [self.tR]])
        self.diag = U + tpad[:-1] + tpad[1:]
        if eta is None:
            self.eta = None
        else:
            e = np.broadcast_to(np.asarray(eta, float), U.shape).astype(float)
            if np.any(e < 0):
                raise ValueError("probe broadening must be >= 0")
            self.eta = e if np.any(e > 0) else None

    def band_bottoms(self):
        return float(self.U[0]), float(self.U[-1])

    def _recursions(self, E):
        """Diagonal of M = E - H - Sigma (nE, N) and the left- and
        right-connected functions gL, gR."""
        E = np.atleast_1d(np.asarray(E, float))
        sL, gamL = lead_self_energy(E, self.U[0], self.tL)
        sR, gamR = lead_self_energy(E, self.U[-1], self.tR)
        a = (E[:, None] - self.diag[None, :]).astype(complex)
        a[:, 0] -= sL
        a[:, -1] -= sR
        etaE = None
        if self.eta is not None:
            # a probe acts on electrons above the local band bottom U_j
            # only (no Lorentzian tails into the gap)
            etaE = np.where(E[:, None] > self.U[None, :], self.eta[None, :], 0.0)
            a = a + 0.5j * etaE
        b2 = self.tl ** 2
        nE, N = a.shape
        gL = np.empty((nE, N), complex)
        gR = np.empty((nE, N), complex)
        gL[:, 0] = 1.0 / a[:, 0]
        for j in range(1, N):
            gL[:, j] = 1.0 / (a[:, j] - b2[j - 1] * gL[:, j - 1])
        gR[:, -1] = 1.0 / a[:, -1]
        for j in range(N - 2, -1, -1):
            gR[:, j] = 1.0 / (a[:, j] - b2[j] * gR[:, j + 1])
        return E, a, gL, gR, gamL, gamR, etaE

    def solve(self, E):
        """Per energy: transmission T (nE), the contacts' spectral diagonals
        A_L, A_R (nE, N; 1/eV), and with probes the probe data."""
        E, a, gL, gR, gamL, gamR, etaE = self._recursions(E)
        nE, N = a.shape
        b = self.tl
        # column 0 of G: G_00 = gR_0, G_j0 = -gR_j b_(j-1) G_(j-1)0
        c0 = np.empty((nE, N), complex)
        c0[:, 0] = gR[:, 0]
        for j in range(1, N):
            c0[:, j] = -gR[:, j] * b[j - 1] * c0[:, j - 1]
        # column N-1: G_(N-1)(N-1) = gL_(N-1), G_j(N-1) = -gL_j b_j G_(j+1)(N-1)
        cN = np.empty((nE, N), complex)
        cN[:, -1] = gL[:, -1]
        for j in range(N - 2, -1, -1):
            cN[:, j] = -gL[:, j] * b[j] * cN[:, j + 1]
        out = dict(E=E, gamL=gamL, gamR=gamR,
                   T=gamL * gamR * np.abs(c0[:, -1]) ** 2,
                   AL=np.abs(c0) ** 2 * gamL[:, None],
                   AR=np.abs(cN) ** 2 * gamR[:, None])
        if self.eta is not None:
            # diagonal of G and the ratios that give every G_ij (i >= j):
            # G_ij = G_jj prod_(k=j+1..i) r_k, r_k = -gR_k b_(k-1)
            Gd = 1.0 / (a - np.concatenate([np.zeros((nE, 1)),
                                            (b ** 2)[None, :] * gL[:, :-1]], 1)
                        - np.concatenate([(b ** 2)[None, :] * gR[:, 1:],
                                          np.zeros((nE, 1))], 1))
            r = np.empty((nE, N), complex)
            r[:, 0] = 1.0
            r[:, 1:] = -gR[:, 1:] * b[None, :]
            out.update(Gd=Gd, r=r, c0=c0, cN=cN, eta=etaE)
        return out


def energy_grid(bottoms, top, step):
    """Quadrature nodes and weights for energy integrals over [min(bottoms),
    top] (eV) whose integrands have 1/sqrt(E - U) edges at the lead band
    bottoms U: the range is split at the bottoms, and each piece is mapped
    so that the nodes crowd into its singular ends (E - U grows as the
    square of the mapped variable), which makes the edges integrable at
    the rate of a smooth integrand. No step is wider than `step`."""
    cuts = sorted(set(float(u) for u in bottoms))
    lo = cuts[0]
    if not top > lo:
        return np.zeros(0), np.zeros(0)
    pts = [c for c in cuts if c < top] + [float(top)]
    Es, ws = [], []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        both = k + 1 < len(pts) - 1           # an edge at both ends
        L = b - a
        n = max(int(math.ceil((math.pi / 2 if both else 2.0) * L / step)), 8)
        u = (np.arange(n) + 0.5) / n
        if both:
            E = a + L * 0.5 * (1.0 - np.cos(math.pi * u))
            w = L * 0.5 * math.pi * np.sin(math.pi * u) / n
        else:
            E = a + L * u * u
            w = L * 2.0 * u / n
        Es.append(E)
        ws.append(w)
    return np.concatenate(Es), np.concatenate(ws)


_FM = None


def _fm_table():
    """F_-1/2 (normalized) and its derivative on a fine grid of eta, from
    `fermi_dirac_half` (whose derivative output is F_-1/2), for fast
    interpolation."""
    global _FM
    if _FM is None:
        e = np.arange(-60.0, 260.0 + 1e-9, 0.002)
        f = np.asarray(fermi_dirac_half(e)[1], float)
        d = np.gradient(f, e)
        _FM = (e, f, d)
    return _FM


def _fm12(eta):
    """F_-1/2(eta) and its derivative (exp(eta) below the table; the
    degenerate limit 2 sqrt(eta / pi) above it)."""
    e, f, d = _fm_table()
    eta = np.asarray(eta, float)
    lo = eta < e[0]
    hi = eta > e[-1]
    F = np.interp(eta, e, f)
    D = np.interp(eta, e, d)
    if np.any(lo):
        ex = np.exp(np.maximum(eta[lo], -700.0))
        F[lo], D[lo] = ex, ex
    if np.any(hi):
        F[hi] = 2.0 * np.sqrt(eta[hi] / math.pi)
        D[hi] = 1.0 / np.sqrt(math.pi * eta[hi])
    return F, D


def _n1(eta, m, kT):
    """N(E) per unit width (1/nm): sqrt(m kT / (2 pi hbar^2)) F_-1/2(eta)."""
    return math.sqrt(m * kT / (2.0 * math.pi * 2.0 * _H2M)) * _fm12(eta)[0]


class _SMul:
    """y_i = sum_j |G_ij|^2 x_j for every energy (rows), from |G_jj|^2 and
    |r_k|^2 (|G_ij|^2 = |G_jj|^2 prod_(k=j+1..i) |r_k|^2 for i >= j, and G
    is symmetric): the two first-order recursions
        L_i = |r_i|^2 L_(i-1) + |G_ii|^2 x_i,   B_i = |r_(i+1)|^2 (x_(i+1) + B_(i+1)),
    y = L + |G_ii|^2 B, solved in blocks of 16 sites (products of the |r|^2
    inside each block, then the carries from block to block), with no
    N x N matrix. The block products are kept for repeated use."""

    def __init__(self, Gd2, r2, block=16):
        nE, N = Gd2.shape
        self.nE, self.N, self.b = nE, N, block
        nb = -(-N // block)
        self.nb = nb
        P = nb * block
        r2 = np.clip(r2, 1e-12, None)
        self.Gd2 = Gd2
        # forward: c_i = prod_(k=i0..i) r2_k inside each block
        rf = np.ones((nE, P))
        rf[:, :N] = r2
        self.cf = np.cumprod(rf.reshape(nE, nb, block), axis=2)
        # backward: for k = i (block from its end), R_k = r2_(k+1)
        rb = np.ones((nE, P + 1))
        rb[:, :N] = r2               # past the end: 1 (x is 0 there, so B = 0)
        Rk = rb[:, 1:P + 1].reshape(nE, nb, block)[:, :, ::-1]     # reversed in block
        self.Rk = Rk
        self.cb = np.cumprod(Rk, axis=2)

    def __call__(self, x):
        nE, N, b, nb = self.nE, self.N, self.b, self.nb
        P = nb * b
        g = np.zeros((nE, P))
        g[:, :N] = self.Gd2 * x
        g = g.reshape(nE, nb, b)
        cs = np.cumsum(g / self.cf, axis=2)
        # carry: acc_b = cf_end_b acc_(b-1) + cf_end_b cs_end_b
        acc = np.zeros((nE, nb))
        a = np.zeros(nE)
        ce, se = self.cf[:, :, -1], cs[:, :, -1]
        for q in range(nb):
            acc[:, q] = a
            a = ce[:, q] * (a + se[:, q])
        low = (self.cf * (acc[:, :, None] + cs)).reshape(nE, P)[:, :N]
        # backward: inside block q (sites q b .. q b + b - 1), reversed index
        # t = 0..b-1 is site k = q b + b - 1 - t; X_t = x_(k+1), and the
        # carry B from the next block enters at t = 0
        xp = np.zeros((nE, P + 1))
        xp[:, :N] = x
        X = xp[:, 1:P + 1].reshape(nE, nb, b)[:, :, ::-1]
        cs2 = np.cumsum(X * self.Rk / np.where(self.cb > 0, self.cb, 1.0), axis=2)
        # V_t = cb_t (B_in R_0 / cb_0 + cs2_t) with B_in the carry at the
        # block's end site + 1
        Bq = np.zeros((nE, nb))
        B = np.zeros(nE)
        for q in range(nb - 1, -1, -1):
            Bq[:, q] = B
            # B at the block's first site: V at t = b - 1
            B = self.cb[:, q, -1] * (B * self.Rk[:, q, 0] / np.where(
                self.cb[:, q, 0] > 0, self.cb[:, q, 0], 1.0) + cs2[:, q, -1])
        V = self.cb * (Bq[:, :, None] * (self.Rk[:, :, :1] / np.where(
            self.cb[:, :, :1] > 0, self.cb[:, :, :1], 1.0)) + cs2)
        up = V[:, :, ::-1].reshape(nE, P)[:, :N]
        return low + self.Gd2 * up


def _smul(Gd2, r2, x):
    """One product (see `_SMul`)."""
    return _SMul(Gd2, r2)(x)


def _uniform_density(m, a, eta, kT, zeta):
    """Density (1/nm^2) of an infinite uniform chain (band bottom 0) with
    probe broadening eta above the band bottom (0: none), every site's
    probe at the Fermi level zeta kT, as a 2D gas free along the width."""
    t = _H2M / (m * a * a)
    n_e = 3000
    u = (np.arange(n_e) + 0.5) / n_e
    E = 4.0 * t * u * u
    w = 4.0 * t * 2.0 * u / n_e
    if eta > 0:
        z = E + 0.5j * eta
        x = 1.0 - z / (2.0 * t)
        r = np.sqrt(1.0 - x * x + 0j)
        s1, s2 = x + 1j * r, x - 1j * r
        sl = np.where(np.abs(s1) < 1.0, s1, s2)
        A = -2.0 * np.imag(1.0 / (z - 2.0 * t + 2.0 * t * sl))
    else:
        ka = np.arccos(np.clip(1.0 - E / (2.0 * t), -1.0, 1.0))
        A = 1.0 / (t * np.sin(ka))
    z = np.atleast_1d(np.asarray(zeta, float))
    return 2.0 / (2.0 * math.pi * a) * (_n1(z[:, None] - E[None, :] / kT, m, kT)
                                        @ (A * w))


_DC_CACHE = {}


class _DensityCorrection:
    """The probes broaden every level; where they stop (at the local band
    bottom) the states next to it lose part of their weight, which the
    contacts and probes then do not fill. This adds the weight back, site by
    site, as the difference between the density of an infinite uniform
    chain without and with the probes at the site's band bottom and its
    probe's Fermi level (exact for a uniform chain, of the order of the
    probes' broadening elsewhere)."""

    def __init__(self, m, a, eta, kT):
        self.z = np.linspace(-40.0, 60.0, 401)
        self.c = (_uniform_density(m, a, 0.0, kT, self.z)
                  - _uniform_density(m, a, eta, kT, self.z))

    def __call__(self, zeta):
        return np.interp(zeta, self.z, self.c, left=0.0, right=self.c[-1])


class _Probes:
    """Buttiker probes with their own Fermi levels (the energy of an
    electron is not kept: a probe takes in electrons at one energy and
    gives them back with its own Fermi-Dirac filling, so it also fills
    states that the contacts cannot reach). The Fermi levels F_p make the
    net current into every probe zero:

        I_p = int sum_q T_pq (N_p - N_q) + T_pS (N_p - N_S) + T_pD (N_p - N_D) dE = 0,

    with N_p(E) = N((F_p - E) / kT). Solved by
    Newton's method; each linear step by GMRES with matrix-vector products
    from the O(N) recursions of `_smul` and a banded preconditioner."""

    def __init__(self, chain, sol, w, NS, ND, kT, my, band=16):
        self.ch, self.w, self.kT = chain, w, kT
        self.eta = sol["eta"]                 # (nE, N)
        self.Gd2 = np.abs(sol["Gd"]) ** 2
        r2 = np.abs(sol["r"]) ** 2
        r2[:, 0] = 0.0
        self.r2 = r2
        self.E = sol["E"]
        self.TpS = self.eta * sol["gamL"][:, None] * np.abs(sol["c0"]) ** 2
        self.TpD = self.eta * sol["gamR"][:, None] * np.abs(sol["cN"]) ** 2
        self.NS, self.ND = NS, ND
        self.sm = _SMul(self.Gd2, r2)
        self.rowsum = self.eta * self.sm(self.eta)
        self.band = band
        # a probe with no energy above its band bottom on the grid takes no part
        self.act = (chain.eta > 0) & np.any(self.eta > 0, axis=0)
        self.pref = math.sqrt(my * kT / (2.0 * math.pi * 2.0 * _H2M))

    def occ(self, F):
        """N_p(E) and dN_p/dF_p for every energy and probe."""
        F_, D_ = _fm12((F[None, :] - self.E[:, None]) / self.kT)
        return self.pref * F_, self.pref * D_ / self.kT

    def residual(self, F):
        Np, _ = self.occ(F)
        y = self.sm(self.eta * Np)
        I = (self.rowsum * Np - self.eta * y
             + self.TpS * (Np - self.NS[:, None]) + self.TpD * (Np - self.ND[:, None]))
        R = (I * self.w[:, None]).sum(0)
        return np.where(self.act, R, F - self.F0)

    def jvp(self, dN, v):
        y = self.sm(self.eta * dN * v[None, :])
        J = ((self.rowsum + self.TpS + self.TpD) * dN * v[None, :]
             - self.eta * y)
        return np.where(self.act, (J * self.w[:, None]).sum(0), v)

    def banded(self, dN):
        """The Jacobian's band |p - q| <= band, in solve_banded form."""
        N, b = self.ch.N, self.band
        ab = np.zeros((2 * b + 1, N))
        diag = ((self.rowsum + self.TpS + self.TpD) * dN
                - self.eta ** 2 * self.Gd2 * dN) * self.w[:, None]
        ab[b] = diag.sum(0)
        # |G_(j+d) j|^2 = |G_jj|^2 prod_(k=j+1..j+d) |r_k|^2
        prod = np.ones_like(self.Gd2)
        for d in range(1, b + 1):
            if d >= N:
                break
            prod = prod[:, :N - d] * self.r2[:, d:N]
            G2 = self.Gd2[:, :N - d] * prod             # |G_(j+d) j|^2, j = 0..N-d-1
            # J[j+d, j] = -sum w eta_(j+d) eta_j |G|^2 dN_j ; J[j, j+d] uses dN_(j+d)
            ee = self.eta[:, d:] * self.eta[:, :N - d]
            lowv = -(self.w[:, None] * ee * G2 * dN[:, :N - d]).sum(0)
            upv = -(self.w[:, None] * ee * G2 * dN[:, d:]).sum(0)
            ab[b + d, :N - d] = lowv              # row j+d, column j
            ab[b - d, d:] = upv                   # row j, column j+d
        off = ~self.act
        if np.any(off):
            for d in range(1, b + 1):
                ab[b + d, :N - d][off[d:]] = 0.0      # rows j+d
                ab[b - d, d:][off[:N - d]] = 0.0      # rows j
            ab[b][off] = 1.0
        return ab

    def solve(self, F0, tol=1e-8, rtol=1e-10, max_iter=60):
        """Newton's method until the step is below tol (eV) or every probe's
        net current is below rtol of the largest gross current through a
        probe (a probe with almost no electrons has a loosely defined
        level that does not matter)."""
        from scipy.linalg import solve_banded
        from scipy.sparse.linalg import LinearOperator, gmres
        F = np.array(F0, float)
        self.F0 = F.copy()
        N, b = self.ch.N, self.band
        mid = 0.5 * (self.NS + self.ND)[:, None]
        gross = ((self.rowsum + self.TpS + self.TpD) * mid * self.w[:, None]).sum(0)
        Rmax = float(np.max(gross)) + 1e-300
        R = self.residual(F)
        for it in range(max_iter):
            if float(np.max(np.abs(np.where(self.act, R, 0.0)))) < rtol * Rmax:
                return F, it
            Np, dN = self.occ(F)
            ab = self.banded(dN)
            A = LinearOperator((N, N), matvec=lambda v: self.jvp(dN, np.asarray(v).ravel()))
            M = LinearOperator((N, N), matvec=lambda v: solve_banded(
                (b, b), ab, np.asarray(v).ravel()))
            try:
                dF, info = gmres(A, -R, M=M, rtol=1e-7, atol=0.0, restart=60, maxiter=40)
            except TypeError:                      # SciPy before 1.12: tol
                dF, info = gmres(A, -R, M=M, tol=1e-7, atol=0.0, restart=60, maxiter=40)
            dF = np.clip(dF, -0.1, 0.1)
            F = F + dF
            R = self.residual(F)
            if float(np.max(np.abs(dF))) < tol:
                return F, it + 1
        if float(np.max(np.abs(np.where(self.act, R, 0.0)))) < 1e-6 * Rmax:
            return F, max_iter
        raise RuntimeError("the Buttiker probes' Fermi levels did not settle")


def _segments(bottoms, top):
    """Pieces of [min(bottoms), top] split at the band bottoms, each with
    its map E(u), dE/du on 0 <= u <= 1 (crowding into singular ends)."""
    cuts = sorted(set(float(u) for u in bottoms))
    pts = [c for c in cuts if c < top] + [float(top)]
    out = []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        L = b - a
        if k + 1 < len(pts) - 1:
            out.append((lambda u, a=a, L=L: a + L * 0.5 * (1.0 - np.cos(math.pi * u)),
                        lambda u, L=L: L * 0.5 * math.pi * np.sin(math.pi * u), L))
        else:
            out.append((lambda u, a=a, L=L: a + L * u * u,
                        lambda u, L=L: 2.0 * L * u, L))
    return out


def _adaptive_grid(func, bottoms, top, step, rtol=1e-4, max_nodes=40000):
    """Composite-Simpson nodes and weights over [min(bottoms), top] for
    func(E) -> (nE, P) (the last column a current, the rest densities):
    each piece between band bottoms is mapped to crowd into its singular
    ends (`_segments`), and every interval whose Simpson and trapezoid
    estimates differ by more than its share of rtol (of the largest density,
    and of the current) is halved."""
    segs = _segments(bottoms, top)
    if not segs:
        return np.zeros(0), np.zeros(0)

    def integrand(seg, u):
        Em, dEm, _ = seg
        u = np.clip(u, 1e-6, 1.0 - 1e-6)      # never exactly on a band edge
        return func(Em(u)) * dEm(u)[:, None]

    nodesE, nodesW = [], []
    for seg in segs:
        L = seg[2]
        n0 = max(int(math.ceil(2.0 * L / step)), 8)
        a = np.arange(n0) / n0
        b = a + 1.0 / n0
        m = 0.5 * (a + b)
        u_all = np.unique(np.concatenate([a, b, m]))
        G = integrand(seg, u_all)
        val = dict(zip(u_all.tolist(), G))
        while True:
            ga = np.array([val[v] for v in a.tolist()])
            gb = np.array([val[v] for v in b.tolist()])
            gm = np.array([val[v] for v in m.tolist()])
            h = (b - a)[:, None]
            S = h / 6.0 * (ga + 4.0 * gm + gb)
            T2 = h / 4.0 * (ga + 2.0 * gm + gb)
            err = np.abs(S - T2)
            tot = np.abs(S.sum(0))
            scale = np.concatenate([np.full(tot.size - 1, max(tot[:-1].max(), 1e-300)),
                                    [max(tot[-1], 1e-300)]])
            rel = err / scale[None, :]
            bad = rel.max(1) > rtol / max(len(a), 1) * 4.0
            if rel.sum(0).max() < rtol or not np.any(bad) or len(val) > max_nodes:
                break
            # halve the bad intervals
            na, nb = a[bad], b[bad]
            nm = m[bad]
            keep = ~bad
            a = np.concatenate([a[keep], na, nm])
            b = np.concatenate([b[keep], nm, nb])
            m = 0.5 * (a + b)
            new = np.array([v for v in m.tolist() if v not in val])
            if new.size:
                Gn = integrand(seg, new)
                val.update(zip(new.tolist(), Gn))
            order = np.argsort(a)
            a, b, m = a[order], b[order], m[order]
        Em, dEm, _ = seg
        h = b - a
        u = np.concatenate([a, m, b])
        wu = np.concatenate([h / 6.0, 4.0 * h / 6.0, h / 6.0])
        uu, inv = np.unique(u, return_inverse=True)
        ww = np.bincount(inv, wu)
        uc = np.clip(uu, 1e-6, 1.0 - 1e-6)
        nodesE.append(Em(uc))
        nodesW.append(ww * dEm(uc))
    return np.concatenate(nodesE), np.concatenate(nodesW)


def adaptive_energies(chain: Chain, FS: float, FD: float, kT: float, mass_y: float,
                      step: float, top: float, rtol: float = 1e-4, max_nodes: int = 40000):
    """Nodes and weights (composite Simpson) for the energy integrals of a
    chain, refined where they need it (see `_adaptive_grid`): long flat
    stretches of a band, whose 1D density of states has a sharp edge
    anywhere in energy, are found this way."""
    bottoms = list(chain.band_bottoms())
    if chain.eta is not None:
        bottoms.append(float(np.min(chain.U)))
    Fm = max(FS, FD)

    def f(E):
        sol = chain.solve(E)
        NS = _n1((FS - E) / kT, mass_y, kT)
        ND = _n1((FD - E) / kT, mass_y, kT)
        if chain.eta is not None:
            Atot = -2.0 * sol["Gd"].imag
            d = Atot * _n1((Fm - E) / kT, mass_y, kT)[:, None]
        else:
            d = sol["AL"] * NS[:, None] + sol["AR"] * ND[:, None]
        cur = sol["T"] * np.abs(NS - ND) + sol["T"] * (NS + ND) * 1e-3
        return np.concatenate([d, cur[:, None]], axis=1)
    return _adaptive_grid(f, bottoms, top, step, rtol, max_nodes)


def chain_transport(chain: Chain, FS: float, FD: float, kT: float,
                    energies=None, energy_step: float = None, emax_kT: float = 30.0,
                    probe_guess=None, mass_y: float = None, rtol: float = 1e-4):
    """Density (1/nm^2, per site) and current per width (A/nm) of a chain
    treated as one subband of a 2D gas that is free along the width, with
    mass along the width mass_y (m0; default the mean of the site masses:
    the transverse momentum is kept, so one mass along the width keeps the
    energy along x as well and detailed balance exact), filled from a
    source at Fermi level FS (left) and a drain at FD (right), eV. Returns
    dict(n, I, E, T, I_drain, probe_F).

    The energies run from the lower lead band bottom (with probes, the
    lowest site potential) to max(FS, FD) + emax_kT kT, unless given: a
    start grid of steps of at most energy_step (default kT / 10) refined by
    `adaptive_energies` to rtol."""
    my = float(np.mean(chain.m)) if mass_y is None else float(mass_y)
    if energies is None:
        step = kT / 10.0 if energy_step is None else float(energy_step)
        E, w = adaptive_energies(chain, FS, FD, kT, my, step,
                                 max(FS, FD) + emax_kT * kT, rtol=rtol)
        if E.size == 0:
            return dict(n=np.zeros(chain.N), I=0.0, I_drain=0.0, E=E, T=np.zeros(0),
                        probe_F=None)
    else:
        E = np.asarray(energies, float)
        w = np.gradient(E) if E.size > 1 else np.ones(1)
    sol = chain.solve(E)
    NS = _n1((FS - E) / kT, my, kT)
    ND = _n1((FD - E) / kT, my, kT)
    pref = 2.0 / (2.0 * math.pi * chain.a)
    dens = sol["AL"] * NS[:, None] + sol["AR"] * ND[:, None]
    IS = sol["T"] * (NS - ND)
    ID = sol["T"] * (NS - ND)
    Fp = None
    if chain.eta is not None:
        pr = _Probes(chain, sol, w, NS, ND, kT, my)
        if probe_guess is None:
            x = np.linspace(0.0, 1.0, chain.N)
            probe_guess = FS + (FD - FS) * x
        Fp, _ = pr.solve(probe_guess)
        Np, _ = pr.occ(Fp)
        dens = dens + pr.sm(pr.eta * Np)
        IS = IS + (pr.TpS * (NS[:, None] - Np)).sum(1)
        ID = ID + (pr.TpD * (Np - ND[:, None])).sum(1)
    n = pref * (dens * w[:, None]).sum(0)
    if chain.eta is not None:
        act = pr.act
        dc = np.zeros(chain.N)
        for e_ in np.unique(chain.eta[act]):
            sel = act & (chain.eta == e_)
            # (the correction changes little with the mass: masses are
            # rounded to 0.001 m0 for the cache)
            key = (round(my, 3), round(chain.a, 9), float(e_), round(kT, 9))
            C = _DC_CACHE.get(key)
            if C is None:
                if len(_DC_CACHE) > 256:
                    _DC_CACHE.clear()
                C = _DC_CACHE[key] = _DensityCorrection(key[0], chain.a, float(e_), kT)
            dc[sel] = C((Fp[sel] - chain.U[sel]) / kT)
        n = n + dc
    return dict(n=n, I=_G0 * float(np.sum(IS * w)), I_drain=_G0 * float(np.sum(ID * w)),
                E=E, T=sol["T"], probe_F=Fp)


# --------------------------------------------- coupled mode space (0.27.0)
def lead_surface_green(E, D, C, eta=1e-9, tol=1e-12, max_iter=200):
    """Surface Green's function of a semi-infinite lead of identical slices
    with on-slice block D and symmetric coupling C between neighbors, at the
    energies E (nE,), by the decimation of M. P. Lopez Sancho, J. M. Lopez
    Sancho and J. Rubio, J. Phys. F 15, 851 (1985),
    doi:10.1088/0305-4608/15/4/009, at E + i eta. Returns (nE, M, M)."""
    E = np.atleast_1d(np.asarray(E, float))
    M = D.shape[0]
    I = np.eye(M)
    z = (E + 1j * eta)[:, None, None] * I[None]
    eps_s = np.broadcast_to(D, (E.size, M, M)).astype(complex)
    eps = eps_s.copy()
    al = np.broadcast_to(C, (E.size, M, M)).astype(complex)
    be = al.copy()
    for _ in range(max_iter):
        g = np.linalg.inv(z - eps)
        agb = al @ g @ be
        bga = be @ g @ al
        eps_s = eps_s + agb
        eps = eps + agb + bga
        al = al @ g @ al
        be = be @ g @ be
        if float(np.max(np.abs(al))) < tol and float(np.max(np.abs(be))) < tol:
            break
    else:
        raise RuntimeError("the lead's surface Green's function did not converge")
    return np.linalg.inv(z - eps_s)


class Slab:
    """A 2D effective-mass region (x along the transport, z across), uniform
    in x with spacing a (nm) and on the same z nodes in every column, joined
    at both ends to leads that continue the end columns (new in 0.27.0).

    U, mass : (Nx, Nz) potential (eV) and mass (m0) at the nodes. z : the
    nodes (nm), zl, zr : the walls (psi = 0) below and above them. The
    z-part of each column is the box-method operator of `column_states`;
    neighbors along x are joined by hbar^2 / (2 m a^2) (link mass the mean of
    the two nodes). n_modes : the lowest eigenstates of every column kept
    (coupled mode space: R. Venugopal et al., J. Appl. Phys. 95, 292 (2004),
    doi:10.1063/1.1631754); all Nz is exact (a change of basis of the real-
    space problem). mass_x : the mass along x where it differs from the
    one across (default mass), as for holes with an in-plane mass of their
    own."""

    def __init__(self, U, mass, a, z, zl, zr, n_modes=None, mass_x=None, eta=None):
        U = np.asarray(U, float)
        m = np.broadcast_to(np.asarray(mass, float), U.shape).astype(float)
        mx = m if mass_x is None else np.broadcast_to(np.asarray(mass_x, float),
                                                      U.shape).astype(float)
        if not np.all(mx > 0):
            raise ValueError("spacing and masses must be positive")
        z = np.asarray(z, float)
        if U.ndim != 2 or U.shape[1] != z.size or U.shape[0] < 2 or z.size < 2:
            raise ValueError("U must be (Nx >= 2, Nz) with Nz = len(z) >= 2")
        if not (a > 0 and np.all(m > 0)):
            raise ValueError("spacing and masses must be positive")
        Nx, Nz = U.shape
        self.U, self.m, self.a, self.z = U, m, float(a), z
        M = Nz if n_modes is None else int(min(n_modes, Nz))
        if M < 1:
            raise ValueError("n_modes must be >= 1")
        self.M = M
        zz = np.concatenate([[zl], z, [zr]])
        h = np.diff(zz)
        if np.any(h <= 0):
            raise ValueError("z must increase, with the walls outside it")
        w = 0.5 * (h[:-1] + h[1:])
        self.w = w
        # x couplings on every link (Nx - 1, Nz) and the lead ones
        tx = _H2M / (0.5 * (mx[1:] + mx[:-1]) * a * a)
        tL = _H2M / (mx[0] * a * a)
        tR = _H2M / (mx[-1] * a * a)
        self.mx = mx
        self.tx, self.tL, self.tR = tx, tL, tR
        from scipy.linalg import eigh_tridiagonal
        V, Ez = [], []
        Dt, Ct = [], []
        for j in range(Nx):
            mm = np.concatenate([[m[j, 0]], m[j], [m[j, -1]]])
            c = _H2M / (0.5 * (mm[1:] + mm[:-1]) * h)
            diag = (U[j] * w + c[:-1] + c[1:]) / w
            off = -c[1:-1] / np.sqrt(w[:-1] * w[1:])
            e, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, M - 1))
            V.append(v)
            Ez.append(e)
        self.V, self.Ez = V, np.array(Ez)
        for j in range(Nx):
            tl = tL if j == 0 else tx[j - 1]
            tr = tR if j == Nx - 1 else tx[j]
            Dt.append(np.diag(self.Ez[j]) + V[j].T @ ((tl + tr)[:, None] * V[j]))
        for j in range(Nx - 1):
            Ct.append(-V[j].T @ (tx[j][:, None] * V[j + 1]))
        self.Dt, self.Ct = Dt, Ct
        # leads: slices equal to the end columns
        self.DL = np.diag(self.Ez[0]) + V[0].T @ ((2 * tL)[:, None] * V[0])
        self.CL = -V[0].T @ (tL[:, None] * V[0])
        self.DR = np.diag(self.Ez[-1]) + V[-1].T @ ((2 * tR)[:, None] * V[-1])
        self.CR = -V[-1].T @ (tR[:, None] * V[-1])
        self.Nx, self.Nz = Nx, Nz
        if eta is None:
            self.eta = None
        else:
            e = np.broadcast_to(np.asarray(eta, float), (Nx,)).astype(float)
            if np.any(e < 0):
                raise ValueError("probe broadening must be >= 0")
            self.eta = e if np.any(e > 0) else None

    def _probe_eta(self, E):
        """(nE, Nx, M): each column's probe broadening on each of its modes,
        above that mode's own bottom only (as in `Chain`)."""
        return np.where(E[:, None, None] > self.Ez[None, :, :], self.eta[None, :, None], 0.0)

    def band_bottoms(self):
        """The leads' subband bottoms (eV): eigenvalues of D + 2 C."""
        return np.concatenate([np.linalg.eigvalsh(self.DL + 2 * self.CL),
                               np.linalg.eigvalsh(self.DR + 2 * self.CR)])

    def solve(self, E, eta=1e-9, weights=None):
        """Per energy: the transmission T (nE) and the contacts' spectral
        functions in the mode basis of each column, diagonals only: PL, PR
        (nE, Nx, M). With weights = (wL, wR) (nE each), also the node
        densities sum_E wL A_L + wR A_R (Nx, Nz; per node of the column,
        psi^2 normalization), summed in the mode basis first."""
        E = np.atleast_1d(np.asarray(E, float))
        nE, Nx, M = E.size, self.Nx, self.M
        I = np.eye(M)
        gsL = lead_surface_green(E, self.DL, self.CL, eta)
        gsR = lead_surface_green(E, self.DR, self.CR, eta)
        SL = self.CL @ gsL @ self.CL
        SR = self.CR @ gsR @ self.CR
        GL = 1j * (SL - np.conj(np.swapaxes(SL, 1, 2)))
        GR = 1j * (SR - np.conj(np.swapaxes(SR, 1, 2)))
        z = (E + 1j * eta)[:, None, None] * I[None]
        A = [z - self.Dt[j][None] for j in range(Nx)]
        A[0] = A[0] - SL
        A[-1] = A[-1] - SR
        if self.eta is not None:
            etaE = self._probe_eta(E)
            for j in range(Nx):
                A[j] = A[j] + 0.5j * etaE[:, j, :, None] * I[None]
        # left- and right-connected Green's functions
        gl = [None] * Nx
        gl[0] = np.linalg.inv(A[0])
        for j in range(1, Nx):
            C = self.Ct[j - 1]
            gl[j] = np.linalg.inv(A[j] - C.T[None] @ gl[j - 1] @ C[None])
        gr = [None] * Nx
        gr[-1] = np.linalg.inv(A[-1])
        for j in range(Nx - 2, -1, -1):
            C = self.Ct[j]
            gr[j] = np.linalg.inv(A[j] - C[None] @ gr[j + 1] @ C.T[None])
        # first column G_j0 and last column G_jN
        c0 = [None] * Nx
        c0[0] = gr[0]
        for j in range(1, Nx):
            c0[j] = gr[j] @ self.Ct[j - 1].T[None] @ c0[j - 1]
        cN = [None] * Nx
        cN[-1] = gl[-1]
        for j in range(Nx - 2, -1, -1):
            cN[j] = gl[j] @ self.Ct[j][None] @ cN[j + 1]
        T = np.real(np.trace(GL @ cN[0] @ GR @ np.conj(np.swapaxes(cN[0], 1, 2)),
                             axis1=1, axis2=2))
        self._last = dict(gl=gl, gr=gr, A=A, c0=c0, cN=cN, GL=GL, GR=GR, E=E)
        Atot = None
        if self.eta is not None:
            # the whole spectral function on each column's modes, 2 Im of
            # the diagonal of G_jj (for the energy grid)
            Atot = np.empty((nE, Nx, M))
            for j in range(Nx):
                S = A[j]
                if j > 0:
                    S = S - self.Ct[j - 1].T[None] @ gl[j - 1] @ self.Ct[j - 1][None]
                if j < Nx - 1:
                    S = S - self.Ct[j][None] @ gr[j + 1] @ self.Ct[j].T[None]
                Atot[:, j] = -2.0 * np.imag(np.diagonal(np.linalg.inv(S), axis1=1, axis2=2))
        PL = np.empty((nE, Nx, M))
        PR = np.empty((nE, Nx, M))
        dens = None if weights is None else np.zeros((Nx, self.Nz))
        for j in range(Nx):
            P = c0[j] @ GL @ np.conj(np.swapaxes(c0[j], 1, 2))
            Q = cN[j] @ GR @ np.conj(np.swapaxes(cN[j], 1, 2))
            PL[:, j] = np.real(np.diagonal(P, axis1=1, axis2=2))
            PR[:, j] = np.real(np.diagonal(Q, axis1=1, axis2=2))
            if weights is not None:
                S = np.real(np.tensordot(weights[0], P, axes=1)
                            + np.tensordot(weights[1], Q, axes=1))
                Vj = self.V[j]
                dens[j] = np.einsum("km,mn,kn->k", Vj, S, Vj) / self.w
        return dict(E=E, T=T, PL=PL, PR=PR, density=dens, Atot=Atot)

    def node_spectral(self, E, eta=1e-9):
        """AL, AR on the nodes (nE, Nx, Nz), one energy at a time (for
        checks: slow)."""
        E = np.atleast_1d(np.asarray(E, float))
        AL = np.empty((E.size, self.Nx, self.Nz))
        AR = np.empty((E.size, self.Nx, self.Nz))
        for k, e in enumerate(E):
            one = np.ones(1)
            AL[k] = self.solve(np.array([e]), eta, weights=(one, 0 * one))["density"]
            AR[k] = self.solve(np.array([e]), eta, weights=(0 * one, one))["density"]
        return AL, AR


def _slab_blocks(slab, E):
    """Every block G_jq (j >= q) of a slab with probes, one column q at a
    time: yields (q, list of G_jq for j = q..Nx-1), each (nE, M, M)."""
    L = slab._last
    gl, gr, A = L["gl"], L["gr"], L["A"]
    Nx = slab.Nx
    Ct = slab.Ct
    for q in range(Nx):
        S = A[q]
        if q > 0:
            S = S - Ct[q - 1].T[None] @ gl[q - 1] @ Ct[q - 1][None]
        if q < Nx - 1:
            S = S - Ct[q][None] @ gr[q + 1] @ Ct[q].T[None]
        G = [np.linalg.inv(S)]
        for j in range(q + 1, Nx):
            G.append(gr[j] @ Ct[j - 1].T[None] @ G[-1])
        yield q, G


class _SlabProbes:
    """Buttiker probes on a Slab (new in 0.32.0): one probe per column,
    joined to every mode of the column above that mode's bottom with the
    broadening eta (the projection of a probe on every node of the column),
    with one Fermi level that makes the column's net probe current zero, as
    `_Probes` does for a chain. T_pq = Tr[Gamma_p G_pq Gamma_q G_pq^+] for
    every pair of columns is formed from the blocks of G (Nx^2 M^3 work per
    energy) and the levels are found by Newton's method with the full
    Jacobian (Nx x Nx)."""

    def __init__(self, slab, E, w, NS, ND, kT, my):
        sol = slab.solve(E)
        self.slab, self.E, self.w, self.kT = slab, E, w, kT
        self.NS, self.ND = NS, ND
        L = slab._last
        Nx = slab.Nx
        eta = slab._probe_eta(E)                                  # (nE, Nx, M)
        self.eta = eta
        GL, GR = L["GL"], L["GR"]
        nE = E.size
        Tpq = np.zeros((nE, Nx, Nx))
        TpS = np.zeros((nE, Nx))
        TpD = np.zeros((nE, Nx))
        for q, G in _slab_blocks(slab, E):
            for k, Gjq in enumerate(G):
                j = q + k
                a2 = np.abs(Gjq) ** 2
                if j != q:
                    v = np.einsum("em,emn,en->e", eta[:, j], a2, eta[:, q])
                    Tpq[:, j, q] = v
                    Tpq[:, q, j] = v
                if q == 0:
                    # G_j0 Gamma_L G_j0^+
                    TpS[:, j] = np.real(np.einsum("em,emk,ekl,eml->e", eta[:, j], Gjq, GL,
                                                  np.conj(Gjq)))
                if j == Nx - 1:
                    # G_(N-1) q = G_q(N-1)^T: Gamma_q G_q(N-1) Gamma_R G_q(N-1)^+
                    Gq = np.swapaxes(Gjq, 1, 2)
                    TpD[:, q] = np.real(np.einsum("em,emk,ekl,eml->e", eta[:, q], Gq, GR,
                                                  np.conj(Gq)))
        self.T, self.Tpq, self.TpS, self.TpD = sol["T"], Tpq, TpS, TpD
        self.sol = sol
        self.act = np.any(eta > 0, axis=(0, 2))
        self.pref = math.sqrt(my * kT / (2.0 * math.pi * 2.0 * _H2M))

    def occ(self, F):
        F_, D_ = _fm12((F[None, :] - self.E[:, None]) / self.kT)
        return self.pref * F_, self.pref * D_ / self.kT

    def currents(self, F):
        Np, _ = self.occ(F)
        out = (self.Tpq.sum(2) * Np - np.einsum("epq,eq->ep", self.Tpq, Np)
               + self.TpS * (Np - self.NS[:, None]) + self.TpD * (Np - self.ND[:, None]))
        return (out * self.w[:, None]).sum(0)

    def solve(self, F0, tol=1e-9, max_iter=80):
        F = np.array(F0, float)
        mid = 0.5 * (self.NS + self.ND)[:, None]
        gross = ((self.Tpq.sum(2) + self.TpS + self.TpD) * mid * self.w[:, None]).sum(0)
        Rmax = float(np.max(gross)) + 1e-300
        for it in range(max_iter):
            R = np.where(self.act, self.currents(F), 0.0)
            if float(np.max(np.abs(R))) < 1e-10 * Rmax:
                return F, it
            _, dN = self.occ(F)
            J = (np.diag(((self.Tpq.sum(2) + self.TpS + self.TpD) * dN * self.w[:, None]).sum(0))
                 - np.einsum("e,epq,eq->pq", self.w, self.Tpq, dN))
            off = ~self.act
            J[off, :] = 0.0
            J[off, off] = 1.0
            dF = np.clip(np.linalg.solve(J, -R), -0.1, 0.1)
            F = F + dF
            if float(np.max(np.abs(dF))) < tol:
                return F, it + 1
        R = np.where(self.act, self.currents(F), 0.0)
        if float(np.max(np.abs(R))) < 1e-6 * Rmax:
            return F, max_iter
        raise RuntimeError("the Buttiker probes' Fermi levels did not settle")

    def density(self, F):
        """(Nx, Nz) node density of the probes' electrons (sum over energies
        with the weights, before the 2 / (2 pi a) prefactor)."""
        slab = self.slab
        Np, _ = self.occ(F)
        wq = self.eta * (Np * self.w[:, None])[:, :, None]               # (nE, Nx, M)
        S = [np.zeros((slab.M, slab.M)) for _ in range(slab.Nx)]
        for q, G in _slab_blocks(slab, self.E):
            for k, Gjq in enumerate(G):
                j = q + k
                # G_jq diag(w_q) G_jq^+ summed over energies, and the mirror
                # block G_qj = G_jq^T for j > q
                S[j] += np.real(np.einsum("emk,ek,enk->mn", Gjq, wq[:, q], np.conj(Gjq)))
                if j != q:
                    S[q] += np.real(np.einsum("ekm,ek,ekn->mn", Gjq, wq[:, j],
                                              np.conj(Gjq)))
        dens = np.zeros((slab.Nx, slab.Nz))
        for j in range(slab.Nx):
            Vj = slab.V[j]
            dens[j] = np.einsum("km,mn,kn->k", Vj, S[j], Vj) / slab.w
        return dens


def slab_transport(slab: Slab, FS: float, FD: float, kT: float, mass_y: float = None,
                   energy_step: float = None, emax_kT: float = 30.0, rtol: float = 1e-4,
                   energies=None, probe_guess=None):
    """Density (nm^-3, (Nx, Nz)) and current per width (A/nm) of a Slab
    whose third direction (the width) is free with one mass mass_y (default
    the mean of the masses), filled from a source at FS (x = 0 side) and a
    drain at FD (eV), ballistic. The energy grid is refined on the modal
    densities and the current (`_adaptive_grid`). Returns dict(n, I, E, T)."""
    my = float(np.mean(slab.m)) if mass_y is None else float(mass_y)
    top = max(FS, FD) + emax_kT * kT
    step = kT / 10.0 if energy_step is None else float(energy_step)
    Nx, Nz = slab.Nx, slab.Nz

    def f(E):
        s = slab.solve(E)
        NS = _n1((FS - E) / kT, my, kT)
        ND = _n1((FD - E) / kT, my, kT)
        if slab.eta is not None:
            # with probes every state can be filled: refine on all of them
            d = (s["Atot"] * _n1((max(FS, FD) - E) / kT, my, kT)[:, None, None]).reshape(
                E.size, -1)
        else:
            d = (s["PL"] * NS[:, None, None] + s["PR"] * ND[:, None, None]).reshape(E.size, -1)
        cur = s["T"] * np.abs(NS - ND) + s["T"] * (NS + ND) * 1e-3
        return np.concatenate([d, cur[:, None]], axis=1)
    if energies is None:
        bottoms = list(slab.band_bottoms())
        if slab.eta is not None:
            bottoms.append(float(np.min(slab.Ez)))
        E, w = _adaptive_grid(f, bottoms, top, step, rtol)
    else:
        E = np.asarray(energies, float)
        w = np.gradient(E) if E.size > 1 else np.ones(1)
    if E.size == 0:
        return dict(n=np.zeros((Nx, Nz)), I=0.0, E=E, T=np.zeros(0), probe_F=None)
    NS = _n1((FS - E) / kT, my, kT)
    ND = _n1((FD - E) / kT, my, kT)
    s = slab.solve(E, weights=(NS * w, ND * w))
    pref = 2.0 / (2.0 * math.pi * slab.a)
    I = _G0 * float(np.sum(s["T"] * (NS - ND) * w))
    if slab.eta is None:
        return dict(n=pref * s["density"], I=I, E=E, T=s["T"], probe_F=None)
    # with probes: their Fermi levels, the electrons they hold, the
    # current they carry into the drain, and the states the cut at each
    # mode's bottom leaves out (added back as for a chain)
    pr = _SlabProbes(slab, E, w, NS, ND, kT, my)
    if probe_guess is None:
        probe_guess = FS + (FD - FS) * np.linspace(0.0, 1.0, Nx)
    Fp, _ = pr.solve(probe_guess)
    Np, _ = pr.occ(Fp)
    n = pref * (s["density"] + pr.density(Fp))
    I_D = I + _G0 * float(np.sum((pr.TpD * (Np - ND[:, None])).sum(1) * w))
    I_S = I + _G0 * float(np.sum((pr.TpS * (NS[:, None] - Np)).sum(1) * w))
    for j in range(Nx):
        e_ = float(slab.eta[j])
        if e_ <= 0:
            continue
        key = (round(my, 3), round(slab.a, 9), e_, round(kT, 9))
        C = _DC_CACHE.get(key)
        if C is None:
            if len(_DC_CACHE) > 256:
                _DC_CACHE.clear()
            C = _DC_CACHE[key] = _DensityCorrection(key[0], slab.a, e_, kT)
        dc = C((Fp[j] - slab.Ez[j]) / kT)                               # (M,) 1/nm^2
        n[j] += (slab.V[j] ** 2 / slab.w[:, None]) @ dc
    return dict(n=n, I=I_D, I_source=I_S, E=E, T=s["T"], probe_F=Fp)
