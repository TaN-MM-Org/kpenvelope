"""Scattering lifetimes and mobility of one subband of a 2D gas (new in
0.27.0).

The rates are the standard Born-approximation (Fermi golden rule) rates
of a single subband, intrasubband only, worked out on the subband's own
envelope psi(z) (from any of the package's solvers, or yours), with the
material numbers you pass. The package ships none of those numbers (the
deformation potential, the alloy potential, the roughness height and
length, the impurity densities): give cited values for your structure.

Energies are kinetic energies above the subband bottom (eV); a rate is
1/tau (1/s). Every mechanism returns a function rate(E, quantum=False):
quantum=False gives the transport (momentum relaxation) rate, with the
weight 1 - cos(theta) on the scattering angle; quantum=True gives the
single-particle (quantum) rate, weight 1, which sets the Dingle factor
of the magneto-oscillations (`fit_sdh_dingle`).

Sources of the formulas (all rederived for the checks in docs/checks.md):
T. Ando, A. B. Fowler and F. Stern, Rev. Mod. Phys. 54, 437 (1982)
(screened Coulomb scattering, form factors, Thomas-Fermi screening of a
2D gas); P. J. Price, Ann. Phys. (1981), doi:10.1016/0003-4916(81)90250-5
(acoustic deformation-potential scattering of a 2D gas); G. Bastard,
Appl. Phys. Lett. 43, 591 (1983) (alloy scattering); H. Sakaki et al.,
Appl. Phys. Lett. 51, 1934 (1987) and S. M. Goodnick et al., Phys. Rev.
B 32, 8171 (1985) (interface roughness); S. Das Sarma and F. Stern,
Phys. Rev. B 32, 8442 (1985) (transport and quantum lifetimes).
"""
from __future__ import annotations

import math

import numpy as np

from .electrons import EPS0, HBAR, KB, M0
from .nitride import QE

__all__ = ["Subband"]


def _gauss_theta(n):
    """Nodes and weights on (0, pi), crowded toward theta = 0 (theta =
    pi t^2), where forward scattering by distant charges sits."""
    t, wt = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t + 1.0)
    wt = 0.5 * wt
    th = math.pi * t * t
    return th, wt * 2 * math.pi * t


class Subband:
    """One subband of a 2D electron (or hole) gas.

    z : nodes (nm); psi2 : |psi(z)|^2 on them (1/nm), normalized with the
    weights (default: the trapezoid weights of z) to sum(w psi2) = 1, as
    `column_states` returns it. mass : the in-plane mass (m0). eps_r : the
    background permittivity (one value: image charges at permittivity steps
    are not included). Give the filling as fermi (eV above the subband
    bottom) or sheet_cm2 (the gas of this subband alone);
    temperature_K sets the thermal average and the screening.
    screening : True for Thomas-Fermi screening of the Coulomb and
    roughness potentials by this subband (with its own form factor),
    False for none.
    """

    def __init__(self, z, psi2, mass, eps_r, fermi=None, sheet_cm2=None,
                 temperature_K=300.0, weights=None, screening=True, n_theta=400):
        z = np.asarray(z, float)
        p = np.asarray(psi2, float)
        if z.ndim != 1 or z.size < 3 or p.shape != z.shape or np.any(np.diff(z) <= 0):
            raise ValueError("z must increase (at least 3 nodes) and psi2 match it")
        if not (mass > 0 and eps_r > 0 and temperature_K > 0):
            raise ValueError("mass, eps_r and temperature_K must be > 0")
        if np.any(p < 0):
            raise ValueError("psi2 must be >= 0")
        if weights is None:
            h = np.diff(z)
            w = np.zeros(z.size)
            w[:-1] += 0.5 * h
            w[1:] += 0.5 * h
        else:
            w = np.asarray(weights, float)
            if w.shape != z.shape or np.any(w < 0):
                raise ValueError("weights must match z and be >= 0")
        norm = float((w * p).sum())
        if not abs(norm - 1.0) < 1e-3:
            raise ValueError(f"psi2 must be normalized (sum(w psi2) = {norm:.6g}, not 1)")
        self.z, self.p, self.w = z, p / norm, w
        self.mass = float(mass)
        self.m = self.mass * M0
        self.eps = float(eps_r) * EPS0
        self.T = float(temperature_K)
        self.kT = KB * self.T
        self.dos = self.m / (math.pi * HBAR ** 2) * QE            # 1/(eV m^2), spin 2
        if (fermi is None) == (sheet_cm2 is None):
            raise ValueError("give fermi or sheet_cm2 (one of them)")
        if fermi is None:
            if not sheet_cm2 > 0:
                raise ValueError("sheet_cm2 must be > 0")
            ns = sheet_cm2 * 1e4
            x = ns / (self.dos * self.kT)
            # n = D kT ln(1 + e^(EF/kT))
            fermi = self.kT * (x + math.log(-math.expm1(-x)) if x < 700 else x)
        self.fermi = float(fermi)
        self.sheet = self.dos * self.kT * float(np.logaddexp(0.0, self.fermi / self.kT))
        self.screening = bool(screening)
        # dn/dE_F (1/(eV m^2)) for the Thomas-Fermi wave number
        self.dn_dEF = self.dos / (1.0 + math.exp(-min(self.fermi / self.kT, 700.0)))
        self.q_s = QE ** 2 / (2 * self.eps) * self.dn_dEF / QE      # 1/m
        self._th, self._wth = _gauss_theta(int(n_theta))
        self._Fee_cache = {}

    # ----------------------------------------------------------- pieces
    def k(self, E):
        """Wave number (1/m) at kinetic energy E (eV)."""
        return np.sqrt(2 * self.m * np.maximum(np.asarray(E, float), 0.0) * QE) / HBAR

    def overlap(self, mask=None):
        """integral of psi^4 (1/nm), over the nodes in mask (default all)."""
        p4 = self.w * self.p ** 2
        return float(p4.sum() if mask is None else p4[np.asarray(mask, bool)].sum())

    def _pair_sum(self, q_nm, a, b):
        """sum_ij a_i b_j exp(-q |z_i - z_j|) for every q (1/nm), by the
        two one-sided recursions (stable for any q)."""
        z = self.z
        q = np.atleast_1d(q_nm)
        tot = np.zeros(q.size)
        S = np.zeros(q.size)
        for k_ in range(z.size):                          # j < k
            if k_:
                S = np.exp(-q * (z[k_] - z[k_ - 1])) * (S + b[k_ - 1])
            tot += a[k_] * S
        S = np.zeros(q.size)
        for k_ in range(z.size - 1, -1, -1):              # j > k
            if k_ < z.size - 1:
                S = np.exp(-q * (z[k_ + 1] - z[k_])) * (S + b[k_ + 1])
            tot += a[k_] * S
        return tot + (a * b).sum()

    def form_factor_ee(self, q_nm):
        """F(q) = int int psi^2(z) psi^2(z') exp(-q |z - z'|) (the screening
        form factor; 1 for a sheet of zero thickness)."""
        q = np.ascontiguousarray(np.atleast_1d(np.asarray(q_nm, float)))
        key = hash(q.tobytes())
        if key not in self._Fee_cache:
            pw = self.p * self.w
            self._Fee_cache = {key: self._pair_sum(q, pw, pw)}
        return self._Fee_cache[key]

    def form_factor_charge(self, q_nm, z0):
        """int psi^2(z) exp(-q |z - z0|) dz for a charge at z0 (nm)."""
        q = np.atleast_1d(np.asarray(q_nm, float))
        d = np.abs(self.z - float(z0))
        pw = self.p * self.w
        out = np.empty(q.size)
        for a in range(0, q.size, 2048):
            out[a:a + 2048] = np.exp(-q[a:a + 2048, None] * d[None, :]) @ pw
        return out

    def _charge_ff_nodes(self, q_nm):
        """F(q, z_j) for a charge on every node j: (nq, Nz)."""
        z = self.z
        pw = self.p * self.w
        q = np.atleast_1d(q_nm)
        F = np.empty((q.size, z.size))
        S = np.zeros(q.size)
        for k_ in range(z.size):
            if k_:
                S = np.exp(-q * (z[k_] - z[k_ - 1])) * (S + pw[k_ - 1])
            F[:, k_] = S + pw[k_]
        S = np.zeros(q.size)
        for k_ in range(z.size - 1, -1, -1):
            if k_ < z.size - 1:
                S = np.exp(-q * (z[k_ + 1] - z[k_])) * (S + pw[k_ + 1])
            F[:, k_] += S
        return F

    def _eps_q(self, q):
        """Dielectric function of the screening gas at q (1/m)."""
        if not self.screening:
            return np.ones_like(q)
        return 1.0 + self.q_s / q * self.form_factor_ee(q * 1e-9)

    def _angles(self, E):
        """theta, weights (2 x the integral over (0, pi)) and q (1/m) per
        energy: arrays (nE, n_theta)."""
        kk = self.k(E)
        th = self._th[None, :]
        q = 2 * kk[:, None] * np.sin(th / 2)
        return th, 2 * self._wth[None, :], q

    def _elastic(self, E, W, quantum):
        """1/tau from |V_q|^2 (W, J^2 m^2: |V_q|^2 times the scatterers per area) on the
        angles: m / (2 pi hbar^3) int dtheta W (1 - cos theta)."""
        th, wt, _ = self._angles(E)
        ang = 1.0 if quantum else (1.0 - np.cos(th))
        return self.m / (2 * math.pi * HBAR ** 3) * (W * ang * wt).sum(1)

    # ------------------------------------------------------- mechanisms
    def acoustic(self, deformation_eV, density_kg_m3, sound_velocity_m_s):
        """Longitudinal acoustic phonons through the deformation potential,
        in the equipartition (elastic, high-temperature) limit:
        1/tau = m D^2 kT int psi^4 / (hbar^3 rho v^2), the same for every
        energy and for the transport and quantum rates (P. J. Price 1981).
        Valid while kT is well above the phonon energy hbar v 2 k_F; at
        lower temperatures (the Bloch-Gruneisen regime) it overestimates
        the rate."""
        if not (density_kg_m3 > 0 and sound_velocity_m_s > 0):
            raise ValueError("density and sound velocity must be > 0")
        r = (self.m * (deformation_eV * QE) ** 2 * self.kT * QE * self.overlap() * 1e9
             / (HBAR ** 3 * density_kg_m3 * sound_velocity_m_s ** 2))

        def rate(E, quantum=False):
            return np.full(np.shape(np.atleast_1d(E)), r)
        return rate

    def alloy(self, x, potential_eV, cell_volume_nm3, mask=None):
        """Alloy disorder in a random alloy A_x B_(1-x) on the nodes in mask
        (the alloy layers; default all): 1/tau = m Omega V^2 x (1 - x)
        int_alloy psi^4 / hbar^3 (G. Bastard 1983), with Omega the volume
        per alloy site and V the alloy scattering potential (yours: the
        values in the literature differ by factors of two and more)."""
        if not (0.0 <= x <= 1.0 and cell_volume_nm3 > 0):
            raise ValueError("0 <= x <= 1 and cell_volume_nm3 > 0")
        r = (self.m * cell_volume_nm3 * 1e-27 * (potential_eV * QE) ** 2 * x * (1 - x)
             * self.overlap(mask) * 1e9 / HBAR ** 3)

        def rate(E, quantum=False):
            return np.full(np.shape(np.atleast_1d(E)), r)
        return rate

    def impurities(self, z0=None, sheet_cm2=None, profile_cm3=None, charge=1.0):
        """Ionized impurities (charge in units of e), screened Coulomb
        scattering: a sheet of sheet_cm2 at z0 (nm), or a density profile
        profile_cm3 on the nodes z (zeros where there are none; extend z
        and psi2 with zeros to reach impurities outside the gas). The
        in-plane transform of one charge is Z e^2 F(q, z0) / (2 eps q
        eps(q)), F(q, z0) = int psi^2 exp(-q |z - z0|)."""
        if (sheet_cm2 is None) == (profile_cm3 is None):
            raise ValueError("give sheet_cm2 (with z0) or profile_cm3")
        if sheet_cm2 is not None:
            if z0 is None or not sheet_cm2 >= 0:
                raise ValueError("a sheet needs z0 and sheet_cm2 >= 0")
            items = [(float(z0), sheet_cm2 * 1e4)]
        else:
            prof = np.asarray(profile_cm3, float)
            if prof.shape != self.z.shape or np.any(prof < 0):
                raise ValueError("profile_cm3 must match z and be >= 0")
            items = None
            dens = prof * 1e6 * self.w * 1e-9                    # per m^2 at each node
        pre = (charge * QE ** 2 / (2 * self.eps)) ** 2

        def rate(E, quantum=False):
            E = np.atleast_1d(np.asarray(E, float))
            _, _, q = self._angles(E)
            qn = q.ravel() * 1e-9
            if items is not None:
                F2 = sum(Ns * self.form_factor_charge(qn, zz) ** 2 for zz, Ns in items)
            else:
                # sum_j N_j F(q, z_j)^2 with F(q, z_j) = sum_i p_i w_i e^(-q|z_i-z_j|)
                F2 = np.empty(qn.size)
                for a in range(0, qn.size, 2048):
                    F2[a:a + 2048] = self._charge_ff_nodes(qn[a:a + 2048]) ** 2 @ dens
            F2 = F2.reshape(q.shape)
            qq = np.maximum(q, 1e-30)
            W = pre * F2 / (qq * self._eps_q(qq.ravel()).reshape(q.shape)) ** 2
            return self._elastic(E, W, quantum)
        return rate

    def roughness(self, z_interface, step_eV, rms_nm, correlation_nm,
                  correlation="gaussian"):
        """Roughness of one interface at z_interface (nm), where the band
        steps by step_eV: a shift Delta of the interface moves the subband
        by step |psi(z_i)|^2 Delta (first order, exact for the box method on
        the nodes), so |V_q|^2 = (step psi^2(z_i))^2 |Delta_q|^2 / eps(q)^2,
        with |Delta_q|^2 = pi Delta^2 L^2 exp(-q^2 L^2 / 4) (gaussian) or
        2 pi Delta^2 L^2 / (1 + q^2 L^2)^(3/2) (exponential); Delta =
        rms_nm, L = correlation_nm. psi^2 at z_interface is interpolated
        between the nodes. A wave function that vanishes at the interface
        (an infinite barrier, as in the Fang-Howard form) gives no rate in
        this model: use the one with the real, finite barrier."""
        if not (rms_nm >= 0 and correlation_nm > 0):
            raise ValueError("rms_nm >= 0 and correlation_nm > 0")
        if correlation not in ("gaussian", "exponential"):
            raise ValueError('correlation must be "gaussian" or "exponential"')
        pz = float(np.interp(z_interface, self.z, self.p)) * 1e9    # 1/m
        M2 = (step_eV * QE * pz) ** 2
        D, L = rms_nm * 1e-9, correlation_nm * 1e-9

        def rate(E, quantum=False):
            E = np.atleast_1d(np.asarray(E, float))
            _, _, q = self._angles(E)
            if correlation == "gaussian":
                S = math.pi * D * D * L * L * np.exp(-(q * L) ** 2 / 4)
            else:
                S = 2 * math.pi * D * D * L * L / (1 + (q * L) ** 2) ** 1.5
            qq = np.maximum(q, 1e-30)
            W = M2 * S / self._eps_q(qq.ravel()).reshape(q.shape) ** 2
            return self._elastic(E, W, quantum)
        return rate

    def polar_optical(self, phonon_eV, eps_static, eps_high, temperature_K=None,
                      screening=False):
        """Polar optical (LO) phonons through the Froehlich coupling, bulk
        phonons of one energy phonon_eV (hbar w_LO) with the subband's own
        form factor: |M_q|^2 = e^2 hbar w H(q) / (4 A eps_p q) times N or
        N + 1, 1/eps_p = (1/eps_high - 1/eps_static) / eps_0, H(q) the
        form factor of `form_factor_ee` (B. K. Ridley, Quantum Processes in
        Semiconductors; P. J. Price, Ann. Phys. (1981)). The phonons are in
        equilibrium at temperature_K (default the subband's).

        Inelastic: it has no momentum relaxation time, so `mobility` solves
        the linearized Boltzmann equation exactly on an energy grid in
        steps of a fraction of hbar w (the method of D. L. Rode, Phys. Rev.
        B 2, 1012 (1970), in 2D, with the Pauli factors of a degenerate
        gas). rate(E, quantum=True) is the out-scattering rate of a state
        at E (emission and absorption, with the final states' occupancy);
        rate(E) with quantum=False refuses. screening=True divides |M_q|^2
        by the static eps(q)^2 of the gas (an upper bound on screening,
        which at the phonon frequency is weaker); False (default) leaves
        it unscreened."""
        if not (phonon_eV > 0 and eps_static > eps_high > 0):
            raise ValueError("phonon_eV > 0 and eps_static > eps_high > 0")
        T = self.T if temperature_K is None else float(temperature_K)
        return _Polar(self, float(phonon_eV), 1.0 / (EPS0 * eps_high) - 1.0 / (EPS0 * eps_static),
                      1.0 / math.expm1(float(phonon_eV) / (KB * T)), bool(screening))

    # --------------------------------------------------------- results
    def lifetimes(self, *mechanisms, energy=None):
        """Transport and quantum lifetimes (s) at one kinetic energy (eV;
        default the Fermi energy, or kT for a non-degenerate gas), all
        mechanisms together (Matthiessen's rule). With polar optical
        phonons among them the transport lifetime is None (an inelastic
        process has none; `mobility` handles it exactly)."""
        E = np.atleast_1d(self.fermi if energy is None else energy)
        if energy is None and self.fermi <= 0:
            E = np.atleast_1d(self.kT)
        rq = sum(np.asarray(f(E, quantum=True), float) for f in mechanisms)
        if any(isinstance(f, _Polar) for f in mechanisms):
            tt = None                  # inelastic: no transport time (see mobility)
        else:
            rt = sum(np.asarray(f(E, quantum=False), float) for f in mechanisms)
            tt = float(1 / rt[0])
        return dict(transport_s=tt, quantum_s=float(1 / rq[0]), energy_eV=float(E[0]))

    def mobility(self, *mechanisms, n_energy=160):
        """Mobility (cm^2/Vs) of the subband: mu = q <tau> / m with
        <tau> = int tau(E) E (-df/dE) dE / int f dE (the 2D Boltzmann
        average), the rates of all mechanisms added at each energy. Returns
        dict(mobility_cm2Vs, tau_s, each: the mobility of each mechanism
        alone, in the order given). With polar optical phonons among the
        mechanisms, the linearized Boltzmann equation is solved exactly
        instead (see `polar_optical`); tau_s is then m mu / q, and
        n_energy sets the grid steps per phonon energy (default 160 means
        an automatic choice)."""
        if not mechanisms:
            raise ValueError("give at least one mechanism")
        if any(isinstance(f, _Polar) for f in mechanisms):
            return self._mobility_bte(mechanisms, n_energy)
        top = max(self.fermi, 0.0) + 40 * self.kT
        u, wu = np.polynomial.legendre.leggauss(int(n_energy))
        u = 0.5 * (u + 1.0) * math.sqrt(top)
        wu = 0.5 * wu * math.sqrt(top)
        E = u * u                                              # crowd small E
        wE = wu * 2 * u
        x = (E - self.fermi) / self.kT
        mdf = 0.25 / self.kT / np.cosh(0.5 * np.clip(x, -700, 700)) ** 2      # -df/dE
        nf = self.kT * float(np.logaddexp(0.0, self.fermi / self.kT))         # int f dE
        rates = [np.asarray(f(E, quantum=False), float) for f in mechanisms]

        def avg(r):
            with np.errstate(divide="ignore"):
                return float((wE * E * mdf / r).sum() / nf)
        tau = avg(sum(rates))
        each = [QE * avg(r) / self.m * 1e4 for r in rates]
        return dict(mobility_cm2Vs=QE * tau / self.m * 1e4, tau_s=tau, each=each)

    def _mobility_bte(self, mechanisms, n_energy):
        """Exact solution of the linearized 2D Boltzmann equation with one
        polar optical phonon and any elastic mechanisms (Rode's method):
        f = f0 + g(E) cos(phi) per unit field, on energies (j + 1/2) dE with
        hbar w a whole number of steps, so emission and absorption join
        grid points; a sparse linear solve, no iteration."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        pol = [f for f in mechanisms if isinstance(f, _Polar)]
        if len({p.hw for p in pol}) != 1:
            raise ValueError("one phonon energy at a time (add the couplings of modes "
                             "with the same energy into one polar_optical)")
        hw = pol[0].hw
        top = max(self.fermi, 0.0) + 40 * self.kT
        # steps of at most kT / 10 and hbar w / 4
        n_sub = max(4, int(math.ceil(hw / (self.kT / 10))))
        if n_energy != 160:
            n_sub = int(n_energy)
        dE = hw / n_sub
        nE = int(math.ceil(top / dE))
        E = (np.arange(nE) + 0.5) * dE
        kT = self.kT
        f0 = 0.5 * (1 - np.tanh(0.5 * (E - self.fermi) / kT))
        mdf = 0.25 / kT / np.cosh(0.5 * np.clip((E - self.fermi) / kT, -700, 700)) ** 2
        v = np.sqrt(2 * E * QE / self.m)
        elastic = [f for f in mechanisms if not isinstance(f, _Polar)]

        def solve(use_pol, use_el):
            nu = np.zeros(nE)
            for f in use_el:
                nu += np.asarray(f(E, quantum=False), float)
            diag = nu.copy()
            rows, cols, vals = [], [], []
            for p in use_pol:
                So, cm, cp = p._operator(E, f0, n_sub)
                diag += So
                j = np.arange(n_sub, nE)
                rows += j.tolist()
                cols += (j - n_sub).tolist()
                vals += (-cm[j]).tolist()
                j = np.arange(0, nE - n_sub)
                rows += j.tolist()
                cols += (j + n_sub).tolist()
                vals += (-cp[j]).tolist()
            M = sp.csr_matrix((vals, (rows, cols)), shape=(nE, nE)) + sp.diags(diag)
            g = spl.spsolve(M.tocsc(), v * mdf)
            nf = kT * float(np.logaddexp(0.0, self.fermi / kT))         # int f dE
            # mu = int v g dE / (2 int f dE): g = tau v (-df/dE) gives q <tau> / m
            return float((v * g).sum() * dE) / (2 * nf) * 1e4          # cm^2/Vs
        mu = solve(pol, elastic)
        each = []
        for f in mechanisms:
            if isinstance(f, _Polar):
                each.append(solve([f], []))
            else:
                each.append(self.mobility(f)["mobility_cm2Vs"])
        return dict(mobility_cm2Vs=mu, tau_s=mu * 1e-4 * self.m / QE, each=each,
                    energy_step_eV=dE)


class _Polar:
    """Polar optical phonon scattering of a `Subband` (see
    `Subband.polar_optical`)."""

    def __init__(self, sb, hw, inv_eps_p, N, screening):
        self.sb, self.hw, self.N, self.screening = sb, hw, N, screening
        self.C = sb.m * QE ** 2 * (hw * QE / HBAR) / (8 * math.pi * HBAR ** 2) * inv_eps_p
        # H(q) is smooth in log q: a cubic spline on 1200 points from
        # 1e-6 to 1e3 1/nm (checked against the direct sum to 1e-9)
        from scipy.interpolate import CubicSpline
        lq = np.linspace(math.log(1e-6), math.log(1e3), 1200)
        self._H = CubicSpline(lq, sb.form_factor_ee(np.exp(lq)))

    def H(self, q_nm):
        q = np.asarray(q_nm, float)
        return self._H(np.log(np.clip(q, 1e-6, 1e3)))

    def angular(self, k, kp):
        """int_0^2pi dtheta H(q) s(q) / q and the same with cos(theta), for
        wave numbers k and k' (1/m, arrays); q = |k - k'|."""
        sb = self.sb
        k = np.asarray(k, float)[:, None]
        kp = np.asarray(kp, float)[:, None]
        # theta = t0 sinh(u): 1/q is sharp near theta = 0 on the scale
        # t0 = |k - k'| / sqrt(k k'); in u it is smooth
        t0 = np.clip(np.abs(k - kp) / np.sqrt(np.maximum(k * kp, 1e-300)), 1e-12, math.pi)
        x, wx = np.polynomial.legendre.leggauss(sb._th.size)
        umax = np.arcsinh(math.pi / t0)
        u = 0.5 * (x[None, :] + 1.0) * umax
        th = t0 * np.sinh(u)
        wt = 2 * (0.5 * wx[None, :] * umax) * t0 * np.cosh(u)          # 2 x int_0^pi
        q = np.sqrt(np.maximum((k - kp) ** 2 + 4 * k * kp * np.sin(th / 2) ** 2, 1e-300))
        H = self.H(q * 1e-9)
        if self.screening:
            H = H / sb._eps_q(q.ravel()).reshape(q.shape) ** 2
        base = H / q * wt
        return base.sum(1), (base * np.cos(th)).sum(1)

    def _sums(self, E):
        sb = self.sb
        k = sb.k(E)
        I0a, I1a = self.angular(k, sb.k(E + self.hw))
        em = E > self.hw
        I0e = np.zeros(E.size)
        I1e = np.zeros(E.size)
        if em.any():
            I0e[em], I1e[em] = self.angular(k[em], sb.k(E[em] - self.hw))
        return I0a, I1a, I0e, I1e, em

    def _f0(self, E):
        sb = self.sb
        return 0.5 * (1 - np.tanh(0.5 * (E - sb.fermi) / sb.kT))

    def __call__(self, E, quantum=False):
        if not quantum:
            raise ValueError("polar optical phonons are inelastic: there is no transport "
                             "rate; use Subband.mobility, which solves the Boltzmann "
                             "equation, or quantum=True for the out-scattering rate")
        E = np.atleast_1d(np.asarray(E, float))
        I0a, _, I0e, _, em = self._sums(E)
        N = self.N
        fa, fe = self._f0(E + self.hw), self._f0(E - self.hw)
        return self.C * (I0a * ((N + 1) * fa + N * (1 - fa))
                         + np.where(em, I0e * (N * fe + (N + 1) * (1 - fe)), 0.0))

    def _operator(self, E, f0, n_sub):
        """Out-scattering rate and the in-scattering coefficients of g at
        E - hbar w (cm) and E + hbar w (cp), on the grid E."""
        I0a, I1a, I0e, I1e, em = self._sums(E)
        N, C = self.N, self.C
        fa, fe = self._f0(E + self.hw), self._f0(E - self.hw)
        So = C * (I0a * ((N + 1) * fa + N * (1 - fa))
                  + np.where(em, I0e * (N * fe + (N + 1) * (1 - fe)), 0.0))
        cm = C * I1e * (N * (1 - f0) + (N + 1) * f0)
        cp = C * I1a * ((N + 1) * (1 - f0) + N * f0)
        return So, cm, cp

