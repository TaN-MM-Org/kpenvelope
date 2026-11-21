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



# ------------------------------------------------------------ many subbands
def _pair(z, q, a, b):
    """sum_ij a_i b_j exp(-q |z_i - z_j|) for every q (1/nm), by the two
    one-sided recursions."""
    q = np.atleast_1d(np.asarray(q, float))
    tot = np.zeros(q.size)
    S = np.zeros(q.size)
    for k_ in range(z.size):
        if k_:
            S = np.exp(-q * (z[k_] - z[k_ - 1])) * (S + b[k_ - 1])
        tot += a[k_] * S
    S = np.zeros(q.size)
    for k_ in range(z.size - 1, -1, -1):
        if k_ < z.size - 1:
            S = np.exp(-q * (z[k_ + 1] - z[k_])) * (S + b[k_ + 1])
        tot += a[k_] * S
    return tot + (a * b).sum()


class Subbands:
    """Several subbands of a 2D electron (or hole) gas, with scattering
    within and between them (new in 0.29.0).

    z : nodes (nm). psi : (len(z), n) the envelopes themselves (signed,
    1/sqrt(nm)), orthonormal with the weights w (default trapezoid):
    sum(w psi_i psi_j) = delta_ij, as `column_states(..., signed=True)`
    returns them. energies : the subband bottoms (eV), on the same scale as
    fermi. mass : the in-plane mass (m0, one for all). eps_r : background
    permittivity. Give fermi (eV) or sheet_cm2 (all subbands together).
    screening : Thomas-Fermi screening by every subband's own charge with
    its own form factor, eps(q) = 1 + sum_i q_s,i F_ii(q) / q (the
    polarization between subbands is left out).

    The mechanisms are those of `Subband`, now with form factors between
    subbands i and j (psi_i psi_j in place of psi^2), so an electron can
    leave its subband. `mobility` solves the linearized Boltzmann equation
    of all subbands together exactly: at each energy for the elastic
    mechanisms (S. D. Siggia and P. C. Kwok, Phys. Rev. B 2, 1024 (1970)),
    and on an energy grid in steps of a fraction of hbar w when polar
    optical phonons are among them (Rode's method)."""

    def __init__(self, z, psi, energies, mass, eps_r, fermi=None, sheet_cm2=None,
                 temperature_K=300.0, weights=None, screening=True, n_theta=200):
        z = np.asarray(z, float)
        psi = np.asarray(psi, float)
        if psi.ndim == 1:
            psi = psi[:, None]
        if z.ndim != 1 or z.size < 3 or psi.shape[0] != z.size or np.any(np.diff(z) <= 0):
            raise ValueError("z must increase (at least 3 nodes) and psi be (len(z), n)")
        Es = np.atleast_1d(np.asarray(energies, float))
        if Es.size != psi.shape[1]:
            raise ValueError("one energy per subband")
        if not (mass > 0 and eps_r > 0 and temperature_K > 0):
            raise ValueError("mass, eps_r and temperature_K must be > 0")
        if weights is None:
            h = np.diff(z)
            w = np.zeros(z.size)
            w[:-1] += 0.5 * h
            w[1:] += 0.5 * h
        else:
            w = np.asarray(weights, float)
            if w.shape != z.shape or np.any(w < 0):
                raise ValueError("weights must match z and be >= 0")
        G = psi.T @ (w[:, None] * psi)
        if np.max(np.abs(G - np.eye(Es.size))) > 1e-3:
            raise ValueError("psi must be orthonormal: sum(w psi_i psi_j) = delta_ij")
        self.z, self.psi, self.w, self.E = z, psi, w, Es
        self.n = Es.size
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
            target = sheet_cm2 * 1e4
            lo, hi = float(Es.min()) - 60 * self.kT, float(Es.max()) + target / self.dos + 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self._sheets(mid).sum() < target:
                    lo = mid
                else:
                    hi = mid
            fermi = 0.5 * (lo + hi)
        self.fermi = float(fermi)
        self.sheets = self._sheets(self.fermi)                    # 1/m^2 per subband
        self.sheet = float(self.sheets.sum())
        self.screening = bool(screening)
        x = (self.fermi - Es) / self.kT
        dn = self.dos / (1.0 + np.exp(-np.clip(x, -700, 700)))
        self.q_s = QE ** 2 / (2 * self.eps) * dn / QE              # 1/m per subband
        self.n_theta = int(n_theta)
        self._spl = {}

    def _sheets(self, F):
        return self.dos * self.kT * np.logaddexp(0.0, (F - self.E) / self.kT)

    # ----------------------------------------------------------- pieces
    def k(self, i, E):
        """Wave number (1/m) in subband i at total energy E (eV); 0 below."""
        return np.sqrt(2 * self.m * np.maximum(np.asarray(E, float) - self.E[i], 0.0)
                       * QE) / HBAR

    def overlap(self, i, j, mask=None):
        """int psi_i^2 psi_j^2 (1/nm), over the nodes in mask."""
        v = self.w * self.psi[:, i] ** 2 * self.psi[:, j] ** 2
        return float(v.sum() if mask is None else v[np.asarray(mask, bool)].sum())

    def form_factor(self, i, j, q_nm):
        """H_ij(q) = int int psi_i psi_j(z) psi_i psi_j(z') exp(-q |z - z'|)
        (1 at q = 0 for i = j, 0 for i != j), from a spline in log q."""
        key = (min(i, j), max(i, j))
        if key not in self._spl:
            from scipy.interpolate import CubicSpline
            lq = np.linspace(math.log(1e-6), math.log(1e3), 1200)
            a = self.w * self.psi[:, i] * self.psi[:, j]
            self._spl[key] = CubicSpline(lq, _pair(self.z, np.exp(lq), a, a))
        return self._spl[key](np.log(np.clip(np.asarray(q_nm, float), 1e-6, 1e3)))

    def _eps_q(self, q):
        """Static dielectric function at q (1/m)."""
        if not self.screening:
            return np.ones_like(q)
        e = np.ones_like(q)
        for i in range(self.n):
            if self.q_s[i] > 0:
                e = e + self.q_s[i] / np.maximum(q, 1e-30) * self.form_factor(i, i, q * 1e-9)
        return e

    def _angles(self, ki, kj):
        """theta nodes and weights (2 x int_0^pi) per pair of wave numbers,
        crowded where |k_i - k_j e^(i theta)| is smallest; q (1/m)."""
        ki = np.asarray(ki, float)[:, None]
        kj = np.asarray(kj, float)[:, None]
        t0 = np.clip(np.abs(ki - kj) / np.sqrt(np.maximum(ki * kj, 1e-300)), 1e-6, math.pi)
        x, wx = np.polynomial.legendre.leggauss(self.n_theta)
        umax = np.arcsinh(math.pi / t0)
        u = 0.5 * (x[None, :] + 1.0) * umax
        th = t0 * np.sinh(u)
        wt = 2 * (0.5 * wx[None, :] * umax) * t0 * np.cosh(u)
        q = np.sqrt(np.maximum((ki - kj) ** 2 + 4 * ki * kj * np.sin(th / 2) ** 2, 1e-300))
        return th, wt, q

    def _elastic_pair(self, i, j, E, W):
        """m / (2 pi hbar^3) int dtheta W(q) and the same with cos theta,
        from subband i at energy E to j (0 where j is closed); W(q, i, j)
        in J^2 m^2."""
        E = np.atleast_1d(np.asarray(E, float))
        nu0 = np.zeros(E.size)
        nu1 = np.zeros(E.size)
        op = (E > self.E[i]) & (E > self.E[j])
        if op.any():
            th, wt, q = self._angles(self.k(i, E[op]), self.k(j, E[op]))
            Wq = W(q, i, j)
            pre = self.m / (2 * math.pi * HBAR ** 3)
            nu0[op] = pre * (Wq * wt).sum(1)
            nu1[op] = pre * (Wq * np.cos(th) * wt).sum(1)
        return nu0, nu1

    # ------------------------------------------------------- mechanisms
    def acoustic(self, deformation_eV, density_kg_m3, sound_velocity_m_s):
        """Acoustic phonons (deformation potential, equipartition): the rate
        from i to j is m D^2 kT int psi_i^2 psi_j^2 / (hbar^3 rho v^2) where
        j is open, the same in every direction."""
        if not (density_kg_m3 > 0 and sound_velocity_m_s > 0):
            raise ValueError("density and sound velocity must be > 0")
        c = (self.m * (deformation_eV * QE) ** 2 * self.kT * QE * 1e9
             / (HBAR ** 3 * density_kg_m3 * sound_velocity_m_s ** 2))
        O = np.array([[self.overlap(i, j) for j in range(self.n)] for i in range(self.n)])
        return _Isotropic(self, c * O)

    def alloy(self, x, potential_eV, cell_volume_nm3, mask=None):
        """Alloy disorder on the nodes in mask: the rate from i to j is
        m Omega V^2 x (1 - x) int_alloy psi_i^2 psi_j^2 / hbar^3."""
        if not (0.0 <= x <= 1.0 and cell_volume_nm3 > 0):
            raise ValueError("0 <= x <= 1 and cell_volume_nm3 > 0")
        c = (self.m * cell_volume_nm3 * 1e-27 * (potential_eV * QE) ** 2 * x * (1 - x) * 1e9
             / HBAR ** 3)
        O = np.array([[self.overlap(i, j, mask) for j in range(self.n)] for i in range(self.n)])
        return _Isotropic(self, c * O)

    def impurities(self, z0=None, sheet_cm2=None, profile_cm3=None, charge=1.0):
        """Ionized impurities (a sheet at z0, or a profile on the nodes), the
        transform between subbands Z e^2 F_ij(q, z0) / (2 eps q eps(q)),
        F_ij = int psi_i psi_j exp(-q |z - z0|)."""
        if (sheet_cm2 is None) == (profile_cm3 is None):
            raise ValueError("give sheet_cm2 (with z0) or profile_cm3")
        if sheet_cm2 is not None:
            if z0 is None or not sheet_cm2 >= 0:
                raise ValueError("a sheet needs z0 and sheet_cm2 >= 0")
            pos, dens = np.array([float(z0)]), np.array([sheet_cm2 * 1e4])
        else:
            prof = np.asarray(profile_cm3, float)
            if prof.shape != self.z.shape or np.any(prof < 0):
                raise ValueError("profile_cm3 must match z and be >= 0")
            sel = prof > 0
            pos, dens = self.z[sel], (prof * 1e6 * self.w * 1e-9)[sel]
        pre = (charge * QE ** 2 / (2 * self.eps)) ** 2
        spl = {}

        def F2_of(i, j):
            """sum over the charges of F_ij(q, z0)^2: a cubic spline in log q
            (1e-6 to 1e3 1/nm), F_ij at every node by the two one-sided
            recursions"""
            key = (min(i, j), max(i, j))
            if key not in spl:
                from scipy.interpolate import CubicSpline
                lq = np.linspace(math.log(1e-6), math.log(1e3), 900)
                q = np.exp(lq)
                a = self.w * self.psi[:, i] * self.psi[:, j]
                zz = np.concatenate([self.z, pos])
                order = np.argsort(zz, kind="stable")
                zs = zz[order]
                aa = np.concatenate([a, np.zeros(pos.size)])[order]
                Fz = np.empty((q.size, zs.size))
                S = np.zeros(q.size)
                for k_ in range(zs.size):
                    if k_:
                        S = np.exp(-q * (zs[k_] - zs[k_ - 1])) * (S + aa[k_ - 1])
                    Fz[:, k_] = S + aa[k_]
                S = np.zeros(q.size)
                for k_ in range(zs.size - 1, -1, -1):
                    if k_ < zs.size - 1:
                        S = np.exp(-q * (zs[k_ + 1] - zs[k_])) * (S + aa[k_ + 1])
                    Fz[:, k_] += S
                at = np.argsort(order)[self.z.size:]           # where the charges went
                spl[key] = CubicSpline(lq, (Fz[:, at] ** 2) @ dens)
            return spl[key]

        def W(q, i, j):
            F2 = F2_of(i, j)(np.log(np.clip(q * 1e-9, 1e-6, 1e3)))
            return pre * F2 / (q * self._eps_q(q)) ** 2
        return _Elastic(self, W)

    def roughness(self, z_interface, step_eV, rms_nm, correlation_nm, correlation="gaussian"):
        """Roughness of one interface (see `Subband.roughness`), the matrix
        element between subbands step psi_i(z_i) psi_j(z_i) Delta."""
        if not (rms_nm >= 0 and correlation_nm > 0):
            raise ValueError("rms_nm >= 0 and correlation_nm > 0")
        if correlation not in ("gaussian", "exponential"):
            raise ValueError('correlation must be "gaussian" or "exponential"')
        pz = np.array([np.interp(z_interface, self.z, self.psi[:, i]) for i in range(self.n)])
        D, L = rms_nm * 1e-9, correlation_nm * 1e-9

        def W(q, i, j):
            M2 = (step_eV * QE * pz[i] * pz[j] * 1e9) ** 2
            if correlation == "gaussian":
                S = math.pi * D * D * L * L * np.exp(-(q * L) ** 2 / 4)
            else:
                S = 2 * math.pi * D * D * L * L / (1 + (q * L) ** 2) ** 1.5
            return M2 * S / self._eps_q(q) ** 2
        return _Elastic(self, W)

    def polar_optical(self, phonon_eV, eps_static, eps_high, temperature_K=None,
                      screening=False):
        """Polar optical phonons (see `Subband.polar_optical`) within and
        between subbands, with the form factors H_ij(q)."""
        if not (phonon_eV > 0 and eps_static > eps_high > 0):
            raise ValueError("phonon_eV > 0 and eps_static > eps_high > 0")
        T = self.T if temperature_K is None else float(temperature_K)
        return _PolarMany(self, float(phonon_eV),
                          1.0 / (EPS0 * eps_high) - 1.0 / (EPS0 * eps_static),
                          1.0 / math.expm1(float(phonon_eV) / (KB * T)), bool(screening))

    # --------------------------------------------------------- results
    def _elastic_matrices(self, mechanisms, E):
        """nu0, nu1 (n, n, nE): rates from i to j at total energy E."""
        n = self.n
        nu0 = np.zeros((n, n, E.size))
        nu1 = np.zeros((n, n, E.size))
        for f in mechanisms:
            a, b = f.kernels(E)
            nu0 += a
            nu1 += b
        return nu0, nu1

    def lifetimes(self, *mechanisms, energy=None):
        """Per subband, at one total energy (default the Fermi level, or kT
        above the lowest bottom for a non-degenerate gas): the quantum
        lifetime (1 / total out-scattering rate) and, if every mechanism is
        elastic, the transport lifetime of the exact solution at that
        energy (the momentum each subband's electrons carry decays with
        it; None for closed subbands)."""
        E = self.fermi if energy is None else float(energy)
        if energy is None and self.fermi <= self.E.min():
            E = float(self.E.min()) + self.kT
        Ea = np.array([E])
        el = [f for f in mechanisms if not isinstance(f, _PolarMany)]
        nu0, nu1 = self._elastic_matrices(el, Ea)
        out = nu0[:, :, 0].sum(1)
        for f in mechanisms:
            if isinstance(f, _PolarMany):
                out = out + f.out_rate(Ea)[:, 0]
        op = E > self.E
        tq = [float(1 / out[i]) if op[i] and out[i] > 0 else None for i in range(self.n)]
        if len(el) < len(mechanisms):
            tt = None
        else:
            tau = self._tau_elastic(nu0[:, :, 0], nu1[:, :, 0], Ea[0])
            tt = [float(tau[i]) if op[i] else None for i in range(self.n)]
        return dict(quantum_s=tq, transport_s=tt, energy_eV=E)

    def _tau_elastic(self, nu0, nu1, E):
        """Exact transport times at one energy (Siggia-Kwok): v_i = sum_j
        (nu0_ij tau_i v_i - nu1_ij tau_j v_j), open subbands only."""
        op = np.flatnonzero(E > self.E)
        tau = np.full(self.n, np.nan)
        if op.size == 0:
            return tau
        v = np.array([self.k(i, E) for i in op])
        A = np.diag(nu0[np.ix_(op, op)].sum(1) * v) - nu1[np.ix_(op, op)] * v[None, :]
        tau[op] = np.linalg.solve(A, v)
        return tau

    def mobility(self, *mechanisms, n_energy=None):
        """Mobility (cm^2/Vs) of the gas: the linearized Boltzmann equation
        of all subbands solved together, with g_i(E) constant on cells of
        total energy (steps of kT / 20 for elastic mechanisms, or hbar w / n
        with steps of at most kT / 10 and hbar w / 4 with polar optical
        phonons; n_energy sets the steps per kT, or per hbar w). Each
        equation is the Boltzmann equation integrated over the part of its
        cell where the states it couples exist (a subband's bottom, or the
        threshold of emission into another, can fall inside a cell), so
        detailed balance holds on the grid and a single subband gets
        exactly the grid of `Subband`. Returns dict(mobility_cm2Vs, each
        (each mechanism alone; inf where it does not scatter), subbands
        (the mobility of each subband's electrons), sheets_cm2)."""
        if not mechanisms:
            raise ValueError("give at least one mechanism")
        pol = [f for f in mechanisms if isinstance(f, _PolarMany)]
        if len({p.hw for p in pol}) > 1:
            raise ValueError("one phonon energy at a time")
        if pol:
            hw = pol[0].hw
            n_sub = (max(4, int(math.ceil(hw / (self.kT / 10)))) if n_energy is None
                     else int(n_energy))
            dE = hw / n_sub
        else:
            hw, n_sub = 0.0, 0
            dE = self.kT / (20 if n_energy is None else float(n_energy))
        E0 = float(self.E.min())
        top = max(self.fermi, float(self.E.max())) + 40 * self.kT
        # cell edges: every subband bottom plus whole steps, so each bottom
        # and each threshold E_j + m hbar w is an edge; the pattern repeats
        # every step, and hbar w shifts cells by n_sub times its length
        offs = np.unique(np.round(np.mod(self.E - E0, dE) / dE, 12)) * dE
        offs = offs[offs < dE * (1 - 1e-12)]
        K = offs.size
        m_ = int(math.ceil((top - E0) / dE)) + 1
        edges = (E0 + (np.arange(m_)[:, None] * dE + offs[None, :])).ravel()
        lo, hi = edges[:-1], edges[1:]
        nE = lo.size
        shift = n_sub * K
        n = self.n
        kT, F = self.kT, self.fermi

        def f0(E):
            return 0.5 * (1 - np.tanh(0.5 * (E - F) / kT))

        def mdf(E):
            return 0.25 / kT / np.cosh(0.5 * np.clip((E - F) / kT, -700, 700)) ** 2

        tolE = 1e-9 * dE

        def region(t):
            """the cells above t (t is always an edge): their lengths (0
            below) and middles"""
            L = np.where(lo >= t - tolE, hi - lo, 0.0)
            return L, 0.5 * (lo + hi)
        own = [region(self.E[i]) for i in range(n)]
        op = np.array([own[i][0] > 0 for i in range(n)])
        idx = -np.ones((n, nE), int)
        idx[op] = np.arange(int(op.sum()))
        Nn = int(op.sum())
        vmid = [self.k(i, own[i][1]) * HBAR / self.m for i in range(n)]
        rhs = np.concatenate([(own[i][0] * vmid[i] * mdf(own[i][1]))[op[i]] for i in range(n)])
        rhs_full = np.zeros((n, nE))
        for i in range(n):
            rhs_full[i] = own[i][0] * vmid[i] * mdf(own[i][1])
        nf = kT * np.logaddexp(0.0, (F - self.E) / kT)                     # eV

        cache = {}

        def solve(mechs):
            import scipy.sparse as sp
            import scipy.sparse.linalg as spl
            diag = np.zeros((n, nE))
            rows, cols, vals = [], [], []
            for f in mechs:
                if id(f) not in cache:
                    cache[id(f)] = blocks(f)
                d_, r_, c_, v_ = cache[id(f)]
                diag += d_
                rows += r_
                cols += c_
                vals += v_
            if np.any(diag[op] <= 0):
                return math.inf, np.full(n, math.inf)
            M = sp.csr_matrix((vals, (rows, cols)), shape=(Nn, Nn)) + sp.diags(diag[op])
            # kept for the checks: the matrix and each unknown's cell middle
            self._last_system = (M, np.concatenate([(0.5 * (lo + hi))[op[i]]
                                                    for i in range(n)]))
            g = np.zeros((n, nE))
            g[op] = spl.spsolve(M.tocsc(), rhs)
            per = np.array([(own[i][0] * vmid[i] * g[i]).sum() / 2 for i in range(n)])
            return float(per.sum() / nf.sum()) * 1e4, per / np.maximum(nf, 1e-300) * 1e4

        def blocks(f):
            """the diagonal and the couplings one mechanism adds"""
            diag = np.zeros((n, nE))
            rows, cols, vals = [], [], []

            def put(i, j, shift, coef):
                c = np.arange(nE)
                ok = (coef != 0) & (c + shift >= 0) & (c + shift < nE)
                c = c[ok]
                ok2 = op[i][c] & op[j][c + shift]
                c = c[ok2]
                rows.extend(idx[i][c].tolist())
                cols.extend(idx[j][c + shift].tolist())
                vals.extend((-coef[ok][ok2]).tolist())
            if True:
                for i in range(n):
                    for j in range(n):
                        if isinstance(f, _PolarMany):
                            N, C = f.N, f.C
                            # emission to j at E - hw
                            L, Em = region(max(self.E[i], self.E[j] + f.hw))
                            if np.any(L > 0):
                                I0, I1 = f.pair(i, j, Em, Em - f.hw, L > 0)
                                fe, fm = f0(Em - f.hw), f0(Em)
                                diag[i] += L * C * I0 * (N * fe + (N + 1) * (1 - fe))
                                put(i, j, -shift, L * C * I1 * (N * (1 - fm) + (N + 1) * fm))
                            # absorption to j at E + hw
                            L, Em = region(max(self.E[i], self.E[j] - f.hw))
                            if np.any(L > 0):
                                I0, I1 = f.pair(i, j, Em, Em + f.hw, L > 0)
                                fa, fm = f0(Em + f.hw), f0(Em)
                                diag[i] += L * C * I0 * ((N + 1) * fa + N * (1 - fa))
                                put(i, j, shift, L * C * I1 * ((N + 1) * (1 - fm) + N * fm))
                        else:
                            L, Em = region(max(self.E[i], self.E[j]))
                            if np.any(L > 0):
                                a0, a1 = f.pair(i, j, Em)
                                diag[i] += L * a0
                                put(i, j, 0, L * a1)
            return diag, rows, cols, vals
        each = [solve([f])[0] for f in mechanisms]
        mu, mus = solve(mechanisms)
        return dict(mobility_cm2Vs=mu, each=each, subbands=mus,
                    sheets_cm2=self.sheets * 1e-4, energy_step_eV=dE)


class _Elastic:
    def __init__(self, sb, W):
        self.sb, self.W = sb, W

    def pair(self, i, j, E):
        """rates from i to j at the energies E: int dtheta W and with cos"""
        return self.sb._elastic_pair(i, j, E, self.W)

    def kernels(self, E):
        n = self.sb.n
        nu0 = np.zeros((n, n, E.size))
        nu1 = np.zeros((n, n, E.size))
        for i in range(n):
            for j in range(i, n):
                a, b = self.sb._elastic_pair(i, j, E, self.W)
                nu0[i, j], nu1[i, j] = a, b
                if j != i:
                    # the rate from j to i at the same energy: same |M|^2 and
                    # final density of states (one mass), so nu_ji = nu_ij
                    nu0[j, i], nu1[j, i] = a, b
        return nu0, nu1


class _Isotropic:
    def __init__(self, sb, C):
        self.sb, self.C = sb, np.asarray(C, float)

    def pair(self, i, j, E):
        E = np.atleast_1d(np.asarray(E, float))
        ok = (E > self.sb.E[i]) & (E > self.sb.E[j])
        return np.where(ok, self.C[i, j], 0.0), np.zeros(E.size)

    def kernels(self, E):
        n = self.sb.n
        op = np.array([E > self.sb.E[j] for j in range(n)])            # (n, nE)
        nu0 = self.C[:, :, None] * (op[None, :, :] & op[:, None, :])
        return nu0, np.zeros_like(nu0)


class _PolarMany:
    def __init__(self, sb, hw, inv_eps_p, N, screening):
        self.sb, self.hw, self.N, self.screening = sb, hw, N, screening
        self.C = sb.m * QE ** 2 * (hw * QE / HBAR) / (8 * math.pi * HBAR ** 2) * inv_eps_p

    def _I(self, i, j, ki, kj):
        sb = self.sb
        th, wt, q = sb._angles(ki, kj)
        H = sb.form_factor(i, j, q * 1e-9)
        if self.screening:
            H = H / sb._eps_q(q) ** 2
        base = H / q * wt
        return base.sum(1), (base * np.cos(th)).sum(1)

    def _f0(self, E):
        sb = self.sb
        return 0.5 * (1 - np.tanh(0.5 * (E - sb.fermi) / sb.kT))

    def pair(self, i, j, Ei, Ej, mask):
        """I0, I1 between subband i at Ei and j at Ej (where mask)"""
        I0 = np.zeros(Ei.size)
        I1 = np.zeros(Ei.size)
        ki, kj = self.sb.k(i, Ei), self.sb.k(j, Ej)
        ok = mask & (ki > 0) & (kj > 0)
        if ok.any():
            I0[ok], I1[ok] = self._I(i, j, ki[ok], kj[ok])
        return I0, I1

    def _blocks(self, E):
        """for every i, j: I0, I1 for absorption (to j at E + hw) and
        emission (to j at E - hw), zero where a state is missing."""
        sb, n = self.sb, self.sb.n
        out = {}
        for i in range(n):
            ki = sb.k(i, E)
            for j in range(n):
                for tag, Ef in (("a", E + self.hw), ("e", E - self.hw)):
                    kj = sb.k(j, Ef)
                    ok = (ki > 0) & (kj > 0)
                    I0 = np.zeros(E.size)
                    I1 = np.zeros(E.size)
                    if ok.any():
                        I0[ok], I1[ok] = self._I(i, j, ki[ok], kj[ok])
                    out[(i, j, tag)] = (I0, I1)
        return out

    def out_rate(self, E):
        """Out-scattering rate of subband i at total energy E: (n, nE)."""
        E = np.atleast_1d(np.asarray(E, float))
        B = self._blocks(E)
        N = self.N
        fa, fe = self._f0(E + self.hw), self._f0(E - self.hw)
        n = self.sb.n
        So = np.zeros((n, E.size))
        for i in range(n):
            for j in range(n):
                So[i] += self.C * (B[(i, j, "a")][0] * ((N + 1) * fa + N * (1 - fa))
                                   + B[(i, j, "e")][0] * (N * fe + (N + 1) * (1 - fe)))
        return So

    def operator(self, E, f0):
        """Out-scattering rate (n, nE) and the in-scattering coefficients of
        g_j at E - hw (cm) and E + hw (cp), (n, n, nE)."""
        B = self._blocks(E)
        N, C, n = self.N, self.C, self.sb.n
        fa, fe = self._f0(E + self.hw), self._f0(E - self.hw)
        So = np.zeros((n, E.size))
        cm = np.zeros((n, n, E.size))
        cp = np.zeros((n, n, E.size))
        for i in range(n):
            for j in range(n):
                I0a, I1a = B[(i, j, "a")]
                I0e, I1e = B[(i, j, "e")]
                So[i] += C * (I0a * ((N + 1) * fa + N * (1 - fa))
                              + I0e * (N * fe + (N + 1) * (1 - fe)))
                cm[i, j] = C * I1e * (N * (1 - f0) + (N + 1) * f0)
                cp[i, j] = C * I1a * ((N + 1) * (1 - f0) + N * f0)
        return So, cm, cp

    def __call__(self, E, quantum=False):
        raise ValueError("use Subbands.lifetimes or Subbands.mobility")
