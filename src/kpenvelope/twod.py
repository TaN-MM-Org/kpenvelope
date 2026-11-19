"""A two-dimensional cross-section of a transistor: gate, access regions
and the electron layer under and beside the gate (new in v0.18.0).

The layers run along x (they are uniform in x) and are stacked along z
(bottom first, as everywhere in the package). On top sit a metal gate
from x0 to x1 and, elsewhere, the free surface between the gate and the
source and drain. The model is the two-dimensional version of
`solve_quasi_equilibrium`:

    div (eps0 eps_r grad psi) = -q (N_d+ - n [+ p]) - (sheet charges),

with the electrons at the Fermi level of the source and drain (0 eV), as
in a C-V measurement or a transistor with no drain bias. The gate holds
E_c - E_F(metal) = barrier at its nodes, with E_F(metal) = -V_gate; the
free surface is pinned at E_c - E_F = surface barrier (surface states);
the bottom and the two sides have no field across them (a thick base,
and the far source and drain sides).

What it adds over one dimension: the edges of the gate, where the field
spreads sideways (fringing), so a short gate holds more charge per length
than the same layers in 1D and its electron layer empties at a different
voltage; and the electron density along x, from under the gate into the
access regions. What it does not do: current along the channel (no drain
bias, no transport), holes unless you ask, quantum electrons.

Discretization: finite volumes on a tensor-product grid that is fine near
the surface, the interfaces and the gate edges and coarse deep in the
buffer and far from the gate (`device_grid`), five-point fluxes with the
permittivity of each half cell, interface sheets on the interface nodes.
On a grid that is uniform in z and a gate that covers the whole width it
is exactly the one-dimensional solver (checked in the tests). Newton steps
are solved with SciPy's sparse LU when SciPy is installed, otherwise with
conjugate gradients (the Jacobian is symmetric negative definite).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .electrons import EPS0, HBAR, KB, M0, fermi_dirac_half
from .interfaces import StackLayer, interface_charges
from .nitride import QE

__all__ = ["Gate", "CrossSection", "device_grid", "solve_cross_section",
           "cv_cross_section"]

_CP = QE / EPS0 * 1e9          # V nm


@dataclasses.dataclass(frozen=True)
class Gate:
    """A metal gate on top from x0 to x1 (nm), with its barrier (eV) and
    voltage (V)."""

    x0: float
    x1: float
    barrier: float
    voltage: float = 0.0

    def __post_init__(self):
        for f in ("x0", "x1", "barrier", "voltage"):
            if not math.isfinite(getattr(self, f)):
                raise ValueError(f"Gate.{f} must be finite")
        if not self.x1 > self.x0:
            raise ValueError("the gate needs x1 > x0")

    @property
    def length(self):
        return self.x1 - self.x0


@dataclasses.dataclass
class CrossSection:
    """Solution of `solve_cross_section` (lengths nm, energies eV).

    x, z : grid. ec : conduction band, shape (len(x), len(z)). n : electrons
    (cm^-3), same shape. sheet_x : electrons per cm^2 below the top
    `sheet_depth` nm, at each x. gate_charge : charge on the gate per
    centimeter of gate width (C/cm). gate : the `Gate`. iterations,
    converged.
    """

    x: np.ndarray
    z: np.ndarray
    psi: np.ndarray
    ec: np.ndarray
    n: np.ndarray
    sheet_x: np.ndarray
    sheet_depth: float
    gate_charge: float
    gate: Gate
    iterations: int
    converged: bool


def _graded(a, b, h_a, h_b, h_max, growth):
    """Points from a to b (a < b), spacing h_a at a and h_b at b, growing
    by `growth` up to h_max in between."""
    if b - a <= max(h_a, h_b):
        return np.array([a, b])
    left, right = [a], [b]
    ha, hb = h_a, h_b
    while left[-1] + ha < right[-1] - hb:
        if ha <= hb:
            left.append(left[-1] + ha)
            ha = min(ha * growth, h_max)
        else:
            right.append(right[-1] - hb)
            hb = min(hb * growth, h_max)
    pts = np.array(left + right[::-1])
    # spread the last gap evenly over its neighbors
    return np.unique(np.round(pts, 9))


def device_grid(layers, width: float, gate: Gate, dz_fine: float = 0.25,
                fine_depth: float = 60.0, dz_max: float = 20.0,
                dx_fine: float = 2.0, dx_max: float = 50.0,
                growth: float = 1.15):
    """A tensor-product grid (x, z) for `solve_cross_section`: z spacing
    `dz_fine` over the top `fine_depth` nm and at every interface there,
    growing by `growth` up to `dz_max` into the buffer; x spacing `dx_fine`
    at the gate edges, growing up to `dx_max` away from them. Layer edges
    and gate edges are grid lines."""
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    top = edges[-1]
    zf = top - min(fine_depth, top)
    zs = [np.linspace(zf, top, max(2, int(round((top - zf) / dz_fine)) + 1))]
    if zf > 0:
        zs.append(_graded(0.0, zf, dz_max, dz_fine, dz_max, growth))
    z = np.unique(np.concatenate(zs + [edges]))
    # merge points closer than a fifth of the fine step to an edge
    keep = np.ones(z.size, bool)
    for e in edges:
        close = (np.abs(z - e) < 0.2 * dz_fine) & (z != e)
        keep &= ~close
    z = z[keep]
    if not (0.0 <= gate.x0 < gate.x1 <= width):
        raise ValueError("the gate must lie inside 0 <= x0 < x1 <= width")
    xs = []
    marks = sorted({0.0, gate.x0, gate.x1, width})
    for a, b in zip(marks[:-1], marks[1:]):
        ha = dx_fine if a in (gate.x0, gate.x1) else dx_max
        hb = dx_fine if b in (gate.x0, gate.x1) else dx_max
        xs.append(_graded(a, b, ha, hb, dx_max, growth))
    x = np.unique(np.concatenate(xs))
    return x, z


def _cv(nodes):
    """Control-volume widths of a 1D node array (half cells at the ends)."""
    h = np.diff(nodes)
    v = np.zeros(nodes.size)
    v[:-1] += 0.5 * h
    v[1:] += 0.5 * h
    return v


def _need(v, what, label):
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"layer {label!r} has no {what} (NaN)")
    return float(v)


def solve_cross_section(layers, gate: Gate, width: float,
                        surface_barrier: float, temperature_K: float = 300.0,
                        line_charge=None, reference: str = "H",
                        interfaces=None, extra_sheet: dict = None,
                        x=None, z=None, statistics: str = "fermi",
                        hole_mass=None, sheet_depth: float = None,
                        tol: float = 1e-10, max_iter: int = 300,
                        guess=None, use_scipy: bool = True,
                        **grid_kwargs) -> CrossSection:
    """Solve the cross-section (see the module docstring).

    width : total width (nm), from x = 0 to width; the gate is `gate`.
    surface_barrier : E_c - E_F at the free surface beside the gate (eV).
    x, z : your own grid (z must contain the layer edges), else
        `device_grid(layers, width, gate, **grid_kwargs)`.
    sheet_depth : depth (nm) below the surface over which `sheet_x` counts
        electrons (default: the top 150 nm, or the whole stack).
    guess : an earlier `CrossSection` on the same grid (warm start).
    """
    for L in layers:
        if not isinstance(L, StackLayer):
            raise ValueError("layers must be StackLayer objects, bottom first")
        if L.acceptors > 0 and hole_mass is None:
            raise ValueError("a layer has acceptors: give hole_mass")
    if not isinstance(gate, Gate):
        raise ValueError("gate must be a Gate")
    if not (math.isfinite(width) and width > 0):
        raise ValueError("width must be > 0 (nm)")
    if not math.isfinite(surface_barrier):
        raise ValueError("surface_barrier must be finite (eV)")
    if statistics not in ("fermi", "boltzmann"):
        raise ValueError("statistics must be 'fermi' or 'boltzmann'")
    if x is None or z is None:
        gx, gz = device_grid(layers, width, gate, **grid_kwargs)
        x = gx if x is None else np.asarray(x, float)
        z = gz if z is None else np.asarray(z, float)
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    edges = np.cumsum([0.0] + [L.thickness for L in layers])
    if not (abs(x[0]) < 1e-9 and abs(x[-1] - width) < 1e-9):
        raise ValueError("x must run from 0 to width")
    if not (abs(z[0]) < 1e-9 and abs(z[-1] - edges[-1]) < 1e-9):
        raise ValueError("z must run from 0 to the top of the stack")
    for e in edges[1:-1]:
        if np.min(np.abs(z - e)) > 1e-6:
            raise ValueError("z must contain every layer edge")
    if interfaces is None:
        interfaces = interface_charges(layers, line_charge=line_charge,
                                       reference=reference)
    extra = dict(extra_sheet or {})
    T = float(temperature_K)
    kT = KB * T
    nx, nz = x.size, z.size
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(layers) - 1)
    zm = 0.5 * (z[1:] + z[:-1])
    cidx = np.clip(np.searchsorted(edges, zm, side="right") - 1, 0, len(layers) - 1)
    eps_l = [_need(L.material.eps_r, "eps_r", L.label) for L in layers]
    ec_l = [_need(L.material.ec, "conduction-band edge ec", L.label) for L in layers]
    me_l = [_need(L.material.me, "electron mass me", L.label) for L in layers]
    Ec0 = np.array([ec_l[i] for i in idx])
    pref = (KB * T * QE / (2 * np.pi * HBAR ** 2)) ** 1.5 * 2.0 * 1e-27
    Nc = pref * (np.array([me_l[i] for i in idx]) * M0) ** 1.5
    holes = hole_mass is not None
    if holes:
        hm = ([float(hole_mass)] * len(layers) if np.ndim(hole_mass) == 0
              else [float(v) for v in hole_mass])
        eg_l = [_need(L.material.eg, "band gap eg", L.label) for L in layers]
        Eg = np.array([eg_l[i] for i in idx])
        Nv = pref * (np.array([hm[i] for i in idx]) * M0) ** 1.5
    ND = np.array([layers[i].donors for i in idx], float)
    Ed = np.array([np.nan if layers[i].donor_depth is None else layers[i].donor_depth
                   for i in idx])
    NA = np.array([layers[i].acceptors for i in idx], float)
    eps_cell = np.array([eps_l[i] for i in cidx])          # (nz-1,)
    hz = np.diff(z)
    hx = np.diff(x)
    vz = _cv(z)
    vx = _cv(x)
    # permittivity at x faces: CV-weighted average of the two half cells
    eps_node = np.zeros(nz)
    w = np.zeros(nz)
    eps_node[:-1] += eps_cell * 0.5 * hz
    eps_node[1:] += eps_cell * 0.5 * hz
    w[:-1] += 0.5 * hz
    w[1:] += 0.5 * hz
    eps_node /= w
    sheet = np.zeros(nz)
    for c in interfaces:
        s = c.sigma_nm2 + extra.get(c.index, 0.0)
        sheet[int(np.argmin(np.abs(z - edges[c.index + 1])))] += s
    # Laplacian as a sparse matrix on the full grid (index i*nz + j)
    N = nx * nz
    I = np.arange(N).reshape(nx, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros((nx, nz))
    # z faces
    cz = eps_cell[None, :] / hz[None, :] * vx[:, None]       # (nx, nz-1)
    rows += [I[:, :-1].ravel(), I[:, 1:].ravel()]
    cols += [I[:, 1:].ravel(), I[:, :-1].ravel()]
    vals += [cz.ravel(), cz.ravel()]
    diag[:, :-1] -= cz
    diag[:, 1:] -= cz
    # x faces
    cxf = eps_node[None, :] / hx[:, None] * vz[None, :]       # (nx-1, nz)
    rows += [I[:-1, :].ravel(), I[1:, :].ravel()]
    cols += [I[1:, :].ravel(), I[:-1, :].ravel()]
    vals += [cxf.ravel(), cxf.ravel()]
    diag[:-1, :] -= cxf
    diag[1:, :] -= cxf
    off_r = np.concatenate(rows)
    off_c = np.concatenate(cols)
    off_v = np.concatenate(vals)
    vol = vx[:, None] * vz[None, :]
    sheet_area = vx[:, None] * sheet[None, :]
    # boundary values on the top row
    top_gate = (x >= gate.x0 - 1e-9) & (x <= gate.x1 + 1e-9)
    psi_top = np.where(top_gate, Ec0[-1] + gate.voltage - gate.barrier,
                       Ec0[-1] - surface_barrier)
    dirichlet = np.zeros((nx, nz), bool)
    dirichlet[:, -1] = True

    def occ(eta):
        eta = np.clip(eta, -700, 700)
        if statistics == "boltzmann":
            e = np.exp(eta)
            return e, e
        F, dF = fermi_dirac_half(eta)
        return np.asarray(F), np.asarray(dF)

    def charge(psi):
        eta = (psi - Ec0[None, :]) / kT
        F, dF = occ(eta)
        n = Nc[None, :] * F
        dn = Nc[None, :] * dF / kT
        rho = ND[None, :] - n
        drho = -dn
        m = np.isfinite(Ed) & (ND > 0)
        if m.any():
            xx = np.exp(np.clip((psi[:, m] - Ec0[m] + Ed[m]) / kT, -700, 700))
            rho[:, m] += -ND[m] + ND[m] / (1 + 2 * xx)
            drho[:, m] += -ND[m] * 2 * xx / (1 + 2 * xx) ** 2 / kT
        if holes:
            etap = (Ec0[None, :] - psi - Eg[None, :]) / kT
            Fp, dFp = occ(etap)
            rho += Nv[None, :] * Fp - NA[None, :]
            drho += -Nv[None, :] * dFp / kT
        return rho, drho, n

    def balance(psi):
        lap = diag * psi
        flat = psi.ravel()
        np.add.at(lap.reshape(-1), off_r, off_v * flat[off_c])
        rho, drho, n = charge(psi)
        return lap + _CP * (vol * rho + sheet_area), drho, n

    try:
        if not use_scipy:
            raise ImportError
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        A0 = sp.csr_matrix((off_v, (off_r, off_c)), shape=(N, N))
        have_scipy = True
    except ImportError:
        have_scipy = False

    if guess is not None:
        if guess.psi.shape != (nx, nz):
            raise ValueError("guess must be on the same grid")
        psi = guess.psi.copy()
    else:
        psi = np.repeat(psi_top[:, None], nz, axis=1).astype(float)
    psi[:, -1] = psi_top
    free = ~dirichlet.ravel()
    conv = False
    it = 0
    for it in range(1, max_iter + 1):
        R, drho, _ = balance(psi)
        R = R.ravel()
        R[~free] = 0.0
        d = (diag + _CP * vol * drho).ravel()
        if have_scipy:
            J = A0 + sp.diags(d)
            Jf = J[free][:, free]
            step = np.zeros(N)
            step[free] = spl.spsolve(Jf.tocsc(), -R[free])
        else:
            step = np.zeros(N)
            step[free] = _pcg(off_r, off_c, off_v, d, free, -R[free], N)
        step = np.clip(step, -0.2, 0.2)
        psi = psi + step.reshape(nx, nz)
        if np.max(np.abs(step)) < tol:
            conv = True
            break
    R, _, n = balance(psi)
    # charge held by the gate: minus the imbalance at its top nodes
    qg = -float(np.sum(R[top_gate, -1])) / _CP          # e per nm of width
    depth = sheet_depth if sheet_depth is not None else min(150.0, z[-1])
    m = z >= z[-1] - depth - 1e-9
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    sheet_x = trap(n[:, m], z[m], axis=1) * 1e14
    return CrossSection(x=x, z=z, psi=psi, ec=Ec0[None, :] - psi, n=n * 1e21,
                        sheet_x=sheet_x, sheet_depth=depth,
                        gate_charge=qg * QE * 1e7, gate=gate,
                        iterations=it, converged=conv)


def _pcg(off_r, off_c, off_v, d, free, b, N, tol=1e-12, max_iter=20000):
    """Jacobi-preconditioned conjugate gradients on -J (symmetric positive
    definite) restricted to the free nodes."""
    fidx = np.flatnonzero(free)
    pos = -np.ones(N, int)
    pos[fidx] = np.arange(fidx.size)
    keep = free[off_r] & free[off_c]
    r_, c_, v_ = pos[off_r[keep]], pos[off_c[keep]], off_v[keep]
    dd = d[fidx]

    def mv(u):
        y = dd * u
        np.add.at(y, r_, v_ * u[c_])
        return -y

    # solve J x = b, i.e. (-J) x = -b
    x = np.zeros(fidx.size)
    r = -b - mv(x)
    Minv = 1.0 / (-dd)
    zv = Minv * r
    p = zv.copy()
    rz = r @ zv
    bn = np.linalg.norm(b) or 1.0
    for _ in range(max_iter):
        Ap = mv(p)
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        if np.linalg.norm(r) < tol * bn:
            break
        zv = Minv * r
        rz2 = r @ zv
        p = zv + (rz2 / rz) * p
        rz = rz2
    return x


def cv_cross_section(layers, gate: Gate, voltages, width: float,
                     surface_barrier: float, dv: float = 0.01, **kwargs):
    """Gate C-V of the cross-section: C = dQ/dV per gate area (F/cm^2,
    the charge per width divided by the gate length), by a central
    difference of +-dv/2. Returns (voltages, C, gate charge per width
    (C/cm), results)."""
    if not (math.isfinite(dv) and dv > 0):
        raise ValueError("dv must be > 0")
    x, z = device_grid(layers, width, gate,
                       **{k: kwargs.pop(k) for k in list(kwargs)
                          if k in ("dz_fine", "fine_depth", "dz_max", "dx_fine",
                                   "dx_max", "growth")})
    kwargs.setdefault("x", x)
    kwargs.setdefault("z", z)
    C, Q, res, prev = [], [], [], None
    for v in voltages:
        g = dataclasses.replace(gate, voltage=float(v))
        r = solve_cross_section(layers, g, width, surface_barrier, guess=prev, **kwargs)
        ra = solve_cross_section(layers, dataclasses.replace(g, voltage=v + dv / 2),
                                 width, surface_barrier, guess=r, **kwargs)
        rb = solve_cross_section(layers, dataclasses.replace(g, voltage=v - dv / 2),
                                 width, surface_barrier, guess=r, **kwargs)
        C.append((ra.gate_charge - rb.gate_charge) / dv / (gate.length * 1e-7))
        Q.append(r.gate_charge)
        res.append(r)
        prev = r
    return np.asarray(voltages, float), np.array(C), np.array(Q), res
