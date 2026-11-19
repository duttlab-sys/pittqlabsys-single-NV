"""Full NV + 15N model used for the Paper 1 predictions (electron S=1 x 15N I=1/2, tilted field).

Same physics and numbers as the analysis scripts used for every full-model result in the paper:
- static Hamiltonian H0/h = D Sz^2 + gamma_e B.S + A_par Sz Iz + A_perp (Sx Ix + Sy Iy) - gamma_n B.I   (MHz)
- the four eigenstates in the ms = 0 and ms = -1 manifolds, frame rotating at the MW frequency, RWA
- drive from the played I/Q envelope: Rabi frequency = kappa * f_I(|I + iQ|) (measured Rabi curve), phase = arg(I + iQ)
- 1 ns time steps; P0 = population left in the ms = 0 manifold, averaged over the two initial nuclear states
Defaults: field angle 10.5 deg and line centre 2397.5825 MHz (four-line Ramsey + ODMR, Nov 16), MW 2397.5 MHz.
kappa is the drive scale refitted within THIS model (1.020 original setup, 0.995 Nov 13, about 1.005-1.010 Nov 16-18).

Usage:
    from nv15n_model import NV15NModel
    m = NV15NModel(theta_deg=10.5, kappa=1.020)
    p0 = m.P0(I, Q)      # I, Q: arrays in AWG full-scale units, one sample per ns
"""
import numpy as np
from scipy.optimize import brentq

D, GE, GN = 2870.0, 28024.95, -4.316          # MHz, MHz/T (gamma_e), MHz/T (gamma_n of 15N)
APAR, APERP = 3.03, 3.65                       # MHz, 15N hyperfine (Felton et al. 2009)
RABI_CURVE = [(0.05, 1.302), (0.1, 2.587), (0.2, 5.043), (0.3, 7.254), (0.4, 9.144), (0.5, 10.690), (0.6, 11.943)]  # (AWG FS, MHz)

_s2 = 1 / np.sqrt(2)
_Sx = np.array([[0, _s2, 0], [_s2, 0, _s2], [0, _s2, 0]])
_Sy = np.array([[0, -1j * _s2, 0], [1j * _s2, 0, -1j * _s2], [0, 1j * _s2, 0]])
_Sz = np.diag([1.0, 0.0, -1.0])
_Ix = np.array([[0, 0.5], [0.5, 0]]); _Iy = np.array([[0, -0.5j], [0.5j, 0]]); _Iz = np.diag([0.5, -0.5])
_K = np.kron; _E3, _E2 = np.eye(3), np.eye(2)


def f_rabi_mhz(a):
    """Measured Rabi curve f_I(a) in MHz, interpolated exactly as in the analysis (nv_data_gui GxRabiCurve):
    the slope f/a is interpolated linearly in |a| and extrapolated linearly outside the measured points."""
    a = np.abs(np.asarray(a, float)); xa = np.array([p[0] for p in RABI_CURVE]); sl = np.array([p[1] / p[0] for p in RABI_CURVE])
    out = np.interp(a, xa, sl); lo = (sl[1] - sl[0]) / (xa[1] - xa[0]); hi = (sl[-1] - sl[-2]) / (xa[-1] - xa[-2])
    out = np.where(a < xa[0], sl[0] + lo * (a - xa[0]), out); out = np.where(a > xa[-1], sl[-1] + hi * (a - xa[-1]), out)
    return a * out


def h_lab(B_T, th):
    """Static NV + 15N Hamiltonian (MHz) for field magnitude B_T (tesla) at angle th (rad) from the NV axis."""
    bz, bx = B_T * np.cos(th), B_T * np.sin(th)
    return (D * _K(_Sz @ _Sz, _E2) + GE * (bz * _K(_Sz, _E2) + bx * _K(_Sx, _E2))
            + APAR * _K(_Sz, _Iz) + APERP * (_K(_Sx, _Ix) + _K(_Sy, _Iy))
            - GN * (bz * _K(_E3, _Iz) + bx * _K(_E3, _Ix)))


def subspace(B_T, th):
    """Energies (MHz) of the two ms=0-like and two ms=-1-like eigenstates, and Sx, Sy in that basis."""
    w, v = np.linalg.eigh(h_lab(B_T, th))
    ms = np.array([np.real(v[:, k].conj() @ _K(_Sz, _E2) @ v[:, k]) for k in range(6)])
    i0 = np.argsort(np.abs(ms))[:2]; i1 = np.argsort(np.abs(ms + 1))[:2]
    idx = list(i0[np.argsort(w[i0])]) + list(i1[np.argsort(w[i1])])
    V = v[:, idx]
    return w[idx], V.conj().T @ _K(_Sx, _E2) @ V, V.conj().T @ _K(_Sy, _E2) @ V


def field_for_center(th, center_mhz):
    """Field (gauss) that puts the centre of the two strong ms=0 <-> -1 lines at center_mhz, at angle th (rad)."""
    def f(Bg):
        E, SX, _ = subspace(Bg * 1e-4, th)
        M = np.abs(SX[2:, :2]) ** 2
        strong = sorted([(i, j) for i in range(2) for j in range(2)], key=lambda p: -M[p[1], p[0]])[:2]
        return np.mean([E[2 + j] - E[i] for i, j in strong]) - center_mhz
    return brentq(f, 100, 400)


class NV15NModel:
    def __init__(self, theta_deg=10.5, kappa=1.020, center_mhz=2397.5825, f_mw_mhz=2397.5):
        th = np.radians(theta_deg)
        self.B_gauss = field_for_center(th, center_mhz)
        E, SX, SY = subspace(self.B_gauss * 1e-4, th)
        Er = E.copy(); Er[2:] -= f_mw_mhz                       # rotating frame at the MW
        self.Hs = np.diag(2 * np.pi * Er) * 1e-3                 # rad/ns
        self.MX = np.zeros((4, 4), complex); self.MY = np.zeros((4, 4), complex)
        for M_, S_ in ((self.MX, SX), (self.MY, SY)):          # RWA: keep only the ms=0 <-> ms=-1 blocks
            M_[2:, :2] = np.sqrt(2) * S_[2:, :2]; M_[:2, 2:] = np.sqrt(2) * S_[:2, 2:]
        self.kappa = kappa

    def P0(self, I, Q):
        """Population of the ms=0 manifold after the waveform (I, Q: AWG full-scale units, 1 ns per sample)."""
        I = np.asarray(I, float); Q = np.asarray(Q, float)
        A = np.hypot(I, Q); Om = 2 * np.pi * self.kappa * f_rabi_mhz(A) * 1e-3   # rad/ns
        ux = np.where(A > 0, I / np.where(A > 0, A, 1), 0); uy = np.where(A > 0, Q / np.where(A > 0, A, 1), 0)
        U = np.eye(4, dtype=complex)
        for n in range(len(I)):
            H = self.Hs + 0.5 * Om[n] * (ux[n] * self.MX + uy[n] * self.MY)
            w, v = np.linalg.eigh(H); U = (v * np.exp(-1j * w)) @ v.conj().T @ U
        return float(sum(0.5 * np.sum(np.abs(U[:2, n0]) ** 2) for n0 in range(2)))
