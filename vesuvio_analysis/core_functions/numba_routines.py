"""Numba-accelerated resolution and NCP functions for VESUVIO analysis.

This module provides ``@njit(cache=True)`` versions of the computational
bottleneck functions from ``analysis_functions.py``.  All functions
operate on plain NumPy arrays and scalars — no Mantid API or Python
objects are used inside JIT-compiled code.

Toggle between the legacy NumPy path and the Numba path via the
``USE_NUMBA`` flag in ``analysis_functions.py``.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Physical constants (mirrors analysis_functions.loadConstants)
# ---------------------------------------------------------------------------

@njit(cache=True)
def loadConstants():
    """Return VESUVIO physical constants as a 5-tuple of floats.

    Returns
    -------
    mN : float
        Neutron mass in atomic mass units.
    Ef : float
        Final neutron energy (meV).
    en_to_vel : float
        sqrt(energy)-to-velocity conversion factor.
    vf : float
        Final neutron velocity (m/μs).
    hbar : float
        Reduced Planck constant in Å⁻¹·a.m.u.·m/μs.
    """
    mN = 1.008
    Ef = 4906.0
    en_to_vel = 4.3737e-4
    vf = np.sqrt(Ef) * en_to_vel
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


# ---------------------------------------------------------------------------
# Elementary line-shape functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def gaussian(x, sigma):
    """Normalised Gaussian centred at zero.

    Parameters
    ----------
    x : ndarray
        Abscissa values.
    sigma : ndarray
        Standard deviation (broadcastable with *x*).

    Returns
    -------
    ndarray
        Gaussian values, same shape as *x*.
    """
    g = np.exp(-x ** 2 / 2.0 / sigma ** 2)
    g = g / (np.sqrt(2.0 * np.pi) * sigma)
    return g


@njit(cache=True)
def lorentizian(x, gamma):
    """Normalised Lorentzian centred at zero.

    Parameters
    ----------
    x : ndarray
        Abscissa values.
    gamma : ndarray
        Half-width at half-maximum (broadcastable with *x*).

    Returns
    -------
    ndarray
        Lorentzian values, same shape as *x*.
    """
    return gamma / np.pi / (x ** 2 + gamma ** 2)


# ---------------------------------------------------------------------------
# Manual trapezoidal integration (replaces np.trapz for Numba)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _trapz_axis1(y, x):
    """Trapezoidal integration along axis 1 of a 2-D array.

    Parameters
    ----------
    y : ndarray, shape (M, N)
        Function values.
    x : ndarray, shape (M, N)
        Abscissa values.

    Returns
    -------
    ndarray, shape (M,)
        Integral for each row.
    """
    m = y.shape[0]
    n = y.shape[1]
    result = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(n - 1):
            s += 0.5 * (y[i, j] + y[i, j + 1]) * (x[i, j + 1] - x[i, j])
        result[i] = s
    return result


# ---------------------------------------------------------------------------
# Pseudo-Voigt profile
# ---------------------------------------------------------------------------

@njit(cache=True)
def pseudoVoigt(x, sigma, gamma, normVoigt):
    """Approximate pseudo-Voigt profile (Thompson–Cox–Hastings).

    Parameters
    ----------
    x : ndarray, shape (n_masses, n_bins)
        Abscissa values (y - centre).
    sigma : ndarray, shape (n_masses, 1)
        Gaussian standard deviation.
    gamma : ndarray, shape (n_masses, 1)
        Lorentzian HWHM.
    normVoigt : bool
        If True, normalise each row by its trapezoidal integral.

    Returns
    -------
    ndarray, shape (n_masses, n_bins)
        Pseudo-Voigt profile.
    """
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    f = 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)
    eta = 1.36603 * fl / f - 0.47719 * (fl / f) ** 2 + 0.11116 * (fl / f) ** 3
    sigma_v = f / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma_v = f / 2.0
    pv = eta * lorentizian(x, gamma_v) + (1.0 - eta) * gaussian(x, sigma_v)

    if normVoigt:
        norms = np.abs(_trapz_axis1(pv, x))
        for i in range(pv.shape[0]):
            if norms[i] != 0.0:
                pv[i] = pv[i] / norms[i]
    return pv


# ---------------------------------------------------------------------------
# Numerical third derivative (FSE term)
# ---------------------------------------------------------------------------

@njit(cache=True)
def numericalThirdDerivative(x, fun):
    """Compute the third derivative via a 13-point stencil.

    Parameters
    ----------
    x : ndarray, shape (n_masses, n_bins)
        Abscissa values.
    fun : ndarray, shape (n_masses, n_bins)
        Function values.

    Returns
    -------
    ndarray, shape (n_masses, n_bins)
        Approximate third derivative (zero-padded edges).
    """
    k6 = (-fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (-fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+fun[:, 9:-3] - fun[:, 3:-9]) * 488
    k2 = (+fun[:, 8:-4] - fun[:, 4:-8]) * 387
    k1 = (-fun[:, 7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dx = x[:, 7:-5] - x[:, 6:-6]
    dev = dev / (dx * dx * dx)
    dev = dev / (12 ** 3)

    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    return derivative


# ---------------------------------------------------------------------------
# Kinematics at y-space centres
# ---------------------------------------------------------------------------

@njit(cache=True)
def kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays):
    """Evaluate kinematics at the y-space bin closest to each NCP centre.

    Parameters
    ----------
    ySpacesForEachMass : ndarray, shape (n_masses, n_bins)
    centers : ndarray, shape (n_masses, 1)
    kinematicArrays : ndarray, shape (4, n_bins)

    Returns
    -------
    v0, E0, deltaE, deltaQ : ndarray, each shape (n_masses, 1)
    """
    n_masses = centers.shape[0]
    v0_out = np.empty((n_masses, 1))
    E0_out = np.empty((n_masses, 1))
    dE_out = np.empty((n_masses, 1))
    dQ_out = np.empty((n_masses, 1))

    v0_row = kinematicArrays[0]
    E0_row = kinematicArrays[1]
    dE_row = kinematicArrays[2]
    dQ_row = kinematicArrays[3]

    for m in range(n_masses):
        best_j = 0
        best_dist = np.abs(ySpacesForEachMass[m, 0] - centers[m, 0])
        for j in range(1, ySpacesForEachMass.shape[1]):
            d = np.abs(ySpacesForEachMass[m, j] - centers[m, 0])
            if d < best_dist:
                best_dist = d
                best_j = j
        v0_out[m, 0] = v0_row[best_j]
        E0_out[m, 0] = E0_row[best_j]
        dE_out[m, 0] = dE_row[best_j]
        dQ_out[m, 0] = dQ_row[best_j]

    return v0_out, E0_out, dE_out, dQ_out


# ---------------------------------------------------------------------------
# Gaussian resolution
# ---------------------------------------------------------------------------

@njit(cache=True)
def calcGaussianResolution(masses, v0, E0, delta_E, delta_Q,
                           resolutionPars, instrPars):
    """Compute Gaussian resolution width in y-space (Å⁻¹).

    Parameters
    ----------
    masses : ndarray, shape (n_masses, 1)
    v0, E0, delta_E, delta_Q : ndarray, shape (n_masses, 1)
    resolutionPars : ndarray, shape (6,)
        [dE1, dTOF, dTheta, dL0, dL1, dE1_lorz]
    instrPars : ndarray, shape (6,)
        [det, plick, angle, T0, L0, L1]

    Returns
    -------
    ndarray, shape (n_masses, 1)
    """
    dE1 = resolutionPars[0]
    dTOF = resolutionPars[1]
    dTheta = resolutionPars[2]
    dL0 = resolutionPars[3]
    dL1 = resolutionPars[4]
    # dE1_lorz = resolutionPars[5]  # not used here

    angle_deg = instrPars[2]
    L0 = instrPars[4]
    L1 = instrPars[5]

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    angle = angle_deg * np.pi / 180.0

    dWdE1 = 1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)
    dWdTOF = 2.0 * E0 * v0 / L0
    dWdL1 = 2.0 * E0 ** 1.5 / Ef ** 0.5 / L0
    dWdL0 = 2.0 * E0 / L0

    dW2 = (dWdE1 ** 2 * dE1 ** 2
           + dWdTOF ** 2 * dTOF ** 2
           + dWdL1 ** 2 * dL1 ** 2
           + dWdL0 ** 2 * dL0 ** 2)
    dW2 = dW2 * (masses / hbar ** 2 / delta_Q) ** 2

    dQdE1 = (1.0
             - (E0 / Ef) ** 1.5 * L1 / L0
             - np.cos(angle) * ((E0 / Ef) ** 0.5 - L1 / L0 * E0 / Ef))
    dQdTOF = 2.0 * E0 * v0 / L0
    dQdL1 = 2.0 * E0 ** 1.5 / L0 / Ef ** 0.5
    dQdL0 = 2.0 * E0 / L0
    dQdTheta = 2.0 * np.sqrt(E0 * Ef) * np.sin(angle)

    dQ2 = (dQdE1 ** 2 * dE1 ** 2
           + (dQdTOF ** 2 * dTOF ** 2
              + dQdL1 ** 2 * dL1 ** 2
              + dQdL0 ** 2 * dL0 ** 2)
           * np.abs(Ef / E0 * np.cos(angle) - 1.0)
           + dQdTheta ** 2 * dTheta ** 2)
    dQ2 = dQ2 * (mN / hbar ** 2 / delta_Q) ** 2

    return np.sqrt(dW2 + dQ2)


# ---------------------------------------------------------------------------
# Lorentzian resolution
# ---------------------------------------------------------------------------

@njit(cache=True)
def calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q,
                             resolutionPars, instrPars):
    """Compute Lorentzian resolution HWHM in y-space (Å⁻¹).

    Parameters
    ----------
    masses : ndarray, shape (n_masses, 1)
    v0, E0, delta_E, delta_Q : ndarray, shape (n_masses, 1)
    resolutionPars : ndarray, shape (6,)
    instrPars : ndarray, shape (6,)

    Returns
    -------
    ndarray, shape (n_masses, 1)
    """
    dE1_lorz = resolutionPars[5]
    angle_deg = instrPars[2]
    L0 = instrPars[4]
    L1 = instrPars[5]

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    angle = angle_deg * np.pi / 180.0

    dWdE1_lor = (1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)) ** 2
    dWdE1_lor = dWdE1_lor * (masses / hbar ** 2 / delta_Q) ** 2

    dQdE1_lor = (1.0
                 - (E0 / Ef) ** 1.5 * L1 / L0
                 - np.cos(angle) * ((E0 / Ef) ** 0.5 + L1 / L0 * E0 / Ef)) ** 2
    dQdE1_lor = dQdE1_lor * (mN / hbar ** 2 / delta_Q) ** 2

    return np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz


# ---------------------------------------------------------------------------
# Full NCP orchestrator (flattened IC attributes)
# ---------------------------------------------------------------------------

@njit(cache=True)
def calculateNcpSpec_numba(masses_1d, pars, ySpacesForEachMass,
                           resolutionPars, instrPars, kinematicArrays,
                           normVoigt):
    """Synthesise C(t) for one spectrum — Numba-accelerated.

    This is the Numba equivalent of ``calculateNcpSpec`` in
    ``analysis_functions.py``.  The ``IC`` object is *unrolled*: its
    ``masses`` array is passed as ``masses_1d`` (1-D, shape
    ``(n_masses,)``) and the boolean ``normVoigt`` flag is passed
    directly.

    Parameters
    ----------
    masses_1d : ndarray, shape (n_masses,)
        Atomic masses.
    pars : ndarray, shape (3 * n_masses,)
        Interleaved [I, W, C, I, W, C, …] fit parameters.
    ySpacesForEachMass : ndarray, shape (n_masses, n_bins)
    resolutionPars : ndarray, shape (6,)
    instrPars : ndarray, shape (6,)
    kinematicArrays : ndarray, shape (4, n_bins)
    normVoigt : bool

    Returns
    -------
    ncpForEachMass : ndarray, shape (n_masses, n_bins)
    ncpTotal : ndarray, shape (n_bins,)
    """
    n_masses = masses_1d.shape[0]

    # --- unroll pars (mirrors prepareArraysFromPars) ---
    masses = masses_1d.reshape((n_masses, 1))
    intensities = np.empty((n_masses, 1))
    widths = np.empty((n_masses, 1))
    centers = np.empty((n_masses, 1))
    for i in range(n_masses):
        intensities[i, 0] = pars[3 * i]
        widths[i, 0] = pars[3 * i + 1]
        centers[i, 0] = pars[3 * i + 2]

    # --- kinematics at y-centres ---
    v0, E0, deltaE, deltaQ = kinematicsAtYCenters(
        ySpacesForEachMass, centers, kinematicArrays
    )

    # --- resolution ---
    gaussRes = calcGaussianResolution(
        masses, v0, E0, deltaE, deltaQ, resolutionPars, instrPars
    )
    lorzRes = calcLorentzianResolution(
        masses, v0, E0, deltaE, deltaQ, resolutionPars, instrPars
    )

    totalGaussWidth = np.sqrt(widths ** 2 + gaussRes ** 2)

    # --- pseudo-Voigt J(y) ---
    JOfY = pseudoVoigt(
        ySpacesForEachMass - centers, totalGaussWidth, lorzRes, normVoigt
    )

    # --- FSE term ---
    FSE = (-numericalThirdDerivative(ySpacesForEachMass, JOfY)
           * widths ** 4 / deltaQ * 0.72)

    # --- NCP per mass ---
    ncpForEachMass = (intensities * (JOfY + FSE)
                      * E0 * E0 ** (-0.92) * masses / deltaQ)

    # --- total ---
    n_bins = ySpacesForEachMass.shape[1]
    ncpTotal = np.zeros(n_bins)
    for i in range(n_masses):
        for j in range(n_bins):
            ncpTotal[j] += ncpForEachMass[i, j]

    return ncpForEachMass, ncpTotal
