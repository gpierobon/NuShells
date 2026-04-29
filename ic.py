import scipy
import numpy as np


# -----------------------------------------------------------------------
#  Fermi-Dirac sampler
# -----------------------------------------------------------------------
def _build_fd_icdf(shells, qmax=20.0, ngrid=10_000):
    """Build inverse CDF for f(hat_q) * hat_q^2 (momentum-space FD)."""
    q   = np.linspace(0.0, qmax, ngrid)
    pdf = q**2 / (np.exp(np.clip(q, 0, 500)) + 1.0)
    cdf = scipy.integrate.cumulative_trapezoid(pdf, q, initial=0.0)
    cdf /= cdf[-1]
    shells.inv_cdf = scipy.interpolate.interp1d(
        cdf, q, kind='linear',
        bounds_error=False,
        fill_value=(0.0, qmax)
    )

def sample_q(shells, N):
    """Return N samples of hat_q from the Fermi-Dirac distribution."""
    if shells.inv_cdf is None:
        _build_fd_icdf(shells)
    shells.log.debug(f"[IC] Sampled {N} momenta from Fermi-Dirac")
    return shells.inv_cdf(np.random.rand(N))


# -----------------------------------------------------------------------
# Weight computation
# -----------------------------------------------------------------------
def compute_weights(r, dr, q, Psi, log):
    """
    Phase-space weight for one shell.

    Parameters
    ----------
    r   : hat_r
    dr  : radial bin width
    q   : hat_q_total = q/T_nu, O(1)
    Psi : dimensionless Newtonian perturbation potential

    Returns
    -------
    w  : float, proportional to r^2 dr * q^2 * f(q) * (1 + perturbation)
    df : float, delta f perturbation
    """

    f0     = 1.0 / (np.exp(q) + 1.0)
    dfdlnq = -(q * np.exp(q)) / (np.exp(q) + 1.0)**2   # = q * df/dq
    pert   = 1.0 + (dfdlnq / f0) * Psi

    weight = (
        8.0 * np.pi**2    # solid angle factor
        * r**2 * dr       # radial volume element [1/m_phi]^3
        * (1.0 - Psi)     # metric perturbation correction
        * f0 * pert       # linearised perturbation (delta f)
    )

    log.debug(f"[IC] Computed weights")
    return weight, dfdlnq * Psi


# -----------------------------------------------------------------------
# Grav. potential computation
# -----------------------------------------------------------------------
def compute_Psi(R, shells):
    """ """
    a      = shells.a
    mphi   = shells.m_phi_hat
    rho    = R / shells.R0
    delta0 = shells.delta0
    omega0 = shells.omega0

    norm = - 1.0 / (16.0 * 3.0 * np.sqrt(3) * np.pi**3)
    psi = norm * omega0 * delta0 / (a**3 *  mphi**2) * np.exp(1.5-0.5*rho**2)
    shells.log.debug(f"[IC] Computed gravitational potential Psi")
    return psi


