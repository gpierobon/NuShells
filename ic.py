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
    return shells.inv_cdf(np.random.rand(N))


# -----------------------------------------------------------------------
# Weight computation
# -----------------------------------------------------------------------
def compute_weight(r, dr, mu, q, Psi):
    """
    Phase-space weight for one shell.

    Parameters
    ----------
    r   : hat_r
    dr  : radial bin width
    mu  : cos(theta) in [-1,1]
    q   : hat_q_total = q/T_nu, O(1)
    Psi : dimensionless Newtonian perturbation potential

    Returns
    -------
    w : float, proportional to r^2 dr * q^2 * f(q) * (1 + perturbation)

    Perturbation from linearised Boltzmann equation:
        delta_f / f_0 = -(d ln f_0 / d ln q) * Psi * mu
    where d ln f_0 / d ln q = -q * exp(q) / (exp(q)+1)  / f_0
    """
    Nq     = 1.5 * scipy.special.zeta(3) # normalisation: int q^2/(e^q+1)dq 

    f0     = 1.0 / (np.exp(q) + 1.0)
    dfdlnq = -(q * np.exp(q)) / (np.exp(q) + 1.0)**2   # = q * df/dq
    pert   = 1.0 + (dfdlnq / f0) * Psi * mu

    weight = (
        8.0 * np.pi**2    # solid angle factor
        * r**2 * dr       # radial volume element [1/m_phi]^3
        * (1.0 - Psi)     # metric perturbation correction
        * 2.0 * Nq        # FD normalisation
        * pert            # linearised perturbation (delta f) 
    )
    return weight


