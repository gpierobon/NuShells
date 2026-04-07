import numpy as np
from timing import timed


@timed("YukawaForce")
def solveYukawaForce(shells, xi_cap=500.0):
    """
    """
    N = shells.N
    if N > 1 and not np.all(shells.R[:-1] <= shells.R[1:]):
        raise ValueError("Shells must be sorted ascending by tilde_r.")

    cache = shells._cache
    xi, capped    = cache['xi'],       cache['capped']
    e, sinh_xi    = cache['e'],        cache['sinh_xi']
    sum_outer     = cache['sum_outer']
    sum_inner     = cache['sum_inner']

    a, R, m, eps, w, alpha, ell = shells.a, shells.R, shells.m, shells.eps, \
                                  shells.w, shells.alpha, shells.ell

    R2 = R**2

    out_kernel = e * (1.0 + xi) / R2

    # xi*coth(xi) - 1 = xi^2/3 + xi^4/45 + ...
    xi_coth_m1 = np.where( xi > 1e-4,
                    xi * np.cosh(xi) / np.sinh(np.maximum(xi, 1e-300)) - 1.0,
                    xi**2/3.0 + xi**4/45.0)
    inner_kernel = np.where(capped, 0.0, -sinh_xi * xi_coth_m1 / R2)

    # Kernel
    forceK = (out_kernel * sum_outer + inner_kernel * sum_inner) / (4.0*np.pi)

    fs = ell**2 / (eps * R**3)
    lr = a**2 * alpha * m / eps * forceK

    return fs, lr


@timed("GravityForce")
def solveGravityForce(shells):
    """ """
    a      = shells.a
    R      = shells.R
    eps    = shells.eps
    beta   = shells.beta
    rhobar = shells.rhobar

    # M(<R_i) = cumulative sum of w_j * eps_j for R_j < R_i
    M_enc  = shells._cumMass
    M_mean = (4.0 * np.pi / 3.0) * R * rhobar

    F_grav = beta * eps / a**2 * ( M_enc / R**2 - M_mean)

    return F_grav
