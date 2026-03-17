import numpy as np
from constants import XI_CAP

# ---------------------------------------------------------------------------
class ForceSolver:
# ---------------------------------------------------------------------------
    def __init__(self, shells):
        if shells.data is None:
            raise RuntimeError("Shells.initialise() must be called first.")

        N = shells.N
        if N > 1 and not np.all(shells.R[:-1] <= shells.R[1:]):
            raise ValueError("Shells must be sorted ascending by tilde_r.")

        # See Shells class in shells.py 
        a   = shells.a
        R   = shells.R
        m   = shells.m
        eps = shells.eps
        w   = shells.w
        m0  = shells.m0

        # xi = a * hat_r  (Yukawa argument; xi~1 at the force range)
        # For numerics, we cap at XI_CAP: 
        #     for xi > XI_CAP, exp(-xi) ~ 0 and sinh(xi) overflows,
        #     and such shells are fully Yukawa-screened
        xi_raw   = a * R
        xi       = np.minimum(xi_raw, XI_CAP)
        screened = xi_raw >= XI_CAP


        # w_i * hat_m_i / hat_eps_i (dimensionless)
        moe = w * m / eps
        cut = 1e6

        sinh_over_xi = np.where(xi > cut, np.sinh(xi) / xi,
                                1.0 + xi**2/6.0 + xi**4/120.0)

        # exp(-xi)/xi, used for B_i = moe_i * exp(-xi_i)/xi_i
        # inner-shell weights, only relevant for xi << 1)
        exp_over_xi = np.exp(-xi) / np.where(xi > cut, xi, cut)

        # Cumulative sums (shells sorted ascending by hat_r) ----
        # A_i = moe_i * sinh(xi_i)/xi_i (outer-kernel weight: shell i inside r)
        # B_i = moe_i * exp(-xi_i)/xi_i (inner-kernel weight: shell i outside r)
        #
        # sum_outer[i] = sum_{j < i} A_j
        # sum_inner[i] = sum_{j > i} B_j
        A      = moe * sinh_over_xi
        B      = moe * exp_over_xi
        cumA   = np.cumsum(A)
        cumB   = np.cumsum(B)
        totalB = cumB[-1]

        sum_outer      = np.empty(N)
        sum_outer[0]   = 0.0
        sum_outer[1:]  = cumA[:-1]          # sum_{j < i} A_j

        sum_inner      = np.empty(N)
        sum_inner[-1]  = 0.0
        sum_inner[:-1] = totalB - cumB[:-1] # sum_{j > i} B_j

        # ---- Phi_code: dimensionless Yukawa potential kernel sum ----
        # hat_phi = pref * Phi_code,  where pref = g*T^2_nu/hat_m_phi
        # Phi_code = -1/(4pi) * [exp(-xi)/r * sum_outer  +  sinh(xi)/r * sum_inner]
        exp_over_r  = np.exp(-xi) / R
        sinh_over_r = np.where(screened, 0.0,
                               np.where(xi > cut, np.sinh(xi) / R, xi / R))

        self.hat_phi = -(exp_over_r * sum_outer + sinh_over_r * sum_inner)
        self.hat_phi *= shells.alpha / (4.0 * np.pi) * m0**2

        # ---- F_kernel = dPhi_code / d(tilde_r) ----
        # Outer kernel (shells j < i, i.e. inside r):
        #   d/dr[-exp(-xi)/r] = exp(-xi)*(1+xi)/r^2
        #
        # Inner kernel (shells j > i, i.e. outside r):
        #   d/dr[sinh(xi)/r] = sinh(xi)*(xi*coth(xi)-1)/r^2
        R2 = R**2

        outer_kernel = np.exp(-xi) * (1.0 + xi) / R2

        # xi*coth(xi) - 1 = xi^2/3 + xi^4/45 + ...
        xi_coth_m1 = np.where(
            xi > 1e-4,
            xi * np.cosh(xi) / np.sinh(np.maximum(xi, 1e-300)) - 1.0,
            xi**2/3.0 + xi**4/45.0
        )
        inner_kernel = np.where(
            screened,
            0.0,
            np.sinh(xi) * xi_coth_m1 / R2
        )

        self.F_kernel = (outer_kernel * sum_outer + inner_kernel * sum_inner) / (4.0*np.pi)




