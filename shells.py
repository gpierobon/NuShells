import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Shells:
    def __init__(self):
        self.dtype = np.dtype([
            ('ID',  np.int32),
            ('R',   np.float64),
            ('q',   np.float64),
            ('ell', np.float64),
            ('w',   np.float64),
            ('phi', np.float64),
            ('m',   np.float64),
            ('eps', np.float64),
        ])

        self.data = None
        self.inv_cdf = None
        self.Rmax = None
        self.m0 = 1.0
        self.a = 1.0

    def initialise(self, Nshells,
                   Rmin=1e-5, Rmax=10.0,
                   Psi0=1e-5, R0=2.0, w_min=1e-6):
        """
        """
        self.Rmin = Rmin
        self.Rmax = Rmax
        r_grid = np.geomspace(Rmin, Rmax, Nshells)
        dr = r_grid[1:] - r_grid[:-1]
        dr = np.empty(Nshells)
        dr[:-1] = r_grid[1:] - r_grid[:-1]
        dr[-1]  = dr[-2]

        q_samples = self.sample_q(Nshells)
        mu_samples = np.random.uniform(-1.0, 1.0, Nshells)

        self.data = np.zeros(Nshells, dtype=self.dtype)

        for i in range(Nshells):
            r    = r_grid[i]
            dr_i = dr[i]
            q    = q_samples[i]
            mu   = mu_samples[i]
            dmu = 2.0 / Nshells

            Psi = Psi0 * np.exp(-r**2 / (2 * R0**2))
            w = self.compute_weight(r, dr_i, mu, dmu, q, Psi)

            qr  = q * mu
            ell = r * q * np.sqrt(1.0 - mu**2)

            self.data['ID'][i]   = i
            self.data['R'][i]    = r
            self.data['q'][i]    = qr
            self.data['ell'][i]  = ell
            self.data['w'][i]    = w
            self.data['phi'][i]  = Psi

        # weights adjustment
        self.data['w'] /= np.max(self.data['w'])
        self.data['w'] = np.maximum(self.data['w'], w_min)

        self.update_mass() # mass and eps

    def setup_FD_sampler(self, qmax, ngrid):
        """
        Prepare the Fermi-Dirac inverse CDF.
        """
        q = np.linspace(0.0, qmax, ngrid)
        pdf = q**2 / (np.exp(q) + 1.0)
        cdf = scipy.integrate.cumulative_trapezoid(pdf, q, initial=0)
        cdf /= cdf[-1]
        self.inv_cdf = scipy.interpolate.interp1d(cdf, q, kind='linear')

    def sample_q(self, N, qmax=20.0, ngrid=10000):
        """
        Sample N momenta using the precomputed inverse CDF.
        """
        self.setup_FD_sampler(qmax, ngrid)
        u = np.random.rand(N)
        return self.inv_cdf(u)

    def compute_weight(self, r, dr, mu, dmu, q, Psi):
        f0 = 1.0 / (np.exp(q) + 1.0)
        dfdlnq = -(q * np.exp(q)) / (np.exp(q) + 1.0)**2
        pert = 1.0 + (dfdlnq / f0) * Psi * mu
        return 8 * np.pi**2 * r**2 * dr * (1.0 - Psi) * dmu * pert

    def update_mass(self):
        self.data['m'] = self.m0 + self.data['phi']
        self.data['eps'] = self.a * np.sqrt(
            self.data['q']**2 +
            self.data['ell']**2 +
            self.data['m']**2
        )

    def sort(self):
        """
        Stable in-place sort by radius.
        Use mergesort for nearly sorted arrays in time-stepping simulations.
        """
        self.data.sort(order='R', kind='mergesort')

    #def enclosed_mass(self):
    #    """
    #    Compute cumulative mass assuming sorted by R.
    #    """
    #    return np.cumsum(self.data['w'])

    def step(self, dt):
        """
        Advance system by one conformal time step deta
        using kick-drift-kick (leapfrog).
        """

        # --- First compute force from current configuration
        solver = ForceSolver(self)
        dmdr_term = solver.computeF(include_self=False)
        accel = self.ell**2 / (self.eps * self.R**3) - self.m * dmdr_term

        # ---- Kick (half step in momentum)
        self.data['q'] += 0.5 * dt * accel
        self.update_mass()

        # ---- Drift (full step in position)
        self.data['R'] += dt * (self.q / self.eps)

        # --- Handling of boundary conditions ---
        mask = self.data['R'] < self.Rmin
        self.data['R'][mask] = 2*self.Rmin - self.data['R'][mask]
        self.data['q'][mask] *= -1.0

        mask = self.data['R'] > self.Rmax
        self.data['R'][mask] = 2*self.Rmax - self.data['R'][mask]
        self.data['q'][mask] *= -1.0

        # ---- Recompute force at new positions
        self.sort()
        solver = ForceSolver(self)
        dmdr_term = solver.computeF(include_self=False)
        accel = self.ell**2 / (self.eps * self.R**3) - self.m * dmdr_term

        # ---- Final half kick
        self.data['q'] += 0.5 * dt * accel
        self.update_mass()

    def phase_space(self):
        fig = plt.figure(figsize=(8,6))
        sizes = 50 * self.w
        #plt.scatter(self.R, self.q,
        #            s=sizes,
        #            alpha=0.5)
        plt.scatter(self.R, self.q, s=5, alpha=0.5)
        plt.xlabel(r"$R$")
        plt.ylabel(r"$q_r$")
        plt.xscale('log')
        plt.xlim(self.Rmin, self.Rmax)
        plt.ylim(-10, 10)
        plt.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)

        return fig

    @property
    def ID(self):
        return self.data['ID']

    @property
    def R(self):
        return self.data['R']

    @property
    def q(self):
        return self.data['q']

    @property
    def ell(self):
        return self.data['ell']

    @property
    def w(self):
        return self.data['w']

    @property
    def m(self):
        return self.data['m']

    @property
    def eps(self):
        return self.data['eps']

    @property
    def phi(self):
        return self.data['phi']


class ForceSolver:
    def __init__(self, shells):
        """
        R must be sorted ascending, radii of previous step shells
        """

        self.a = shells.a
        self.w = shells.w
        self.R = shells.R
        self.q = shells.q
        self.ell = shells.ell
        self.eps = shells.eps
        self.phi_prev = shells.phi
        self.m = shells.m

        if not np.all(self.R[:-1] <= self.R[1:]):
            raise ValueError("R must be sorted in ascending order")

        A = self.w * self.m / self.eps * np.sinh(self.R) / self.R
        B = self.w * self.m / self.eps * np.exp(-self.R) / self.R

        self.cumA = np.cumsum(A)
        self.cumB = np.cumsum(B)
        self.totalB = self.cumB[-1]

    def computeF(self, include_self=True):
        r = self.R

        sum_outer = np.concatenate(([0.0], self.cumA[:-1]))
        sum_inner = self.totalB - np.concatenate(([0.0], self.cumB[:-1]))

        if not include_self: # Remove self-contribution
            #A_self = self.cumA - np.concatenate(([0.0], self.cumA[:-1]))
            B_self = self.cumB - np.concatenate(([0.0], self.cumB[:-1]))
            sum_inner = sum_inner - B_self


        sinh_r = np.sinh(r)
        coth_r = np.cosh(r) / sinh_r

        exp_term = np.exp(-r) / r
        r_term = sinh_r / r

        ## New potential 
        self.phi = - exp_term * sum_outer - r_term * sum_inner # again, g=1

        # Force
        exp_r_term = np.exp(-r) / r**2 * (1+r)
        inner_r_term = sinh_r / r**2 * (1 - r * coth_r)

        #exp_r_term = exp_term * (1 + r) / r
        #inner_r_term = r_term / r * (1 - r * coth_r)

        force = exp_r_term * sum_outer + inner_r_term * sum_inner
        force *= self.a**2 / self.eps

        return force


