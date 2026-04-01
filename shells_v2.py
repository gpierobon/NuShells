import time
import scipy
import numpy as np
import warnings

from force import ForceSolver
from constants import *


# ---------------------------------------------------------------------------
class Shells:
# ---------------------------------------------------------------------------
    def __init__(self):
        self._dtype = np.dtype([
            ('ID',  np.int32),     # Shell ID
            ('R',   np.float64),   # hat_r   = r / r_phi 
            ('q',   np.float64),   # hat_qr  = q_r / T_nu
            ('ell', np.float64),   # hat_ell = r * q_perp * H0 / T_nu
            ('w',   np.float64),   # weight (dimensionless)
            ('phi', np.float64),   # hat_phi = phi / T_nu
            ('m',   np.float64),   # hat_m   = m_tilde / T_nu
            ('eps', np.float64),   # hat_eps = eps / T_nu
        ])

        self.data    = None
        self.inv_cdf = None
        self.cumMass = None

        # Physical inputs [eV]
        self.g      = None
        self.m_phi  = None
        self.r_phi  = None
        self.m_nu   = None
        self.T_nu   = None
        self.H0     = None

        # Derived dimensionless quantities
        self.alpha     = None   # g^2 * T^2 r_phi^2
        self.hat_M_phi = None   # M_phi / H0
        self.m0_hat    = None   # m_nu / T_nu 
        self.m0        = None   # alias for m0_hat
        self.m_bkg     = None   # Effective mass in T units
        self.phi_bkg   = None   # <phi> in T/g units
        self.F         = None   # Force kernel

        # Grid bounds [tilde_r]
        self.Rmin  = None
        self.Rmax  = None

        # --- Time ---
        self.a      = None
        self.eta    = None
        self.dt     = None
        self.a_ini  = None
        self.a_end  = None   # when force range shrinks below Rmin


    def init(self, Nshells,
             g       = 1e-26,
             m_phi   = 1e-29,    # eV
             m_nu    = 0.1,      # eV
             T_nu    = T_NU_EV,  # eV (defined in constants.py) 
             H0      = H0_EV,    # eV (defined in constants.py)
             kappa   = 0.75,     # a_ini = kappa * a_NR
             kappa2  = 0.75,     # a_end = kappa2 * 1 / R_min
             dt_frac = 0.01,     # dt = dt_frac * sqrt(a_ini)
             Psi0    = 1e-5,     # amplitude of initial perturbation
             soft    = 1e-3,     # "softening length"
             R0      = None,     # perturbation scale [1/H0]; default below
             w_min   = None,     # weight floor
             verb    = False
            ):
        """
        Initialise N shells in dimensionless units.

        All masses / temperatures given in eV.
        After this call, no eV quantities appear in the dynamics.

        Parameters
        ----------
        Nshells  : int    number of shells
        g        : float  coupling constant           (default: 1e-26)
        m_phi    : float  mediator mass [eV]          (default: 1e-29 eV)
        m_nu     : float  neutrino mass [eV]          (default: 0.1 eV)
        kappa    : float  a_ini = kappa * a_NR        (default 0.1)
        dt_frac  : float  dt = dt_frac * sqrt(a_ini)  (default 0.01)
        Psi0     : float  amplitude of perturbation   (default: 1e-5)
        R0       : float  scale of perturbation
        w_min    : float  minimum weight floor
        verb     : bool   print diagnostics
        """

        start = time.time()

        # -------------------------------------------------------------------
        # Store eV inputs for reference
        # -------------------------------------------------------------------
        self.g     = g
        self.m_phi = m_phi
        self.r_phi = 1/m_phi
        self.m_nu  = m_nu
        self.T_nu  = T_nu
        self.H0    = H0
        frange = self.r_phi*EV_2_PC*1e-6 # Mpc

        # -------------------------------------------------------------------
        # Derived dimensionless ratios
        # -------------------------------------------------------------------
        self.alpha     = g**2 * m_nu**2 / m_phi**2
        self.m_phi_hat = m_phi / H0       # for the da/d(tilde_eta) update
        self.m0_hat    = m_nu  / T_nu
        self.m0        = self.m0_hat

        # -------------------------------------------------------------------
        # Scale factors
        # -------------------------------------------------------------------
        a_NR       = 1.0 / self.m0_hat
        a_ini      = kappa * a_NR
        self.a     = a_ini
        self.a_ini = a_ini
        self.eta   = 2/H0*np.sqrt(self.a)

        # -------------------------------------------------------------------
        # Background value
        # -------------------------------------------------------------------
        m_bkg, phi_bkg = self._solve_background()
        self.m_bkg     = m_bkg
        self.phi_bkg   = phi_bkg

        # -------------------------------------------------------------------
        # Length/time scales
        # -------------------------------------------------------------------
        lambda_phi_ini = 1.0 / a_ini
        lambda_FS_rad  = self._lambdaFS_rad()
        lambda_FS_NR   = lambda_FS_rad + 2.0 / self.m_phi_hat * \
                         (1.0/np.sqrt(a_ini) - 1.0/np.sqrt(a_NR))

        self.Rmin  = 0.01 * lambda_phi_ini
        self.Rmax  = 50.0 * max(lambda_phi_ini, lambda_FS_NR)
        self.a_end = 1.0 / self.Rmin * kappa2

        if R0 is None:
            R0 = lambda_phi_ini

        self.dt = dt_frac * self.Rmin
        self.soft = soft

        # -------------------------------------------------------------------
        # Redshifts
        # -------------------------------------------------------------------
        z_ini = 1/self.a_ini-1
        z_end = 1/self.a_end-1

        # -------------------------------------------------------------------
        # r_hat, q_hat
        # -------------------------------------------------------------------
        r_grid  = np.geomspace(self.Rmin, self.Rmax, Nshells)
        dr      = np.empty(Nshells)
        dr[:-1] = r_grid[1:] - r_grid[:-1]
        dr[-1]  = dr[-2]

        q_total = self._sample_q(Nshells)
        mu_samples  = np.random.uniform(-1.0, 1.0, Nshells)      # cos(theta)

        self.data = np.zeros(Nshells, dtype=self._dtype)


        # -------------------------------------------------------------------
        # Initialisation loop
        # -------------------------------------------------------------------
        for i in range(Nshells):
            r    = r_grid[i]
            dr_i = dr[i]
            q    = q_total[i]
            mu   = mu_samples[i]

            # Initial dimensionless Newtonian potential (Gaussian profile)
            Psi = Psi0 * np.exp(-r**2 / (2.0 * R0**2))

            # Weight (dimensionless, factor (T_nu/M_phi)^3)
            w_i = self._compute_weight(r, dr_i, mu, q, Psi)

            # Radial momentum (hat_q_r)
            hat_qr = q * mu

            # Angular momentum  hat_ell = hat_r * hat_q_T
            hat_qT = q * np.sqrt(max(1.0 - mu**2, 0.0))
            hat_ell = r * hat_qT

            # Initial hat_phi, guess is the background
            hat_phi = self.phi_bkg

            self.data['ID'][i]  = i
            self.data['R'][i]   = r
            self.data['q'][i]   = hat_qr
            self.data['ell'][i] = hat_ell
            self.data['w'][i]   = w_i
            self.data['phi'][i] = hat_phi

        # -------------------------------------------------------------------
        # Normalise weights 
        # -------------------------------------------------------------------
        self.data['w'] /= np.max(self.data['w'])
        if w_min is None:
            w_min = 1e-3
        self.data['w'] = np.maximum(self.data['w'], w_min)

        # -------------------------------------------------------------------
        # Initial mass, energy, enclosed mass
        # -------------------------------------------------------------------
        self._update_mass()
        self._update_enclosed_mass()

        # -------------------------------------------------------------------
        # Compute initial hat_phi self-consistently 
        # -------------------------------------------------------------------
        solver = ForceSolver(self)
        self.data['phi'] = solver.hat_phi
        self._update_mass()
        fs = self.data['ell']**2 / (self.data['eps'] * self.data['R']**3)
        lr = self.a**2 * self.alpha * self.data['m'] / self.data['eps'] * solver.F_kernel
        self.F = fs - lr

        # -------------------------------------------------------------------
        # Summary print
        # -------------------------------------------------------------------
        if verb:
            print("")
            print("=" * 60)
            print("  Simulation parameters")
            print("=" * 60)
            print(f" \n   Nshells = {self.N}\n")
            print(f"   g       = {g:.3e}")
            print(f"   m_phi   = {m_phi:.3e} eV")
            print(f"   m_nu    = {m_nu:.3e} eV")
            print(f"   T_nu    = {T_nu:.3e} eV")
            print(f"   H0      = {H0:.3e} eV")
            print(f"   Range   = {frange:.3e} Mpc\n")
            print(f"   alpha   = {self.alpha:.3e}  ")
            print(f"   m_nu/T_nu = {self.m0_hat:.3e} ")
            print(f"   m_phi/H0  = {self.m_phi_hat:.3e}\n")
            print(f"   a_NR  = {a_NR:.3e}   ")
            print(f"   a_ini = {self.a_ini:.3e}  (z={z_ini:.1f})")
            print(f"   a_end = {self.a_end:.3e}  (z={z_end:.1f})")
            print(f"   dt    = {self.dt:.3e}     \n")
            print(f"   lambda_FS  = {lambda_FS_NR:.3e}  (free-streaming)")
            print(f"   lambda_phi = {lambda_phi_ini:.3e}  (Yukawa range)")
            print(f"   Rmin       = {self.Rmin:.3e}")
            print(f"   Rmax       = {self.Rmax:.3e}")
            print("=" * 60)
            print(f" \n ICs took {time.time()-start:.5f} s\n")

    # -----------------------------------------------------------------------
    #  Fermi-Dirac sampler
    # -----------------------------------------------------------------------
    def _build_fd_icdf(self, qmax=20.0, ngrid=10_000):
        """Build inverse CDF for f(hat_q) * hat_q^2 (momentum-space FD)."""
        q   = np.linspace(0.0, qmax, ngrid)
        pdf = q**2 / (np.exp(np.clip(q, 0, 500)) + 1.0)
        cdf = scipy.integrate.cumulative_trapezoid(pdf, q, initial=0.0)
        cdf /= cdf[-1]
        self.inv_cdf = scipy.interpolate.interp1d(
            cdf, q, kind='linear',
            bounds_error=False,
            fill_value=(0.0, qmax)
        )

    def _sample_q(self, N):
        """Return N samples of hat_q from the Fermi-Dirac distribution."""
        if self.inv_cdf is None:
            self._build_fd_icdf()
        return self.inv_cdf(np.random.rand(N))

    # -----------------------------------------------------------------------
    # Weight computation
    # -----------------------------------------------------------------------
    def _compute_weight(self, r, dr, mu, q, Psi):
        """
        Phase-space weight for one shell.

        Parameters
        ----------
        r   : hat_r [1/H0]
        dr  : radial bin width [1/H0]
        mu  : cos(theta) in [-1,1]
        q   : hat_q_total = q/T_nu, O(1)
        Psi : dimensionless Newtonian perturbation potential

        Returns
        -------
        w : float, proportional to r^2 dr * q^2 * f(q) * (1 + dipole perturbation)
            Units: [1/H0]^3 before normalisation (cancels after max-normalisation).

        Perturbation from linearised Boltzmann equation:
            delta_f / f_0 = -(d ln f_0 / d ln q) * Psi * mu
        where d ln f_0 / d ln q = -q * exp(q) / (exp(q)+1)  / f_0
        """
        Nq     = 1.5 * scipy.special.zeta(3) # normalisation: int q^2/(e^q+1)dq 

        f0     = 1.0 / (np.exp(q) + 1.0)
        dfdlnq = -(q * np.exp(q)) / (np.exp(q) + 1.0)**2   # = q * df/dq
        pert   = 1.0 + (dfdlnq / f0) * Psi * mu

        weight = (
            8.0 * np.pi**2    # solid angle factor (4pi * 2pi from phi integral)
            * r**2 * dr       # radial volume element [1/H0]^3
            * (1.0 - Psi)     # metric perturbation correction
            * 2.0 * Nq        # FD normalisation
            * pert            # linearised perturbation (delta f) 
        )
        return weight

    # -----------------------------------------------------------------------
    # Background potential and effective mass
    # -----------------------------------------------------------------------
    def _solve_background(self, max_iter=100, tol=1e-5):
        """
        Solve for the self-consistent effective mass hat_m = hat_m0 + <hat_phi>
        by iterating hat_m = hat_m0 / (1 + alpha/(4pi) * I(a, hat_m)), with
               I(a, hat_m) = int dq q^2/(exp(q)+1) * 1/sqrt(q^2 + a^2*M^2)
        This form guarantees hat_m > 0
        Returns
        -------
        hat_m     : float   self-consistent effective mass  hat_m0 + <hat_phi>
        phi_bg    : float   background <hat_phi> = hat_m - hat_m0
                            Units are 1/T and g/T, respectively.
        """
        alpha  = self.alpha
        m0     = self.m0_hat
        a      = self.a

        def I(M):
            if M <= 0:
                return np.inf
            def integrand(q):
                den = np.sqrt(q**2 + a**2 * M**2)
                return q**2 / (np.exp(np.clip(q, 0, 500)) + 1) / den
            result, _ = scipy.integrate.quad(integrand, 0, 30)
            return result

        # Initial guess: start from bare mass (phi=0)
        M = m0

        for i in range(max_iter):
            M_new = m0 / (1.0 + alpha / (4.0 * np.pi) * I(M))

            if M_new <= 0:
                raise ValueError(f"M went negative at iteration {i}. "
                                 f"alpha={alpha:.3e} may be too large.")

            if abs(M_new - M) < tol * m0:
                phi_bkg = M_new - m0       # always negative
                return M_new, phi_bkg

            M = M_new

        raise ValueError(f"Background phi did not converge after {max_iter} iterations. "
                         f"Last M={M:.6e}, m0={m0:.6e}")

    # -----------------------------------------------------------------------
    # Mass/energy update
    # -----------------------------------------------------------------------
    def _update_mass(self):
        """
        """
        self.data['m']   = self.m0 + self.data['phi']
        self.data['eps'] = np.sqrt(
            self.data['q']**2
            + self.data['ell']**2 / self.data['R']**2
            + self.a**2 * self.data['m']**2          # a^2 inside sqrt
        )

    # -----------------------------------------------------------------------
    # Update cumulative weighted mass
    # -----------------------------------------------------------------------
    def _update_enclosed_mass(self):
        """
        Cumulative sum M(<tilde_r) = sum_{R_i <= tilde_r} w_i * hat_m_i / hat_eps_i

        Assumes data is already sorted ascending by R.
        Used by ForceSolver for the gravity term.
        """
        self._cumMass = np.cumsum(
            self.data['w'] * self.data['m'] / self.data['eps']
        )

    # -----------------------------------------------------------------------
    # Scale factor update
    # -----------------------------------------------------------------------
    def _update_a(self):
        """
        Scale factor update
        Matter domination: da/deta = H0*sqrt(a)
                           da/d(hat_eta) = sqrt(a) / m_phi_hat
        """
        self.a += self.dt * np.sqrt(self.a) / self.m_phi_hat

    # -----------------------------------------------------------------------
    # Sorting routine
    # -----------------------------------------------------------------------
    def _sort(self):
        """Sort shells in ascending hat_r order (stable / merge sort)."""
        idx = np.argsort(self.R, kind='stable')
        for arr in (self.ID, self.R, self.q, self.ell, self.w, \
                    self.m, self.eps, self.phi):
            arr[:] = arr[idx]

    # -----------------------------------------------------------------------
    #  Time step: kick-drift-kick leapfrog
    # -----------------------------------------------------------------------
    def step(self, include_yukawa=True): #, include_gravity=False, G_code=1.0):
        """
        Advance by one tilde_eta step using kick-drift-kick (KDK) leapfrog.

        EOM:
            d(hat_r)/d(hat_eta) = hat_q / hat_eps

            d(hat_q)/d(hat_eta) = hat_ell^2 / (hat_eps * hat_r^3)          [FS]
                                - a^2 * alpha * hat_m / hat_eps * F_kernel [LR]

            da      /d(hat_eta) = sqrt(a) / m_phi_hat                [update_a]

        Force prefactor:
            alpha = g^2 * T^2_nu / m^2_phi

        Parameters
        ----------
        include_yukawa  : bool   include long-range Yukawa force
        """
        dt = self.dt
        soft = self.soft
        F_prev = self.F
        FP = self.alpha

        def _force():

            # Here add interpolation!!
            #
            #
            solver = ForceSolver(self)

            # Update hat_phi from the newly computed potential
            self.data['phi'] = solver.hat_phi
            self._update_mass()
            F = solver.F_kernel  # dPhi_code/d(tilde_r) for each shell  [O(1)]

            ## Free-streaming: tilde_ell^2 / (hat_eps * tilde_r^3)
            fs = self.data['ell']**2 / (self.data['eps'] * self.data['R']**3)

            # Long-range: a^2 * alpha * hat_m / hat_eps * F_kernel
            lr = np.zeros_like(fs)
            if include_yukawa:
                lr = self.a**2 * FP * self.data['m'] / self.data['eps'] * F

            ## Gravity (optional): -G * M(<r) / tilde_r^2
            grav = np.zeros_like(fs)
            #if include_gravity:
            #    M_enc = np.concatenate(([0.0], self._cumMass[:-1]))
            #    grav  = G_code * M_enc / self.data['R']**2

            return fs - lr - grav

        # -- Half kick --
        self.data['q'] += 0.5 * dt * F_prev
        self._update_mass()

        # -- Full drift --
        self.data['R'] += dt * self.data['q'] / self.data['eps']

        # -- Reflecting boundary conditions --
        lo = self.data['R'] < soft
        self.data['R'][lo]  = 2.0*self.Rmin - self.data['R'][lo]
        self.data['q'][lo] *= -1.0

        hi = self.data['R'] > self.Rmax
        self.data['R'][hi]  = 2.0*self.Rmax - self.data['R'][hi]
        self.data['q'][hi] *= -1.0

        # -- Update a and sort --
        self._update_a()
        self._sort()
        #self._update_enclosed_mass() // For gravity 

        # -- Second half kick  --
        F_new = _force()
        self.F = F_new
        self.data['q'] += 0.5 * dt * F_new

    # -----------------------------------------------------------------------
    # Free-streaming utility
    # -----------------------------------------------------------------------
    def _lambdaFS_rad(self, ai=1e-8):
        """Return the free-streaming scale in R.D. in units of 1/m_phi."""
        Omega_r = 9.2e-5
        def v(a):
            p = Q_MEAN * T_NU_EV / a
            E = np.sqrt(p**2 + self.m_nu**2)
            return p / E

        I, err = scipy.integrate.quad(v, ai, self.a_ini)
        lambda_FS_H0 = I / np.sqrt(Omega_r)

        return lambda_FS_H0/self.m_phi_hat

    # -----------------------------------------------------------------------
    # Save configuration
    # -----------------------------------------------------------------------
    def _save(self, path, step_index):
        """Save shell state to a text file."""
        header = (
            f"ID hat_r  hat_q  hat_ell  hat_m  hat_eps  hat_w  hat_phi\n"
            f"a={self.a:.6e}\n"
            f"Rmin={self.Rmin:.2e}, Rmax={self.Rmax:.2e}"
        )
        np.savetxt(
            f"{path}/shells_{step_index:05d}.txt",
            np.column_stack([
                self.data['ID'],  self.data['R'],
                self.data['q'], self.data['ell'],
                self.data['m'], self.data['eps'],
                self.data['w'], self.data['phi'],
            ]),
            header=header,
            fmt="%d %.3e %.3e %.3e %.3e %.3e %.3e %.3e"
        )

    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    def _load(self, path, step_index):
        """Load shell state from text file."""
        fname = f"{path}/shells_{step_index:05d}.txt"

        # Read header to extract scale factor
        with open(fname, "r") as f:
            f.readline()
            line = f.readline()
            self.a = float(line.split("=")[-1])
            line = f.readline()
            self.Rmin = float(line.split(",")[0].split('=')[1])
            self.Rmax = float(line.split(",")[1].split('=')[1])

        raw = np.loadtxt(fname)

        N = raw.shape[0]
        self.data = np.zeros(N, dtype=self._dtype)

        self.data['ID']  = raw[:,0].astype(np.int32)
        self.data['R']   = raw[:,1]
        self.data['q']   = raw[:,2]
        self.data['ell'] = raw[:,3]
        self.data['m']   = raw[:,4]
        #self.data['eps'] = raw[:,5]
        self.data['w']   = raw[:,6]
        #self.data['phi'] = raw[:,7]

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    def diagnostics(self):
        """Print key dimensionless quantities."""
        a = self.a
        print(f"  a                = {a:.4e}")
        print(f"  a / a_NR         = {a * self.m0_hat:.4e}  (>1 => NR)")
        print(f"  a / a_end        = {a / self.a_end:.4e}  (>1 => force off)")
        print(f"  xi_min = a*Rmin  = {a*self.Rmin:.4e}  (0.01 at a_ini)")
        print(f"  xi_max = a*Rmax  = {a*self.Rmax:.4e}  (>>1 => screened at Rmax)")
        print(f"  <|hat_q_r|>      = {np.mean(np.abs(self.data['q'])):.4e}  (~O(1))")
        print(f"  <tilde_ell/R>    = {np.mean(self.data['ell']/self.data['R']):.4e}  (~O(1))")
        print(f"  <hat_m>          = {np.mean(self.data['m']):.4e}  (~m0_hat={self.m0_hat:.3e})")
        print(f"  <hat_eps>        = {np.mean(self.data['eps']):.4e}  (~O(1) relativistic)")
        print(f"  <a^2*hat_m^2>    = {np.mean(a**2*self.data['m']**2):.4e}  (<<1 if rel.)")

    # -----------------------------------------------------------------------
    # Neutrino number density
    # -----------------------------------------------------------------------
    def density(self, nbins=200):
        """
        Bin shells by weight in hat_r and return (r_bins, delta).

        Returns
        -------
        r_c : array  hat_r bin centres
        n   : array  number density
        """
        self._sort()
        edges = np.geomspace(np.min(self.R), np.max(self.R), nbins + 1)
        r_c   = np.sqrt(edges[:-1] * edges[1:])
        vol   = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

        mass, _ = np.histogram(self.data['R'], bins=edges,
                               weights=self.data['w'])

        occ = mass > 0
        n      = np.full_like(r_c, np.nan)
        n[occ] = mass[occ] / vol[occ]
        return r_c, n

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------
    @property
    def ID(self):   return self.data['ID']
    @property
    def R(self):    return self.data['R']
    @property
    def q(self):    return self.data['q']
    @property
    def ell(self):  return self.data['ell']
    @property
    def w(self):    return self.data['w']
    @property
    def m(self):    return self.data['m']
    @property
    def eps(self):  return self.data['eps']
    @property
    def phi(self):  return self.data['phi']
    @property
    def N(self):    return len(self.data)


