import time
import scipy
import h5py as h5
import numpy as np
import warnings

import ic
from timing import timed
from force import solveForce
from phi import solvePhi, interpPhi



## Constants
T_NU_EV = 1.676e-4             # Neutrino temperature today [eV]
Q_MEAN  = 3.1514               # <q/T_nu> for Fermi-Dirac distribution
EV_2_PC = 1/(1.57e23)          # eV to pc conversion
PC_2_M  = 3.08567758128e16     # pc to meter conversion
H0_EV   = 1.444e-33            # Hubble constant today  [eV]




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
            ('F_fs', np.float64),  # Free-streaming term
            ('F_lr', np.float64),  # Long-range term
        ])

        self.data     = None
        self.inv_cdf  = None
        self.cumMass  = None
        self.hdf5_io  = None
        self.iter_m   = None
        self.iter_tol = None

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

        # Grid bounds [tilde_r]
        self.Rmin  = None
        self.Rmax  = None

        # Time
        self.a      = None
        self.eta    = None
        self.dt     = None
        self.a_ini  = None
        self.a_end  = None   # when force range shrinks below Rmin


    @timed("Init")
    def init(self, Nshells,
             g        = 1e-26,
             m_phi    = 1e-29,          # eV
             m_nu     = 0.1,            # eV
             T_nu     = T_NU_EV,        # eV (defined in constants.py) 
             H0       = H0_EV,          # eV (defined in constants.py)
             kappa    = 0.75,           # a_ini = kappa * a_NR
             kappa2   = 0.75,           # a_end = kappa2 * 1 / R_min
             dt_frac  = 0.01,           # dt = dt_frac * Rmin
             Psi0     = 1e-5,           # amplitude of initial perturbation
             soft     = 1e-3,           # softening length
             iter_m   = 'anderson',     # method in phi iteration
             iter_tol = 1e-3,           # tolerance in phi iteration
             R0       = None,           # perturbation scale [1/m_phi]
             w_min    = None,           # weight floor
             hdf5_io  = False,          # HDF5 I/O
             verb     = False           # Print summary 
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
        self.hdf5_io  = hdf5_io
        self.iter_m   = iter_m
        self.iter_tol = iter_tol

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

        q_total = ic.sample_q(self, Nshells)
        mu_samples  = np.random.uniform(-1.0, 1.0, Nshells)  # cos(theta)

        self.data = np.zeros(Nshells, dtype=self._dtype)

        phi_guess = -0.01

        # -------------------------------------------------------------------
        # Initialisation loop
        # -------------------------------------------------------------------
        for i in range(Nshells):
            r    = r_grid[i]
            dr_i = dr[i]
            q    = q_total[i]
            mu   = mu_samples[i]

            # Initial dimensionless Newtonian potential (Gaussian profile)
            #Psi = ic.get_profile(ptype=ic_prof, Psi0, R0) ## TODO
            Psi = Psi0 * np.exp(-r**2 / (2.0 * R0**2))

            # Weight (dimensionless, factor (T_nu/M_phi)^3)
            w_i = ic.compute_weight(r, dr_i, mu, q, Psi)

            # Radial momentum (hat_q_r)
            hat_qr = q * mu

            # Angular momentum  hat_ell = hat_r * hat_q_T
            hat_qT = q * np.sqrt(max(1.0 - mu**2, 0.0))
            hat_ell = r * hat_qT

            # Initial hat_phi, guess is the background
            hat_phi = phi_guess

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
        # Compute initial hat_phi self-consistently 
        # Then, mass, energy, cumulative mass are updated due to new phi value
        # For the first step we always choose anderson to make the convergence
        # is reached, however the tolerance can still be chosen
        # -------------------------------------------------------------------
        _ = solvePhi(self, method="anderson", tol=iter_tol, verbose=verb)

        self._update_mass()
        self._update_cumMass()
        min_m = np.min(self.m/self.m0)
        max_m = np.max(self.m/self.m0)

        # Commit forces for the first half-kick
        F_fs, F_lr = solveForce(self)
        self.data["F_fs"] = F_fs
        self.data["F_lr"] = F_lr

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
            print(f"   m/m0    = [{min_m:.5f}, {max_m:5f}]\n")
            print(f"   m_nu/T_nu = {self.m0_hat:.3e} ")
            print(f"   m_phi/H0  = {self.m_phi_hat:.3e}\n")
            print(f"   a_NR  = {a_NR:.3e}   ")
            print(f"   a_ini = {self.a_ini:.3e}  (z={z_ini:.1f})")
            print(f"   a_end = {self.a_end:.3e}  (z={z_end:.1f})")
            print(f"   dt    = {self.dt:.3e}     \n")
            print(f"   lambda_FS  = {lambda_FS_NR:.3e}  (free-streaming)")
            print(f"   lambda_phi = {lambda_phi_ini:.3e}  (Yukawa range)")
            print(f"   Rmin       = {self.Rmin:.3e}")
            print(f"   Rmax       = {self.Rmax:.3e}\n")
            print("=" * 60)
            print(f" \n ICs took {time.time()-start:.5f} s\n")


    # -----------------------------------------------------------------------
    # Mass/energy update
    # -----------------------------------------------------------------------
    @timed("update_mass")
    def _update_mass(self):
        """
        """
        self.data['m']   = self.m0 + self.phi
        self.data['eps'] = np.sqrt(self.q**2 + self.ell**2 / self.R**2
                                 + self.a**2 * self.m**2)

    # -----------------------------------------------------------------------
    # Update cumulative weighted mass
    # -----------------------------------------------------------------------
    @timed("update_cumMass")
    def _update_cumMass(self):
        """
        Cumulative sum M(<tilde_r) = sum_{R_i <= tilde_r} w_i * m_i / eps_i
        Assumes data is already sorted ascending by R.
        """
        self._cumMass = np.cumsum(self.w * self.m / self.eps)

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
    @timed("Sort")
    def _sort(self):
        """Sort shells in ascending hat_r order (stable / merge sort)."""
        idx = np.argsort(self.R, kind='stable')
        for arr in (self.ID, self.R, self.q, self.ell, self.w, \
                    self.m, self.eps, self.phi):
            arr[:] = arr[idx]

    # -----------------------------------------------------------------------
    #  Time step: kick-drift-kick leapfrog
    # -----------------------------------------------------------------------
    def step(self, verb=False):
        """
        Advance by one tilde_eta step using kick-drift-kick (KDK) leapfrog.

        EOM:
            d(hat_r)/d(hat_eta) = hat_q / hat_eps

            d(hat_q)/d(hat_eta) = hat_ell^2 / (hat_eps * hat_r^3)          [FS]
                                - a^2 * alpha * hat_m / hat_eps * F_kernel [LR]

            da      /d(hat_eta) = sqrt(a) / m_phi_hat                [update_a]

        Force prefactor:
            alpha = g^2 * T^2_nu / m^2_phi
        """
        dt = self.dt
        soft = self.soft
        F_fs_prev = self.F_fs
        F_lr_prev = self.F_lr
        ## ADD GRAV

        # -- Half kick --
        self.data['q'] += 0.5 * dt * (F_fs_prev - F_lr_prev)
        self._update_mass()

        # -- Store pre-drift state for phi interpolation --
        R_old   = self.data['R'].copy()
        phi_old = self.data['phi'].copy()

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
        self._update_mass()
        self._sort()
        #self._update_cumMmass() // For gravity 

        # -- Phi and Force updates
        phi0_interp = interpPhi(R_old, phi_old, self.data['R'])
        self.data['phi'] = phi0_interp
        _ = solvePhi(self, method=self.iter_m, tol=self.iter_tol, verbose=verb)

        self._update_mass()
        self._update_cumMass()
        min_m = np.min(self.m/self.m0)
        max_m = np.max(self.m/self.m0)

        F_fs, F_lr = solveForce(self)
        self.data["F_fs"] = F_fs
        self.data["F_lr"] = F_lr
        ## ADD GRAV

        # -- Second half kick  --
        self.data['q'] += 0.5 * dt * (F_fs - F_lr)


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
    def _save_hdf5(self, path, step_index):
        """Save shell state to a hdf5 file."""
        with h5.File(f"{path}/shells_{step_index:05d}.hdf5", 'w') as f:

            head = f.create_group("Header")
            head.attrs['N'] = self.N
            head.attrs['a'] = self.a
            head.attrs['g'] = self.g
            head.attrs['m_phi'] = self.m_phi
            head.attrs['alpha'] = self.alpha
            head.attrs['Rmin'] = self.Rmin
            head.attrs['Rmax'] = self.Rmax
            head.attrs['m0'] = self.m0
            head.attrs['dt'] = self.dt

            data = f.create_group("Data")
            data.create_dataset("ID",   data=self.ID,    dtype=np.int32)
            data.create_dataset("R",    data=self.R,     dtype=np.float32)
            data.create_dataset("q",    data=self.q,     dtype=np.float32)
            data.create_dataset("m",    data=self.m,     dtype=np.float32)
            data.create_dataset("w",    data=self.w,     dtype=np.float32)
            data.create_dataset("phi",  data=self.phi,   dtype=np.float32)
            data.create_dataset("F_fs", data=self.F_fs,  dtype=np.float32)
            data.create_dataset("F_lr", data=self.F_lr,  dtype=np.float32)


    @timed("I/O")
    def _save(self, path, step_index):
        """Save shell state to a text or hdf5 file."""
        if self.hdf5_io:
            self._save_hdf5(path, step_index)
        else:
            header = (
                f"ID R q ell m w phi F_fs F_lr\n"
                f"a={self.a:.6e}\n"
                f"Rmin={self.Rmin:.2e}, Rmax={self.Rmax:.2e}"
            )
            np.savetxt(
                f"{path}/shells_{step_index:05d}.txt",
                np.column_stack([
                    self.data['ID'], self.data['R'],
                    self.data['q'],  self.data['ell'],
                    self.data['m'],  self.data['w'],
                    self.data['phi'], self.data['F_fs'], self.data['F_lr']
                ]),
                header=header,
                fmt="%d %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e"
            )


    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    def _load_hdf5(self, path, step_index):
        """Load shell state from hdf5 file."""
        with h5.File(f"{path}/shells_{step_index:05d}.hdf5", 'r') as f:
            N = int(f['Header'].attrs['N'])
            a = int(f['Header'].attrs['a'])
            g = int(f['Header'].attrs['g'])
            #...

            self.data = np.zeros(N, dtype=self._dtype)

            self.data['ID']  = f['Data/ID']
            self.data['R']  = f['Data/R']
            self.data['q']  = f['Data/q']
            self.data['m']  = f['Data/m']
            self.data['w']  = f['Data/w']
            self.data['phi']  = f['Data/phi']
            self.data['F_fs']  = f['Data/F_fs']
            self.data['F_lr']  = f['Data/F_lr']


    @timed("I/O")
    def _load(self, path, step_index, hdf5_io=True):
        """Load shell state from text or hdf5 file."""
        if hdf5_io:
            self._load_hdf5(path, step_index)
        else:
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
            self.data['w']   = raw[:,5]
            self.data['phi'] = raw[:,6]
            self.data['F_fs'] = raw[:,7]
            self.data['F_lr'] = raw[:,8]


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
    def F_fs(self):  return self.data['F_fs']
    @property
    def F_lr(self):  return self.data['F_lr']
    @property
    def N(self):    return len(self.data)


