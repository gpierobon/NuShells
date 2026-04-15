import os
import tqdm
import shutil
import numpy as np

from timing import timed, report, reset, start_wall, stop_wall
from shells import Shells
from phi import solvePhi, interpPhi
from force import solveYukawaForce, solveGravityForce


# === Main params ============================================================
Nshells = 1000         # Number of simulation shells
g = 1e-26              # Yukawa coupling constant
m_nu = 0.1             # Neutrino  mass (eV)
m_phi = 1e-29          # Scalar field mass (eV)

# === Time params ============================================================
dt_frac = 0.3          # Time step fractions (Courant-like)
ti_frac = 0.75         # Initial time controller (a_i=ti_frac*a_NR)
tf_frac = 1.0          # Final time controller (a_f=tf_frac*1/mphi)

# === IC params ==============================================================
ic_only = False        # Only create and save initial conditions
Psi0 = 1e-5            # Initial perturbation amplitude
ic_type = 'tophat'     # Perturbation profile: 'tophat', 'tophat_c', 
                       #                       'gaussian', 'gaussian_c',
                       #                       'exp_c, 'poly_c'
                       #                       '_c' profiles are compensated
w_min = 1e-12          # Minimum weight
seed = 9               # IC random seed 

# === Output params ==========================================================
nmeas = 50             # Number of measurements
logmeas = True         # Logarithmic spacing (in a) for measurements
odir = 'output'        # Output directory (data and logs)
hdf5_io = True         # Saved in HDF5 format (True) or .txt (False)
to_file = True         # Print to file (True) or to screen (False)
verb = 1               # Verbosity level (0:INFO, 1:DEBUG)

# === Force params ===========================================================
method = 'anderson'    # Iteration method:
                       #   'naive'    : brute force, ok for small g 
                       #   'anderson' : uses scipy.optimise.anderson
                       #                to accelerate the iteration
                       #   'noiter'   : one evaluation only
tol = 1e-5             # Tolerance level for the iterations
grav = False           # Keep gravitational forces in the time loop (True)
soft = 1e-2            # Softening parameter to define minimum radius 

# ============================================================================

if __name__ == "__main__":

    start_wall()
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)
    os.makedirs(odir+"/states")

    shells = Shells()
    shells.init(Nshells, g=g, m_phi=m_phi, m_nu=m_nu, grav=grav,
                kappa=ti_frac, kappa2=tf_frac, dt_frac=dt_frac,
                iter_m=method, iter_tol=tol, soft=soft, w_min=w_min,
                hdf5_io=hdf5_io, seed=seed, odir=odir,verb=verb,
                to_file=to_file)

    if ic_only:
        shells._save(odir, 0)
        print(f"ICs saved in {odir}")
        exit(0)

    if logmeas:
        saves_a = np.geomspace(shells.a_ini, shells.a_end, nmeas)
    else:
        saves_a = np.linspace(shells.a_ini, shells.a_end, nmeas)

    j = 0
    pbar = tqdm.tqdm(total=1)

    while True:
        shells.step()

        z = 1/shells.a - 1
        pbar.set_description(f"z={z:.1f}")
        pbar.update(1)

        while j < len(saves_a) and shells.a >= saves_a[j]:

            shells._save(odir, j)
            j += 1

        if shells.a >= shells.a_end:
            pbar.close()
            z_end = 1/shells.a_end-1
            print(f'\nReached z_end={z_end:.1f}, ending simulation!')
            break

    stop_wall()
    report(unit="s", path=odir+'/profile.log')


