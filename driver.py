import os
import tqdm
import shutil
import numpy as np

from timing import timed, report, reset, start_wall, stop_wall
from shells import Shells
from phi import solvePhi, interpPhi
from force import solveYukawaForce, solveGravityForce

# Main params
Nshells = 1000
g       = 1e-26
m_nu    = 0.1
m_phi   = 1e-29

# Times
nt      = 10000
dt_frac = 0.5
kappa   = 0.75
kappa2  = 1.0

# Initial conditions
Psi0    = 1e-5
ic_type = 'tophat'
w_min   = 1e-12
seed    = 9

# Outputs
nmeas   = 100
odir    = 'output'
hdf5_io = True

# Iteration
method  = 'anderson'
tol     = 1e-2
soft    = 1e-2


if __name__ == "__main__":

    start_wall()
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    shells = Shells()
    shells.init(Nshells, g=g, m_phi=m_phi, m_nu=m_nu, kappa=kappa, kappa2=kappa2,
                dt_frac=dt_frac, iter_m=method, iter_tol=tol, soft=soft,
                w_min=w_min, hdf5_io=hdf5_io, seed=seed, verb=True)

    saves = set(np.linspace(0, nt - 1, nmeas, dtype=int))

    j = 0; t = 0
    pbar = tqdm.tqdm(total=1)

    while True:
        shells.step()

        z = 1/shells.a - 1
        pbar.set_description(f"z={z:.1f}")
        pbar.update(1)

        if t in saves:
            shells._save(odir, j)
            j += 1

        t += 1
        if shells.a >= shells.a_end:
            pbar.close()
            z_end = 1/shells.a_end-1
            print(f'\nReached z_end={z_end:.1f}, ending simulation!')
            break

    stop_wall()
    report(unit="s", path=odir+'/profile.log')


