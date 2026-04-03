import os
import tqdm
import shutil
import numpy as np

from timing import timed, report, reset, start_wall, stop_wall
from shells_v2 import Shells
from force import solveForce
from phi import solvePhi

Nshells = 10_000
nt      = 100_000
g       = 1e-27
m_nu    = 0.05

nmeas   = 100
odir    = 'output'

method  = 'naive'
tol     = 1e-2
seed    = 9



if __name__ == "__main__":

    start_wall()
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    saves = set(np.linspace(0, nt - 1, nmeas, dtype=int))
    np.random.seed(seed)
    shells = Shells()
    shells.init(Nshells, g=g, m_nu=m_nu, dt_frac=0.005,\
                w_min=1e-12, kappa2=0.8,verb=True)

    j = 0; t = 0
    pbar = tqdm.tqdm(total=1)

    while True:
        shells.step(method=method, tol=tol)

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


