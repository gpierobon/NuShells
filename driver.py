import os
import tqdm
import shutil
import numpy as np
#import matplotlib.pyplot as plt

from profile import timed, report, reset
import shells as sh

Nshells = 10_000
nt      = 1_000
dt      = 0.0002
nmeas   = 50
odir    = 'o1'
seed    = 9

lr      = True
grav    = False

# Profilers
sh.Shells.initialise      = timed()(sh.Shells.initialise)
sh.Shells.sort            = timed()(sh.Shells.sort)
sh.Shells.save            = timed()(sh.Shells.save)
sh.ForceSolver.__init__   = timed()(sh.ForceSolver.__init__)
sh.ForceSolver.computeF   = timed()(sh.ForceSolver.computeF)

if __name__ == "__main__":

    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    saves = set(np.linspace(0, nt - 1, nmeas, dtype=int))
    np.random.seed(seed)
    shells = sh.Shells()
    shells.initialise(Nshells, verb=True, Rmin=1e-3, w_min=1e-12, Psi0=1)

    j = 0
    for t in tqdm.tqdm(range(nt)):
        shells.step(dt, include_lr=lr, include_gravity=grav)
        if t in saves:
            shells.save(odir, j, delta=False)
            j += 1

    report(unit="s")
