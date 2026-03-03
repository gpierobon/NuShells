import os
import tqdm
import shutil
import numpy as np

from profile import timed, report, reset
import shells as sh

Nshells = 10_000
nt      = 1000
dt      = 0.0001
nmeas   = 30
odir    = 'output'
seed    = 9

# Profilers
sh.Shells.initialise      = timed("ICs")(sh.Shells.initialise)
sh.Shells.sort            = timed("Sorting shells")(sh.Shells.sort)
sh.ForceSolver.__init__   = timed("Precompute force sums")(sh.ForceSolver.__init__)
sh.ForceSolver.computeF   = timed("Force compuation")(sh.ForceSolver.computeF)

if __name__ == "__main__":

    j = 0
    k = 100

    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    saves = set(np.linspace(0, nt - 1, nmeas, dtype=int))
    np.random.seed(seed)
    shells = sh.Shells()
    shells.initialise(Nshells, verb=True)

    for t in tqdm.tqdm(range(nt)):
        shells.step(dt)
        if t in saves:
            shells.save(odir, j)
            j += 1

    report(unit="s")
