import os
import tqdm
import shutil
import numpy as np

from profile import timed, report, reset
from shells_v2 import Shells
from force import ForceSolver

Nshells = 10_000
nt      = 30_000
nmeas   = 100
odir    = 'output'
seed    = 9

# Profilers
Shells.init  = timed("Init")(Shells.init)
Shells._sort = timed("Sort")(Shells._sort)
Shells._save = timed("I/O") (Shells._save)
ForceSolver.__init__ = timed("Force")(ForceSolver.__init__)

if __name__ == "__main__":

    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)

    saves = set(np.linspace(0, nt - 1, nmeas, dtype=int))
    np.random.seed(seed)
    shells = Shells()
    shells.init(Nshells, verb=True)

    j = 0
    pbar = tqdm.tqdm(range(nt))

    for t in pbar:
        shells.step()

        z = 1/shells.a - 1
        pbar.set_description(f"z={z:.1f}")

        if t in saves:
            shells._save(odir, j)
            j += 1

    report(unit="s")


