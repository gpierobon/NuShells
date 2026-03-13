import os
import tqdm
import shutil
import numpy as np

from profile import timed, report, reset
from shells_v2 import Shells
from force import ForceSolver

Nshells = 10_000
nt      = 100_000
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
    shells.init(Nshells, g=1e-24, dt_frac=0.005, w_min=1e-9, verb=True)

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

    report(unit="s")


