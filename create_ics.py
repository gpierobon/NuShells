import os
import tqdm
import shutil
import numpy as np

from timing import timed, report, reset, start_wall, stop_wall
from shells import Shells
from phi import solvePhi, interpPhi
from force import solveYukawaForce, solveGravityForce


m_nu = 0.1             # Neutrino  mass (eV)
m_phi = 1e-29          # Scalar field mass (eV)
w_min = 1e-12          # Minimum weight
seed = 9               # IC random seed 
hdf5_io = True         # Saved in HDF5 format (True) or .txt (False)
to_file = True         # Print to file (True) or to screen (False)
verb = 0               # Verbosity level (0:INFO, 1:DEBUG)


def create_ics(Nshells, g, ic_type, Psi0, odir):
    odir ='ic_tests/'+odir
    if os.path.exists(odir):
        shutil.rmtree(odir)
    os.makedirs(odir)
    os.makedirs(odir+"/states")

    shells = Shells()
    shells.init(Nshells, g=g, m_phi=m_phi, m_nu=m_nu,
                w_min=w_min, hdf5_io=hdf5_io, seed=seed, odir=odir,
                verb=verb, to_file=to_file)
    shells._save(odir, 0)
    print(f"ICs saved in {odir}")


if __name__ == "__main__":

    Psi0 = 1e-5
    for N in [1_000, 10_000, 100_000, 1_000_000]:
        for g in [1e-24]:
            for it in ['gaussian', 'gaussian_c', 'tophat_c']:
                odir = f"o_N{int(np.log10(N)):d}g{-int(np.log10(g)):d}_ic{it}"
                create_ics(N, g, it, Psi0, odir)




