import os, sys
import glob
import numpy as np
import scipy.ndimage as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells_v2 import Shells

data_dir = sys.argv[1]
speed    = float(sys.argv[2])

pattern = os.path.join(data_dir, "shells_*.txt")
files   = sorted(glob.glob(pattern))
if len(files) < 1:
    print(f"No field files found, check the output directory!")
    exit(0)
print(f"Found {len(files)} field files")

shells = Shells()
shells._load(data_dir, 0)

sig = 1.0

plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

r_c, n = shells.density()
valid  = np.isfinite(n)
n_bar = np.nanmean(n[valid])

line_n,   = ax.loglog(r_c[valid], sn.gaussian_filter(n[valid]/n_bar, sigma=sig), color='steelblue', lw=1.8,
                             label=r'$n(r)$')
line_nbar = ax.axhline(n_bar/n_bar, color='k', lw=2, ls='--',alpha=0.6)
line_phi = ax.axvline(1/shells.a, color='navy', lw=1.5, label=r'$\lambda_\phi$')

ax.set_xlabel(r'$\hat{r}$')
ax.set_ylabel(r'$\delta_\nu$')
ax.legend()
ax.set_xlim(5e-2, 1e4)
ax.set_ylim(1e-1, 1e9)

title = fig.suptitle(r'$z=%.2f$' % (1/shells.a - 1))

for i in range(len(files)):
    shells._load(data_dir, i)

    r_c, n = shells.density()
    valid  = np.isfinite(n)
    line_n.set_data(r_c[valid], sn.gaussian_filter(n[valid]/n_bar, sigma=sig))
    line_phi.set_xdata([1/shells.a, 1/shells.a])

    z = 1/shells.a - 1
    title.set_text(r'$z=%.2f$' % z)

    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
