import os, sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.style.use('sty.mplstyle')

cdir = os.path.dirname(os.path.abspath(__file__))
pdir = os.path.dirname(cdir)
sys.path.append(pdir)

from shells_v2 import Shells
from plot import circles

data_dir = "output" #sys.argv[1]
speed    = 0.005 #float(sys.argv[2])
pattern = os.path.join(data_dir, "shells_*.txt")
files = sorted(glob.glob(pattern))

if len(files) < 1:
    print(f"No field files found (set --meas 8!)")
    exit(0)

print(f"Found {len(files)} field files")


shells = Shells()
shells._load(data_dir, 0)
#norm = mpl.colors.Normalize(vmin=shells.w.min(), vmax=shells.w.max())
norm = mpl.colors.LogNorm(vmin=shells.w.min(), vmax=shells.w.max())
cmap = plt.get_cmap("Blues")
skip = 5 if shells.N > 1000 else 1

plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, label=r"$w$")

# Create circles once
circles = []
for r, w in zip(shells.R[::skip], shells.w[::skip]):
    color = cmap(norm(w))
    c = Circle((0,0), r, fill=False, color=color, alpha=0.8)
    ax.add_patch(c)
    circles.append(c)

ax.set_xlim(-shells.R.max(), shells.R.max())
ax.set_ylim(-shells.R.max(), shells.R.max())
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

for s in ax.spines.values():
    s.set_visible(False)

# Update loop
for i in range(len(files)):
    shells._load(data_dir, i)

    for circle, r, w in zip(circles, shells.R[::skip], shells.w[::skip]):
        circle.set_radius(r)
        circle.set_color(cmap(norm(w)))

    z = 1/shells.a-1
    ax.set_title(r'$z=%.2f$'%z)
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()
    plt.pause(speed)

plt.ioff()
plt.show()
