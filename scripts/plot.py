import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def phase_space(shells):
    fig = plt.figure(figsize=(8,6))
    sizes = 50 * shells.w
    #plt.scatter(self.R, self.q,
    #            s=sizes,
    #            alpha=0.5)
    plt.scatter(shells.R, shells.q, s=5, alpha=0.5)
    plt.xlabel(r"$R$")
    plt.ylabel(r"$q_r$")
    plt.xscale('log')
    plt.xlim(shells.Rmin, shells.Rmax)
    plt.ylim(-10, 10)
    plt.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    return fig

def circles(shells, skip=10, save=False, odir='output', num=0, cmap='Grays'):

    norm = mpl.colors.LogNorm(vmin=shells.w.min(), vmax=shells.w.max())
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(8,8))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r"$w$")
    #for r in shells.R[::skip]:
    for r, w in zip(shells.R[::skip], shells.w[::skip]):
        color = cmap(norm(w))
        circle = Circle((0,0), r, fill=False, color=color, alpha=0.8)
        ax.add_patch(circle)

    ax.set_xlim(-shells.R.max(), shells.R.max())
    ax.set_ylim(-shells.R.max(), shells.R.max())
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    z = 1/shells.a-1
    ax.set_title(r'$z=%.2f$'%z)
    if save:
        plt.savefig(odir+'/circ_%.3d.png'%num, bbox_inches='tight')
    return fig

