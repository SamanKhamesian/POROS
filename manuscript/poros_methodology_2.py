import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(123)

# ── Synthetic data ─────────────────────────────────────────────────────────
n = 80
pts = np.random.uniform(0.02, 0.98, (n, 2))
health = 0.65 * pts[:, 0] + 0.25 * pts[:, 1] + np.random.normal(0, 0.08, n)
health = (health - health.min()) / (health.max() - health.min())
good = health > 0.5

# ── Graph construction ─────────────────────────────────────────────────────
eps = 0.22
D = cdist(pts, pts)
G = nx.DiGraph()
G.add_nodes_from(range(n))
edges = [(i, j) for i in range(n) for j in range(n)
         if i != j and D[i, j] <= eps and health[j] > health[i]]
for i, j in edges:
    G.add_edge(i, j, weight=D[i, j])

# ── Source / target ────────────────────────────────────────────────────────
red_idx  = np.where(~good)[0]
blue_idx = np.where( good)[0]
source = red_idx[np.argmin(pts[red_idx,  0])]
target = blue_idx[np.argmax(pts[blue_idx, 0])]
if not nx.has_path(G, source, target):
    reachable = nx.descendants(G, source) & set(blue_idx.tolist())
    if reachable:
        target = max(reachable, key=lambda t: pts[t, 0])
try:
    path = nx.dijkstra_path(G, source, target, weight='weight')
except (nx.NetworkXNoPath, nx.NodeNotFound):
    path = []
path_set = set(path)

# ── Style constants ────────────────────────────────────────────────────────
C_RED    = '#C0392B'
C_DARK_RED = '#7B241C'
C_DARK_BLUE = '#1A5276'
C_BLUE   = '#2471A3'
C_YELLOW = 'gold'
C_GREEN  = '#27AE60'

# ── Helpers ────────────────────────────────────────────────────────────────
def make_bg(ax):
    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    Zr = np.exp(-((X - 0.25)**2 + (Y - 0.65)**2) / 0.10)
    Zb = np.exp(-((X - 0.75)**2 + (Y - 0.55)**2) / 0.10)
    Z  = Zb - Zr
    cmap = LinearSegmentedColormap.from_list(
        'rb', ['#d44f4f', '#eaa0a0', '#f2f2f2', '#9abcd8', '#4a85b8'])
    ax.contourf(X, Y, Z, levels=16, cmap=cmap, alpha=0.75, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('$f_1$', fontsize=13)
    ax.set_ylabel('$f_2$', fontsize=13)
    ax.tick_params(labelsize=10)

def draw_nodes(ax, exclude=(), zorder=5, size=55):
    mask = np.ones(n, dtype=bool)
    for e in exclude:
        mask[e] = False
    ax.scatter(pts[mask & ~good, 0], pts[mask & ~good, 1],
               c=C_RED,  s=size, zorder=zorder, edgecolors='white', linewidths=0.5)
    ax.scatter(pts[mask &  good, 0], pts[mask &  good, 1],
               c=C_BLUE, s=size, zorder=zorder, edgecolors='white', linewidths=0.5)

def draw_src_tgt(ax, zorder=10):
    for node, lbl, color in [(source, "x", C_DARK_RED), (target, "x'", C_DARK_BLUE)]:
        nc  = C_BLUE if good[node] else C_RED
        # Solid filled dot with black border
        ax.scatter(*pts[node], s=180, c=nc,
                   edgecolors='black', linewidths=2.5, zorder=zorder)
        # Bold, upright label below node
        lx = np.clip(pts[node, 0], 0.04, 0.93)
        ax.text(lx, pts[node, 1] - 0.07, lbl,
                fontsize=14, fontweight='bold', fontstyle='normal',
                ha='center', va='top', color=color, zorder=zorder + 1)

def draw_edges(ax, alpha_dark=0.45, alpha_light=0.22, zorder_base=2):
    # Near-miss disconnected pairs (light gray)
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if eps < d <= 1.5 * eps:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color='#cccccc', alpha=alpha_light, lw=0.5,
                        zorder=zorder_base)
    # Connected directed edges (dark gray arrows)
    for i, j in edges:
        ax.annotate('', xy=pts[j], xytext=pts[i],
                    arrowprops=dict(arrowstyle='->', color='#444444',
                                    lw=0.55, alpha=alpha_dark,
                                    shrinkA=3, shrinkB=3),
                    zorder=zorder_base + 1)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

# ── Panel 1: Node Classification ───────────────────────────────────────────
ax = axes[0]
make_bg(ax)
draw_nodes(ax)
ax.text(0.03, 0.03, 'Poor outcome', color='#7B241C', fontsize=11,
        transform=ax.transAxes, fontstyle='italic',
        bbox=dict(facecolor='white', alpha=0.65, edgecolor='none',
                  boxstyle='round,pad=0.25'))
ax.text(0.63, 0.03, 'Better outcome', color='#1A5276', fontsize=11,
        transform=ax.transAxes, fontstyle='italic',
        bbox=dict(facecolor='white', alpha=0.65, edgecolor='none',
                  boxstyle='round,pad=0.25'))
ax.set_title('(1)  Node Classification', fontweight='bold', fontsize=13, pad=9)

# ── Panel 2: Graph Construction ────────────────────────────────────────────
ax = axes[1]
make_bg(ax)
draw_edges(ax)
draw_nodes(ax, exclude=(source, target))
draw_src_tgt(ax)
ax.text(0.5, 0.04, r'$\varepsilon = 0.22$', fontsize=12, ha='center',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#aaaaaa', alpha=0.9))
ax.set_title('(2)  Graph Construction', fontweight='bold', fontsize=13, pad=9)

# ── Panel 3: Counterfactual Path ───────────────────────────────────────────
ax = axes[2]
make_bg(ax)
draw_edges(ax)                              # same as panel 2
draw_nodes(ax, exclude=(source, target))

# Yellow path
if len(path) > 1:
    for k in range(len(path) - 1):
        ax.annotate('',
                    xy=pts[path[k + 1]], xytext=pts[path[k]],
                    arrowprops=dict(arrowstyle='->', color=C_YELLOW, lw=2.8,
                                    shrinkA=3, shrinkB=3),
                    zorder=8)
    for p in path:
        nc = C_BLUE if good[p] else C_RED
        ax.scatter(*pts[p], s=75, zorder=9, c=nc,
                   edgecolors=C_YELLOW, linewidths=2.2)

# Dashed green arc (infeasible direct jump)
arc = FancyArrowPatch(pts[source], pts[target],
                      connectionstyle='arc3,rad=-0.42',
                      arrowstyle='->', color=C_GREEN,
                      linewidth=2.2, linestyle='dashed',
                      zorder=6, mutation_scale=16)
ax.add_patch(arc)

draw_src_tgt(ax, zorder=10)
ax.set_title('(3)  Counterfactual Path', fontweight='bold', fontsize=13, pad=9)

# ── Shared legend ──────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc=C_RED,   ec='white', label='Poor health outcome'),
    mpatches.Patch(fc=C_BLUE,  ec='white', label='Better health outcome'),
    Line2D([0],[0], color='#444444', lw=1.5,
           label=r'Directed edge ($d \leq \varepsilon$, $\rightarrow$ better outcome)'),
    Line2D([0],[0], color='#cccccc', lw=1.2,
           label=r'Disconnected ($d > \varepsilon$)'),
    Line2D([0],[0], color=C_YELLOW, lw=2.8, label='Shortest path (Dijkstra)'),
    Line2D([0],[0], color=C_GREEN,  lw=2.2, ls='dashed',
           label='Direct jump (infeasible)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 0.01), fontsize=14,
           frameon=True, edgecolor='#cccccc', fancybox=False)

plt.tight_layout(rect=[0, 0.13, 1, 1])
plt.savefig('poros_methodology_2.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig('poros_methodology_2.pdf', bbox_inches='tight',
            facecolor='white')