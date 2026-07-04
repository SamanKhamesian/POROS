import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import heapq

np.random.seed(12)

N = 75
x1_raw = np.random.beta(2, 2, N)
x1 = np.sort(x1_raw)
x2 = np.random.uniform(0.08, 0.92, N)
tir = x1 + 0.15 * np.random.randn(N)
labels = np.where(tir >= np.median(tir), 'blue', 'red')

# ── Gaussian circular background (red left, blue right) ───────────────────
g = 300
XX, YY = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g))
Z_blue = np.exp(-((XX - 0.80)**2 + (YY - 0.50)**2) / (2 * 0.32**2))
Z_red  = np.exp(-((XX - 0.20)**2 + (YY - 0.50)**2) / (2 * 0.32**2))
Z = Z_blue - Z_red
Z = gaussian_filter(Z + 0.015 * np.random.randn(g, g), sigma=10)
Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap_tir = LinearSegmentedColormap.from_list(
    'tir',
    ['#c0392b', '#e8877c', '#f5cfc9', '#f5f5f5', '#c5dff0', '#6baed6', '#1a5276'],
    N=256
)

# ── Edges ─────────────────────────────────────────────────────────────────
eps = 0.22
edges_conn, edges_near = [], []
for i in range(N):
    for j in range(i + 1, N):
        d = np.hypot(x1[i] - x1[j], x2[i] - x2[j])
        if d <= eps:
            edges_conn.append((i, j, d))
        elif d <= 1.5 * eps:
            edges_near.append((i, j, d))

# ── Source / target ───────────────────────────────────────────────────────
red_idx  = np.where(labels == 'red')[0]
blue_idx = np.where(labels == 'blue')[0]
src = red_idx[np.argmin(x1[red_idx])]
tgt = blue_idx[np.argmax(x1[blue_idx])]

# ── Dijkstra (bidirectional adjacency, d² weights) ────────────────────────
adj = defaultdict(list)
for i, j, d in edges_conn:
    adj[i].append((d**2, j))
    adj[j].append((d**2, i))

dist_d = {n: float('inf') for n in range(N)}
dist_d[src] = 0
prev = {n: None for n in range(N)}
pq = [(0, src)]
while pq:
    cost, u = heapq.heappop(pq)
    if cost > dist_d[u]:
        continue
    for w, v in adj[u]:
        nc = cost + w
        if nc < dist_d[v]:
            dist_d[v] = nc
            prev[v] = u
            heapq.heappush(pq, (nc, v))

path, cur = [], tgt
while cur is not None:
    path.append(cur)
    cur = prev[cur]
path.reverse()
path_set = set(path)

# ── Style constants ───────────────────────────────────────────────────────
BANANA       = 'gold'
GREEN        = '#27AE60'
COLOR_POOR   = '#7B241C'   # dark brick red
COLOR_BETTER = '#1A5276'   # dark navy blue
colors_map   = {'red': 'tab:red', 'blue': 'tab:blue'}
panel_titles = [
    '(1)  Node Classification',
    '(2)  Graph Construction',
    '(3)  Counterfactual Path',
]

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
fig.patch.set_facecolor('whitesmoke')

for pi, ax in enumerate(axes):
    ax.set_facecolor('whitesmoke')
    ax.contourf(XX, YY, Z, levels=22, cmap=cmap_tir, alpha=0.60, zorder=0)
    ax.contour(XX, YY, Z, levels=10, colors='gray', alpha=0.12,
               linewidths=0.4, zorder=1)

    # ── Edges (panels 2 and 3) ────────────────────────────────────────────
    if pi >= 1:
        for i, j, _ in edges_near:
            ax.plot([x1[i], x1[j]], [x2[i], x2[j]],
                    color='#cccccc', alpha=0.22, lw=0.5, zorder=2)
        for i, j, _ in edges_conn:
            ax.plot([x1[i], x1[j]], [x2[i], x2[j]],
                    color='#444444', alpha=0.35, lw=0.65, zorder=3)

    # ── Green arc + yellow path (panel 3) ────────────────────────────────
    if pi == 2:
        ax.annotate('', xy=(x1[tgt], x2[tgt]), xytext=(x1[src], x2[src]),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.2,
                                   linestyle='dashed',
                                   connectionstyle='arc3,rad=-0.45'), zorder=5)
        for k in range(len(path) - 1):
            ax.annotate('',
                        xy=(x1[path[k+1]], x2[path[k+1]]),
                        xytext=(x1[path[k]], x2[path[k]]),
                        arrowprops=dict(arrowstyle='->', color=BANANA, lw=2.6),
                        zorder=8)

    # ── Regular nodes ─────────────────────────────────────────────────────
    for i in range(N):
        if pi >= 1 and i in (src, tgt):
            continue
        c  = colors_map[labels[i]]
        ec = BANANA if (pi == 2 and i in path_set) else 'white'
        lw = 1.8 if ec == BANANA else 0.6
        s  = 72 if (pi == 2 and i in path_set) else 58
        ax.scatter(x1[i], x2[i], s=s, c=c, edgecolors=ec,
                   linewidths=lw, alpha=0.90, zorder=6)

    # ── Source / target markers (panels 2 and 3) ──────────────────────────
    if pi >= 1:
        for node, lbl, color in [(src, 'x', COLOR_POOR), (tgt, "x'", COLOR_BETTER)]:
            node_color = colors_map[labels[node]]
            # Solid filled dot with black border
            ax.scatter(x1[node], x2[node], s=180,
                       c=node_color, edgecolors='black',
                       linewidths=2.5, zorder=10)
            # Bold, upright label below node
            lx = np.clip(x1[node], 0.04, 0.95)
            ax.text(lx, x2[node] - 0.07, lbl,
                    fontsize=14, fontweight='bold', fontstyle='normal',
                    ha='center', va='top', color=color, zorder=11)

    # ── Epsilon box (panel 2) ─────────────────────────────────────────────
    if pi == 1:
        ax.text(0.50, 0.035, r'$\varepsilon = 10.95$', fontsize=12,
                ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='gray', alpha=0.9))

    # ── Poor / Better outcome labels (panel 1) ────────────────────────────
    if pi == 0:
        ax.text(0.02, 0.025, 'Poor outcome', color=COLOR_POOR,
                fontsize=11, transform=ax.transAxes, fontstyle='italic',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none',
                          boxstyle='round,pad=0.25'))
        ax.text(0.68, 0.025, 'Better outcome', color=COLOR_BETTER,
                fontsize=11, transform=ax.transAxes, fontstyle='italic',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none',
                          boxstyle='round,pad=0.25'))

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xlabel('$f_1$', fontsize=14)
    ax.set_ylabel('$f_2$', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(panel_titles[pi], fontsize=16, fontweight='bold', pad=10)
    ax.tick_params(labelsize=13)
    ax.grid(True, color='gray', alpha=0.07, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)

# ── Shared legend ─────────────────────────────────────────────────────────
leg = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:red',
           markeredgecolor='white', markersize=12, label='Poor health outcome'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='tab:blue',
           markeredgecolor='white', markersize=12, label='Better health outcome'),
    Line2D([0],[0], color='#444444', lw=1.6, alpha=0.8,
           label=r'Directed edge  ($d \leq \varepsilon$, $\rightarrow$ better outcome)'),
    Line2D([0],[0], color='#cccccc', lw=1.6,
           label=r'Disconnected  ($d > \varepsilon$)'),
    Line2D([0],[0], color=BANANA, lw=2.8, label='Shortest path (Dijkstra)'),
    Line2D([0],[0], color=GREEN,  lw=2.2, ls='--',
           label='Direct jump  (infeasible)'),
]
fig.legend(handles=leg, loc='lower center', ncol=3, fontsize=14,
           frameon=True, edgecolor='lightgray',
           bbox_to_anchor=(0.5, -0.12), framealpha=0.95)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('poros_methodology.png', dpi=300, bbox_inches='tight',
            facecolor='whitesmoke')
plt.savefig('poros_methodology.pdf', bbox_inches='tight',
            facecolor='whitesmoke')