import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use('Agg')

OUT_PATH = 'results/path_progression_example.png'

# ── Data ───────────────────────────────────────────────────────────────────────
# Source: subject_14, 2025-06-19, TIR = 68.1%
# Gradual path: 12 steps, cross-patient nodes
grad_nodes = [
    {'step': 0,  'tir': 68.1},
    {'step': 1,  'tir': 68.8},
    {'step': 2,  'tir': 69.4},
    {'step': 3,  'tir': 72.9},
    {'step': 4,  'tir': 80.2},
    {'step': 5,  'tir': 84.1},
    {'step': 6,  'tir': 85.7},
    {'step': 7,  'tir': 86.7},
    {'step': 8,  'tir': 86.8},
    {'step': 9,  'tir': 88.8},
    {'step': 10, 'tir': 90.3},
    {'step': 11, 'tir': 92.4},
    {'step': 12, 'tir': 92.7},
]

# Direct edge: same source → subject_11, 2025-06-01, TIR = 85.1%
direct_nodes = [
    {'step': 0, 'tir': 68.1},
    {'step': 1, 'tir': 85.1},
]

# ── Colors ─────────────────────────────────────────────────────────────────────
BLUE   = 'tab:blue'
GREEN  = 'tab:green'
RED    = 'tab:red'
BLACK  = 'black'
GRAY   = 'silver'

# ── Box style (shared by all three main labels) ────────────────────────────────
BOX_FS = 12

def create_box(ec):
    return dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor=ec, linewidth=1.0)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# ── Multi-hop path — segment 1: steps 0-6 ─────────────────────────────────────
seg1 = grad_nodes[:7]
ax.plot([n['step'] for n in seg1], [n['tir'] for n in seg1], '-', color=BLUE, linewidth=2, zorder=3)
ax.scatter([n['step'] for n in seg1], [n['tir'] for n in seg1], s=50, color=BLUE, zorder=5, linewidths=1)

# ── Multi-hop path — segment 2: steps 6-12 ────────────────────────────────────
seg2 = grad_nodes[6:]
ax.plot([n['step'] for n in seg2], [n['tir'] for n in seg2], '-', color=GREEN, linewidth=2, zorder=3)
ax.scatter([n['step'] for n in seg2], [n['tir'] for n in seg2], s=50, color=GREEN, zorder=5, linewidths=1)

# ── Direct path ────────────────────────────────────────────────────────────────
ax.plot([n['step'] for n in direct_nodes], [n['tir'] for n in direct_nodes], '--', color=RED, linewidth=2, zorder=3)
ax.scatter([n['step'] for n in direct_nodes], [n['tir'] for n in direct_nodes], s=50, color=RED, zorder=5, linewidths=1)

# ── SOURCE box ─────────────────────────────────────────────────────────────────
ax.text(0.5, 65, 'Source\nTIR = 68.1%', fontsize=BOX_FS, ha='center', va='top', color=BLACK, bbox=create_box(BLACK), zorder=6)

# ── DIRECT TARGET box ──────────────────────────────────────────────────────────
ax.text(1, 88.0, 'Direct Target\nTIR = 85.1%', fontsize=BOX_FS, ha='center', va='bottom', color=RED, bbox=create_box(RED), zorder=6)

# ── PATH TARGET box ────────────────────────────────────────────────────────────
ax.text(11.5, 95.5, 'Path Target\nTIR = 92.7%', fontsize=BOX_FS, ha='center', va='bottom', color=GREEN, bbox=create_box(GREEN), zorder=6)

# ── Axis labels, limits, ticks ────────────────────────────────────────────────
ax.set_xlabel('Step', fontsize=15)
ax.set_ylabel('Time in Range  (%)', fontsize=15)
ax.set_xlim(-0.9, 12.9)
ax.set_ylim(57, 104)
ax.tick_params(axis='both', labelsize=13)

# ── Legend ────────────────────────────────────────────────────────────────────
leg = [
    Line2D([0], [0], color=RED, linestyle='--', linewidth=2, marker='o', markersize=6,
           label='Direct Path'),
    Line2D([0], [0], color=BLUE, linestyle='-', linewidth=2, marker='o', markersize=6,
           label='Multi-hop Path (steps 0--6)'),
    Line2D([0], [0], color=GREEN, linestyle='-', linewidth=2, marker='o', markersize=6,
           label='Multi-hop Path (steps 6--12)'),
    Line2D([0], [0], color=BLACK, linestyle='--', linewidth=1, alpha=0.5,
           label='70% TIR Threshold'),
]

ax.legend(handles=leg, fontsize=11, loc='lower right')

# ── Grid ───────────────────────────────────────────────────────────────────────
ax.set_axisbelow(True)
ax.grid(True, which='major', color=GRAY, linewidth=1.0, zorder=0, alpha=0.2)
ax.set_xticks(range(13))
ax.set_yticks(range(60, 101, 5))
ax.axhline(70, color=BLACK, linestyle='--', linewidth=1, zorder=1, alpha=0.5)
ax.set_facecolor("whitesmoke")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved → {OUT_PATH}")