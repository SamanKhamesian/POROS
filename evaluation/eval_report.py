"""
eval_report.py — All console output and figure functions. No computation.

Functions
  _print_tir_table        : 3-column TIR gain table (direct / per-step / path avg)
  _print_feature_profile  : side-by-side feature Δv profile (direct vs multi-hop)
  _print_feature_single   : single-column feature Δv profile (combos in B and D)
  _print_direct_combo     : stats block for a direct path combo
  _print_graph_combo      : stats block for a graph path combo
  _print_nearest_section  : full 4-combo printout for B or D
  _print_teachers         : role model subject frequency table
  print_summary           : top-level report across all four queries
  plot_path_length_histogram : P1 figure — multi-hop path length distribution
"""

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from node import FEATURE_KEYS


# ── Style ─────────────────────────────────────────────────────────────────────

FEATURE_LABEL = {
    "total_carbs":                  "total_carbs",
    "avg_carbs_per_meal":           "avg_carbs_per_meal",
    "avg_time_between_meals":       "avg_time_between_meals",
    "total_daily_insulin":          "total_daily_insulin",
    "bolus_per_meal":               "bolus_per_meal",
    "total_correction_insulin":     "total_correction_insulin",
    "avg_meal_bolus_delta_minutes": "avg_meal_bolus_delta_min",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(np.median(a)), float(a.min()), float(a.max())


def _ascii_bar(value, max_value, width=14):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


# ── Console output ────────────────────────────────────────────────────────────

def _print_tir_table(sh_vals, mh_flat, mh_path_means, indent="  "):
    sm,  smed,  smin,  smax  = _stats(sh_vals)
    fm,  fmed,  fmin,  fmax  = _stats(mh_flat)
    pm,  pmed,  pmin,  pmax  = _stats(mh_path_means)

    n_sh = len(sh_vals);  n_st = len(mh_flat);  n_pa = len(mh_path_means)

    L = 14;  COL = 14;  N = COL - 3
    p   = f"{indent}  "
    sep = f"{p}{'':─<{L}}  {'':─<{COL}}  {'':─<{COL}}  {'':─<{COL}}"

    def num_row(label, s, f, pv):
        return f"{p}{label:<{L}}  {s:>{N}.1f} pp  {f:>{N}.1f} pp  {pv:>{N}.1f} pp"
    def count_row(label, s, f, pv):
        return f"{p}{label:<{L}}  {s:>{COL},}  {f:>{COL},}  {pv:>{COL},}"
    def str_row(label, s, f, pv):
        return f"{p}{label:<{L}}  {s:>{COL}}  {f:>{COL}}  {pv:>{COL}}"

    print(f"{p}{'':>{L}}  {'Direct path':>{COL}}  {'Multi-hop':>{COL}}  {'Multi-hop':>{COL}}")
    print(f"{p}{'':>{L}}  {'':>{COL}}  {'per step':>{COL}}  {'path avg':>{COL}}")
    print(sep)
    print(num_row("Mean",   sm,   fm,   pm))
    print(num_row("Median", smed, fmed, pmed))
    print(num_row("Min",    smin, fmin, pmin))
    print(num_row("Max",    smax, fmax, pmax))
    print(sep)
    print(count_row("n",   n_sh, n_st, n_pa))
    print(str_row("unit",  "pairs", "steps", "paths"))
    if fm and pm:
        print(f"{p}→  direct path mean ({sm:.1f} pp) is "
              f"{sm/fm:.1f}×  multi-hop/step ({fm:.1f} pp)  and  "
              f"{sm/pm:.1f}×  multi-hop/path avg ({pm:.1f} pp)")


def _print_feature_profile(sh_feat, mh_feat, indent="  "):
    LABEL_W = 30;  BAR_W = 12;  CELL_W = 5 + 2 + BAR_W
    sh_means = {k: float(np.mean(sh_feat[k])) if sh_feat[k] else 0 for k in FEATURE_KEYS}
    mh_means = {k: float(np.mean(mh_feat[k])) if mh_feat[k] else 0 for k in FEATURE_KEYS}
    max_val  = max(max(sh_means.values()), max(mh_means.values()), 1e-9)
    p = f"{indent}  "
    print(f"{p}{'Feature':<{LABEL_W}}  {'Direct path':^{CELL_W}}  {'Multi-hop/step':^{CELL_W}}")
    print(f"{p}{'':─<{LABEL_W}}  {'':─<{CELL_W}}  {'':─<{CELL_W}}")
    for k in FEATURE_KEYS:
        s = sh_means[k];  m = mh_means[k]
        print(f"{p}{FEATURE_LABEL[k]:<{LABEL_W}}  "
              f"{s:5.2f}  {_ascii_bar(s, max_val, BAR_W)}  "
              f"{m:5.2f}  {_ascii_bar(m, max_val, BAR_W)}")


def _print_feature_single(feat, indent="  "):
    """Single-column feature profile — used for the 4 combos in B and D."""
    LABEL_W = 30;  BAR_W = 12
    means   = {k: float(np.mean(feat[k])) if feat[k] else 0.0 for k in FEATURE_KEYS}
    max_val = max(max(means.values()), 1e-9)
    p = f"{indent}  "
    print(f"{p}{'Feature':<{LABEL_W}}  {'Δv (MCID units)'}")
    print(f"{p}{'':─<{LABEL_W}}  {'':─<22}")
    for k in FEATURE_KEYS:
        v = means[k]
        print(f"{p}{FEATURE_LABEL[k]:<{LABEL_W}}  {v:5.2f}  {_ascii_bar(v, max_val, BAR_W)}")


def _print_direct_combo(c, indent="  "):
    """Stats for a direct path combo (one value per source node)."""
    tm, tmed, _, _ = _stats(c["tir"])
    cm, cmed, _, _ = _stats(c["cbtd"])
    p = f"{indent}  "
    L = 16;  N = 8
    print(f"{p}{'TIR gain  mean':<{L}}  {tm:>{N}.1f} pp")
    print(f"{p}{'TIR gain  median':<{L}}  {tmed:>{N}.1f} pp")
    print(f"{p}{'CBTD  mean':<{L}}  {cm:>{N}.2f}")
    print(f"{p}{'CBTD  median':<{L}}  {cmed:>{N}.2f}")
    print(f"{p}{'n':<{L}}  {len(c['tir']):>{N},}")
    print()
    print(f"{p}Mean behavioral change  (MCID units)")
    _print_feature_single(c["feat"], indent)


def _print_graph_combo(c, indent="  "):
    """Stats for a graph path combo (per-step values)."""
    sm, smed, _, _ = _stats(c["steps"])
    pm, pmed, _, _ = _stats(c["pmeans"])
    p = f"{indent}  "
    L = 16;  N = 8
    print(f"{p}{'TIR gain  mean':<{L}}  {sm:>{N}.1f} pp/step   {pm:>{N}.1f} pp/path avg")
    print(f"{p}{'TIR gain  median':<{L}}  {smed:>{N}.1f} pp/step   {pmed:>{N}.1f} pp/path avg")
    print(f"{p}{'n  steps / paths':<{L}}  {len(c['steps']):>{N},}           {len(c['pmeans']):>{N},}")
    print()
    print(f"{p}Mean behavioral change per step  (MCID units)")
    _print_feature_single(c["feat"], indent)


def _print_nearest_section(r, indent="  "):
    """Prints all 4 combinations for a nearest-BLUE query (B or D)."""
    n       = r["n"]
    agree   = r["agree"]
    unreach = r["c3"]["unreachable"]
    n_c3    = n - unreach

    p  = f"{indent}"
    sp = f"{indent}  "

    # hop breakdown
    total_paths = sum(r["hop_counts"].values())
    one_hop     = r["hop_counts"].get(1, 0)
    multi_hop   = total_paths - one_hop
    print(f"{p}Graph path breakdown   "
          f"1-hop: {one_hop}/{total_paths} ({one_hop/total_paths*100:.1f}%)   "
          f"multi-hop: {multi_hop}/{total_paths} ({multi_hop/total_paths*100:.1f}%)")
    print(f"{p}Target agreement   {agree}/{n}  ({agree/n*100:.1f}%)  "
          f"— both methods select the same nearest BLUE node")
    print(f"{p}Combo (3) gap      {unreach}/{n}  ({unreach/n*100:.1f}%)  "
          f"— graph cannot reach direct's target (excluded from combo 3)")

    # ── (1) ───────────────────────────────────────────────────────────────────
    print(f"\n{p}(1)  direct path  →  direct's target")
    print(f"{sp}target : nearest BLUE by minimum raw CBTD  (trivially minimum distance)")
    print(f"{sp}path   : raw CBTD jump, no ε, no graph")
    _print_direct_combo(r["c1"], indent + "  ")

    # ── (2) ───────────────────────────────────────────────────────────────────
    print(f"\n{p}(2)  direct path  →  graph's target")
    print(f"{sp}target : nearest BLUE by minimum Dijkstra path cost")
    print(f"{sp}path   : raw CBTD jump, no ε, no graph")
    _print_direct_combo(r["c2"], indent + "  ")

    # ── (3) ───────────────────────────────────────────────────────────────────
    print(f"\n{p}(3)  graph path   →  direct's target  [{n_c3}/{n} sources reachable]")
    print(f"{sp}target : nearest BLUE by minimum raw CBTD")
    print(f"{sp}path   : Dijkstra through ε-graph  (d² edge weights)")
    if r["c3"]["steps"]:
        _print_graph_combo(r["c3"], indent + "  ")
    else:
        print(f"{sp}no sources have a reachable direct target via the ε-graph")

    # ── (4) ───────────────────────────────────────────────────────────────────
    print(f"\n{p}(4)  graph path   →  graph's target")
    print(f"{sp}target : nearest BLUE by minimum Dijkstra path cost  (trivially minimum)")
    print(f"{sp}path   : Dijkstra through ε-graph  (d² edge weights)")
    _print_graph_combo(r["c4"], indent + "  ")


def _print_teachers(paths, top_n=7, indent="  "):
    appearances = defaultdict(int)
    for path in paths:
        for node in path[1:]:
            appearances[node.subject] += 1
    if not appearances:
        return
    total  = sum(appearances.values())
    ranked = sorted(appearances.items(), key=lambda x: -x[1])[:top_n]
    max_c  = ranked[0][1]
    print(f"{indent}  {'Subject':<22}  {'n':>6}  {'%':>7}")
    print(f"{indent}  {'':─<22}  {'':─>6}  {'':─>7}")
    for subj, cnt in ranked:
        print(f"{indent}  {subj:<22}  {cnt:>6}  {cnt/total*100:>6.1f}%  "
              f"{_ascii_bar(cnt, max_c, 12)}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(ap, nr):
    W   = 72
    SEP = f"  {'─' * (W - 2)}"

    print(f"\n{'═' * W}")
    print("  Evaluation summary")
    print(f"{'═' * W}\n")

    # ── A ─────────────────────────────────────────────────────────────────────
    print(SEP)
    print(f"  A.  All-pairs   "
          f"({ap['a_n_sh']:,} direct pairs  |  {ap['a_n_mh']:,} multi-hop paths)")
    print(SEP)
    print(f"  Direct path  {ap['a_n_sh']:>8,}   "
          f"cross-patient: {ap['a_cross_sh']/ap['a_n_sh']*100:.1f}%")
    print(f"  Multi-hop    {ap['a_n_mh']:>8,}   "
          f"cross-patient: {ap['a_cross_mh']/ap['a_n_mh']*100:.1f}%")
    print()
    print("  TIR gain per step")
    _print_tir_table(ap["a_sh_tir"], ap["a_mh_flat"], ap["a_mh_path_means"])
    print()
    print("  Mean behavioral change per step  (MCID units)")
    _print_feature_profile(ap["a_sh_feat"], ap["a_mh_feat"])

    # ── B ─────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  B.  Nearest-BLUE   any source → nearest BLUE   "
          f"({nr['b']['n']} sources)")
    print(SEP)
    _print_nearest_section(nr["b"], indent="  ")
    print()
    print("  Subjects most referenced as behavioral role models  (combo 4 paths)")
    _print_teachers(nr["b"]["c4"]["paths"])

    # ── C ─────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  C.  RED → BLUE   "
          f"({ap['c_n_sh']:,} direct pairs  |  {ap['c_n_mh']:,} multi-hop paths)")
    print(SEP)
    print(f"  Direct path  {ap['c_n_sh']:>8,}   "
          f"cross-patient: {ap['c_cross_sh']/ap['c_n_sh']*100:.1f}%")
    print(f"  Multi-hop    {ap['c_n_mh']:>8,}   "
          f"cross-patient: {ap['c_cross_mh']/ap['c_n_mh']*100:.1f}%")
    print()
    print("  TIR gain per step")
    _print_tir_table(ap["c_sh_tir"], ap["c_mh_flat"], ap["c_mh_path_means"])
    print()
    print("  Mean behavioral change per step  (MCID units)")
    _print_feature_profile(ap["c_sh_feat"], ap["c_mh_feat"])

    # ── D ─────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  D.  RED → nearest-BLUE   "
          f"({nr['d']['n']} RED sources)")
    print(SEP)
    _print_nearest_section(nr["d"], indent="  ")
    print()
    print("  Subjects most referenced as behavioral role models  (combo 4 paths)")
    _print_teachers(nr["d"]["c4"]["paths"])

    print(f"\n{'═' * W}\n")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_path_length_histogram(a_mh_hops, c_mh_hops, save_path):
    """
    P1 figure: distribution of multi-hop path lengths (in hops).

    Parameters
    ----------
    a_mh_hops   : list[int]  — hop counts, all-pairs multi-hop
    c_mh_hops   : list[int]  — hop counts, poorly-controlled source multi-hop
    save_path   : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    datasets = [
        (a_mh_hops, "tab:blue", "All Source Nodes — Path Length Distribution"),
        (c_mh_hops, "tab:red",  "Poorly-Controlled Sources — Path Length Distribution"),
    ]

    for ax, (hops, color, subtitle) in zip(axes, datasets):
        counts  = Counter(hops)
        max_hop = max(counts)
        xs_int  = list(range(2, max_hop + 1))
        total   = sum(counts.values())
        pcts    = [counts.get(x, 0) / total * 100 for x in xs_int]
        x_pos   = np.arange(len(xs_int))
        labels  = [str(x) for x in xs_int]

        ax.bar(x_pos, pcts, width=0.75, color=color,  alpha=0.4, edgecolor="none")
        ax.bar(x_pos, pcts, width=0.75, color="none", edgecolor=color, linewidth=2)

        for x, pct in zip(x_pos, pcts):
            if pct >= 0.5:
                ax.text(x, pct + 0.8, f"{pct:.1f}%",
                        ha="center", va="bottom", fontsize=11, color="dimgray")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("Path length (Hops)", fontsize=13)
        ax.set_ylabel("% of Paths", fontsize=13)
        ax.set_title(subtitle, fontsize=16)
        ax.set_ylim(0, max(pcts) * 1.25)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
        ax.set_facecolor("whitesmoke")

    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  saved → {save_path}")

    plt.close(fig)