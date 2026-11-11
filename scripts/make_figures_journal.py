#!/usr/bin/env python3
"""
make_figures.py - Journal-ready figures (The Journal of Pain / Elsevier artwork specs)

Re-creates every manuscript figure from the CSVs written by analyze.py, in ONE
consistent style, without re-running the analysis.

Usage (from the project root, or call main([...]) in an interactive window)
---------------------------------------------------------------------------
    python scripts/make_figures.py                          # all figures, all runs
    python scripts/make_figures.py --list
    python scripts/make_figures.py --only main_figure_combined --runs placebo_0
    python scripts/make_figures.py --only "correlation_*"
    python scripts/make_figures.py --draft                  # fast 100-dpi PNG preview
    python scripts/make_figures.py --formats pdf png tiff   # choose export formats

Inputs  (per run folder, written by analyze.py)
    data_wide_dat_withexcl.csv, data_all_eval_frame_withexcl.csv,
    data_all_discrim_task_long_withexcl.csv
    stats/*.csv   (optional: used only to print p-values / BF10 on the figures)
Outputs -> <run>/figures_journal/   (originals in figures/ are never touched)

Style summary (edit the CONFIG block below to change it everywhere at once)
    * Exact final size in mm (Elsevier: 90 / 140 / 190 mm wide). No bbox_inches="tight",
      so the file really is the size you asked for.
    * Arial (falls back to Liberation Sans/DejaVu with a warning), 7 pt text.
    * Vector PDF with embedded fonts + 600 dpi PNG. White background.
    * TENS on = blue circles, TENS off = red squares (colour AND shape differ).
    * Panel letters bold lowercase, same position logic in every figure.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import fnmatch
import os
import sys
from os.path import join as opj
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CONFIG
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DERIVATIVES_DIR = opj(PROJECT_DIR, "derivatives")

RUNS = None  # None = auto-discover every folder in derivatives/ with the CSVs
OUT_SUBDIR = "figures_journal"
FORMATS = ["pdf", "png"]  # "tiff" and "svg" also work
DPI = 600  # for PNG / TIFF (Elsevier: >= 500 dpi for combination art)
SEED = 1  # fixes the jitter so figures are identical on every re-run

# ---- Elsevier artwork widths (mm) --------------------------------------------
W_SINGLE, W_MID, W_FULL = 90, 140, 190

# ---- typography (points, at final printed size) --------------------------------
FS_TEXT = 7  # tick labels, legends
FS_LABEL = 8  # axis labels
FS_SMALL = 6.5  # annotations, phase labels (never below 6)
FS_PANEL = 10  # bold panel letters

# ---- colours / markers ---------------------------------------------------------
COLOR_TENS_ON = "#1a80bb"  # condition "active"   (blue)
COLOR_TENS_OFF = "#a00000"  # condition "inactive" (red)
MARKER = {"active": "o", "inactive": "s"}  # shape differs too (colour-blind / greyscale safe)
PAL = {"active": COLOR_TENS_ON, "inactive": COLOR_TENS_OFF}
LABEL = {"active": "TENS on", "inactive": "TENS off"}
COLOR_PLACEBO_BG = "#E6E6E6"  # light grey band = placebo-test phases (neutral, so it
#                               is not confused with the blue "TENS on" colour)
COLOR_NEUTRAL = "#3a3a3a"  # dots/lines of the correlation plots

# ---- wording used on several figures ------------------------------------------------
PAIN_LABEL = "Pain rating (0\u2013100)"
ACC_LABEL = "Proportion correct"

# ---- axis ranges --------------------------------------------------------------------
PAIN_YLIM = (0, 80)  # time-series panels
ACC_YLIM = (0.4, 1.0)  # time-series + paired panels (chance = 0.5 stays visible)
PAIN_YTICKS = [0, 20, 40, 60, 80]
ACC_YTICKS = [0.4, 0.6, 0.8, 1.0]
HEADROOM = 0.16  # extra room above paired panels for the p-value bracket

# ---- main_figure_combined -----------------------------------------------------------
# True : right column = pain (c, row 1) over accuracy (d, row 2), so each ROW is one
#        outcome (a+c pain, b+d accuracy).  False: old order (c = accuracy, d = pain).
MAIN_PAIN_ON_TOP_RIGHT = True

PLACEBO_TRIALS = [8, 9, 10, 11, 20, 21, 22, 23]  # 0-indexed placebo-test rating trials
PLACEBO_THRESHOLD_LINE = 10  # dashed line on placebo_effect_distribution (%)
SHOW_N = False  # True: print "n = ..." in the paired panels (handy while drafting)

# x-ranges of the four experimental phases (trial index units)
PLACEBO_SPANS = [(7.5, 11.5), (19.5, 23.5)]
PHASES = [((-0.5, 7.5), "Conditioning"), ((7.5, 11.5), "Placebo test"),
          ((11.5, 19.5), "Conditioning"), ((19.5, 23.5), "Placebo test")]
XLIM_TRIALS = (-0.7, 23.7)

MM = 1 / 25.4


# =============================================================================
# STYLE / FONTS
# =============================================================================
def setup_style():
    avail = {f.name for f in fm.fontManager.ttflist}
    prefs = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    chosen = [p for p in prefs if p in avail] or ["DejaVu Sans"]
    if chosen[0] != "Arial":
        print(f"  [warn] Arial not found - using '{chosen[0]}'. Install Arial for the "
              "final submission files (Elsevier asks for Arial/Helvetica).")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chosen,
        "font.size": FS_TEXT,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_SMALL,
        "xtick.labelsize": FS_TEXT,
        "ytick.labelsize": FS_TEXT,
        "legend.fontsize": FS_TEXT,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType fonts
        "svg.fonttype": "none",
        # math text (e.g. d_z) in the same font as everything else
        "mathtext.fontset": "custom", "mathtext.rm": "sans",
        "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "mathtext.cal": "sans",
    })


# =============================================================================
# LAYOUT HELPERS (everything in millimetres, origin bottom-left)
# =============================================================================
def new_fig(w_mm, h_mm):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def add_ax(fig, left, bottom, width, height, **kw):
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def panel_label(fig, ax, letter, dx=-11.0, dy=1.0, y_from=None):
    """Bold lowercase panel letter at the axes' top-left corner. y_from: take the height
    from another axes so letters of one row sit on the same line."""
    W, H = fig.get_size_inches() / MM
    pos = ax.get_position()
    y1 = (y_from or ax).get_position().y1
    fig.text(pos.x0 + dx / W, y1 + dy / H, letter, fontsize=FS_PANEL,
             fontweight="bold", ha="left", va="bottom")


# =============================================================================
# STATS HELPERS (read what analyze.py already computed)
# =============================================================================
def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "p < .001"
    s = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    return "p = " + s.lstrip("0")


def read_stat(D, fname):
    path = opj(D.run_dir, "stats", fname)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path).iloc[0]
    except Exception:
        return None


def _first(row, names):
    if row is None:
        return None
    for n in names:
        if n in row.index:
            return row[n]
    return None


def stat_p(row):
    v = _first(row, ["p-val", "p_val", "p-value", "pval", "p"])
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def stat_bf(row):
    v = _first(row, ["BF10"])
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return None if v is None else str(v)


# =============================================================================
# DATA LOADING
# =============================================================================
WIDE_NUMERIC = [
    "perc_placebo_all", "acc_all", "active_acc_all", "inactive_acc_all",
    "d_prime_active", "d_prime_inactive", "beta_active", "beta_inactive",
    "c_active", "c_inactive", "pcs_total", "iastay1_total", "iastay2_total",
] + [f"placebo_b{b}_{t}" for b in (1, 2) for t in (1, 2, 3, 4)]


def load_run(run_dir):
    need = {
        "data_wide_dat_withexcl.csv": "wide_dat",
        "data_all_eval_frame_withexcl.csv": "all_eval_frame",
        "data_all_discrim_task_long_withexcl.csv": "all_discrim_task_long",
    }
    D = SimpleNamespace(run_dir=run_dir, name=os.path.basename(os.path.normpath(run_dir)))
    for fname, attr in need.items():
        path = opj(run_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run analyze.py for this folder first.")
        setattr(D, attr, pd.read_csv(path))
    for col in WIDE_NUMERIC:
        if col in D.wide_dat.columns:
            D.wide_dat[col] = pd.to_numeric(D.wide_dat[col], errors="coerce")
    D.n = D.wide_dat["participant"].nunique()

    # participant x condition x trial means (pain ratings over the session)
    D.eval_ts = (D.all_eval_frame.groupby(["participant", "condition", "trial"])
                 .mean(numeric_only=True).reset_index())
    # participant x condition x discrimination trial means (accuracy over the session)
    acc = (D.all_discrim_task_long.groupby(["participant", "condition", "actual_trial"])
           .mean(numeric_only=True).reset_index())
    acc["actual_trial"] = acc["actual_trial"].astype(int) - 0.8  # same x-offset as analyze.py
    D.acc_ts = acc

    # participant-level pain rating during the placebo-test trials
    placeb = D.all_eval_frame[D.all_eval_frame.trial.isin(PLACEBO_TRIALS)]
    pv = (placeb.groupby(["participant", "condition"]).mean(numeric_only=True)
          .reset_index().pivot(index="participant", columns="condition", values="ratings"))
    D.pain_on, D.pain_off = pv["active"], pv["inactive"]

    w = D.wide_dat
    D.corr_df = pd.DataFrame({
        "placebo_effect": w["perc_placebo_all"], "accuracy_all": w["acc_all"],
        "pcs_total": w["pcs_total"], "iastay1_total": w["iastay1_total"],
        "iastay2_total": w["iastay2_total"],
    })
    return D


def wide_pair(D, col_on, col_off):
    w = D.wide_dat.set_index("participant")
    return w[col_on].astype(float), w[col_off].astype(float)


# =============================================================================
# BUILDING BLOCKS (shared by every figure -> guarantees a consistent look)
# =============================================================================
def shade_phases(ax):
    for a, b in PLACEBO_SPANS:
        ax.axvspan(a, b, color=COLOR_PLACEBO_BG, lw=0, zorder=0)


def label_phases(ax):
    tr = ax.get_xaxis_transform()
    for (a, b), txt in PHASES:
        ax.text((a + b) / 2, 1.025, txt, transform=tr, ha="center", va="bottom",
                fontsize=FS_SMALL, color="0.25", clip_on=False)


def draw_timeseries(ax, df, xcol, ycol, ylim, ylabel, yticks=None, chance=None, dodge=0.08):
    """Mean +/- SE across participants per trial and condition. Lines are broken at
    phase boundaries, because the stimulus intensity changes there."""
    shade_phases(ax)
    ts = df.groupby(["condition", xcol])[ycol].agg(["mean", "sem"]).reset_index()
    for cond in ("inactive", "active"):  # draw 'active' last (on top)
        sub = ts[ts.condition == cond]
        off = -dodge if cond == "active" else dodge
        for a, b in [s for s, _ in PHASES]:
            s = sub[(sub[xcol] > a) & (sub[xcol] < b)]
            if s.empty:
                continue
            ax.errorbar(s[xcol] + off, s["mean"], yerr=s["sem"], color=PAL[cond],
                        marker=MARKER[cond], ms=3.4, mew=0.4, mec="white", lw=1.0,
                        elinewidth=0.8, capsize=0, zorder=3)
    ax.set_xlim(*XLIM_TRIALS)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if chance is not None:
        ax.axhline(chance, color="0.4", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(XLIM_TRIALS[0] + 0.15, chance, "chance", ha="left", va="bottom",
                fontsize=FS_SMALL, color="0.35")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(np.arange(1, 25))


def timeseries_legend(ax, loc="upper left"):
    h = [Line2D([], [], color=PAL[c], marker=MARKER[c], ms=3.4, mew=0.4, mec="white",
                lw=1.0, label=LABEL[c]) for c in ("active", "inactive")]
    ax.legend(handles=h, loc=loc, ncol=2, frameon=False, handlelength=1.6,
              columnspacing=1.2, borderaxespad=0.1, handletextpad=0.5)


def _nice_ticks(lo, hi, n=5):
    t = MaxNLocator(nbins=n).tick_values(lo, hi)
    return [x for x in t if lo - 1e-9 <= x <= hi + 1e-9]


def draw_paired(ax, on, off, ylabel, lo, hi, yticks=None, chance=None, p=None,
                jitter=0.0, show_xlabels=True, headroom=HEADROOM):
    """TENS on vs off, one dot per participant, thin paired lines, narrow box
    (median + IQR), white diamond = mean. Bracket on top gives the p-value."""
    on, off = on.dropna(), off.dropna()
    common = on.index.intersection(off.index)
    on, off = on.loc[common], off.loc[common]
    rng = np.random.default_rng(SEED)
    j = rng.normal(0, jitter, len(on)) if jitter else np.zeros(len(on))

    for i in range(len(on)):  # paired lines
        ax.plot([0 + j[i], 1 + j[i]], [on.iloc[i], off.iloc[i]], color="0.55",
                lw=0.45, alpha=0.45, zorder=1, solid_capstyle="round")
    bp = ax.boxplot([on.values, off.values], positions=[0, 1], widths=0.34,
                    patch_artist=True, showfliers=False, showcaps=False, whis=1.5,
                    manage_ticks=False)
    for k, cond in enumerate(("active", "inactive")):
        c = PAL[cond]
        bp["boxes"][k].set(facecolor=to_rgba(c, 0.18), edgecolor=c, lw=0.9, zorder=2)
        for w_ in bp["whiskers"][2 * k:2 * k + 2]:
            w_.set(color=c, lw=0.9, zorder=2)
        bp["medians"][k].set(color="black", lw=1.2, zorder=4)
    for x0, vals, cond in ((0, on, "active"), (1, off, "inactive")):
        xs = x0 + j
        ax.scatter(xs, vals.values, s=8, color=PAL[cond], alpha=0.8, edgecolors="white",
                   linewidths=0.25, zorder=3)
        ax.plot([x0], [vals.mean()], marker="D", ms=4, mfc="white", mec="black", mew=0.8,
                zorder=5, ls="none")

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(lo, hi + headroom * (hi - lo))
    ax.spines["left"].set_bounds(lo, hi)
    ax.set_yticks(yticks if yticks is not None else _nice_ticks(lo, hi))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL["active"], LABEL["inactive"]] if show_xlabels else [])
    ax.set_ylabel(ylabel)
    if chance is not None:
        ax.axhline(chance, color="0.4", lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.text(1.58, chance, "chance", ha="right", va="bottom", fontsize=FS_SMALL,
                color="0.35")
    txt = fmt_p(p)
    if txt:
        y0 = hi + 0.035 * (hi - lo)
        d = 0.02 * (hi - lo)
        ax.plot([0, 0, 1, 1], [y0 - d, y0, y0, y0 - d], color="black", lw=0.7,
                clip_on=False, zorder=6)
        ax.text(0.5, y0 + 0.01 * (hi - lo), txt, ha="center", va="bottom",
                fontsize=FS_SMALL, clip_on=False)
    return ax


def pain_limits(D):
    hi = max(PAIN_YLIM[1], float(np.ceil(max(D.pain_on.max(), D.pain_off.max()) / 10) * 10))
    return 0, hi


def acc_limits(on, off):
    lo = ACC_YLIM[0]
    m = min(on.min(), off.min())
    if m < lo:
        lo = np.floor((m - 0.02) * 10) / 10
    return lo, ACC_YLIM[1]


def n_note(ax, D):
    if SHOW_N:
        ax.text(0.02, 0.02, f"n = {D.n}", transform=ax.transAxes, fontsize=FS_SMALL,
                color="0.3", va="bottom")


# =============================================================================
# FIGURES
# =============================================================================
def fig_group_eval_task(D):
    """Pain ratings across the session (140 x 70 mm)."""
    fig = new_fig(W_MID, 70)
    ax = add_ax(fig, 14, 13, 124, 46)
    draw_timeseries(ax, D.eval_ts, "trial", "ratings", PAIN_YLIM, PAIN_LABEL, yticks=PAIN_YTICKS)
    ax.set_xlabel("Trial")
    label_phases(ax)
    timeseries_legend(ax)
    return fig


def fig_group_discrim_task(D):
    """Pain ratings (top) + discrimination accuracy (bottom), shared x (140 x 100 mm)."""
    fig = new_fig(W_MID, 100)
    ax_a = add_ax(fig, 14, 14 + 32 + 5, 124, 40)
    ax_b = add_ax(fig, 14, 14, 124, 32, sharex=ax_a)
    draw_timeseries(ax_a, D.eval_ts, "trial", "ratings", PAIN_YLIM, PAIN_LABEL, yticks=PAIN_YTICKS)
    ax_a.tick_params(labelbottom=False)
    label_phases(ax_a)
    timeseries_legend(ax_a)
    draw_timeseries(ax_b, D.acc_ts, "actual_trial", "accuracy", ACC_YLIM, ACC_LABEL, yticks=ACC_YTICKS, chance=0.5)
    ax_b.set_xlabel("Trial")
    return fig


def fig_mean_placebo_effect(D):
    """Pain rating TENS on vs off, placebo-test trials (90 x 75 mm)."""
    fig = new_fig(W_SINGLE, 75)
    ax = add_ax(fig, 15, 12, 72, 56)
    lo, hi = pain_limits(D)
    draw_paired(ax, D.pain_on, D.pain_off, PAIN_LABEL, lo, hi,
                p=stat_p(read_stat(D, "t_test_placebo_eval.csv")))
    n_note(ax, D)
    return fig


def fig_placebo_effect_distribution(D):
    """Placebo effect of every participant, ranked (90 x 75 mm)."""
    fig = new_fig(W_SINGLE, 75)
    ax = add_ax(fig, 15, 12, 72, 58)
    v = np.sort(D.wide_dat["perc_placebo_all"].dropna().values)
    ax.axhline(PLACEBO_THRESHOLD_LINE, color="0.4", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.scatter(np.arange(1, len(v) + 1), v, s=10, color=COLOR_NEUTRAL, alpha=0.85,
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.text(len(v), PLACEBO_THRESHOLD_LINE, f"{PLACEBO_THRESHOLD_LINE}%", ha="right",
            va="bottom", fontsize=FS_SMALL, color="0.35")
    ax.set_xlabel("Participants (ranked)")
    ax.set_ylabel("Placebo effect (%)")
    ax.set_xlim(0, len(v) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    return fig


def _corr_fig(D, x, y, xlabel, ylabel):
    df = D.corr_df[[x, y]].dropna()
    xv, yv = df[x].values.astype(float), df[y].values.astype(float)
    n = len(df)
    fig = new_fig(W_SINGLE, 82)
    ax = add_ax(fig, 15, 13, 72, 60)

    # OLS line + analytic 95% CI band (deterministic, no bootstrap)
    b, a = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 100)
    resid = yv - (a + b * xv)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (xs - xv.mean()) ** 2 / np.sum((xv - xv.mean()) ** 2))
    t = sps.t.ppf(0.975, n - 2)
    ax.fill_between(xs, a + b * xs - t * se, a + b * xs + t * se, color="0.85", lw=0, zorder=1)
    ax.plot(xs, a + b * xs, color="black", lw=1.0, zorder=2)
    ax.scatter(xv, yv, s=14, color=COLOR_NEUTRAL, alpha=0.8, edgecolors="white",
               linewidths=0.3, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    r, p = sps.pearsonr(xv, yv)
    txt = f"r = {r:.2f}, {fmt_p(p)}"
    bf = stat_bf(read_stat(D, CORR_STAT_FILES.get((x, y), "")))
    if bf:
        txt += f", BF\u2081\u2080 = {bf}"
    ax.set_title(txt, loc="left", fontsize=FS_TEXT, pad=3)
    return fig


CORR_STAT_FILES = {
    ("placebo_effect", "accuracy_all"): "corr_placebo_accuracy.csv",
    ("placebo_effect", "pcs_total"): "corr_2_pcs.csv",
    ("placebo_effect", "iastay1_total"): "corr_2_iasta1.csv",
    ("placebo_effect", "iastay2_total"): "corr_2_iasta2.csv",
    ("accuracy_all", "pcs_total"): "corr_3_pcs.csv",
    ("accuracy_all", "iastay1_total"): "corr_3_iasta1.csv",
    ("accuracy_all", "iastay2_total"): "corr_3_iasta2.csv",
}
PLACEBO_L, ACC_L = "Placebo effect (%)", "Discrimination accuracy"


def fig_correlation_placebo_accuracy(D):
    return _corr_fig(D, "placebo_effect", "accuracy_all", PLACEBO_L, ACC_L)


def fig_correlation_placebo_pcs(D):
    return _corr_fig(D, "placebo_effect", "pcs_total", PLACEBO_L, "PCS score")


def fig_correlation_placebo_iasta1(D):
    return _corr_fig(D, "placebo_effect", "iastay1_total", PLACEBO_L, "STAI-Y1 score")


def fig_correlation_placebo_iasta2(D):
    return _corr_fig(D, "placebo_effect", "iastay2_total", PLACEBO_L, "STAI-Y2 score")


def fig_correlation_accuracy_pcs(D):
    return _corr_fig(D, "accuracy_all", "pcs_total", ACC_L, "PCS score")


def fig_correlation_accuracy_iasta1(D):
    return _corr_fig(D, "accuracy_all", "iastay1_total", ACC_L, "STAI-Y1 score")


def fig_correlation_accuracy_iasta2(D):
    return _corr_fig(D, "accuracy_all", "iastay2_total", ACC_L, "STAI-Y2 score")


def fig_tost_discrim(D):
    """TOST equivalence plot, Cohen's dz with 90% CI (90 x 42 mm)."""
    w = D.wide_dat
    bound = 0.45
    diff = w["inactive_acc_all"] - w["active_acc_all"]
    sd = np.std(w["active_acc_all"] - w["inactive_acc_all"], ddof=1)
    dz = diff.mean() / sd
    h = sps.sem(diff) * sps.t.ppf(0.95, len(w) - 1) / sd  # 90% CI half-width
    fig = new_fig(W_SINGLE, 42)
    ax = add_ax(fig, 6, 13, 78, 20)
    lim = max(bound, abs(dz) + h) * 1.35
    ax.axvspan(-bound, bound, color=COLOR_PLACEBO_BG, lw=0, zorder=0)
    for xb in (-bound, bound):
        ax.axvline(xb, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color="0.7", lw=0.5, zorder=1)
    ax.errorbar([dz], [0], xerr=[[h], [h]], fmt="o", color="black", ms=4.5, lw=1.0,
                capsize=2.5, capthick=1.0, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-bound, 0, bound])
    ax.set_xticklabels(["\u22120.45", "0", "0.45"])
    ax.set_xlabel("Cohen\u2019s $d_z$ (TENS off \u2212 TENS on), 90% CI")
    p = stat_p(read_stat(D, "tost_accuracy.csv"))
    txt = "Equivalence bounds \u00b10.45" + (f"; TOST {fmt_p(p)}" if p is not None else "")
    ax.set_title(txt, loc="left", fontsize=FS_SMALL, pad=3)
    return fig


def _paired_wide_fig(D, col_on, col_off, ylabel, stat_file, chance=None, acc=False):
    on, off = wide_pair(D, col_on, col_off)
    if acc:
        lo, hi = acc_limits(on, off)
    else:
        m, M = min(on.min(), off.min()), max(on.max(), off.max())
        pad = 0.06 * (M - m)
        lo, hi = m - pad, M + pad
    fig = new_fig(W_SINGLE, 75)
    ax = add_ax(fig, 15, 12, 72, 56)
    draw_paired(ax, on, off, ylabel, lo, hi, chance=chance,
                p=stat_p(read_stat(D, stat_file)), jitter=0.04 if acc else 0.0,
                yticks=ACC_YTICKS if (acc and lo == ACC_YLIM[0]) else None)
    n_note(ax, D)
    return fig


def fig_discrim_acc_cond(D):
    return _paired_wide_fig(D, "active_acc_all", "inactive_acc_all", ACC_LABEL,
                            "t_test_accuracy.csv", chance=0.5, acc=True)


def fig_discrim_dprime_cond(D):
    return _paired_wide_fig(D, "d_prime_active", "d_prime_inactive", "d\u2032 (sensitivity)",
                            "t_test_dprime.csv")


def fig_discrim_beta_cond(D):
    return _paired_wide_fig(D, "beta_active", "beta_inactive", "\u03b2 (bias)",
                            "t_test_beta.csv")


def fig_discrim_c_cond(D):
    return _paired_wide_fig(D, "c_active", "c_inactive", "Criterion (c)", "t_test_c.csv")


def fig_extinction_placebo(D):
    """Placebo effect across the four placebo-test trials of each block (90 x 75 mm)."""
    fig = new_fig(W_SINGLE, 75)
    ax = add_ax(fig, 15, 12, 72, 58)
    styles = {1: ("#222222", "o", -0.04), 2: ("#8c8c8c", "s", 0.04)}
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)), zorder=0)
    for b, (c, m, off) in styles.items():
        cols = [f"placebo_b{b}_{t}" for t in (1, 2, 3, 4)]
        d = D.wide_dat[cols]
        ax.errorbar(np.arange(1, 5) + off, d.mean().values, yerr=d.sem().values, color=c,
                    marker=m, ms=3.8, mew=0.4, mec="white", lw=1.0, elinewidth=0.8,
                    capsize=0, label=f"Block {b}", zorder=3)
    ax.set_xlim(0.5, 4.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Placebo-test trial within block")
    ax.set_ylabel("Placebo effect (TENS on \u2212 TENS off)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]) if ax.get_ylim()[0] < 0 else 0)
    return fig


def fig_main_figure_combined(D):
    """Main 4-panel figure, 190 x 118 mm (full page width).
    Right-hand panels are (1 + HEADROOM) taller than the left ones, so that rows share the
    same baseline and the same scale (the extra height holds the p-value bracket)."""
    fig = new_fig(W_FULL, 118)
    left, lw_ = 14, 112
    rleft, rw = 142, 45
    b0, hb, gap, ha = 14, 34, 11, 48
    ax_b = add_ax(fig, left, b0, lw_, hb)
    ax_a = add_ax(fig, left, b0 + hb + gap, lw_, ha, sharex=ax_b)
    ax_d = add_ax(fig, rleft, b0, rw, hb * (1 + HEADROOM))
    ax_c = add_ax(fig, rleft, b0 + hb + gap, rw, ha * (1 + HEADROOM))

    # left column: time courses
    draw_timeseries(ax_a, D.eval_ts, "trial", "ratings", PAIN_YLIM, PAIN_LABEL,
                    yticks=PAIN_YTICKS)
    ax_a.tick_params(labelbottom=False)
    label_phases(ax_a)
    timeseries_legend(ax_a)
    draw_timeseries(ax_b, D.acc_ts, "actual_trial", "accuracy", ACC_YLIM, ACC_LABEL,
                    yticks=ACC_YTICKS, chance=0.5)
    ax_b.set_xlabel("Trial")

    # right column: TENS on vs off summaries (x labels only on the lower one)
    lo_p, hi_p = pain_limits(D)
    on_a, off_a = wide_pair(D, "active_acc_all", "inactive_acc_all")
    lo_a, hi_a = acc_limits(on_a, off_a)
    pain_kw = dict(ylabel=PAIN_LABEL, lo=lo_p, hi=hi_p,
                   p=stat_p(read_stat(D, "t_test_placebo_eval.csv")))
    acc_kw = dict(ylabel=ACC_LABEL, lo=lo_a, hi=hi_a, chance=0.5, jitter=0.04,
                  p=stat_p(read_stat(D, "t_test_accuracy.csv")),
                  yticks=ACC_YTICKS if lo_a == ACC_YLIM[0] else None)
    if MAIN_PAIN_ON_TOP_RIGHT:
        draw_paired(ax_c, D.pain_on, D.pain_off, show_xlabels=False, **pain_kw)
        draw_paired(ax_d, on_a, off_a, show_xlabels=True, **acc_kw)
    else:  # old order: c = accuracy, d = pain (rows then no longer share a scale)
        draw_paired(ax_c, on_a, off_a, show_xlabels=False, **acc_kw)
        draw_paired(ax_d, D.pain_on, D.pain_off, show_xlabels=True, **pain_kw)

    for ax, letter, row in ((ax_a, "a", ax_a), (ax_b, "b", ax_b),
                            (ax_c, "c", ax_a), (ax_d, "d", ax_b)):
        panel_label(fig, ax, letter, y_from=row)
    return fig


# =============================================================================
# REGISTRY + RUNNER
# =============================================================================
FIGURES = {
    "group_eval_task": fig_group_eval_task,
    "group_discrim_task": fig_group_discrim_task,
    "mean_placebo_effect": fig_mean_placebo_effect,
    "placebo_effect_distribution": fig_placebo_effect_distribution,
    "correlation_placebo_accuracy": fig_correlation_placebo_accuracy,
    "correlation_placebo_pcs": fig_correlation_placebo_pcs,
    "correlation_placebo_iasta1": fig_correlation_placebo_iasta1,
    "correlation_placebo_iasta2": fig_correlation_placebo_iasta2,
    "correlation_accuracy_pcs": fig_correlation_accuracy_pcs,
    "correlation_accuracy_iasta1": fig_correlation_accuracy_iasta1,
    "correlation_accuracy_iasta2": fig_correlation_accuracy_iasta2,
    "tost_discrim": fig_tost_discrim,
    "discrim_acc_cond": fig_discrim_acc_cond,
    "discrim_dprime_cond": fig_discrim_dprime_cond,
    "discrim_beta_cond": fig_discrim_beta_cond,
    "discrim_c_cond": fig_discrim_c_cond,
    "extinction_placebo": fig_extinction_placebo,
    "main_figure_combined": fig_main_figure_combined,
}


def discover_runs(requested):
    def is_run(p):
        return os.path.exists(opj(p, "data_wide_dat_withexcl.csv"))

    if requested:
        out = []
        for r in requested:
            cand = r if os.path.isabs(r) else opj(DERIVATIVES_DIR, r)
            if not is_run(cand) and is_run(os.path.abspath(r)):
                cand = os.path.abspath(r)
            out.append(cand)
        return out
    found = []
    if os.path.isdir(DERIVATIVES_DIR):
        if is_run(DERIVATIVES_DIR):
            found.append(DERIVATIVES_DIR)
        for name in sorted(os.listdir(DERIVATIVES_DIR)):
            p = opj(DERIVATIVES_DIR, name)
            if os.path.isdir(p) and is_run(p):
                found.append(p)
    return found


def select_figures(patterns):
    if not patterns:
        return list(FIGURES)
    chosen = []
    for pat in patterns:
        hits = fnmatch.filter(FIGURES, pat)
        if not hits:
            sys.exit(f"No figure matches '{pat}'. Use --list to see names.")
        chosen += [h for h in hits if h not in chosen]
    return chosen


def save_figure(fig, name, out_dir, formats, dpi):
    for fmt in formats:
        kw = {}
        if fmt in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        # NB: no bbox_inches="tight" -> the file keeps its exact size in mm
        fig.savefig(opj(out_dir, f"{name}.{fmt}"), dpi=dpi, transparent=False, **kw)


def main(argv=None):
    """argv=None reads the command line; in an interactive window defaults are used,
    or call main(["--only", "main_figure_combined", "--runs", "placebo_0"])."""
    if argv is None and ("ipykernel" in sys.modules or hasattr(sys, "ps1")):
        argv = []
    ap = argparse.ArgumentParser(description="Journal-ready figures from analyze.py outputs.",
                                 allow_abbrev=False)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--draft", action="store_true", help="PNG only, 100 dpi")
    ap.add_argument("--out-subdir", default=OUT_SUBDIR)
    ap.add_argument("--list", action="store_true")
    args, _unknown = ap.parse_known_args(argv)

    runs = discover_runs(args.runs)
    if args.list:
        print("Runs found:")
        for r in runs:
            print("  ", r)
        print("Figures:")
        for n in FIGURES:
            print("  ", n)
        return
    if not runs:
        sys.exit(f"No run folders with data_wide_dat_withexcl.csv found in {DERIVATIVES_DIR}.")

    names = select_figures(args.only)
    formats = ["png"] if args.draft else args.formats
    dpi = 100 if args.draft else (args.dpi or DPI)
    setup_style()

    for run_dir in runs:
        print(f"\n=== {os.path.basename(os.path.normpath(run_dir))} ===")
        try:
            D = load_run(run_dir)
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        out_dir = opj(run_dir, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"  n = {D.n} participants -> {out_dir}")
        if not os.path.isdir(opj(run_dir, "stats")):
            print("  [note] no stats/ folder: p-values will not be printed on the figures.")
        for name in names:
            try:
                fig = FIGURES[name](D)
                save_figure(fig, name, out_dir, formats, dpi)
                print(f"  ok   {name}")
            except Exception as e:
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
            finally:
                plt.close("all")


if __name__ == "__main__":
    main()
