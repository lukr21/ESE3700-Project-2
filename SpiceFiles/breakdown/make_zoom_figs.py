"""Three figures (all at T = 320 ps) — now with slimmer stage sets and tables
written to separate markdown files so the PNGs stay readable.

Outputs (all written to project 2/figures/breakdown/):
    worst_case_read_zoom.png      + worst_case_read_events.md
    worst_case_write_zoom.png     + worst_case_write_events.md
    full_period_320ps.png         (caption only, no event table)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use Times New Roman where available; fall back to Liberation Serif / Nimbus Roman
# (both are metric-compatible Times clones) so renders look identical on Linux.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "Liberation Serif", "Nimbus Roman No9 L", "DejaVu Serif",
]
plt.rcParams["mathtext.fontset"] = "stix"

HERE   = Path(__file__).parent
FIGDIR = HERE.parent.parent / "figures" / "breakdown"
FIGDIR.mkdir(parents=True, exist_ok=True)

BASE_COLS = (HERE / "signal_columns.txt").read_text().split()
COL_NAMES = BASE_COLS + ["cell_n3", "cell_n35"]
VDD    = 0.8
VTH    = VDD / 2
BL_DEV = 0.050
TPER   = 320e-12


# ----------------------- loaders / helpers -----------------------

def load(path=HERE / "probed_320ps_v2_results.txt"):
    data = np.loadtxt(path, skiprows=1)
    t = data[:, 0]
    sig = {name: data[:, i + 1] for i, name in enumerate(COL_NAMES)}
    return t, sig


def cross(t, y, thresh, rising=True, t_min=None, t_max=None):
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None: mask &= t >= t_min
    if t_max is not None: mask &= t <= t_max
    idx = np.where(mask)[0]
    if len(idx) < 2: return None
    yy, tt = y[idx], t[idx]
    s = (yy >= thresh).astype(int) if rising else (yy <= thresh).astype(int)
    d = np.diff(s)
    hits = np.where(d == 1)[0]
    if len(hits) == 0: return None
    k = hits[0]
    if yy[k + 1] == yy[k]: return float(tt[k + 1])
    return float(tt[k] + (thresh - yy[k]) * (tt[k + 1] - tt[k]) / (yy[k + 1] - yy[k]))


def any_cross(t, y, thresh, **kw):
    r = cross(t, y, thresh, rising=True, **kw)
    f = cross(t, y, thresh, rising=False, **kw)
    if r is None: return f
    if f is None: return r
    return min(r, f)


def diff_cross(t, a, b, sep, t_min=None, t_max=None):
    diff = np.abs(a - b)
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None: mask &= t >= t_min
    if t_max is not None: mask &= t <= t_max
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    if diff[idx[0]] >= sep:
        return float(t[idx[0]])
    return cross(t, diff, sep, rising=True, t_min=t_min, t_max=t_max)


def phi2_rise_in_cycle(t, sig, cyc):
    lo, hi = (cyc - 1) * TPER, cyc * TPER
    return cross(t, sig["phi2"], VTH, rising=True, t_min=lo, t_max=hi)


# ----------------------- plot helpers -----------------------

def draw_markers(ax, events, t0, show_labels):
    """events = [(label, t_abs, description)]. Labels alternate two heights."""
    heights = [1.05, 1.14]
    for i, (lab, t_abs, _) in enumerate(events):
        if t_abs is None: continue
        dt = (t_abs - t0) * 1e12
        ax.axvline(dt, color="tab:gray", lw=1.0, ls="--", alpha=0.7)
        if show_labels:
            ax.text(dt, heights[i % 2], lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=11, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="tab:gray", lw=0.7, alpha=0.95))


def write_events_md(path, title, events, t0):
    def plain(s):
        return (s.replace("$_{DD}$", "_DD")
                 .replace("V$_{DD}$", "V_DD")
                 .replace("$_0$", "0")
                 .replace("$_1$", "1")
                 .replace("$_{n3}$", "_n3")
                 .replace("$_{n35}$", "_n35")
                 .replace("$_", "").replace("$", ""))
    lines = [f"# {title}", "", "| Mark | Offset from Phi2↑ | Event |", "|---|---|---|"]
    for lab, t_abs, desc in events:
        if t_abs is None:
            lines.append(f"| {plain(lab)} | — | {plain(desc)} |")
        else:
            dt = (t_abs - t0) * 1e12
            lines.append(f"| {plain(lab)} | {dt:+.1f} ps | {plain(desc)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ----------------------- figure 1: READ zoom -----------------------

def fig_read_zoom(t, sig):
    # Worst-case read: cyc 4 (read 0 after 1) — R0 must flip 1 -> 0.
    t0 = phi2_rise_in_cycle(t, sig, 4)
    win = (t0 - 8e-12, t0 + 110e-12)

    t_wl    = cross(t, sig["wl1"], VTH, rising=True, t_min=t0, t_max=win[1])
    t_bldev = diff_cross(t, sig["bl0"], sig["blb0"], BL_DEV, t_min=t_wl, t_max=win[1])
    t_sae   = cross(t, sig["sae"], VTH, rising=True, t_min=t0, t_max=win[1])
    t_qd    = any_cross(t, sig["qd0"], VTH, t_min=t_sae, t_max=win[1])
    t_r0    = any_cross(t, sig["r0"],  VTH, t_min=t_sae, t_max=win[1])

    events = [
        ("t$_1$", t0,      "Phi2 rises — access begins"),
        ("t$_2$", t_wl,    "WL1 rises — decoder propagation finished"),
        ("t$_3$", t_bldev, "Bitline differential ≥ 50 mV"),
        ("t$_4$", t_sae,   "SAE fires — sense amp armed"),
        ("t$_5$", t_qd,    "Q/D$_0$ at V$_{DD}$/2 — sense amp resolved"),
        ("t$_6$", t_r0,    "R$_0$ at V$_{DD}$/2 — read latched"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw=dict(hspace=0.35))
    fig.suptitle("Worst-case READ critical path  —  T = 320 ps  (cyc 4: read 0 after 1)",
                 fontsize=13)

    td = (t - t0) * 1e12
    msk = (t >= win[0]) & (t <= win[1])

    # Panel 1 — clocks + WL + SAE (the core path)
    ax = axes[0]
    ax.plot(td[msk], sig["phi2"][msk], lw=2.0, color="tab:orange", label="Phi2")
    ax.plot(td[msk], sig["wl1"][msk],  lw=2.0, color="tab:blue",   label="WL1")
    ax.plot(td[msk], sig["sae"][msk],  lw=2.0, color="tab:red",    label="SAE")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=False)

    # Panel 2 — bitline differential
    ax = axes[1]
    ax.plot(td[msk], sig["bl0"][msk],  lw=1.8, color="tab:blue",  label="BL0")
    ax.plot(td[msk], sig["blb0"][msk], lw=1.8, color="tab:blue",  ls="--", alpha=0.85, label="BLb0")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    # Panel 3 — output path
    ax = axes[2]
    ax.plot(td[msk], sig["qd0"][msk], lw=2.0, color="tab:green", label="Q/D$_0$")
    ax.plot(td[msk], sig["r0"][msk],  lw=2.0, color="tab:cyan",  label="R$_0$")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("time since Phi2 rise (ps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=False)

    draw_markers(axes[0], events, t0, show_labels=True)
    for ax in axes[1:]:
        draw_markers(ax, events, t0, show_labels=False)
    for ax in axes:
        ax.set_xlim((win[0] - t0) * 1e12, (win[1] - t0) * 1e12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "worst_case_read_zoom.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")

    write_events_md(FIGDIR / "worst_case_read_events.md",
                    "Worst-case READ timing events (T = 320 ps, cyc 4)",
                    events, t0)


# ----------------------- figure 2: WRITE zoom -----------------------

def fig_write_zoom(t, sig):
    # Worst-case write: cyc 3 (write 0 over stored 1) — cell internal must flip.
    t0 = phi2_rise_in_cycle(t, sig, 3)
    win = (t0 - 8e-12, t0 + 140e-12)

    t_wl   = cross(t, sig["wl1"], VTH, rising=True, t_min=t0, t_max=win[1])
    t_wren = cross(t, sig["wren"], VTH, rising=True, t_min=t0, t_max=win[1])
    t_bl   = cross(t, sig["bl0"], VTH, rising=False, t_min=t_wren, t_max=win[1])
    t_cell = any_cross(t, sig["cell_n3"], VTH, t_min=t_wl, t_max=win[1])

    events = [
        ("t$_1$", t0,     "Phi2 rises — access begins"),
        ("t$_2$", t_wl,   "WL1 rises — decoder propagation finished"),
        ("t$_3$", t_wren, "WrEn rises — column drivers enabled"),
        ("t$_4$", t_bl,   "BL0 crosses V$_{DD}$/2 — data on bitline"),
        ("t$_5$", t_cell, "Cell internal at V$_{DD}$/2 — write committed"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw=dict(hspace=0.35))
    fig.suptitle("Worst-case WRITE critical path  —  T = 320 ps  (cyc 3: write 0 over stored 1)",
                 fontsize=13)

    td = (t - t0) * 1e12
    msk = (t >= win[0]) & (t <= win[1])

    # Panel 1 — clocks + decoder + WrEn
    ax = axes[0]
    ax.plot(td[msk], sig["phi2"][msk], lw=2.0, color="tab:orange", label="Phi2")
    ax.plot(td[msk], sig["wl1"][msk],  lw=2.0, color="tab:blue",   label="WL1")
    ax.plot(td[msk], sig["wren"][msk], lw=2.0, color="tab:green",  label="WrEn")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=False)

    # Panel 2 — bitline being driven
    ax = axes[1]
    ax.plot(td[msk], sig["bl0"][msk],  lw=1.8, color="tab:blue",  label="BL0  (driven)")
    ax.plot(td[msk], sig["blb0"][msk], lw=1.8, color="tab:blue",  ls="--", alpha=0.85, label="BLb0 (driven)")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=False)

    # Panel 3 — cell internal (the thing that actually has to flip)
    ax = axes[2]
    ax.plot(td[msk], sig["cell_n3"][msk],  lw=2.0, color="tab:purple", label="cell net@3")
    ax.plot(td[msk], sig["cell_n35"][msk], lw=2.0, color="tab:purple", ls="--", alpha=0.85, label="cell net@35")
    ax.axhline(VTH, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("time since Phi2 rise (ps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=False)

    draw_markers(axes[0], events, t0, show_labels=True)
    for ax in axes[1:]:
        draw_markers(ax, events, t0, show_labels=False)
    for ax in axes:
        ax.set_xlim((win[0] - t0) * 1e12, (win[1] - t0) * 1e12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "worst_case_write_zoom.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")

    write_events_md(FIGDIR / "worst_case_write_events.md",
                    "Worst-case WRITE timing events (T = 320 ps, cyc 3)",
                    events, t0)


# ----------------------- figure 3: full 320-ps period -----------------------

def fig_full_period(t, sig):
    # Anchor at Phi1 rise of cyc 2 so one period shows precharge -> access -> next precharge.
    phi1_rise_abs = cross(t, sig["phi1"], VTH, rising=True,
                          t_min=1 * TPER, t_max=2 * TPER)
    lo = phi1_rise_abs - 10e-12
    hi = lo + TPER

    phi1_rise = phi1_rise_abs
    phi1_fall = cross(t, sig["phi1"], VTH, rising=False, t_min=phi1_rise, t_max=hi)
    phi2_rise = cross(t, sig["phi2"], VTH, rising=True,  t_min=phi1_fall, t_max=hi)
    phi2_fall = cross(t, sig["phi2"], VTH, rising=False, t_min=phi2_rise, t_max=hi + 40e-12)
    sae_rise  = cross(t, sig["sae"],  VTH, rising=True,  t_min=phi2_rise,
                      t_max=phi2_fall if phi2_fall is not None else hi)

    td = (t - lo) * 1e12
    msk = (t >= lo) & (t <= hi)

    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.suptitle("One full 320-ps cycle  —  read (cyc 2), phases annotated", fontsize=12)

    ax.plot(td[msk], sig["phi1"][msk], lw=1.6, color="tab:orange", label="Phi1")
    ax.plot(td[msk], sig["phi2"][msk], lw=1.6, color="tab:red",    label="Phi2")
    ax.plot(td[msk], sig["pchb"][msk], lw=1.2, color="tab:purple", label="PCHb")
    ax.plot(td[msk], sig["wl1"][msk],  lw=1.6, color="tab:blue",   label="WL1")
    ax.plot(td[msk], sig["bl0"][msk],  lw=1.2, color="tab:cyan",   label="BL0")
    ax.plot(td[msk], sig["blb0"][msk], lw=1.2, color="tab:cyan",   ls="--", alpha=0.85, label="BLb0")
    ax.plot(td[msk], sig["sae"][msk],  lw=1.6, color="tab:green",  label="SAE")
    ax.plot(td[msk], sig["qd0"][msk],  lw=1.2, color="k",          label="Q/D$_0$")

    ax.axhline(VTH, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("time within cycle (ps)")
    ax.grid(True, alpha=0.3)

    Y_HI, Y_LO = 1.09, 1.02
    def shade(x0, x1, color, label, y):
        if x0 is None or x1 is None: return
        ax.axvspan((x0 - lo) * 1e12, (x1 - lo) * 1e12, color=color, alpha=0.09, lw=0)
        ax.text(((x0 + x1) / 2 - lo) * 1e12, y, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=color,
                bbox=dict(facecolor="white", edgecolor=color, lw=0.8,
                          boxstyle="round,pad=0.25", alpha=0.95))

    shade(lo, phi1_fall,               "tab:purple", "precharge",        Y_HI)
    shade(phi1_fall,  phi2_rise,       "tab:gray",   "non-overlap",      Y_LO)
    shade(phi2_rise,  sae_rise,        "tab:blue",   "access pre-SAE",   Y_HI)
    shade(sae_rise,   phi2_fall,       "tab:green",  "access post-SAE",  Y_LO)
    shade(phi2_fall,  hi,              "tab:purple", "precharge",        Y_HI)

    for x in [0, TPER * 1e12]:
        ax.axvline(x, color="black", lw=0.6, alpha=0.5)

    ax.legend(loc="center right", fontsize=9, frameon=False, ncol=2)
    ax.set_xlim(0, TPER * 1e12)

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    out = FIGDIR / "full_period_320ps.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    t, sig = load()
    fig_read_zoom(t, sig)
    fig_write_zoom(t, sig)
    fig_full_period(t, sig)
