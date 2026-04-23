"""Render 3 zoomed annotated PNGs (one per period) and a comparison bar chart.

Run after analyze_breakdown.py. Reads *_results.txt files + measurements.npz,
writes PNGs into project 2/figures/breakdown/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGDIR = HERE.parent.parent / "figures" / "breakdown"
FIGDIR.mkdir(parents=True, exist_ok=True)

COL_NAMES = (HERE / "signal_columns.txt").read_text().split()

RUNS = [
    ("2000 ps", 2000, "probed_2000ps_results.txt", "probed_2000ps_zoom.png"),
    ("320 ps",   320, "probed_320ps_results.txt",  "probed_320ps_zoom.png"),
    ("298 ps",   298, "probed_298ps_results.txt",  "probed_298ps_zoom.png"),
]

VDD = 0.8


def load(path):
    data = np.loadtxt(path, skiprows=1)
    t = data[:, 0]
    sig = {name: data[:, i+1] for i, name in enumerate(COL_NAMES)}
    return t, sig


def annotate(ax, t_abs, t0, label, color, lw=0.8, ls="--", ytext=0.95, ha="left"):
    """Drop a vertical marker at `t_abs` (relative to t0) labeled with ps offset."""
    if t_abs is None or not np.isfinite(t_abs):
        return
    dt = (t_abs - t0) * 1e12
    ax.axvline(dt, color=color, lw=lw, ls=ls, alpha=0.7)
    ax.text(dt, ytext, f"{label}\n{dt:+.1f} ps", color=color, fontsize=7,
            ha=ha, va="top", rotation=90, transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.8))


def plot_one(label, period_ps, results_file, out_name, meas_idx, meas):
    p = HERE / results_file
    if not p.exists():
        print(f"  [skip] {results_file} missing")
        return
    t, sig = load(p)
    t0 = float(meas["t_phi2"][meas_idx])
    if not np.isfinite(t0):
        print(f"  [skip] no valid t0 for {label}")
        return

    # Zoom window: from slightly before Phi2 rise to well past R0.
    pre = 30e-12
    post = max(period_ps * 1e-12 * 1.2, 250e-12)
    xmin = (t0 - pre - t0) * 1e12   # always -pre in ps
    xmax = (t0 + post - t0) * 1e12

    # Data converted to "ps since Phi2 rise"
    td = (t - t0) * 1e12
    mask = (t >= t0 - pre) & (t <= t0 + post)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"Critical-path timing breakdown  \u2014  T = {label}", fontsize=12)

    # Panel 1: clock + decoder + WL
    ax = axes[0]
    ax.plot(td[mask], sig["clk"][mask],       label="Clk",        lw=1.0, color="k", alpha=0.5, ls=":")
    ax.plot(td[mask], sig["phi2"][mask],      label="Phi2",       lw=1.6, color="tab:orange")
    ax.plot(td[mask], sig["dec_nand1"][mask], label="net@314 (pre-buf)", lw=1.0, color="tab:brown")
    ax.plot(td[mask], sig["wl1"][mask],       label="WL1",        lw=1.6, color="tab:blue")
    ax.plot(td[mask], sig["sae"][mask],       label="SAE",        lw=1.6, color="tab:red")
    ax.axhline(VDD/2, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    annotate(ax, meas["t_clk"][meas_idx],      t0, "Clk\u2191",      "k",         ytext=0.98)
    annotate(ax, meas["t_phi2"][meas_idx],     t0, "Phi2\u2191",     "tab:orange",ytext=0.98)
    annotate(ax, meas["t_dec_nand"][meas_idx], t0, "net@314\u2193",  "tab:brown", ytext=0.76)
    annotate(ax, meas["t_wl1"][meas_idx],      t0, "WL1\u2191",      "tab:blue",  ytext=0.98)
    annotate(ax, meas["t_sae"][meas_idx],      t0, "SAE\u2191",      "tab:red",   ytext=0.98)

    # Panel 2: BL development + SA internal development
    ax = axes[1]
    ax.plot(td[mask], sig["bl0"][mask],  label="BL0",  lw=1.3, color="tab:blue")
    ax.plot(td[mask], sig["blb0"][mask], label="BLb0", lw=1.3, color="tab:blue",  ls="--", alpha=0.8)
    ax.plot(td[mask], sig["sa_n3"][mask],  label="SA net@3",  lw=1.0, color="tab:purple")
    ax.plot(td[mask], sig["sa_n35"][mask], label="SA net@35", lw=1.0, color="tab:purple", ls="--", alpha=0.8)
    ax.axhline(VDD/2, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    annotate(ax, meas["t_wl1"][meas_idx],    t0, "WL1\u2191",       "tab:blue",   ytext=0.98)
    annotate(ax, meas["t_bl_dev"][meas_idx], t0, "BL dev 50mV",     "tab:blue",   ytext=0.75)
    annotate(ax, meas["t_sae"][meas_idx],    t0, "SAE\u2191",       "tab:red",    ytext=0.98)
    annotate(ax, meas["t_sa_dev"][meas_idx], t0, "SA dev 50mV",     "tab:purple", ytext=0.60)

    # Panel 3: Q/D and R output
    ax = axes[2]
    ax.plot(td[mask], sig["sae"][mask], label="SAE",   lw=1.2, color="tab:red",  alpha=0.7)
    ax.plot(td[mask], sig["qd0"][mask], label="Q/D0",  lw=1.6, color="tab:green")
    ax.plot(td[mask], sig["r0"][mask],  label="R0",    lw=1.6, color="tab:cyan")
    ax.axhline(VDD/2, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("V"); ax.set_ylim(-0.05, VDD + 0.05)
    ax.set_xlabel("time since Phi2 rise (ps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    annotate(ax, meas["t_sae"][meas_idx], t0, "SAE\u2191",       "tab:red",   ytext=0.98)
    annotate(ax, meas["t_qd"][meas_idx],  t0, "Q/D0 V_DD/2",     "tab:green", ytext=0.98)
    annotate(ax, meas["t_r0"][meas_idx],  t0, "R0 V_DD/2",       "tab:cyan",  ytext=0.98)

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / out_name
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  wrote {out}")


def stacked_bar_chart(meas, labels):
    """Gantt-style stacked bars: one row per period, bars = major stages."""
    fig, ax = plt.subplots(figsize=(12, 3.3))
    y_labels = []
    stage_colors = {
        "Clk \u2192 Phi2":          "#888888",
        "Phi2 \u2192 WL1":          "tab:blue",
        "WL1 \u2192 BL 50 mV":      "tab:cyan",
        "Phi2 \u2192 SAE":          "tab:red",
        "SAE \u2192 Q/D at V_DD/2": "tab:green",
        "Q/D \u2192 R0":            "tab:olive",
    }
    bar_h = 0.6
    for yi, lbl in enumerate(labels):
        i = labels.index(lbl)
        t0    = meas["t_phi2"][i]
        segs = []  # (name, start_ps, width_ps)
        def seg(name, a, b):
            if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                return
            segs.append((name, (a - t0) * 1e12, (b - a) * 1e12))

        seg("Clk \u2192 Phi2",           meas["t_clk"][i],   meas["t_phi2"][i])
        # WL path (parallel to SAE path; draw on same row lightly offset is noisy — just show total sequentially).
        # Here we draw the WL branch above the SAE branch.
        # Top branch: Phi2 -> WL1 -> BL dev (read development path)
        wl_start = meas["t_phi2"][i]
        seg("Phi2 \u2192 WL1",           wl_start,              meas["t_wl1"][i])
        seg("WL1 \u2192 BL 50 mV",       meas["t_wl1"][i],      meas["t_bl_dev"][i])
        # Bottom branch starts at t0 again
        seg("Phi2 \u2192 SAE",           meas["t_phi2"][i],     meas["t_sae"][i])
        seg("SAE \u2192 Q/D at V_DD/2",  meas["t_sae"][i],      meas["t_qd"][i])
        seg("Q/D \u2192 R0",             meas["t_qd"][i],       meas["t_r0"][i])

        for name, x, w in segs:
            ax.barh(yi, w, left=x, height=bar_h, color=stage_colors.get(name, "gray"),
                    edgecolor="k", lw=0.5, label=name if yi == 0 else None)
            if w > 8:
                ax.text(x + w/2, yi, f"{w:.0f}", ha="center", va="center", fontsize=7, color="white")
        y_labels.append(lbl)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("time since Phi2 rise (ps)")
    ax.set_title("Per-stage critical-path delays across the three periods")
    ax.grid(True, axis="x", alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    uniq = dict(zip(lbls, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8, ncol=3, frameon=False)
    plt.tight_layout()
    out = FIGDIR / "breakdown_compare_bars.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  wrote {out}")


def main():
    mfile = HERE / "measurements.npz"
    if not mfile.exists():
        print("measurements.npz not found — run analyze_breakdown.py first")
        return
    meas = np.load(mfile, allow_pickle=True)
    labels = list(meas["labels"])
    for label, period_ps, results_file, out_name in RUNS:
        if label not in labels:
            continue
        idx = labels.index(label)
        plot_one(label, period_ps, results_file, out_name, idx, meas)

    stacked_bar_chart(meas, labels)


if __name__ == "__main__":
    main()
