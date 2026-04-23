"""Measure per-stage timing breakdown from the three probed mainTest1 runs.

For each run, finds the worst read cycle's Phi2 rising edge (t0), then measures
threshold crossings for every stage along the critical path and writes:

    - breakdown_table.md   (human readable, one column per period)
    - breakdown_table.csv  (same data, raw)

Threshold convention:
    V_DD/2 = 0.4 V     for logic edges
    50 mV separation   for BL/BLb development and SA internal differential

Run from project 2/SpiceFiles/breakdown/ after the three results .txt files exist.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
VDD = 0.8
VTH = VDD / 2                # 0.4 V
BL_DEV = 0.050               # 50 mV

# The order of columns in signal_columns.txt matches the data.
COL_NAMES = (HERE / "signal_columns.txt").read_text().split()

RUNS = [
    # label, period_ps (for display), results file
    ("2000 ps", 2000, "probed_2000ps_results.txt"),
    ("320 ps",   320, "probed_320ps_results.txt"),
    ("298 ps",   298, "probed_298ps_results.txt"),
]


def load(path: Path):
    """Load ngspice wrdata output (whitespace columns, with header row)."""
    with path.open() as f:
        header = f.readline().split()
    data = np.loadtxt(path, skiprows=1)
    # Validate: time column + N signals => N+1 columns.
    return header, data


def cross_time(t, y, thresh, rising=True, t_min=None, t_max=None):
    """First time t where y crosses `thresh`. Linear interp. Returns None if no crossing."""
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None
    yy = y[idx]
    tt = t[idx]
    sign = (yy >= thresh).astype(int) if rising else (yy <= thresh).astype(int)
    d = np.diff(sign)
    targets = np.where(d == 1)[0]  # rising transition of boolean
    if len(targets) == 0:
        return None
    k = targets[0]
    y0, y1 = yy[k], yy[k+1]
    t0, t1 = tt[k], tt[k+1]
    if y1 == y0:
        return float(t1)
    return float(t0 + (thresh - y0) * (t1 - t0) / (y1 - y0))


def diff_cross(t, ya, yb, sep, t_min=None, t_max=None):
    """First time |ya - yb| reaches `sep`, in the window.

    If the diff is already above `sep` at window start, returns that first
    sample time (the difference 'developed' before the window began).
    """
    diff = np.abs(ya - yb)
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    if diff[idx[0]] >= sep:
        return float(t[idx[0]])
    return cross_time(t, diff, sep, rising=True, t_min=t_min, t_max=t_max)


def any_cross(t, y, thresh, t_min=None, t_max=None):
    """First time y crosses `thresh` in either direction in the window."""
    r = cross_time(t, y, thresh, rising=True,  t_min=t_min, t_max=t_max)
    f = cross_time(t, y, thresh, rising=False, t_min=t_min, t_max=t_max)
    if r is None: return f
    if f is None: return r
    return min(r, f)


def find_read_phi2_edges(t, phi2, wr_in, tper_ns):
    """Return a list of (cycle_idx, phi2_rise_time) for the two read cycles (cyc2 and cyc4).

    In this design Phi2 rises in the middle of each cycle (NOR-based 2-phase clock).
    Cycle index = floor(tr / tper) + 1. Cyc 2 and cyc 4 are the reads (Wr held low).
    """
    edges = []
    rise = np.where((phi2[:-1] < VTH) & (phi2[1:] >= VTH))[0]
    for i in rise:
        tr = t[i] + (VTH - phi2[i]) * (t[i+1] - t[i]) / (phi2[i+1] - phi2[i])
        edges.append(tr)
    reads = []
    tper_s = tper_ns * 1e-9
    for tr in edges:
        cyc = int(tr // tper_s) + 1  # floor — Phi2 rises inside cycle cyc
        # Sample wr_in well into the cycle (tr is near cyc-midpoint already).
        idx = np.searchsorted(t, tr + 5e-12)
        if idx >= len(t):
            continue
        if wr_in[idx] < VTH:
            reads.append((cyc, tr))
    return reads


def measure_one_run(path: Path, period_ps: int):
    header, data = load(path)
    t = data[:, 0]
    sig = {name: data[:, i+1] for i, name in enumerate(COL_NAMES)}

    reads = find_read_phi2_edges(t, sig["phi2"], sig["wr_in"], period_ps / 1000.0)
    if not reads:
        print(f"  [!] No read cycles found in {path.name}")
        return None

    # Pick the worst read = the one whose R0 takes longest to cross V_DD/2.
    # Window ends at start of the NEXT cycle so we don't catch transitions that
    # belong to the following cycle (e.g. Q/D gets driven by the next write).
    tper_s = period_ps * 1e-12
    worst = None
    worst_delay = -1.0
    for cyc, t0 in reads:
        next_cycle_start = cyc * tper_s  # current cycle ends at cyc*tper
        win_end = next_cycle_start - 2e-12
        tR = any_cross(t, sig["r0"], VTH, t_min=t0, t_max=win_end)
        if tR is None:
            continue  # R0 didn't transition this read; skip
        delay = tR - t0
        if delay > worst_delay:
            worst_delay = delay
            worst = (cyc, t0, win_end)
    if worst is None:
        cyc, t0 = reads[0]
        win_end = cyc * tper_s - 2e-12
    else:
        cyc, t0, win_end = worst

    # Phi2 rises when Clk FALLS (this is a NOR-based 2-phase generator driven off Clk low).
    # Find the Clk falling crossing at V_DD/2 that precedes this Phi2 rise.
    # Search window: back up to half a period.
    t_clk = cross_time(t, sig["clk"], VTH, rising=False,
                       t_min=t0 - (period_ps * 1e-12) * 0.6, t_max=t0 + 5e-12)

    # ---- Stage measurements ----
    # Clock buffer: Clk rise -> net@30 fall -> Phi2 rise
    # (net@30 is the input to the W8 inverter whose output is Phi2 — so it falls when Phi2 rises.)
    t_phi2_pre = cross_time(t, sig["phi2_pre"], VTH, rising=False,
                            t_min=t0 - 60e-12, t_max=t0 + 10e-12)
    t_phi2 = t0

    # Decoder path: Phi2 rise -> NAND@34 (net@314) falls -> WL1 rises
    t_dec_nand = cross_time(t, sig["dec_nand1"], VTH, rising=False,
                            t_min=t0, t_max=win_end)
    t_wl1 = cross_time(t, sig["wl1"], VTH, rising=True,
                       t_min=t0, t_max=win_end)

    # BL development: WL1 rise -> |BL0 - BLb0| = 50 mV
    t_bl_dev = None
    if t_wl1 is not None:
        t_bl_dev = diff_cross(t, sig["bl0"], sig["blb0"], BL_DEV,
                              t_min=t_wl1, t_max=win_end)

    # SAE chain: Phi2 rise -> AND (net@87) -> buffer taps -> SAE
    # net@87 = Phi2 AND ~Wr_lat. Since we're reading, ~Wr_lat is already high, so net@87 rises with Phi2.
    t_and = cross_time(t, sig["sae_and"], VTH, rising=True,
                       t_min=t0 - 10e-12, t_max=win_end)
    # Each buffer is an inverter, so polarity flips each stage.
    # buffer netlist: in -> net@0 (fall) -> net@1 (rise) -> net@6 (fall) -> net@2 (rise) -> net@9 (fall) -> INV_sized -> out(rise)
    t_bufA = cross_time(t, sig["sae_buf0"], VTH, rising=False, t_min=t0, t_max=win_end)  # after 1 inv
    t_bufB = cross_time(t, sig["sae_buf1"], VTH, rising=True,  t_min=t0, t_max=win_end)
    t_bufC = cross_time(t, sig["sae_buf6"], VTH, rising=False, t_min=t0, t_max=win_end)
    t_bufD = cross_time(t, sig["sae_buf2"], VTH, rising=True,  t_min=t0, t_max=win_end)
    t_bufE = cross_time(t, sig["sae_buf9"], VTH, rising=False, t_min=t0, t_max=win_end)  # last INV_min
    t_sae  = cross_time(t, sig["sae"],      VTH, rising=True,  t_min=t0, t_max=win_end)

    # SA differential develops to 50 mV (measured from WL1 rise — the SA nodes
    # track BL/BLb through the pass PMOS before SAE fires, so "dev" can start
    # before SAE rise.).
    t_sa_dev = None
    if t_wl1 is not None:
        t_sa_dev = diff_cross(t, sig["sa_n3"], sig["sa_n35"], BL_DEV,
                              t_min=t_wl1, t_max=win_end)

    # Q/D0 can be rising or falling depending on data; take first V_DD/2 crossing
    # after SAE rise.
    t_qd = None
    if t_sae is not None:
        t_qd = any_cross(t, sig["qd0"], VTH, t_min=t_sae, t_max=win_end)

    # R0 latches at end of Phi2. First V_DD/2 crossing after Q/D settles.
    t_r0 = None
    if t_qd is not None:
        t_r0 = any_cross(t, sig["r0"], VTH, t_min=t_qd - 5e-12, t_max=win_end)

    return dict(
        period_ps=period_ps,
        cyc=cyc,
        t_clk=t_clk, t_phi2_pre=t_phi2_pre, t_phi2=t_phi2,
        t_dec_nand=t_dec_nand, t_wl1=t_wl1,
        t_bl_dev=t_bl_dev,
        t_and=t_and,
        t_bufA=t_bufA, t_bufB=t_bufB, t_bufC=t_bufC, t_bufD=t_bufD, t_bufE=t_bufE,
        t_sae=t_sae, t_sa_dev=t_sa_dev, t_qd=t_qd, t_r0=t_r0,
    )


def ps(val):
    """Format a seconds-valued delta as a ps string; return 'FAIL' if None."""
    if val is None:
        return "FAIL"
    return f"{val * 1e12:.1f} ps"


def delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def build_rows(m):
    """Build the ordered table rows from a measurement dict."""
    rows = [
        ("Clock buffer (final stage)", "net@30 fall \u2192 Phi2 rise",         delta(m["t_phi2_pre"], m["t_phi2"])),
        ("Clock buffer (total)",       "Clk fall \u2192 Phi2 rise",            delta(m["t_clk"],      m["t_phi2"])),
        ("Decoder NAND",               "Phi2 rise \u2192 net@314 fall",        delta(m["t_phi2"],     m["t_dec_nand"])),
        ("WL buffer (INV_sized)",      "net@314 fall \u2192 WL1 rise",         delta(m["t_dec_nand"], m["t_wl1"])),
        ("Decoder total",              "Phi2 rise \u2192 WL1 rise",            delta(m["t_phi2"],     m["t_wl1"])),
        ("BL development",             "WL1 rise \u2192 |BL\u2212BLb|=50 mV",  delta(m["t_wl1"],      m["t_bl_dev"])),
        ("SAE AND gate",               "Phi2 rise \u2192 net@87 rise",         delta(m["t_phi2"],     m["t_and"])),
        ("SAE buffer tap 1",           "net@87 \u2192 buf.net@0 fall",         delta(m["t_and"],      m["t_bufA"])),
        ("SAE buffer tap 2",           "buf.net@0 \u2192 buf.net@1 rise",      delta(m["t_bufA"],     m["t_bufB"])),
        ("SAE buffer tap 3",           "buf.net@1 \u2192 buf.net@6 fall",      delta(m["t_bufB"],     m["t_bufC"])),
        ("SAE buffer tap 4",           "buf.net@6 \u2192 buf.net@2 rise",      delta(m["t_bufC"],     m["t_bufD"])),
        ("SAE buffer tap 5",           "buf.net@2 \u2192 buf.net@9 fall",      delta(m["t_bufD"],     m["t_bufE"])),
        ("SAE sized INV",              "buf.net@9 \u2192 SAE rise",            delta(m["t_bufE"],     m["t_sae"])),
        ("SAE chain total",            "Phi2 rise \u2192 SAE rise",            delta(m["t_phi2"],     m["t_sae"])),
        ("SA differential develops",   "WL1 rise \u2192 |n3\u2212n35|=50 mV",   delta(m["t_wl1"],      m["t_sa_dev"])),
        ("SA resolve to V_DD/2",       "SAE rise \u2192 Q/D0 at V_DD/2",       delta(m["t_sae"],      m["t_qd"])),
        ("Q/D bus \u2192 R latch",     "Q/D0 at V_DD/2 \u2192 R0 at V_DD/2",   delta(m["t_qd"],       m["t_r0"])),
        ("Total read latency",         "Phi2 rise \u2192 R0 at V_DD/2",        delta(m["t_phi2"],     m["t_r0"])),
    ]
    return rows


def main():
    measurements = {}
    for label, period_ps, fname in RUNS:
        p = HERE / fname
        if not p.exists():
            print(f"[!] Missing {fname} - run run_probed.bat first")
            continue
        m = measure_one_run(p, period_ps)
        if m is None:
            continue
        measurements[label] = m
        print(f"\n=== {label} (cyc {m['cyc']} worst read) ===")
        for stage, transition, d in build_rows(m):
            print(f"  {stage:35s}  {transition:42s}  {ps(d)}")

    if not measurements:
        print("No measurements produced.")
        return

    # Build the combined table.
    stages = [(r[0], r[1]) for r in build_rows(next(iter(measurements.values())))]
    md_lines = [
        "# SRAM read critical-path timing breakdown",
        "",
        "Threshold convention: V_DD/2 = 0.4 V for logic; 50 mV separation for BL dev "
        "and SA internal differential. Worst read cycle picked per run.",
        "",
        "| Stage | Signal transition | " + " | ".join(measurements.keys()) + " |",
        "|---|---|" + "|".join("---" for _ in measurements) + "|",
    ]
    for i, (stage, transition) in enumerate(stages):
        cells = []
        for label in measurements:
            rows = build_rows(measurements[label])
            cells.append(ps(rows[i][2]))
        md_lines.append(f"| {stage} | {transition} | " + " | ".join(cells) + " |")
    md_text = "\n".join(md_lines) + "\n"

    (HERE / "breakdown_table.md").write_text(md_text, encoding="utf-8")
    print(f"\nWrote breakdown_table.md")

    with (HERE / "breakdown_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Stage", "Signal transition"] + list(measurements.keys()))
        for i, (stage, transition) in enumerate(stages):
            cells = []
            for label in measurements:
                rows = build_rows(measurements[label])
                v = rows[i][2]
                cells.append(f"{v*1e12:.2f}" if v is not None else "FAIL")
            w.writerow([stage, transition] + cells)
    print("Wrote breakdown_table.csv")


    # Also dump the raw timestamps (for the plot script).
    keys = ("t_clk","t_phi2_pre","t_phi2","t_dec_nand","t_wl1","t_bl_dev",
            "t_and","t_bufA","t_bufB","t_bufC","t_bufD","t_bufE",
            "t_sae","t_sa_dev","t_qd","t_r0")
    arrays = {}
    for k in keys:
        vals = []
        for l in measurements:
            v = measurements[l].get(k)
            vals.append(float(v) if v is not None else float("nan"))
        arrays[k] = np.array(vals)
    np.savez(HERE / "measurements.npz",
             labels=np.array(list(measurements.keys())),
             **arrays)
    print("Wrote measurements.npz")


if __name__ == "__main__":
    main()
