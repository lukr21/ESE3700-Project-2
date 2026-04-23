"""Plot mainTest2.spi — cross-address (wr 1001@0, wr 0110@15, rd @0, rd @15)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "mainTest2_results.txt"
OUT  = HERE.parent / "figures" / "mainTest2_waveforms.png"

with DATA.open() as f:
    header = f.readline().split()
cols = [c.strip().lower() for c in header]
data = np.loadtxt(DATA, skiprows=1)

def pick(name):
    return data[:, cols.index(name.lower())]

t    = pick("time") * 1e9
clk  = pick("v(clk)")
wr   = pick("v(wr)")
phi1 = pick("v(xmain@0.net@227)")
phi2 = pick("v(xmain@0.net@117)")
wr_l = pick("v(xmain@0.net@180)")
wren = pick("v(xmain@0.net@169)")
sae  = pick("v(xmain@0.net@71)")
pchb = pick("v(xmain@0.net@159)")

W = [pick(f"v(w{i})") for i in range(4)]
R = [pick(f"v(r{i})") for i in range(4)]
Q = [pick(f"v(xmain@0.net@{n})") for n in (184, 183, 182, 181)]
BL  = [pick(f"v(xmain@0.net@{n})") for n in (149, 151, 153, 155)]
BLb = [pick(f"v(xmain@0.net@{n})") for n in (150, 152, 154, 156)]

fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
fig.suptitle("mainTest-2: wr 1001@0 / wr 0110@15 / rd @0 / rd @15   (500 MHz, WrEn fix)", fontsize=12)

ax = axes[0]
ax.plot(t, phi1, label="Phi1", lw=1.4)
ax.plot(t, phi2, label="Phi2", lw=1.4, color="tab:orange")
ax.plot(t, clk,  label="Clk (ext)", lw=0.8, color="k", alpha=0.4, ls="--")
ax.set_ylabel("V"); ax.set_ylim(-0.05, 0.9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8, frameon=False)
for x, lbl in [(1.0,"WR 1001 @0"),(3.0,"WR 0110 @15"),(5.0,"RD @0"),(7.0,"RD @15")]:
    ax.text(x, 0.92, lbl, ha="center", va="bottom", fontsize=8, color="dimgray", transform=ax.get_xaxis_transform())

ax = axes[1]
ax.plot(t, wr,   label="Wr (in)", lw=1.3)
ax.plot(t, wr_l, label="Wr_latched (raw)", lw=1.0, ls="--", alpha=0.8)
ax.plot(t, wren, label="WrEn (gated)", lw=1.2, color="tab:green")
ax.plot(t, pchb, label="PCHb", lw=0.9, color="tab:purple", alpha=0.7)
ax.plot(t, sae,  label="SAE", lw=1.6, color="tab:red")
ax.set_ylabel("V"); ax.set_ylim(-0.05, 0.9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=8, frameon=False)

colors = ["tab:blue","tab:orange","tab:green","tab:red"]

# --- Panel 3: W0..W3 stacked, with Q/D overlaid as dashed ---
ax = axes[2]
step = 1.0
for i in range(4):
    y0 = i * step
    ax.axhline(y0, color="lightgray", lw=0.5)
    ax.plot(t, W[i] + y0, color=colors[i], lw=1.3, label=f"W{i}" if i == 0 else None)
    ax.plot(t, Q[i] + y0, color=colors[i], lw=0.9, ls="--", alpha=0.8,
            label="Q/D (dashed)" if i == 0 else None)
ax.set_ylim(-0.1, 4 * step)
ax.set_yticks([i * step + 0.4 for i in range(4)])
ax.set_yticklabels([f"bit {i}" for i in range(4)])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8, frameon=False)

# --- Panel 4: R0..R3 stacked ---
ax = axes[3]
for i in range(4):
    y0 = i * step
    ax.axhline(y0, color="lightgray", lw=0.5)
    ax.plot(t, R[i] + y0, color=colors[i], lw=1.5, label=f"R{i}" if i == 0 else None)
ax.set_ylim(-0.1, 4 * step)
ax.set_yticks([i * step + 0.4 for i in range(4)])
ax.set_yticklabels([f"bit {i}" for i in range(4)])

# --- Panel 5: BL/BLb stacked per bit ---
ax = axes[4]
for i in range(4):
    y0 = i * step
    ax.axhline(y0, color="lightgray", lw=0.5)
    ax.plot(t, BL[i]  + y0, color=colors[i], lw=1.2,
            label="BL"  if i == 0 else None)
    ax.plot(t, BLb[i] + y0, color=colors[i], lw=1.0, ls="--", alpha=0.8,
            label="BLb (dashed)" if i == 0 else None)
ax.set_xlabel("Time (ns)")
ax.set_ylim(-0.1, 4 * step)
ax.set_yticks([i * step + 0.4 for i in range(4)])
ax.set_yticklabels([f"bit {i}" for i in range(4)])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8, frameon=False)

for ax in axes:
    ax.grid(True, alpha=0.3)
    for x in (2,4,6): ax.axvline(x, color="gray", lw=0.5, ls=":")

plt.tight_layout(rect=[0,0,1,0.97])
fig.subplots_adjust(hspace=0.32)
plt.savefig(OUT, dpi=130)
print(f"Saved {OUT}")

def at(sig, when_ns):
    return float(np.interp(when_ns, t, sig))

print()
print("=== Read-back checks (end of each Phi2) ===")
expected = {"cyc1 wr 1001@0":"1001","cyc2 wr 0110@15":"0110",
            "cyc3 rd @0":"1001","cyc4 rd @15":"0110"}
for cyc, when in [("cyc1 wr 1001@0",1.9),("cyc2 wr 0110@15",3.9),
                  ("cyc3 rd @0",5.9),("cyc4 rd @15",7.9)]:
    rv = [at(R[i], when) for i in range(4)]
    rbits = "".join("1" if v>0.4 else "0" for v in reversed(rv))
    rs = " ".join(f"{v:.3f}" for v in reversed(rv))
    print(f"  t={when:.2f}ns  {cyc:17s}  R3..R0=[{rs}]  -> {rbits}  expect {expected[cyc]}")
