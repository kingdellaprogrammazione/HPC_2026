#!/usr/bin/env python3
"""
Analysis of cuda_timing_results.csv: compares the pageable and pinned builds.

Produces
  - a summary table printed on screen and written as a LaTeX table
  - three plots: transfer time vs grid size, effective bandwidth, speedup

Only needs pandas and matplotlib. If they are not available on the cluster,
copy the CSV to your laptop and run it there.

    python3 analyze_cuda.py
"""

import sys

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")          # no display on a compute node
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"missing module: {e}. Try: pip install --user pandas matplotlib")

CSV = "cuda_timing_results.csv"

df = pd.read_csv(CSV)

# Per-frame averages: the totals depend on how many frames were run.
for col in ("h2d", "kernel", "d2h"):
    df[f"{col}_per_frame"] = df[f"{col}_ms"] / df["frames"]
df["transfer_per_frame"] = df["h2d_per_frame"] + df["d2h_per_frame"]

# Bytes moved per frame: M*M ints in, 3*M*M bytes out.
df["h2d_MB"] = df["M"] ** 2 * 4 / 1e6
df["d2h_MB"] = df["M"] ** 2 * 3 / 1e6
# Effective bandwidth in GB/s (MB / ms is already GB/s).
df["h2d_GBs"] = df["h2d_MB"] / df["h2d_per_frame"]
df["d2h_GBs"] = df["d2h_MB"] / df["d2h_per_frame"]

# Repeated runs: the median is robust against the occasional outlier caused by
# another job sharing the node.
med = df.groupby(["mode", "M"]).median(numeric_only=True).reset_index()

pageable = med[med["mode"] == "pageable"].set_index("M")
pinned = med[med["mode"] == "pinned"].set_index("M")
sizes = sorted(set(pageable.index) & set(pinned.index))

if not sizes:
    sys.exit("the CSV does not contain both modes: run ./benchmark_cuda.sh first")

print(f"\n{'M':>6} {'transfer pageable':>18} {'transfer pinned':>16} "
      f"{'speedup':>9} {'BW pageable':>13} {'BW pinned':>11}")
print(f"{'':>6} {'[ms/frame]':>18} {'[ms/frame]':>16} {'':>9} "
      f"{'[GB/s]':>13} {'[GB/s]':>11}")
print("-" * 80)

rows = []
for M in sizes:
    tp = pageable.loc[M, "transfer_per_frame"]
    ti = pinned.loc[M, "transfer_per_frame"]
    bwp = pageable.loc[M, "h2d_GBs"]
    bwi = pinned.loc[M, "h2d_GBs"]
    speedup = tp / ti
    rows.append((M, tp, ti, speedup, bwp, bwi))
    print(f"{M:>6} {tp:>18.4f} {ti:>16.4f} {speedup:>8.2f}x "
          f"{bwp:>13.2f} {bwi:>11.2f}")

# The kernel is untouched by the allocation strategy: showing that it does not
# move is a good sanity check that the experiment is controlled.
print("\nkernel time (should be the same in both builds):")
for M in sizes:
    kp = pageable.loc[M, "kernel_per_frame"]
    ki = pinned.loc[M, "kernel_per_frame"]
    print(f"  M={M:>5}  pageable {kp:.4f} ms   pinned {ki:.4f} ms   "
          f"difference {100*abs(kp-ki)/max(kp,1e-9):.1f}%")

# ---- LaTeX table, ready to be pasted into the report --------------------
with open("cuda_comparison_table.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrr}\n\\hline\n")
    f.write("$M$ & pageable [ms] & pinned [ms] & speedup & BW pinned [GB/s] \\\\\n\\hline\n")
    for M, tp, ti, sp, _, bwi in rows:
        f.write(f"{M} & {tp:.4f} & {ti:.4f} & {sp:.2f}$\\times$ & {bwi:.2f} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print("\nLaTeX table written to cuda_comparison_table.tex")

# ---- plots --------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

ax[0].plot(sizes, [pageable.loc[M, "transfer_per_frame"] for M in sizes],
           "o-", label="pageable")
ax[0].plot(sizes, [pinned.loc[M, "transfer_per_frame"] for M in sizes],
           "s-", label="pinned")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel("grid size $M$"); ax[0].set_ylabel("transfer time per frame [ms]")
ax[0].set_title("H2D + D2H transfer time"); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(sizes, [pageable.loc[M, "h2d_GBs"] for M in sizes], "o-", label="pageable H2D")
ax[1].plot(sizes, [pinned.loc[M, "h2d_GBs"] for M in sizes], "s-", label="pinned H2D")
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("grid size $M$"); ax[1].set_ylabel("effective bandwidth [GB/s]")
ax[1].set_title("Host to device bandwidth"); ax[1].legend(); ax[1].grid(alpha=.3)

ax[2].plot(sizes, [r[3] for r in rows], "d-", color="tab:green")
ax[2].axhline(1.0, ls="--", c="grey", lw=1)
ax[2].set_xscale("log", base=2)
ax[2].set_xlabel("grid size $M$"); ax[2].set_ylabel("speedup, pageable / pinned")
ax[2].set_title("Gain from pinned memory"); ax[2].grid(alpha=.3)

plt.tight_layout()
plt.savefig("cuda_pinned_comparison.png", dpi=150)
print("plots written to cuda_pinned_comparison.png")
