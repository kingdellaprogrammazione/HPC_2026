#!/usr/bin/env python3
"""
OpenMP scaling test results plots:
- Execution time vs. number of threads
- Speedup vs. number of threads (compared to ideal speedup)
- Parallel efficiency vs. number of threads
 
Usage:
    python3 plot_scaling.py
    (reads timing_results.csv from the same folder)
"""

import csv
import matplotlib.pyplot as plt

# ─── Read CSV ───
threads_list = []
times_list = []

with open("timing_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        threads_list.append(int(row["threads"]))
        times_list.append(float(row["time_seconds"]))

# ─── Sort data by number of threads ───
data = sorted(zip(threads_list, times_list))
threads_list, times_list = zip(*data)

# ─── Calculate speedup and efficiency ───
# Speedup = T(1 thread) / T(N thread)
t_serial = times_list[0]  # min number of threads (assumed to be the serial execution time)
speedup = [t_serial / t for t in times_list]
ideal_speedup = list(threads_list)  # Ideal speedup is linear with the number of threads
efficiency = [s / t * 100 for s, t in zip(speedup, threads_list)]  # in %

# ─── Plot: 3 graph near───
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Execution time vs. number of threads
axes[0].plot(threads_list, times_list, marker='o', color='#0F6E56', linewidth=2)
axes[0].set_xlabel("Number of threads")
axes[0].set_ylabel("Execution time (s)")
axes[0].set_title("Execution time vs Number of threads")
axes[0].grid(True, alpha=0.3)

# 2. Speedup (real vs ideal)
axes[1].plot(threads_list, speedup, marker='o', label='Real speedup', color='#185FA5', linewidth=2)
axes[1].plot(threads_list, ideal_speedup, linestyle='--', label='Ideal speedup', color='gray')
axes[1].set_xlabel("Number of threads")
axes[1].set_ylabel("Speedup")
axes[1].set_title("Speedup vs Number of threads")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Parallel efficiency
axes[2].plot(threads_list, efficiency, marker='o', color='#854F0B', linewidth=2)
axes[2].axhline(100, linestyle='--', color='gray', alpha=0.7)
axes[2].set_xlabel("Number of threads")
axes[2].set_ylabel("Parallel Efficiency (%)")
axes[2].set_title("Parallel Efficiency vs Number of threads")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scaling_results.png", dpi=150)
print("Grafico salvato in scaling_results.png")

# ─── Stampa tabella riassuntiva ───
print("\n{:<10} {:<15} {:<12} {:<12}".format("Threads", "Time (s)", "Speedup", "Efficiency"))
print("-" * 50)
for t, time, s, e in zip(threads_list, times_list, speedup, efficiency):
    print("{:<10} {:<15.4f} {:<12.2f} {:<12.1f}%".format(t, time, s, e))
