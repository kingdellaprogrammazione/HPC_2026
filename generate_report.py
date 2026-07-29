#!/usr/bin/env python3
"""
Genera tabella e grafici di performance a partire da:
  - timing_results.csv                 : tempo "pulito" per numero di thread (threads,M,N,time_seconds)
  - threads_N/vtune/hotspots.csv        : CPU Time per funzione (VTune, collection "hotspots")
  - threads_N/vtune/threading_summary.csv : Elapsed Time + Effective CPU Utilization (collection "threading")
  - threads_N/vtune/threading.csv       : Wait Time per thread OS (collection "threading")

Produce:
  - report_table.csv       : tabella riassuntiva
  - scaling_plot.png       : tempo, speedup, efficienza vs numero di thread
  - compute_vs_io_plot.png : barre impilate CPU Time (lavoro) vs Wait Time (attesa/I-O)

Struttura attesa su disco:
    damped_wave/openmp/results/
        threads_1/vtune/hotspots.csv, threading.csv, threading_summary.csv, ...
        threads_2/vtune/...
        ...
    damped_wave/openmp/timing_results.csv

Uso:
    pip install matplotlib          (Windows/PowerShell)
    pip install matplotlib --break-system-packages   (Linux/WSL)
    python generate_report.py       (Windows)
    python3 generate_report.py      (Linux/WSL)
"""

import os
import re
import csv
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIGURAZIONE — modifica questi path in base al tuo progetto
# ------------------------------------------------------------------
RESULTS_DIR = "damped_wave/openmp/results"
TIMING_CSV = "damped_wave/openmp/timing_results.csv"

REPORT_TABLE_CSV = "report_table.csv"
SCALING_PLOT_PNG = "scaling_plot.png"
COMPUTE_IO_PLOT_PNG = "compute_vs_io_plot.png"

# Metti a True per stampare a schermo delimitatore/colonne/righe trovate
# nei CSV di VTune (utile se un campo resta vuoto)
DEBUG = False


# ------------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------------
def _detect_delimiter(sample_line):
    """Auto-rileva se il file usa tab o virgola come separatore."""
    if sample_line.count("\t") >= sample_line.count(","):
        return "\t"
    return ","


def _find_col(fieldnames, exact, prefix=None):
    """Trova una colonna per nome esatto, poi (opzionale) per prefisso."""
    for c in fieldnames:
        if c and c.strip() == exact:
            return c
    if prefix:
        for c in fieldnames:
            if c and c.strip().startswith(prefix):
                return c
    return None


# ------------------------------------------------------------------
# 1) timing_results.csv -> tempo "pulito" per numero di thread
# ------------------------------------------------------------------
def parse_timing_results(path):
    times = {}
    if not os.path.exists(path):
        print(f"WARNING: {path} non trovato — userò l'Elapsed Time di VTune come fallback.")
        return times

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_threads = int(row["threads"])
                t = float(row["time_seconds"])
                times[n_threads] = t   # se ci sono più righe, tiene l'ultima
            except (KeyError, ValueError):
                continue
    return times


# ------------------------------------------------------------------
# 2) threading_summary.csv -> Elapsed Time + Effective CPU Utilization
#    Formato reale (esempio):
#        Hierarchy Level  Metric Name                Metric Value
#        0                Elapsed Time               442.386843
#        0                Effective CPU Utilization  0.3% (0.242 out of 96 logical CPUs)
# ------------------------------------------------------------------
def parse_threading_summary(path, debug=False):
    elapsed = None
    cpu_util = None

    with open(path, newline="") as f:
        first_line = f.readline()
        delim = _detect_delimiter(first_line)
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)
        header = next(reader, None)
        if debug:
            print(f"  [debug] {path}: delimiter={delim!r} header={header}")

        for row in reader:
            if len(row) < 3:
                continue
            metric_name = row[1].strip()
            metric_value = row[2].strip()

            if metric_name == "Elapsed Time":
                try:
                    elapsed = float(metric_value)
                except ValueError:
                    pass

            elif "CPU Utilization" in metric_name:
                # es: "0.3% (0.242 out of 96 logical CPUs)" -> prende il numero prima di "%"
                m = re.match(r"\s*([\d.]+)\s*%", metric_value)
                if m:
                    cpu_util = float(m.group(1))

    if debug:
        print(f"  [debug] {path}: elapsed={elapsed} cpu_util={cpu_util}")
    return elapsed, cpu_util


# ------------------------------------------------------------------
# 3) hotspots.csv -> CPU Time totale (somma su tutte le funzioni)
# ------------------------------------------------------------------
def parse_hotspots_cpu_time(path, debug=False):
    total_cpu_time = 0.0

    with open(path, newline="") as f:
        first_line = f.readline()
        delim = _detect_delimiter(first_line)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            return total_cpu_time

        cpu_col = _find_col(reader.fieldnames, "CPU Time", prefix="CPU Time")
        if debug:
            print(f"  [debug] {path}: delimiter={delim!r} columns={reader.fieldnames} cpu_col={cpu_col!r}")

        for row in reader:
            try:
                total_cpu_time += float(row.get(cpu_col) or 0.0)
            except ValueError:
                continue

    return total_cpu_time


# ------------------------------------------------------------------
# 4) threading.csv -> Wait Time totale (somma su tutti i thread OS)
#    Colonne reali: Thread, CPU Time, ..., Wait Time, Wait Time:Idle, ...
# ------------------------------------------------------------------
def parse_threading_wait_time(path, debug=False):
    total_wait_time = 0.0

    with open(path, newline="") as f:
        first_line = f.readline()
        delim = _detect_delimiter(first_line)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            return total_wait_time

        wait_col = _find_col(reader.fieldnames, "Wait Time", prefix="Wait Time")
        if debug:
            print(f"  [debug] {path}: delimiter={delim!r} columns={reader.fieldnames} wait_col={wait_col!r}")

        for row in reader:
            try:
                total_wait_time += float(row.get(wait_col) or 0.0)
            except ValueError:
                continue

    return total_wait_time


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    thread_dirs = sorted(
        glob.glob(f"{RESULTS_DIR}/threads_*"),
        key=lambda p: int(re.search(r"threads_(\d+)", p).group(1))
    )

    if not thread_dirs:
        print(f"No results found under {RESULTS_DIR}. Check the path.")
        return

    clean_times = parse_timing_results(TIMING_CSV)

    rows = []
    for d in thread_dirs:
        n_threads = int(re.search(r"threads_(\d+)", d).group(1))
        if DEBUG:
            print(f"\nthreads_{n_threads}:")

        threading_summary_path = os.path.join(d, "vtune", "threading_summary.csv")
        hotspots_path = os.path.join(d, "vtune", "hotspots.csv")
        threading_path = os.path.join(d, "vtune", "threading.csv")

        vtune_elapsed, cpu_util = (None, None)
        cpu_time, wait_time = (0.0, 0.0)

        if os.path.exists(threading_summary_path):
            vtune_elapsed, cpu_util = parse_threading_summary(threading_summary_path, debug=DEBUG)
        else:
            print(f"WARNING: missing {threading_summary_path}")

        if os.path.exists(hotspots_path):
            cpu_time = parse_hotspots_cpu_time(hotspots_path, debug=DEBUG)
        else:
            print(f"WARNING: missing {hotspots_path}")

        if os.path.exists(threading_path):
            wait_time = parse_threading_wait_time(threading_path, debug=DEBUG)
        else:
            print(f"WARNING: missing {threading_path}")

        # Tempo da mostrare nel grafico: preferisci il tempo "pulito" (senza
        # overhead di profiling VTune), altrimenti fallback sull'Elapsed Time VTune
        clean_time = clean_times.get(n_threads)
        display_time = clean_time if clean_time is not None else vtune_elapsed

        rows.append({
            "threads": n_threads,
            "display_time": display_time,
            "cpu_utilization_pct": cpu_util,
            "cpu_time": cpu_time,
            "wait_time": wait_time,
        })

    rows = [r for r in rows if r["display_time"] is not None]
    if not rows:
        print("No valid data found. Controlla timing_results.csv e i CSV VTune.")
        return

    rows.sort(key=lambda r: r["threads"])
    baseline = rows[0]["display_time"]
    for r in rows:
        r["speedup"] = baseline / r["display_time"]
        r["efficiency_pct"] = 100 * r["speedup"] / r["threads"]

    # ------------------------------------------------------------------
    # Scrittura report_table.csv
    # ------------------------------------------------------------------
    with open(REPORT_TABLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threads", "elapsed_time_s", "speedup", "efficiency_pct",
            "cpu_utilization_pct", "cpu_time_s", "wait_time_s"
        ])
        for r in rows:
            writer.writerow([
                r["threads"],
                f"{r['display_time']:.4f}",
                f"{r['speedup']:.3f}",
                f"{r['efficiency_pct']:.1f}",
                f"{r['cpu_utilization_pct']:.2f}" if r["cpu_utilization_pct"] is not None else "",
                f"{r['cpu_time']:.4f}",
                f"{r['wait_time']:.4f}",
            ])
    print(f"Scritto {REPORT_TABLE_CSV}")

    # Stampa tabella a schermo
    header = f"{'Threads':>7} | {'Time (s)':>10} | {'Speedup':>8} | {'Eff (%)':>8} | {'CPU (%)':>8} | {'CPU Time (s)':>13} | {'Wait Time (s)':>13}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        cpu_str = f"{r['cpu_utilization_pct']:.2f}" if r["cpu_utilization_pct"] is not None else "n/a"
        print(f"{r['threads']:>7} | {r['display_time']:>10.4f} | {r['speedup']:>8.3f} | "
              f"{r['efficiency_pct']:>8.1f} | {cpu_str:>8} | {r['cpu_time']:>13.4f} | {r['wait_time']:>13.4f}")

    # ------------------------------------------------------------------
    # PLOT 1 — scaling_plot.png: tempo, speedup, efficienza vs numero di thread
    # ------------------------------------------------------------------
    threads = [r["threads"] for r in rows]
    times = [r["display_time"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    efficiencies = [r["efficiency_pct"] for r in rows]
    ideal_speedup = [t / threads[0] for t in threads]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))   # era (16, 5), ora più grande

    #  Aggiungi anche font più grandi per gli assi/titoli:
    plt.rcParams.update({'font.size': 20})

    axes[0].plot(threads, times, "o-", color="tab:blue")
    axes[0].set_xlabel("Number of threads")
    axes[0].set_ylabel("Execution Time (s)")
    axes[0].set_title("Execution Time vs Number of threads")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(threads, speedups, "o-", color="tab:orange", label="Measured Speedup")
    axes[1].plot(threads, ideal_speedup, "--", color="gray", label="Ideal Speedup")
    axes[1].set_xlabel("Number of threads")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup vs Number of threads")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(threads, efficiencies, "o-", color="tab:green")
    axes[2].axhline(100, linestyle="--", color="gray")
    axes[2].set_xlabel("Number of threads")
    axes[2].set_ylabel("Efficiency (%)")
    axes[2].set_title("Efficiency vs Number of threads")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(SCALING_PLOT_PNG, dpi=200)
    plt.close(fig)
    print(f"Scritto {SCALING_PLOT_PNG}")

    # ------------------------------------------------------------------
    # PLOT 2 — compute_vs_io_plot.png: barre impilate CPU Time vs Wait Time
    # ------------------------------------------------------------------
    cpu_times = [r["cpu_time"] for r in rows]
    wait_times = [r["wait_time"] for r in rows]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_labels = [str(t) for t in threads]
    ax2.bar(x_labels, cpu_times, label="CPU Time (lavoro utile)", color="tab:blue")
    ax2.bar(x_labels, wait_times, bottom=cpu_times, label="Wait Time (attesa / I-O)", color="tab:red")
    ax2.set_xlabel("Numero di thread")
    ax2.set_ylabel("Tempo (s)")
    ax2.set_title("CPU Time vs Wait Time (VTune hotspots + threading)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(COMPUTE_IO_PLOT_PNG, dpi=150)
    plt.close(fig2)
    print(f"Scritto {COMPUTE_IO_PLOT_PNG}")


if __name__ == "__main__":
    main()