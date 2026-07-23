#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Scaling test: esegue la simulazione con diversi numeri di
# thread OpenMP e registra i tempi in timing_results.csv
# ═══════════════════════════════════════════════════════════

set -e   # ferma lo script se un comando fallisce

M=1000
N=1000
EXECUTABLE=./wave_sim.out

# Lista di thread da testare — modifica secondo i core disponibili
# (controlla con: nproc)
THREAD_LIST="1 2 4 8"

# Rimuovi risultati precedenti per partire puliti
rm -f timing_results.csv

echo "=== Avvio scaling test (M=$M, N=$N) ==="

for T in $THREAD_LIST; do
    echo ""
    echo ">>> Eseguo con OMP_NUM_THREADS=$T"
    OMP_NUM_THREADS=$T $EXECUTABLE $M $N
done

echo ""
echo "=== Test completato. Risultati in timing_results.csv ==="
cat timing_results.csv
