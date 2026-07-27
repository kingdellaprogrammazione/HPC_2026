#!/bin/bash
THREADS=(1 2 4 8 16 32 48)

for t in "${THREADS[@]}"; do
    sbatch --cpus-per-task=$t --export=ALL,THREADS=$t run_openmp.sh
done