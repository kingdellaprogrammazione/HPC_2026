THREADS=(1 2 4 8 16 32 64)

for t in "${THREADS[@]}"; do
    sbatch --cpus-per-task=$t --export=ALL,THREADS=$t run_openmp.sh
done