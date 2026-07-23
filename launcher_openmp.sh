THREADS=(2)

for t in "${THREADS[@]}"; do
    sbatch --cpus-per-task=$t --export=ALL,THREADS=$t run_openmp.sh
done