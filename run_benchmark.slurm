#!/usr/bin/env bash
#SBATCH --job-name=wave_bench
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-00:20:00
#SBATCH --partition=gpu_a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s345215@studenti.polito.it
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

# Pageable vs pinned host memory, swept over the grid size.
# Only the timings matter here, so WAVE_SKIP_IO=1 (set inside the script)
# removes the file writing, which would otherwise dominate the wall time.
#
# --mem is larger than for a normal run: at M=2048 the three double buffers
# alone are about 100 MB, and pinned memory cannot be swapped out.

module purge
module load gcc/12.4.0
module load nvhpc/25.1

cd $HOME/HPC_2026

# The node is shared: asking for more threads than allocated would oversubscribe.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

chmod +x benchmark_cuda.sh
./benchmark_cuda.sh 128 256 512 1024 2048

echo "=== BENCHMARK DONE ==="
echo "now run:  python3 analyze_cuda.py"
