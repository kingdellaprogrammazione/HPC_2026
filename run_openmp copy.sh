#!/bin/bash
#SBATCH --job-name=vtune
#SBATCH --partition=cpu_sapphire
#SBATCH --nodes=1
#SBATCH --time=00:30:00

module purge
module load oneapi/vtune/2025.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

OUTDIR=damped_wave/openmp/results/threads_${OMP_NUM_THREADS}
RAW_DIR=${OUTDIR}/raw
VTUNE_DIR=${OUTDIR}/vtune

mkdir -p "$RAW_DIR"
mkdir -p "$VTUNE_DIR"

echo "Threads = $OMP_NUM_THREADS"

# To run the file: 

M_GRID=500
N_FRAMES=500

###################################
# 1) CLEAN RUNS (raw time)
###################################

N_RUNS=1

for RUN in $(seq 1 $N_RUNS); do
    echo "Clean run $RUN"
    { time ./wave_sim_omp.out ${M_GRID} ${N_FRAMES}; } \
    > ${RAW_DIR}/stdout_${RUN}.txt \
    2> ${RAW_DIR}/time_${RUN}.txt
done

