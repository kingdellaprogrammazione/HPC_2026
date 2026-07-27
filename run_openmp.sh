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
    { 
        time perf stat \
            -e cache-references,cache-misses,cycles,instructions \
            ./wave_sim_omp.out ${M_GRID} ${N_FRAMES}
    } \
    > ${RAW_DIR}/stdout_${RUN}.txt \
    2> ${RAW_DIR}/time_${RUN}.txt
done

###################################
# 2) VTUNE RUN
###################################

echo "VTune profiling"

# Threading analysis (OpenMP behavior)
"$VTUNE" \
    -collect threading \
    -result-dir ${VTUNE_DIR}/threading_collection \
    ./wave_sim_omp.out ${M_GRID} ${N_FRAMES} \
    > ${VTUNE_DIR}/threading_stdout.txt \
    2> ${VTUNE_DIR}/threading_stderr.txt


# Hotspots analysis (CPU time distribution)
"$VTUNE" \
    -collect hotspots \
    -result-dir ${VTUNE_DIR}/hotspots_collection \
    ./wave_sim_omp.out ${M_GRID} ${N_FRAMES} \
    > ${VTUNE_DIR}/hotspots_stdout.txt \
    2> ${VTUNE_DIR}/hotspots_stderr.txt


# Memory Access analysis
"$VTUNE" \
    -collect memory-access \
    -result-dir ${VTUNE_DIR}/memory_collection \
    ./wave_sim_omp.out ${M_GRID} ${N_FRAMES} \
    > ${VTUNE_DIR}/memory_stdout.txt \
    2> ${VTUNE_DIR}/memory_stderr.txt


###################################
# VTune reports
###################################

# Summary reports
"$VTUNE" \
    -report summary \
    -result-dir ${VTUNE_DIR}/threading_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/threading_summary.csv


"$VTUNE" \
    -report summary \
    -result-dir ${VTUNE_DIR}/hotspots_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/hotspots_summary.csv


"$VTUNE" \
    -report summary \
    -result-dir ${VTUNE_DIR}/memory_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/memory_summary.csv


# Detailed reports
"$VTUNE" \
    -report threading \
    -result-dir ${VTUNE_DIR}/threading_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/threading.csv


"$VTUNE" \
    -report hotspots \
    -result-dir ${VTUNE_DIR}/hotspots_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/hotspots.csv


"$VTUNE" \
    -report memory-access \
    -result-dir ${VTUNE_DIR}/memory_collection \
    -format csv \
    -report-output ${VTUNE_DIR}/memory.csv