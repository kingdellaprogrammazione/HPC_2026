#!/bin/bash
#SBATCH --job-name=mpi_scaling
#SBATCH --partition=cpu_sapphire
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00

module purge
module load oneapi/vtune/2025.0
module load openmpi/4.1.8_gcc11

echo "Loaded modules are:"  
module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

VTUNE="/share/apps/intel/oneapi/vtune/2025.0/bin64/vtune"

N=3000

# Pass 1 as first argument to enable VTune
RUN_VTUNE=${1:-0}

if [ "$RUN_VTUNE" -eq 1 ]; then
    echo "VTune profiling ENABLED"
else
    echo "VTune profiling DISABLED"
fi

# Unique experiment ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")_job${SLURM_JOB_ID}

echo "Experiment timestamp: $TIMESTAMP"

GRID_SIZES=(512)

echo "Running Hybrid MPI+OpenMP with 3 Ranks"
echo "OMP Threads per node = $OMP_NUM_THREADS"


for M in "${GRID_SIZES[@]}"; do

    echo "====================================="
    echo "Testing Grid Size: ${M} x ${M}"
    echo "====================================="

    OUTDIR=damped_wave/MPI/results/grid_${M}/run_${TIMESTAMP}

    RAW_DIR=${OUTDIR}/raw
    VTUNE_DIR=${OUTDIR}/vtune

    mkdir -p "$RAW_DIR"
    mkdir -p "$VTUNE_DIR"


    ###################################
    # 1) CLEAN RUNS (raw time)
    ###################################

    N_RUNS=1

    for RUN in $(seq 1 $N_RUNS); do

        RUN_TIME=$(date +"%Y%m%d_%H%M%S")

        echo "Clean run $RUN for Grid ${M} (${RUN_TIME})"

        { time mpirun -np 3 ./wave_sim_mpi.out ${M} ${N}; } \
            > ${RAW_DIR}/stdout_${RUN}_${RUN_TIME}.txt \
            2> ${RAW_DIR}/time_${RUN}_${RUN_TIME}.txt

    done


    ###################################
    # 2) VTUNE PROFILING (optional)
    ###################################

    if [ "$RUN_VTUNE" -eq 1 ]; then

        echo "VTune profiling for Grid ${M}"


        ###################################
        # Hotspots collection
        ###################################

        mpirun -np 3 \
        "$VTUNE" \
            -collect hotspots \
            -result-dir ${VTUNE_DIR}/hotspots_collection \
            ./wave_sim_mpi.out ${M} ${N} \
            > ${VTUNE_DIR}/hotspots_stdout.txt \
            2> ${VTUNE_DIR}/hotspots_stderr.txt


        ###################################
        # Threading collection
        ###################################

        mpirun -np 3 \
        "$VTUNE" \
            -collect threading \
            -result-dir ${VTUNE_DIR}/threading_collection \
            ./wave_sim_mpi.out ${M} ${N} \
            > ${VTUNE_DIR}/threading_stdout.txt \
            2> ${VTUNE_DIR}/threading_stderr.txt


        ###################################
        # VTune reports
        ###################################

        "$VTUNE" \
            -report summary \
            -result-dir ${VTUNE_DIR}/hotspots_collection \
            -format csv \
            -report-output ${VTUNE_DIR}/summary.csv


        "$VTUNE" \
            -report hotspots \
            -result-dir ${VTUNE_DIR}/hotspots_collection \
            -format csv \
            -report-output ${VTUNE_DIR}/hotspots.csv


        "$VTUNE" \
            -report threading \
            -result-dir ${VTUNE_DIR}/threading_collection \
            -format csv \
            -report-output ${VTUNE_DIR}/threading.csv


    else

        echo "Skipping VTune for Grid ${M}"

    fi

done


echo "All MPI scaling runs completed."