#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Pageable vs pinned host memory: parameter sweep over the grid size.
#
#  Both binaries come from the same sources, the only difference being the
#  -DUSE_PINNED_MEMORY flag used to build the GPU module, so any difference
#  measured here is due to the allocation strategy alone.
#
#  WAVE_SKIP_IO=1 keeps the computation and every GPU transfer but skips the
#  file writing, which would otherwise dominate the wall time and turn the
#  sweep into a disk benchmark.
#
#  Usage:  ./benchmark_cuda.sh            (default sizes)
#          ./benchmark_cuda.sh 256 512    (custom sizes)
# ═══════════════════════════════════════════════════════════════════════
set -e

SIZES=${@:-"128 256 512 1024 2048"}
N=100        # frames per run: enough to average out the noise, quick to run
REPEATS=3    # each configuration is repeated, take the median in the analysis

echo "sizes   : $SIZES"
echo "frames  : $N"
echo "repeats : $REPEATS"
echo

# A previous CSV would mix old and new results: move it aside.
if [ -f cuda_timing_results.csv ]; then
    mv cuda_timing_results.csv cuda_timing_results.csv.bak
    echo "previous CSV saved as cuda_timing_results.csv.bak"
fi

make -f Makefile.cuda cuda-both

export WAVE_SKIP_IO=1
mkdir -p damped_wave/CUDA/sim

for M in $SIZES; do
    for rep in $(seq 1 $REPEATS); do
        echo "--- M=$M  repeat $rep/$REPEATS ---"
        ./wave_cuda.out        $M $N | grep -E "H2D|kernel|D2H|total GPU"
        ./wave_cuda_pinned.out $M $N | grep -E "H2D|kernel|D2H|total GPU"
    done
done

echo
echo "done. Raw data in cuda_timing_results.csv"
echo "run:  python3 analyze_cuda.py"
