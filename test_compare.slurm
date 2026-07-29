#!/usr/bin/env bash
#SBATCH --job-name=wave_cmp
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --partition=edu_a40
#SBATCH --output=compare_%j.out
#SBATCH --error=compare_%j.err

module purge
module load gcc/12.4.0
module load nvhpc/25.1
cd $HOME/HPC_2026
make -f Makefile.cuda clean
make -f Makefile.cuda compare M=256 N=200
echo "=== FATTO ==="
