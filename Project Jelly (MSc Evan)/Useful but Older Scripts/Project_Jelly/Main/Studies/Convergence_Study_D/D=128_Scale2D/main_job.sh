#!/bin/bash
#SBATCH --job-name=jelly_cpu
#SBATCH --output=jelly_cpu.out
#SBATCH --error=jelly_cpu.err
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G

module purge
module load 2025
module load julia
module load openmpi
module load slurm

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on host $(hostname)"
echo "Using $JULIA_NUM_THREADS Julia threads"
echo "Starting at $(date)"

srun --export=ALL julia --project=/home/evanderweide/jelly_project DelftBlueJellyfish.jl

echo "Finished at $(date)"