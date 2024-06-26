#!/bin/bash

#SBATCH --job-name=P12345.678
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=28

source /etc/profile.d/modules.sh

export JULIA_NUM_THREADS=28
julia --threads 28 sim_01_movingJelly_differentKinematics_static.jl
