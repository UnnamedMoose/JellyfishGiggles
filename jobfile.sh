#!/bin/bash

#SBATCH --job-name=P12345.678
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=28

source /etc/profile.d/modules.sh

julia --threads 28 sim_02_movingJelly.jl
