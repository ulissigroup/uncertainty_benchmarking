#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --constraint=knl
#SBATCH --exclusive
#SBATCH --qos=premium
#SBATCH --job-name=lips
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00


python calculate_lips.py
