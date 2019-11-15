#!/bin/sh

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=m1759
#SBATCH --job-name=cgcnn_blocked
#SBATCH --out=blocked.out
#SBATCH --error=blocked.error
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00


source ~/miniconda3/bin/activate
conda activate gaspy
python fit_cgcnn_blocked.py
