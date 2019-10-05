#!/bin/sh

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=m1759
#SBATCH --job-name=cgcnn_ensemble
#SBATCH --out=ensembling.out
#SBATCH --error=ensembling.error
#SBATCH --mail-user=ktran@andrew.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00


source ~/miniconda3/bin/activate
conda activate gaspy
python fit_ensembles.py
