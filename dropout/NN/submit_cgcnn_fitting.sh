#!/bin/sh

#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --account=m1759
#SBATCH --job-name=cgcnn_batch
#SBATCH --out=cgcnn_batch.out
#SBATCH --error=cgcnn_batch.error

export PYTHONPATH=$PYTHONPATH:$HOME/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 1 python d_cgcnn.py 0.05 1 &
srun -N 1 -G 1 python d_cgcnn.py 0.10 2 &
srun -N 1 -G 1 python d_cgcnn.py 0.15 3 &
srun -N 1 -G 1 python d_cgcnn.py 0.20 4 &

wait