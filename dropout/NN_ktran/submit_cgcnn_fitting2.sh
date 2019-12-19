#!/bin/sh

#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:3
#SBATCH --time=01:30:00
#SBATCH --account=m1759
#SBATCH --job-name=cgcnn_batch2
#SBATCH --out=cgcnn_batch2.out
#SBATCH --error=cgcnn_batch2.error

export PYTHONPATH=$PYTHONPATH:$HOME/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 1 python d_cgcnn.py 0.25 1 &
srun -N 1 -G 1 python d_cgcnn.py 0.30 2 &
srun -N 1 -G 1 python d_cgcnn.py 0.00 3 &

wait