#! /bin/bash

SBATCH --job-name="training"
#SBATCH --mail-type=ALL
SBATCH --nodes=1
SBATCH --ntasks-per-node=8
SBATCH --output=PROVAjob%J.out
SBATCH --error=PROVAjob%J.err
SBATCH --partition=normal
SBATCH --gres=gpu:1

module load cuda/10.0

python3 training.py
