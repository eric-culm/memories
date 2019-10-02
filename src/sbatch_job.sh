#! /bin/bash

#SBATCH --job-name="training"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=PROVAjob%J.out
#SBATCH --error=PROVAjob%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:0

module load cuda/9.2

python3 dummy_job.py
