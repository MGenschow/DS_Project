#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source activate DS_Project
python train_multiple_models_slurm.py
