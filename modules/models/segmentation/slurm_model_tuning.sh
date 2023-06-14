#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

#SBATCH --mail-user=aaron.lay@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source activate DS_Project
python tune_potsdam_DeepLab_slurm.py