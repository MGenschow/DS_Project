#!/bin/bash

#SBATCH --partition=gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --mail-user=aaron.lay@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source /pfs/work7/workspace/scratch/tu_zxmav84-ds_project/conda/bin/activate
conda activate DS_Project

python tune_potsdam_DeepLab_slurm.py
