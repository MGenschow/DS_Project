#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000mb
#SBATCH --time=08:00:00

#SBATCH --mail-user=aaron.lay@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source /pfs/work7/workspace/scratch/tu_zxmav84-ds_project/conda/bin/activate
conda activate DS_Project

python slurm_download.py
