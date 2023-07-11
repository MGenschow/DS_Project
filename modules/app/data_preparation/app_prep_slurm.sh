#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000mb
#SBATCH --time=04:00:00

#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source activate DS_Project
python slurm_ortho_processing.py