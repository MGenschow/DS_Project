#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80000mb
#SBATCH --time=72:00:00

#SBATCH --mail-user=stefan.glaisner@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

source /pfs/work7/workspace/scratch/tu_zxmav84-ds_project/conda/bin/activate
conda activate DS_Project

python disaggregateDO.py
