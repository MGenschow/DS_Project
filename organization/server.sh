#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000mb
#SBATCH --time=00:20:00

#SBATCH --mail-user=malte.genschow@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

module load devel/code-server
PASSWORD=test code-server --bind-addr 0.0.0.0:7896 --auth password

