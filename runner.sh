#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ktapp@sju.edu


python run_scot_elec.py