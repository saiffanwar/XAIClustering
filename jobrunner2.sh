#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=33
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00
#SBATCH --output=slurm2.out
#SBATCH --error=slurm2.err


module purge

source ../venv/bin/activate
python rul_phm08.py 2 ensembles
