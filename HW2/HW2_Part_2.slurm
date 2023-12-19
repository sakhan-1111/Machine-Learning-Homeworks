#!/bin/bash
#SBATCH -p shared
#SBATCH -n 1
#SBATCH -t 06:15:00
#SBATCH --mem-per-cpu=8192

# Script Start
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# Run python file
python HW2_Part_2.py

# Exit
echo "Program finished with exit code $? at: `date`"
