#!/bin/bash

#SBATCH --job-name=HW5
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem=64Gb
#SBATCH --time=99:59:59

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Move to the correct directory

cd /home/xvw10/HW5
echo "Current working directory is `pwd`"

# Train & Test the model
python HW5.py

# end of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
