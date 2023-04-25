#!/bin/bash
### This is a sample of submission script to run large number of simulations in parallel on using a slurm array job
#SBATCH --clusters=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=2G
#SBATCH --time=11:59:00
#SBATCH --job-name=Train
###SBATCH --gres=gpu
###--constraint='nvlink:2.0'
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END
#SBATCH --array=0-13
#SBATCH --mail-user=haotian.chen@chem.ox.ac.uk

# Useful job diagnostics
echo "CUDA Devices(s) allocated: $CUDA_VISIBLE_DEVICES"



source activate $DATA/tensor-env   ### Activate your own python job environment

cd $DATA/'RDE CV Levich W Coordinates - Fixed Sigma'  ### Change directory to where your Pyhon Script is located

srun python "RDE CV W.py" $SLURM_ARRAY_TASK_ID  # Run with TASK SLURM TASK ID