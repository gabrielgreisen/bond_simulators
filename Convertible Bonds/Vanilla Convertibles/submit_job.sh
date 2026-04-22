#!/bin/bash
#SBATCH --job-name=vanilla_cb
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load anaconda3
cd ~/bonds_sim/Convertible\ Bonds/Vanilla\ Convertibles
python run_cluster.py
