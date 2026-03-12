#!/bin/bash
#SBATCH -p gpu8_medium
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --mem=150G
#SBATCH --output=logs/tune.log
#SBATCH --job-name=tune
#SBATCH --signal=B:TERM@60

module load python/gpu/3.10.6-cuda12.9
source /gpfs/home/jv2807/dms_contrastive/venv/bin/activate

python -u tune_asha.py