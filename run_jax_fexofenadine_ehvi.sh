#!/bin/bash
#SBATCH --job-name=ehvi_jax
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=ehvi.%j.out
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2) Activate your environment
conda activate seed-mobo
export PYTHONNOUSERSITE=True
echo "Using Python at: $(which python)"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
python /home/s/shuyuan/efficient-mobo/ehvi_jax_fexofenadine.py
