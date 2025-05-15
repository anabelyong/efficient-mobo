#!/bin/sh
#SBATCH --time=2:00:00
eval "$(/home/s/shuyuan/miniconda3/condabin/conda shell.bash hook)"
conda activate seed-mobo
which python 
python ehvi_jax.py