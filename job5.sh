#!/bin/bash -l
#SBATCH --account JANIK-SL3-CPU
#SBATCH --time=0-00:20:00
#SBATCH -p skylake
#SBATCH --nodes 2
#SBATCH --ntasks=64

module load miniconda3
source /home/akr54/torch_env/bin/activate
python eval_0.9.py
