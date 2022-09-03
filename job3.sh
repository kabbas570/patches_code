#!/bin/bash -l
#SBATCH --account JANIK-SL3-GPU
#SBATCH --time=0-00:20:00
#SBATCH -p pascal
#SBATCH --nodes 1
#SBATCH --gres=gpu:1

module load miniconda3
source /home/akr54/torch_env/bin/activate
python only_img1.py
