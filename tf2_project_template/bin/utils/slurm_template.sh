#!/bin/sh
#SBATCH --output={save_path}/out
#SBATCH --error={save_path}/err
#SBATCH --gres=gpu
#SBATCH -J {batch_name}
#SBATCH --time='8:00:00'
#SBATCH -p'gpu4_medium,gpu4_long,gpu4_short,gpu8_short,gpu8_medium,gpu8_long'
#SBATCH --mem=10000
cd /gpfs/home/jastrs01/cooperative_optimization
source /gpfs/home/jastrs01/cooperative_optimization/e_bigpurple.sh
export CUDA_VISIBLE_DEVICES=0
{job}