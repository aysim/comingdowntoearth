#!/bin/bash
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:2,VRAM:48GB
#SBATCH --mem=300GB
#SBATCH --time=240:00:00
#SBATCH --output=/storage/slurm/toker/baseline_checkpoints/doublecheckcdetestcvact-unet512batch4-%j.out
#SBATCH --error=/storage/slurm/toker/baseline_checkpoints/doublecheckcdetestcvact-unet512batch4-%j.out
export https_proxy=http://proxy.in.tum.de:8080/
source ~/py36
python3 -m cvact_try -o '/storage/slurm/toker/baseline_checkpoints/'