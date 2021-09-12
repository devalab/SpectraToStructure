#!/bin/bash

#SBATCH -A research
#SBATCH --mem-per-cpu=3000
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -J Predictor_train

eval "$(conda shell.bash hook)"
conda activate chem

python respredict_pipeline.py

