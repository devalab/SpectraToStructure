#!/bin/bash

#SBATCH --mem-per-cpu=3000
##SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:2
#SBATCH -t 4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --job-name="mctsPretraining"

source activate chem
ulimit -n 40960
ray start --head --num-cpus=40 --num-gpus=2 --object-store-memory 50000000000
python parallel_agent.py
