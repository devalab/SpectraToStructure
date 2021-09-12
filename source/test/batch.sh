#!/bin/bash

#SBATCH -A research
#SBATCH --mem-per-cpu=3000
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00
##SBATCH --mail-type=END,FAIL
#SBATCH --job-name="mctsTest"

source activate chem
ulimit -n 40960
ray start --head --num-cpus=40 --num-gpus=4 --object-store-memory 50000000000
python parallel_agent.py
ray stop


