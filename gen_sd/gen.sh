#!/bin/bash

#SBATCH -p gpu5
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -o gen1000-%j.out
#SBATCH --mem 20G

date
python -u gen1000.py
date

