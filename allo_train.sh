#!/bin/bash

#SBATCH -p gpu3
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres gpu:4
#SBATCH -o allo1-%j.out
#SBATCH --mem 20G

date
echo '10 sets for fluctuation'
python -u train_gifdl.py --outf "allo"
date

