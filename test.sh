#!/bin/bash

#SBATCH -p gpu5
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres gpu:4
#SBATCH -o allo-%j.out
#SBATCH --mem 20G




echo 'allo'
python -u generate_cost.py --config "/config_allo" --netG './allo_1/netG_epoch__72.pth' --datacover './dataset_sd_in_gray/'
date


date
echo 'embed'
module load matlab/R2018b
srun matlab -nodisplay -nosplash -nodesktop -r "clear;\
Payload = 0.4;\
cover_dir = './dataset_sd_in_gray';\
stego_dir = './stego/allo';\
cost_dir = './root/config_allo';\
run('embedding.m');\
exit;"
date

