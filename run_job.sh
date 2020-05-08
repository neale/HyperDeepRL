#!/bin/bash
#SBATCH -J neale_qlearning # name of job
#SBATCH -A eecs # name of my sponsored account, e.g. class or
#SBATCH -p dgx2 # name of partition or queue

#SBATCH -e slurm_outfiles/action_mnormal_ortho_1-0.01a_noclip_random_m1m2.err
#SBATCH -o slurm_outfiles/action_mnormal_ortho_1-0.01a_noclip_random_m1m2.out
#SBATCH --mail-type=BEGIN,END,FAIL # send email when job begins,
#SBATCH --mail-user=ratzlafn@oregonstate.edu # send email to this address
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00
python3 hyperexamples.py
