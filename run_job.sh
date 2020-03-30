#!/bin/bash
#SBATCH -J neale_qlearning # name of job
#SBATCH -A eecs # name of my sponsored account, e.g. class or
#SBATCH -p dgx2 # name of partition or queue

#SBATCH -o slurm_outfiles/10particles_sampler3.out # name of output file for this submission script
#SBATCH -e slurm_outfiles/10particles_sampler3.err # name of error file for this submission script
#SBATCH --mail-type=BEGIN,END,FAIL # send email when job begins,
#SBATCH --mail-user=ratzlafn@oregonstate.edu # send email to this address
#SBATCH --gres=gpu:1
python3 hyperexamples.py
