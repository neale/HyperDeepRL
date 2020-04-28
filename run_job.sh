#!/bin/bash
#SBATCH -J neale_qlearning # name of job
#SBATCH -A eecs # name of my sponsored account, e.g. class or
#SBATCH -p dgx2 # name of partition or queue

#SBATCH -o slurm_outfiles/svgd_converge_anneal0.out # name of output file for this submission script
#SBATCH -e slurm_outfiles/svgd_converge_anneal0.err # name of error file for this submission script
#SBATCH --mail-type=BEGIN,END,FAIL # send email when job begins,
#SBATCH --mail-user=ratzlafn@oregonstate.edu # send email to this address
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00
python3 hyperexamples.py
