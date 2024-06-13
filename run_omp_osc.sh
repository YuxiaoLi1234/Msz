#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --account=PAS2402


module load cuda


parallel -j1 './helloworldomp {1} {2} {3} 0 {4};' ::: "finger,128,128,128" :::: bounds.txt ::: sz3 zfp :::: values.txt
