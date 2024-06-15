#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --account=PAS2402


module load cuda

./helloworldomp finger,128,128,128 1e-6 sz3 0 32
./helloworldomp_d finger,128,128,128 1e-6 sz3 0 32
./helloworld_determin finger,128,128,128 1e-6 sz3 0
./helloworld finger,128,128,128 1e-6 sz3 0
