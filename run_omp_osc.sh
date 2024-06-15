#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --account=PAS2402

values = (5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2)
module load cuda
for i in "${values[@]}":
    ./helloworldomp finger,128,128,128 $i sz3 0 32
    ./helloworldomp_d finger,128,128,128 $i sz3 0 32
    ./helloworld_determin finger,128,128,128 $i sz3 0
    ./helloworld finger,128,128,128 $i sz3 0
