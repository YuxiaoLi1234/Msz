#!/bin/bash
#SBATCH -A m4259
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 2
#SBATCH -c 128
#SBATCH --cpus-per-task=128

module load parallel

# 定义开始和结束的值以及数值的数量


# parallel -j4 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3} 0;' ::: "finger,128,128,128" :::: values.txt ::: sz3 zfp
parallel -j4 './helloworldomp {1} {2} {3} 0 {4};' ::: "finger,128,128,128" ::: 1e-06 ::: sz3 :::: values.txt
# parallel -j2 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3};' ::: "finger,128,128,128" "Red_sea,500,500,50" "earthquake,750,375,100" "NYX,512,512,512" "S3D,500,500,500" ::: $(awk 'BEGIN{for(i=0;i<100;i++) print 1e-6 + i*(1e-5-1e-6)/99}') ::: sz3 zfp

# parallel -j4 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); num=$(echo {1} | awk -F,"'"'{print $NF}'"'"'); if [ "$num" -eq 1 ]; then ./helloworld {1} {2} {3}; else ./helloworld2 {1} {2} {3}; fi' ::: "NYX,512,512,512,0" ::: 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 ::: sz3 zfp