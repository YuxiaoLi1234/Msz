#!/bin/bash
#SBATCH -A m4259
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --cpus-per-task=1

module load parallel
mv nyx2.bin ./experiment_data/
mv ivt.bin ./experiment_data/
# 定义开始和结束的值以及数值的数量
nvcc -c kernel3d.cu -o kernel.o
g++-12 -std=c++17 -O3 -g -fopenmp -c preserve3d.cpp -o hello2.o
g++-12 -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld

parallel -j1 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3} 0;' ::: "finger,128,128,128" "Red_sea,500,500,50" "earthquake,750,375,100" "NYX,512,512,512" "at,177,95,48" "heated,150,450,1" "CSEM,1800,3600,1"  :::: bounds.txt ::: sz3 zfp
parallel -j1 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3} 0;' ::: "ivt,576,361,1" "jet_cropped,100,100,100" "nyx2,50,50,50" ::: 1e-2 ::: sz3 zfp
parallel -j1 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3} 0;' ::: "at,177,95,48" ::: 1e-3 ::: sz3 zfp
# parallel -j4 './helloworldmp {1} {2} {3} 0 {4};' ::: "finger,128,128,128" :::: values.txt ::: sz3
# parallel -j2 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); ./helloworld {1} {2} {3};' ::: "finger,128,128,128" "Red_sea,500,500,50" "earthquake,750,375,100" "NYX,512,512,512" "S3D,500,500,500" ::: $(awk 'BEGIN{for(i=0;i<100;i++) print 1e-6 + i*(1e-5-1e-6)/99}') ::: sz3 zfp

# parallel -j4 'export CUDA_VISIBLE_DEVICE=$(({%}-1)); num=$(echo {1} | awk -F,"'"'{print $NF}'"'"'); if [ "$num" -eq 1 ]; then ./helloworld {1} {2} {3}; else ./helloworld2 {1} {2} {3}; fi' ::: "NYX,512,512,512,0" ::: 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 ::: sz3 zfp