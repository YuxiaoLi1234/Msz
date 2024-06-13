#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --qos=interactive
#SBATCH --constraint=gpu
#SBATCH --account=PAS2402


module load cuda


while IFS= read -r bound; do
    # 读取 values.txt 文件中的每一行作为参数
    while IFS= read -r value; do
        # 运行命令并传入参数
        ./helloworldomp finger 128 128 128 0 "$bound" sz3 zfp "$value"
    done < values.txt
done < bounds.txt
