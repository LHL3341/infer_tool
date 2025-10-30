#!/bin/bash
#SBATCH -J qwen3vl_tikzgen_score
#SBATCH -p belt_road
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --ntasks=4                # 要4个task，对应你要跑的4个part
#SBATCH --cpus-per-task=4         # 每个task给4个CPU，按需改
#SBATCH -o /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen_score_%j.out
#SBATCH -e /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen_score_%j.err
#SBATCH -t 7-00:00:00

set -euo pipefail

# 1) 进入项目目录
cd /mnt/petrelfs/qinchonghan/project/infer_tool

# 2) 激活环境
source activate llamafactory

# 3) 真正跑你现在的脚本（注意：这里是“在 SLURM 里跑 bash”，所以不会因为你 ssh 断开而停）
bash /mnt/petrelfs/qinchonghan/project/infer_tool/scorer_batch_qwen3vl_tikzgen.sh
