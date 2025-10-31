#!/bin/bash
#SBATCH -J qwen3vl_tikzgen2_score
#SBATCH -p belt_road
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --ntasks=4                # 要4个task，对应你要跑的4个part
#SBATCH --cpus-per-task=4         # 每个task给4个CPU，按需改
#SBATCH -o /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen2_score_%j.out
#SBATCH -e /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen2_score_%j.err
#SBATCH -t 7-00:00:00

set -euo pipefail

cd /mnt/petrelfs/qinchonghan/project/infer_tool

source activate llamafactory

bash /mnt/petrelfs/qinchonghan/project/infer_tool/scorer_batch_qwen3vl_tikzgen2.sh
