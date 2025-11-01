#!/bin/bash
#SBATCH -J qwen3vl_tikzgen_score
#SBATCH -p raise
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:0
#SBATCH --nodelist=SH-IDC1-10-140-37-94
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH -o /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen_score_%j.out
#SBATCH -e /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_tikzgen_score_%j.err
#SBATCH -t 7-00:00:00

set -euo pipefail

cd /mnt/petrelfs/qinchonghan/project/infer_tool

source activate llamafactory

bash /mnt/petrelfs/qinchonghan/project/infer_tool/run_qwen3vl_batch_scorer_node.sh
