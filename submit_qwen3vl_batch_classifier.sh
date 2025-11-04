#!/bin/bash
#SBATCH -J qwen3vl_classifier
#SBATCH -p belt_road
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH -o /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_classifier_%j.out
#SBATCH -e /mnt/petrelfs/qinchonghan/project/infer_tool/logs/qwen3vl_classifier_%j.err
#SBATCH -t 7-00:00:00

set -euo pipefail

cd /mnt/petrelfs/qinchonghan/project/infer_tool

source activate llamafactory

bash /mnt/petrelfs/qinchonghan/project/infer_tool/run_qwen3vl_batch_classifier.sh
