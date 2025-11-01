#!/bin/bash
set -euo pipefail

cd /mnt/petrelfs/qinchonghan/project/infer_tool

bash batch_infer_node.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/data_process/api_tool/outputs/qwen3_tikzgen/results_merged/success_merged.jsonl \
  --output_dir /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data_selection/qwen3_tikzgen \
  --model_path /mnt/dhwfile/raise/user/linhonglin/vlm/models/qwen3vl_tikzgen_score \
  --prompt_name scorer \
  --model_name qwen3vl_nothink \
  --parts 8 \
  --gpus 1 \
  --partition raise \
  --save_images \
  --n_sample 1 \
  --chunk_size 1 \
  --temperature 0.1 \
  --backend hf
