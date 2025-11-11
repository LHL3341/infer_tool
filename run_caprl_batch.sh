#!/bin/bash
set -euo pipefail

cd /mnt/petrelfs/qinchonghan/project/infer_tool

  # --model_path /mnt/dhwfile/raise/user/qinchonghan/models/CapRL-InternVL3.5-8B \
  # --model_path /mnt/dhwfile/raise/user/qinchonghan/models/InternVL3_5-8B \
  
bash batch_infer.sh \
  --input /mnt/petrelfs/qinchonghan/project/data_process/caption/datikzv3/test_10.jsonl \
  --output_dir /mnt/petrelfs/qinchonghan/project/data_process/caption/datikzv3 \
  --model_path /mnt/dhwfile/raise/user/qinchonghan/models/CapRL-InternVL3.5-8B \
  --prompt_name caprl \
  --model_name intern_vl_35 \
  --parts 1 \
  --gpus 1 \
  --partition raise \
  --save_images \
  --n_sample 1 \
  --chunk_size 1 \
  --temperature 0.1 \
  --backend hf