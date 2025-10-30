cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/rlpr/results_merged_success_all.jsonl \
  --model_path /mnt/dhwfile/raise/user/linhonglin/vlm/models/qwen3vl_tikzgen_score \
  --prompt_name score \
  --model_name qwen3vl_nothink \
  --parts 1 \
  --gpus 1 \
  --temperature 0.1 \
  --n_sample 1 \
  --backend hf \
  --chunk_size 8 \
  --partition belt_road
