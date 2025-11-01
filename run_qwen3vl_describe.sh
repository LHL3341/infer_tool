cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/rlpr/results_merged_success_others_sample500.jsonl \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --prompt_name describe \
  --model_name qwen3vl_nothink \
  --parts 1 \
  --gpus 1 \
  --temperature 0.1 \
  --n_sample 3 \
  --backend hf \
  --chunk_size 16 \
  --partition belt_road
