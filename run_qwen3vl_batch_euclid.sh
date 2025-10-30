cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input LiamLian0727/Euclid30K \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --prompt_name vqa_euclid \
  --model_name qwen3vl_nothink \
  --parts 1 \
  --gpus 1 \
  --temperature 0.3 \
  --backend hf \
  --chunk_size 8 \
  --partition belt_road
