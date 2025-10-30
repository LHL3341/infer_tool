cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/stages/vqa_data2/qwen3_tikzgen_vqa_20k.jsonl \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --prompt_name vqa \
  --model_name qwen3vl_nothink \
  --parts 1 \
  --gpus 1 \
  --temperature 0.5 \
  --n_sample 4 \
  --backend hf \
  --chunk_size 8 \
  --partition belt_road
