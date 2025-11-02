cd /mnt/petrelfs/qinchonghan/project/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/stages/text_data/AM-Thinking-v1-Distilled-math.jsonl \
  --output_dir /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data_processed \
  --model_path /mnt/dhwfile/raise/user/linhonglin/vlm/models/qwen3_tikzgen_webinstruct_50k \
  --prompt_name convert_amthinking \
  --model_name qwen3_nothink \
  --parts 8 \
  --gpus 1 \
  --temperature 0.6 \
  --n_sample 2 \
  --chunk_size 256 \
  --max_tokens 3072 \
  --partition belt_road