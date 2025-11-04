cd /mnt/petrelfs/qinchonghan/project/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data_processed/AM-Thinking-v1-Distilled-math-qwen3_nothink-convert_amthinking-qwen3_tikzgen_webinstruct_50k/tikz/merged_parsed/success_with_answer_500_random.jsonl \
  --output_dir /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data_processed \
  --model_path /mnt/dhwfile/raise/user/qinchonghan/models/Qwen3-VL-8B-Instruct \
  --prompt_name classifier \
  --model_name qwen3vl_nothink \
  --parts 4 \
  --gpus 1 \
  --temperature 0.1 \
  --n_sample 2 \
  --chunk_size 8 \
  --max_tokens 1024 \
  --backend hf \
  --partition belt_road