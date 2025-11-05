cd /mnt/petrelfs/qinchonghan/project/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data/WebInstruct-verified/annotations.jsonl \
  --output_dir /mnt/dhwfile/raise/user/qinchonghan/llamafactory/data_processed \
  --model_path /mnt/dhwfile/raise/user/qinchonghan/models/Qwen3-8B \
  --prompt_name detailed_classifier \
  --model_name qwen3_nothink \
  --parts 8 \
  --gpus 1 \
  --temperature 0.1 \
  --n_sample 1 \
  --chunk_size 128 \
  --max_tokens 8192 \
  --partition belt_road