cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/stages/text_data/megascience.jsonl \
  --model_path /mnt/dhwfile/raise/user/linhonglin/vlm/models/qwen3_tikzgen_webinstruct_50k \
  --prompt_name convert_megascience \
  --model_name qwen3_nothink \
  --parts 8 \
  --gpus 1 \
  --temperature 0.6 \
  --n_sample 2 \
  --chunk_size 256 \
  --partition belt_road
