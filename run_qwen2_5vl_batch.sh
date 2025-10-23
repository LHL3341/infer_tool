cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input /mnt/dhwfile/raise/user/linhonglin/vlm/stages/vqa_data/megascience_52k_vqa_withans.jsonl \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --prompt_name qwen2_5vl \
  --model_name qwen2_5vl \
  --parts 8 \
  --gpus 1 \
  --partition belt_road
