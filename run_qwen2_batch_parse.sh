cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input LiamLian0727/Euclid30K \
  --model_path /mnt/dhwfile/raise/user/linhonglin/vlm/models/qwen2_5vl_megascience52k_parse_datikzv3_arxiv \
  --prompt_name parse \
  --model_name qwen2_5vl \
  --parts 1 \
  --gpus 1 \
  --partition belt_road \
  --save_images \
  --n_sample 3 \
  --chunk_size 1 \
  --temperature 0.2
