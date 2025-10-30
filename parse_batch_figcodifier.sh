cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool

bash batch_infer.sh \
  --input LiamLian0727/Euclid30K \
  --model_path /mnt/dhwfile/raise/user/linhonglin/hf/huggingface/hub/models--MathLLMs--FigCodifier/snapshots/efcd597750dcbf115fee8adec6bf89d3ad835a2d \
  --prompt_name parse \
  --model_name intern_vl \
  --parts 1 \
  --gpus 1 \
  --partition belt_road \
  --save_images \
  --n_sample 3 \
  --chunk_size 1 \
  --temperature 0.2
