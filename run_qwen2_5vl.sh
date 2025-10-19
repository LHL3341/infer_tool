#!/bin/bash
# ==========================================================
# 通用启动脚本：自动保存日志到 logs/ 目录
# 用法：
#   bash start_vllm.sh \
#     --model_path /path/to/qwen2-vl \
#     --input_jsonl data/input.jsonl \
#     --output_jsonl data/output.jsonl \
#     --prompt_name multimodal
# ==========================================================
source activate vlmeval
cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool
export VLLM_USE_V1=0
GPUS=4
# 1️⃣ 解析参数
# MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
INPUT_JSONL="/mnt/dhwfile/raise/user/linhonglin/vlm/stages/vqa_data/megascience_52k_vqa_withans.jsonl"
OUTPUT_JSONL="/mnt/dhwfile/raise/user/linhonglin/vlm/stages/vqa_data/megascience_52k_vqa_qwen2_5vl_output.jsonl"
PROMPT_NAME="qwen2_5_vl"
MODEL_NAME="qwen2_5_vl"
PROMPT_DIR="prompts"
TEMPERATURE=0.1
TOP_P=0.95
MAX_TOKENS=4096
N=3
CHUNK_SIZE=64

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model_path) MODEL_PATH="$2"; shift ;;
    --input_jsonl) INPUT_JSONL="$2"; shift ;;
    --output_jsonl) OUTPUT_JSONL="$2"; shift ;;
    --prompt_name) PROMPT_NAME="$2"; shift ;;
    --model_name) MODEL_NAME="$2"; shift ;;
    --prompt_dir) PROMPT_DIR="$2"; shift ;;
    --temperature) TEMPERATURE="$2"; shift ;;
    --top_p) TOP_P="$2"; shift ;;
    --max_tokens) MAX_TOKENS="$2"; shift ;;
    --n) N="$2"; shift ;;
    --chunk_size) CHUNK_SIZE="$2"; shift ;;
    *) echo "⚠️ 未知参数: $1"; exit 1 ;;
  esac
  shift
done

# 检查必要参数
if [[ -z "$MODEL_PATH" || -z "$INPUT_JSONL" || -z "$OUTPUT_JSONL" || -z "$PROMPT_NAME" || -z "$MODEL_NAME" ]]; then
  echo "❌ 缺少必要参数：--model_path --input_jsonl --output_jsonl --prompt_name --model_name"
  exit 1
fi

# 2️⃣ 创建日志目录
mkdir -p logs

# 3️⃣ 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${PROMPT_NAME}_${TIMESTAMP}.log"

# 4️⃣ 打印启动信息
echo "🚀 启动任务：${PROMPT_NAME}"
echo "📦 模型: ${MODEL_PATH}"
echo "📂 输入: ${INPUT_JSONL}"
echo "📝 输出: ${OUTPUT_JSONL}"
echo "🧾 日志: ${LOG_FILE}"
echo "--------------------------------------------"

# 5️⃣ 运行主程序并记录日志
srun -p belt_road --gres=gpu:${GPUS} python main.py \
  --model_path "$MODEL_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_dir "$PROMPT_DIR" \
  --prompt_name "$PROMPT_NAME" \
  --model_name "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --n "$N" \
  --chunk_size "$CHUNK_SIZE" \
  2>&1 | tee "$LOG_FILE"

# 6️⃣ 结束标识
echo "✅ 任务完成，日志已保存至：$LOG_FILE"
