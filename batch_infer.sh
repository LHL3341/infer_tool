#!/bin/bash
# ==========================================================
# 批量推理脚本（支持 --save_images）
# 功能：
#   1️⃣ 将输入 JSONL/HF 数据切分为 index 范围；
#   2️⃣ 每份用 srun 启动 main.py；
#   3️⃣ 自动收集日志；
#   4️⃣ 所有任务完成后合并结果。
# ==========================================================

source activate vlmeval
cd /mnt/dhwfile/raise/user/linhonglin/data_process/infer_tool
export VLLM_USE_V1=0

# ========== 捕获退出信号 ==========
cleanup() {
  echo "⚠️ 捕获退出信号，正在终止所有子任务..."
  pkill -P $$ 2>/dev/null
  echo "🛑 所有子任务已中止。"
}
trap cleanup EXIT INT TERM

# ========== 默认参数 ==========
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
QUOTA_TYPE="auto"
PROMPT_NAME="vqa"
MODEL_NAME="qwen2_5_vl"
PROMPT_DIR="prompts"
PARTS=4
GPUS=1
PARTITION="belt_road"
TEMPERATURE=0.0
TOP_P=0.95
MAX_TOKENS=2048
N_SAMPLE=1
CHUNK_SIZE=1
SAVE_IMAGES=${SAVE_IMAGES:-false}  # 🆕 新增：是否保存图片（默认 false）

# ========== 参数解析 ==========
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input) INPUT="$2"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    --parts) PARTS="$2"; shift ;;
    --model_path) MODEL_PATH="$2"; shift ;;
    --prompt_name) PROMPT_NAME="$2"; shift ;;
    --model_name) MODEL_NAME="$2"; shift ;;
    --prompt_dir) PROMPT_DIR="$2"; shift ;;
    --partition) PARTITION="$2"; shift ;;
    --gpus) GPUS="$2"; shift ;;
    --save_images) SAVE_IMAGES=true ;;
    --backend) BACKEND="$2"; shift ;;
    --n_sample) N_SAMPLE="$2"; shift ;;                 
    --chunk_size) CHUNK_SIZE="$2"; shift ;;
    --temperature) TEMPERATURE="$2"; shift ;;
    --top_p) TOP_P="$2"; shift ;;
    --max_tokens) MAX_TOKENS="$2"; shift ;;
    *) echo "⚠️ 未知参数: $1"; exit 1 ;;
  esac
  shift
done

# 默认 backend 为 vllm
BACKEND=${BACKEND:-vllm}


if [[ -z "$INPUT" ]]; then
  echo "❌ 缺少必要参数: --input"
  exit 1
fi

# ========== 输出目录 ==========
BASENAME=$(basename "$INPUT")
EXP_NAME="${BASENAME%.*}"   # 去掉扩展名，比如 dev.jsonl -> dev
MODEL_PATH_NAME=$(basename "$MODEL_PATH")
MODEL_PATH_NAME=${MODEL_PATH_NAME%.*}

# 输出目录直接放在 outputs/ 下一级
OUTPUT_ROOT="${OUTPUT_DIR:-outputs}"   # 一级 outputs
LOG_ROOT="logs"

# 实验子目录
EXP_TAG="${EXP_NAME}-${MODEL_NAME}-${PROMPT_NAME}-${MODEL_PATH_NAME}"

OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_TAG}"  # e.g. outputs/dev-qwen2_5vl-parse
LOG_DIR="${LOG_ROOT}/${EXP_TAG}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "📂 输入: $INPUT"
echo "📤 输出目录: $OUTPUT_DIR"
echo "🗒️ 日志目录: $LOG_DIR"


# ========== 获取输入长度 ==========
echo "🔍 正在检查输入数据条数..."
python - "$INPUT" > /tmp/input_len.txt <<'PYCODE'
import json
from datasets import load_dataset
from pathlib import Path
import sys

input_path = sys.argv[1]
try:
    if Path(input_path).exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
    else:
        ds = load_dataset(input_path, split='train')
        n = len(ds)
    print(n)
except Exception as e:
    print(-1)
PYCODE

TOTAL=$(cat /tmp/input_len.txt)
if [[ "$TOTAL" -le 0 ]]; then
  echo "❌ 无法确定输入长度或输入为空 (INPUT=$INPUT)."
  exit 1
fi

echo "📊 总样本数: $TOTAL"

# ========== 计算切分范围 ==========
PER_PART=$(( (TOTAL + PARTS - 1) / PARTS ))
echo "🔪 每个任务分配约 $PER_PART 条样本"

# ========== 启动任务 ==========
echo "🚀 启动推理任务..."

for ((i=0; i<PARTS; i++)); do
  START_IDX=$(( i * PER_PART ))
  END_IDX=$(( (i + 1) * PER_PART ))
  if [[ $START_IDX -ge $TOTAL ]]; then
    break
  fi
  if [[ $END_IDX -gt $TOTAL ]]; then
    END_IDX=$TOTAL
  fi

  OUT_FILE="${OUTPUT_DIR}/part_${i}.jsonl"
  LOG_FILE="${LOG_DIR}/part_${i}_$(date +%Y%m%d_%H%M%S).log"

  echo "▶️ 启动任务 part_$i: [$START_IDX, $END_IDX)"
  echo "   日志: $LOG_FILE"

  CMD="python main.py \
      --model_path $MODEL_PATH \
      --input_jsonl $INPUT \
      --output_jsonl $OUT_FILE \
      --prompt_name $PROMPT_NAME \
      --model_name $MODEL_NAME \
      --prompt_dir $PROMPT_DIR \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --max_tokens $MAX_TOKENS \
      --n_sample $N_SAMPLE \
      --chunk_size $CHUNK_SIZE \
      --start_idx $START_IDX \
      --end_idx $END_IDX \
      --backend $BACKEND \
      "

  if [[ "$SAVE_IMAGES" == "true" ]]; then
    CMD="$CMD --save_images"   # 🆕 自动追加参数
  fi

  srun -p "$PARTITION" --gres=gpu:${GPUS} --quotatype=$QUOTA_TYPE bash -c "$CMD" > "$LOG_FILE" 2>&1 &
done

echo "⏳ 所有任务已提交，等待完成..."
wait
echo "✅ 所有子任务执行完毕！"

# ========== 合并输出 ==========
MERGED_FILE="${OUTPUT_DIR}/merged.jsonl"
echo "🧩 开始合并结果到: $MERGED_FILE"

find "$OUTPUT_DIR" -maxdepth 1 -type f -name "part_*.jsonl" \
  | sort | xargs cat > "$MERGED_FILE"

echo "🎉 全部完成！最终输出文件: $MERGED_FILE"
