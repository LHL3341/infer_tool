import json
from pathlib import Path
from vllm import LLM, SamplingParams
import torch
from itertools import islice
import argparse
from transformers import AutoTokenizer
from PIL import Image
from utils import load_remaining_records, load_image
from build_prompt import build_prompt

# ========== 参数 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--input_jsonl", type=str, required=True)
parser.add_argument("--output_jsonl", type=str, required=True)
parser.add_argument("--prompt_name", type=str, required=True,
                    help="模板文件名（不含路径、不含后缀）")
parser.add_argument("--model_name", type=str, required=True,
                    help="模型名称")
parser.add_argument("--prompt_dir", type=str, default="prompts",
                    help="存放模板文件的目录")
parser.add_argument("--temperature", type=float, default=0.)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--chunk_size", type=int, default=32)
parser.add_argument("--start_idx", type=int, default=None, help="起始 index（包含）")
parser.add_argument("--end_idx", type=int, default=None, help="结束 index（不包含）")
parser.add_argument("--save_images", action="store_true", help="是否保存输出记录中的 image 字段到文件")  # 🆕 新增参数
args = parser.parse_args()

# ========== 配置 ==========
model_path = args.model_path
input_jsonl = Path(args.input_jsonl)
output_jsonl = Path(args.output_jsonl)
prompt_file = Path(args.prompt_dir) / f"{args.prompt_name}.txt"
assert prompt_file.exists(), f"❌ 找不到模板文件: {prompt_file}"

data_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

# ========== 加载模板 ==========
prompt_template = prompt_file.read_text(encoding="utf-8")
uses_image = "{image_path}" in prompt_template
if uses_image:
    print("🖼️ 模板使用 {image_path} ，将加载对应图片")

# ========== 读取待生成记录 ==========
remaining_records = load_remaining_records(
    input_jsonl=input_jsonl,
    output_jsonl=output_jsonl,
    prompt_template=prompt_template,
    use_image_basename=True,
    start_idx=args.start_idx,
    end_idx=args.end_idx,
)
if not remaining_records:
    exit(0)

# ========== 初始化 LLM ==========
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path,
    max_model_len=args.max_tokens + 2048,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1024 * 1024,
    },
    data_parallel_size=data_parallel_size
)

sampling_params = SamplingParams(
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
)

# ========== 工具函数 ==========
def render_prompt(template: str, fields: dict) -> str:
    """用输入字段填充模板"""
    try:
        return template.format(**fields)
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(f"❌ 模板中使用了未提供的字段: {missing}")

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# ========== 主循环 ==========
output_jsonl.parent.mkdir(parents=True, exist_ok=True)

# 🆕 如果开启了保存图片选项，则创建目录
output_image_dir = None
if args.save_images:
    output_image_dir = output_jsonl.parent / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 图片保存已开启，将输出到: {output_image_dir}")

total = len(remaining_records)
processed = 0

with output_jsonl.open("a", encoding="utf-8") as fout:
    for chunk in chunked_iterable(remaining_records, args.chunk_size):
        prompts = []
        images = []
        valid_records = []

        for r in chunk:
            try:
                text = render_prompt(prompt_template, r)
                text = build_prompt(text, args.model_name)
                if uses_image:
                    if "image" in r and r["image"] is not None:
                        img = r["image"]
                    else:
                        img_path = Path(r["image_path"])
                        if not img_path.exists():
                            print(f"⚠️ 图片不存在，跳过: {img_path}")
                            continue
                        img = load_image(img_path)
                    images.append(img)
                else:
                    images.append(None)
                prompts.append(text)
                valid_records.append(r)
            except Exception as e:
                print(f"⚠️ 构造 prompt 出错，跳过: {r}\n{e}")
                continue

        if not prompts:
            continue

        try:
            outputs = llm.generate(
                [
                    {
                        "prompt": t,
                        "multi_modal_data": {"image": img} if uses_image else None
                    }
                    for t, img in zip(prompts, images)
                ],
                sampling_params=sampling_params
            )
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU 显存不足，建议减小 batch/chunk size 或 max_tokens")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"❌ llm.generate() 出错，本批次跳过。\n错误详情: {repr(e)}")
            continue

        for record, out in zip(valid_records, outputs):
            # 🆕 根据 save_images 参数决定处理方式
            if "image" in record:
                if args.save_images and isinstance(record["image"], Image.Image):
                    img_save_path = output_image_dir / f"{record.get('id', processed):06d}.jpg"
                    try:
                        record["image"].save(img_save_path)
                        record["image_path"] = str(img_save_path)
                    except Exception as e:
                        print(f"⚠️ 保存图片失败: {e}")
                # 无论是否保存，都移除原始 image 对象
                del record["image"]

            results = [g.text.strip() for g in out.outputs]
            out_record = {
                **record,
                "outputs": results,
                "prompt_name": args.prompt_name,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

        print(f"✅ 已处理 {processed}/{total} 条 (chunk size={len(chunk)})")

print(f"🎉 全部生成完成，结果写入：{output_jsonl}")
