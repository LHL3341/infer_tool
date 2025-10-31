import json
from pathlib import Path
from itertools import islice
import torch
import argparse
from PIL import Image
from utils import load_remaining_records, load_image
from models import *

# ========== 参数 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--input_jsonl", type=str, required=True)
parser.add_argument("--output_jsonl", type=str, required=True)
parser.add_argument("--prompt_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--prompt_dir", type=str, default="prompts")
parser.add_argument("--temperature", type=float, default=0.)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--n_sample", type=int, default=1)
parser.add_argument("--chunk_size", type=int, default=32)
parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"], help="选择推理后端 (vllm 或 hf)")
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--start_idx", type=int, default=None)
parser.add_argument("--end_idx", type=int, default=None)
args = parser.parse_args()

# ========== 通用准备 ==========
input_jsonl = Path(args.input_jsonl)
output_jsonl = Path(args.output_jsonl)
prompt_file = Path(args.prompt_dir) / f"{args.prompt_name}.txt"
assert prompt_file.exists(), f"❌ 找不到模板文件: {prompt_file}"

prompt_template = prompt_file.read_text(encoding="utf-8")
uses_image = "{<|image_pad|>}" in prompt_template
if uses_image:
    print("🖼️ 模板使用 {<|image_pad|>}, 将加载对应图片")
if not uses_image:
    print("🖼️ 模板不使用 {<|image_pad|>}, 将不加载图片")

remaining_records = load_remaining_records(
    input_jsonl=input_jsonl,
    output_jsonl=output_jsonl,
    prompt_template=prompt_template,
    use_image_basename=True,
    start_idx=args.start_idx,
    end_idx=args.end_idx,
)
if not remaining_records:
    print("✅ 无待处理记录")
    exit(0)

output_jsonl.parent.mkdir(parents=True, exist_ok=True)
output_image_dir = None
if args.save_images:
    output_image_dir = output_jsonl.parent / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 图片保存已开启，将输出到: {output_image_dir}")

# ========== 根据 backend 初始化模型 ==========
if args.backend == "vllm":
    from vllm import LLM, SamplingParams
    print("🚀 使用 vLLM 推理后端")

    llm = LLM(
        model=args.model_path,
        # max_model_len=args.max_tokens + 2048,
        data_parallel_size=torch.cuda.device_count(),
        mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 1024 * 1024},
        # enforce_eager=False,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=args.n_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

else:
    print("🧠 使用 HuggingFace Transformers 推理后端")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = MODEL_CLASS[args.model_name].from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.float16
    ).cuda()
    model.eval()
    if uses_image:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

print(f"🚀 模型加载完成，开始推理")
# ========== 工具 ==========
def render_prompt(template: str, fields: dict) -> str:
    return template.format(**fields)

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# ========== 主循环 ==========
total = len(remaining_records)
processed = 0

print(f"🚀 开始推理，共 {total} 条记录")
with output_jsonl.open("a", encoding="utf-8") as fout:
    for chunk in chunked_iterable(remaining_records, args.chunk_size):
        prompts, images, image_paths, valid_records = [], [], [], []
        for r in chunk:
            try:
                text = render_prompt(prompt_template, r)
                img = None
                if uses_image:
                    if "image" in r and r["image"] is not None:
                        img = r["image"]
                    elif "images" in r and r["images"] is not None:
                        img = r["images"][0]
                    else:
                        img_path = Path(r["image_path"])
                        if not img_path.exists():
                            print(f"⚠️ 图片不存在，跳过: {img_path}")
                            continue
                        img = load_image(img_path)
                prompts.append(text)
                images.append(img)
                image_paths.append(img_path)
                valid_records.append(r)
            except Exception as e:
                print(f"⚠️ 构造 prompt 出错: {e}")
                continue

        if not prompts:
            continue

        if args.backend == "vllm":
            try:
                outputs = llm.generate(
                    [{"prompt": t, "multi_modal_data": {"image": img} if uses_image else None}
                    for t, img in zip(prompts, images)],
                    sampling_params=sampling_params
                )
                # 每条记录收集所有 n_sample 输出
                generations = [[build_prompt(g.text.strip(), args.model_name) for g in o.outputs] for o in outputs]
            except Exception as e:
                print(f"❌ vLLM 生成失败: {e}")
                continue

        else:
            # HF 批量生成多采样
            generations = []
            if uses_image:
                processor.tokenizer.padding_side = 'left'
                input_messages = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}, {"type": "text", "text": prompt}]}] for prompt, image_path in zip(prompts, image_paths)]
            else:
                processor.tokenizer.padding_side = 'left'
                input_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

            input_messages = [msg for msg in input_messages for _ in range(args.n_sample)]

            with torch.no_grad():
                inputs = processor.apply_chat_template(
                    input_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True # padding should be set for batch generation!
                ).to(model.device)
                # print(inputs)
                # print(inputs.input_ids.shape)

                generated_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_tokens,
                    # num_return_sequences=args.n_sample
                )

            num_original = len(prompts)
            # print(num_original)
            # print(args.n_sample)
            # print(generated_ids.shape)
            for i in range(num_original):
                start = i * args.n_sample
                end = (i + 1) * args.n_sample
                gen_ids_trimmed = [
                    out_ids[len(inputs.input_ids[start]):] for j, out_ids in enumerate(generated_ids[start:end])
                ]
                output_texts = processor.batch_decode(
                    gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generations.append(output_texts)

            print(generations)

        for idx, (record, gen_texts) in enumerate(zip(valid_records, generations)):
            # ✅ 确定 ID：如果原始数据有 id 就沿用，否则用递增的 processed 计数
            record_id = record.get("id", processed)

            if "image" in record or "images" in record:
                if 'images' in record:
                    img = record["images"][0]
                else:
                    img = record["image"]

                try:
                    img_path = (output_image_dir / f"{record_id:06d}.png").resolve()  # ✅ 用 record_id 命名
                    img.save(img_path, format="PNG")
                    record["image_path"] = str(img_path)
                except Exception as e:
                    print(f"⚠️ 保存图片失败 (ID={record_id}): {e}")

                record.pop("image", None)
                record.pop("images", None)

            # ✅ 输出记录时显式写入 id
            out_record = {
                **record,
                "id": record_id,
                "outputs": gen_texts,
                "prompt_name": args.prompt_name,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

        print(f"✅ 已处理 {processed}/{total}")

print(f"🎉 全部生成完成，结果写入：{output_jsonl}")
