import json
import re
from pathlib import Path
from typing import List, Dict, Set
import math
from PIL import Image
from datasets import load_dataset

# utils.py (节选)
from pathlib import Path
import json

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _read_json_lines_chunk(file_path, start_line, end_line):
    """子进程任务：读取文件指定区间的 JSON 行"""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= end_line:
                break
            try:
                results.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ 解析第 {i} 行失败: {e}")
    return results


def load_remaining_records(input_jsonl, output_jsonl=None, prompt_template=None,
                           use_image_basename=False, start_idx=None, end_idx=None,
                           num_workers=8, chunk_size=2000):
    """
    高性能 JSONL 加载器：
    - 自动使用多进程读取大文件
    - 打印总行数、加载条数、首条样本结构
    """

    input_s = str(input_jsonl)
    p = Path(input_s)
    records = []

    # ✅ 支持 HF 数据集
    if ("/" in input_s) and (not p.exists()):
        ds = load_dataset(input_s, split="train",num_proc=num_workers)
        total = len(ds)
        s = 0 if start_idx is None else start_idx
        e = total if end_idx is None else end_idx
        if s < 0: s = 0
        if e > total: e = total
        sub = ds.select(range(s, e))
        records = [dict(r) for r in sub]
        print(f"📚 从 HuggingFace 数据集加载 {len(records)} 条记录")
        print(f"🧩 首条记录结构: {list(records[0].keys()) if records else '空'}")
        return records

    # ✅ 本地 JSONL 文件
    if not p.exists():
        raise FileNotFoundError(f"❌ 文件不存在: {p}")

    # 获取总行数（轻量扫描）
    with open(p, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    s = 0 if start_idx is None else start_idx
    e = total_lines if end_idx is None else min(end_idx, total_lines)

    print(f"📄 文件: {p}")
    print(f"📏 总行数: {total_lines:,}  |  加载范围: [{s}, {e})")

    # === 使用多进程分块读取 ===
    chunks = [(max(s, i), min(e, i + chunk_size))
              for i in range(s, e, chunk_size)]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_read_json_lines_chunk, p, start, end): (start, end)
            for start, end in chunks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="加载中"):
            try:
                chunk_data = fut.result()
                results.extend(chunk_data)
            except Exception as e:
                print(f"⚠️ 子进程出错: {e}")

    print(f"✅ 成功加载 {len(results):,} 条记录")
    if results:
        print(f"🧩 首条记录结构: {list(results[0].keys())}")
    else:
        print("⚠️ 文件为空或范围内无有效记录")

    return results



def load_image(image_path: str, min_pixels: int=262144, max_pixels: int=1003520) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None