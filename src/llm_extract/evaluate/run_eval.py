"""批量推理 + 评测：把「基座」和「微调后」放在同一评测集上跑对比。

用法示例：
    # 基座零样本
    python -m llm_extract.evaluate.run_eval \
        --model Qwen/Qwen2.5-7B-Instruct --dataset data/processed/test_iid.jsonl \
        --tag base_iid --out reports/preds/base_iid.jsonl

    # 微调后（加载 LoRA adapter）
    python -m llm_extract.evaluate.run_eval \
        --model Qwen/Qwen2.5-7B-Instruct --adapter outputs/xxx/final_adapter \
        --dataset data/processed/test_iid.jsonl --tag lora_iid --out reports/preds/lora_iid.jsonl

关键约定：**贪心解码**。评测必须可复现，采样会让两次跑出来的数不一样，
"提升 30 个百分点"这种结论就站不住了。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_extract.evaluate.metrics import evaluate_one, summarize
from llm_extract.prompts import build_messages

ROOT = Path(__file__).resolve().parents[3]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(model_path: str, adapter: str | None, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 生成时必须左侧 padding

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()   # 合并后推理更快，且导出的是纯模型

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(model, tokenizer, texts: list[str], max_new_tokens: int) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
        for t in texts
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=2048, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,               # 贪心，保证可复现
        num_beams=1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="批量推理并评测")
    ap.add_argument("--model", required=True, help="基座模型路径或 HF id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录（不传则是基座零样本）")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="结果标识，如 base_iid / lora_ood")
    ap.add_argument("--out", type=Path, required=True, help="预测明细落盘路径(jsonl)")
    ap.add_argument("--summary-out", type=Path, default=None, help="指标汇总(json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true", help="显存不够时用 4bit 加载基座")
    args = ap.parse_args()

    rows = load_jsonl(ROOT / args.dataset if not args.dataset.is_absolute() else args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[{args.tag}] 评测样本 {len(rows)} 条，模型 {args.model}"
          f"{f' + adapter {args.adapter}' if args.adapter else '（基座零样本）'}")

    model, tokenizer = build_model(args.model, args.adapter, args.load_in_4bit)

    preds: list[str] = []
    t0 = time.time()
    for i in tqdm(range(0, len(rows), args.batch_size), desc=args.tag):
        batch = rows[i: i + args.batch_size]
        preds.extend(generate_batch(model, tokenizer, [r["text"] for r in batch],
                                    args.max_new_tokens))
    elapsed = time.time() - t0

    records = []
    for i, (row, raw) in enumerate(zip(rows, preds)):
        gold = {k: v for k, v in row.items()
                if k not in ("text", "target_json", "template_family")}
        records.append(evaluate_one(i, raw, gold, row.get("template_family", "")))

    summary = summarize(records)
    summary["tag"] = args.tag
    summary["model"] = args.model
    summary["adapter"] = args.adapter
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["sec_per_sample"] = round(elapsed / max(1, len(rows)), 3)
    if torch.cuda.is_available():
        summary["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    out_path = ROOT / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "idx": r.idx,
                "template_family": r.template_family,
                "gold": r.gold,
                "raw_output": r.raw_output,
                "pred": r.pred,
                "json_ok": r.json_ok,
                "schema_ok": r.schema_ok,
                "schema_error": r.schema_error,
                "field_correct": r.field_correct,
                "exact_match": r.exact_match,
            }, ensure_ascii=False) + "\n")

    summary_path = (ROOT / args.summary_out if args.summary_out
                    else ROOT / "reports" / f"summary_{args.tag}.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{args.tag}] JSON 合法率   {summary['json_valid_rate'] * 100:6.1f}%")
    print(f"[{args.tag}] Schema 合规率 {summary['schema_valid_rate'] * 100:6.1f}%")
    print(f"[{args.tag}] 全字段准确率  {summary['exact_match_rate'] * 100:6.1f}%")
    print(f"[{args.tag}] 耗时 {elapsed:.1f}s（{summary['sec_per_sample']}s/条）")
    print(f"明细 → {out_path}\n汇总 → {summary_path}")


if __name__ == "__main__":
    main()
