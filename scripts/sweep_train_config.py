"""7B QLoRA 配置搜索：在**真实数据**上找出 8GB 显卡能跑通的配置。

背景：探针实测 7B QLoRA 在 seq_len=1024 时峰值 8.96 GiB，
而本机可用仅 6.87 GiB。但"跑不动"不是结论，"跑得动的最小代价"才是。

本脚本在**一次模型加载**内横向对比多个配置，回答三个问题：
  1. gradient checkpointing 到底有没有生效（关掉对比就知道）？
  2. 序列长度降到多少能压进显存？
  3. 代价是多少（每秒多少步 / 预计总训练时长）？
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
IGNORE = -100


def gib(x: float) -> float:
    return x / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--configs", default="1024:1,768:1,768:0,512:1",
                    help="seq_len:grad_ckpt 逗号分隔")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1,
                    help="必须与 per_device_train_batch_size 一致，否则测的不是生产配置")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fp32-upcast", action="store_true",
                    help="启用官方 fp32 升级（用于对比它多占多少显存）")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from llm_extract.prompts import build_target
    from llm_extract.schema import InquiryRecord
    from llm_extract.train.sft import SupervisedCollator, build_supervised_example

    mp = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    free, total = torch.cuda.mem_get_info()
    print("=" * 74)
    print(f"模型 {Path(mp).name}")
    print(f"可用显存 {gib(free):.2f} GiB / 总 {gib(total):.2f} GiB")
    print("=" * 74)

    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, quantization_config=qc, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # ⚠️ 这里必须与 src/llm_extract/train/sft.py 的加载路径保持一致。
    # 用 prepare_model_for_kbit_training 会把 embed_tokens / lm_head 升到 fp32
    # （7B 上多占 2.04 GB，常驻 5.34 → 7.38 GiB，必然溢出），
    # 那样扫出来的结论是失真的。生产路径用的是 fp32_upcast_non_quantized=False。
    if not args.fp32_upcast:
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.train()

    base_alloc = torch.cuda.memory_allocated()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"基座+LoRA 常驻显存 : {gib(base_alloc):.2f} GiB")
    print(f"可训练参数         : {n_train:,} / {n_total:,} = {n_train/n_total*100:.4f}%")
    print(f"grad checkpointing : {model.is_gradient_checkpointing}")
    print()

    rows = [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl")][: args.n_samples]

    import bitsandbytes as bnb

    results = []
    for spec in args.configs.split(","):
        seq_len_s, gc_s = spec.split(":")
        seq_len, want_gc = int(seq_len_s), bool(int(gc_s))

        # 切换 gradient checkpointing
        if want_gc and not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        elif not want_gc and model.is_gradient_checkpointing:
            model.gradient_checkpointing_disable()
        model.train()

        # 用真实样本编码，超长则截断
        feats = []
        for r in rows:
            payload = {k: r[k] for k in InquiryRecord.model_fields}
            ex = build_supervised_example(
                tok, r["text"], build_target(InquiryRecord.model_validate(payload)), seq_len)
            if ex["input_ids"]:
                feats.append(ex)
        if not feats:
            print(f"seq_len={seq_len} gc={int(want_gc)} → 无有效样本，跳过")
            continue

        collator = SupervisedCollator(tok, use_logits_to_keep=True)
        # 切成 micro_batch 大小的 batch —— 必须与生产配置的
        # per_device_train_batch_size 一致，否则测出来的是别的配置
        batches = []
        for i in range(0, len(feats), args.micro_batch):
            b = collator(feats[i:i + args.micro_batch])
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            batches.append(b)
        batch = batches[0]
        real_len = batch["input_ids"].shape[1]
        k = batch.get("logits_to_keep", "off")

        # 优化器（每次重建，保证公平）
        params = [p for p in model.parameters() if p.requires_grad]
        opt = bnb.optim.PagedAdamW8bit(params, lr=1e-4)

        peaks, times = [], []
        oom = False
        for step in range(args.steps):
            # 第 0 步包含 cuBLAS 句柄初始化与显存池预热，计时无参考价值，单独丢弃；
            # 但显存峰值要算进去（生产里第一次迭代同样会付这笔钱）。
            warmup = step == 0
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            cur = batches[step % len(batches)]
            try:
                out = model(**cur)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                oom = True
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                break
            if not warmup:
                times.append(time.time() - t0)
            peaks.append(torch.cuda.max_memory_allocated())

        row = {
            "seq_len": seq_len,
            "grad_ckpt": want_gc,
            "real_len": real_len,
            "logits_to_keep": k,
            "n_samples": len(feats),
            "micro_batch": args.micro_batch,
            "n_batches": len(batches),
        }
        if oom:
            row["result"] = "OOM"
            print(f"seq_len={seq_len:<5} gc={int(want_gc)}  ❌ OOM")
        else:
            # 用第一次（含编译/缓存建立）之后的稳态值
            peak = max(peaks)
            step_t = statistics.median(times)
            row.update({
                "result": "ok",
                "peak_allocated_gib": round(gib(peak), 2),
                "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 2),
                "activations_gib": round(gib(peak - base_alloc), 2),
                "sec_per_step": round(step_t, 2),
            })
            print(f"seq_len={seq_len:<5} gc={int(want_gc)}  ✅ peak={gib(peak):.2f} GiB "
                  f"(激活 {gib(peak-base_alloc):.2f})  {step_t:.2f}s/step")
        results.append(row)

        # 清理
        del batch, opt
        gc.collect()
        torch.cuda.empty_cache()

    # ── 估算训练时长 ──
    print()
    print("=" * 74)
    print("训练时长估算（1584 条训练样本，等效 batch=16 → 每 epoch 99 步）")
    print("=" * 74)
    for r in results:
        if r.get("result") != "ok":
            continue
        per_epoch = 99 * r["sec_per_step"] * 16 / max(1, r["n_samples"])
        print(f"  seq_len={r['seq_len']:<5} gc={int(r['grad_ckpt'])}  "
              f"{r['sec_per_step']:.2f}s/step → 2 epoch 约 {per_epoch*2/60:.0f} 分钟")

    out = args.out or (ROOT / "reports" / "sweep_7b_config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": Path(mp).name,
        "available_gib": round(gib(free), 2),
        "base_resident_gib": round(gib(base_alloc), 2),
        "trainable_params": n_train,
        "configs": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细 → {out}")


if __name__ == "__main__":
    main()
