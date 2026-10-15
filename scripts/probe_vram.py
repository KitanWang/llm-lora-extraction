"""显存可行性探针：**实测**每个模型在本机能否跑通项目 A 的每一个环节。

不做"参数量 × 字节数"的纸上估算——那类估算在 QLoRA 场景下经常差 2 倍以上
（激活值、梯度检查点、优化器状态分页、CUDA 上下文开销都算不准）。
这里直接真加载、真前向、真反向，把每个环节的峰值显存打出来。

用法：
    venv/bin/python scripts/probe_vram.py --model models/Qwen2.5-7B-Instruct --qlora
    venv/bin/python scripts/probe_vram.py --model models/Qwen2.5-1.5B-Instruct

探测的四个环节（对应项目 A 的完整流程）：
    [1] 4bit/bf16 加载基座        —— 能不能装进显存
    [2] 前向 + 生成（评测用）      —— 一条样本的推理峰值
    [3] 前向 + 反向（训练用）      —— 一个 micro-batch 的训练峰值
    [4] 优化器步进 + 梯度检查点     —— 真实训练峰值（最吃显存的一步）
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def gib(x: int | float) -> float:
    return x / 1024**3


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def report(label: str, extra: dict | None = None) -> dict:
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    row = {
        "stage": label,
        "peak_allocated_gib": round(gib(peak), 2),
        "peak_reserved_gib": round(gib(reserved), 2),
        "free_after_gib": round(gib(free), 2),
    }
    if extra:
        row.update(extra)
    print(
        f"  {label:<28} peak_alloc={row['peak_allocated_gib']:>5.2f} GiB  "
        f"peak_reserved={row['peak_reserved_gib']:>5.2f} GiB  "
        f"free_after={row['free_after_gib']:>5.2f} GiB"
        + (f"  {extra}" if extra else "")
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qlora", action="store_true", help="4bit NF4 量化加载")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--seq-ladder",
        type=str,
        default=None,
        help="逗号分隔的序列长度阶梯，如 1024,896,768,640。"
             "用于在 OOM 时立刻找到本机可用的最大长度（只跑训练那一步，不重复加载模型）",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，无法探测")

    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    name = Path(model_path).name
    mode = "QLoRA-4bit" if args.qlora else "bf16"

    free, total = torch.cuda.mem_get_info()
    print("=" * 78)
    print(f"显存探针：{name}  [{mode}]")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"探测开始时可用显存: {gib(free):.2f} GiB / {gib(total):.2f} GiB")
    print("=" * 78)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    results: list[dict] = []
    from llm_extract.prompts import build_messages

    # ── [1] 加载基座 ────────────────────────────────────────────
    print("\n[1] 加载基座模型")
    reset_peak()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not args.qlora else None,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    results.append(report("1. 加载基座"))

    # ── [2] 推理/生成 ──────────────────────────────────────────
    print("\n[2] 前向 + 生成（评测环节）")
    prompt = tokenizer.apply_chat_template(
        build_messages("型号 HX-1234，数量 1000 台，请报价。"), tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    reset_peak()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    results.append(report("2. 生成 256 token", {"prompt_tokens": enc["input_ids"].shape[1],
                                               "gen_tokens": out.shape[1] - enc["input_ids"].shape[1]}))
    del out, enc
    reset_peak()

    # ── [3] 训练：前向 + 反向 ───────────────────────────────────
    print("\n[3] LoRA 前向 + 反向（训练环节）")
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数 {n_train:,} / {n_total:,} = {n_train / n_total * 100:.4f}%")

    # 构造一个 max_seq_length 长度的真实训练样本（最坏情况）
    def make_batch(seq_len: int):
        ids = torch.randint(100, 30000, (args.batch_size, seq_len), device="cuda")
        attn = torch.ones_like(ids)
        lbl = ids.clone()
        lbl[:, : int(seq_len * 0.8)] = -100   # 模拟 prompt mask
        return ids, attn, lbl

    ids, attn, labels = make_batch(args.seq_len)

    model.train()
    reset_peak()
    try:
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        results.append(report("3. 前向+反向",
                              {"seq_len": args.seq_len, "micro_bs": args.batch_size,
                               "loss": round(float(loss.detach()), 4)}))
    except torch.cuda.OutOfMemoryError:
        results.append({"stage": "3. 前向+反向", "error": "OOM", "seq_len": args.seq_len})
        print(f"  ❌ OOM（seq_len={args.seq_len}）：训练跑不起来")
        del ids, attn, labels
        reset_peak()
        if args.seq_ladder:
            _run_ladder(model, args, results)
        _dump(results, name, mode, args, model_path)
        return

    # ── [4] 优化器步进（真实峰值）──────────────────────────────
    print("\n[4] 优化器步进（真实训练峰值）")
    optim_name = "paged_adamw_8bit" if args.qlora else "adamw_torch"
    try:
        if args.qlora:
            import bitsandbytes as bnb

            opt = bnb.optim.PagedAdamW8bit(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
        else:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        opt.step()
        opt.zero_grad(set_to_none=True)
        results.append(report("4. 优化器步进", {"optimizer": optim_name}))
    except torch.cuda.OutOfMemoryError:
        results.append({"stage": "4. 优化器步进", "error": "OOM"})
        print("  ❌ OOM：优化器状态放不下")
        _dump(results, name, mode, args, model_path)
        return

    # ── 结论 ───────────────────────────────────────────────────
    peak = max(r.get("peak_reserved_gib", 0) for r in results)
    print("\n" + "=" * 78)
    print(f"结论：{name} [{mode}] 全流程峰值显存 ≈ {peak:.2f} GiB（预留口径）")
    if peak < gib(total) * 0.92:
        print(f"✅ 可以跑通项目 A 全流程（训练 + 评测），余量 {gib(total) - peak:.2f} GiB")
    else:
        print(f"⚠️  余量不足（{gib(total) - peak:.2f} GiB），训练时可能因碎片化 OOM")
    print("=" * 78)
    _dump(results, name, mode, args, model_path)


def _run_ladder(model, args, results: list[dict]) -> None:
    """模型已加载的前提下，逐个序列长度试训练，找到本机能跑的最大长度。

    比"重新加载模型再试"快得多（7B 加载一次要几十秒），
    而且能直接给出「该把 max_seq_length 配成多少」这个可执行结论。
    """
    print("\n  序列长度阶梯测试（找本机可用的最大长度）...")
    ladder = [int(x) for x in args.seq_ladder.split(",") if x.strip()]
    best = None
    for seq_len in ladder:
        if seq_len >= args.seq_len:
            continue
        reset_peak()
        try:
            ids = torch.randint(100, 30000, (args.batch_size, seq_len), device="cuda")
            attn = torch.ones_like(ids)
            lbl = ids.clone()
            lbl[:, : int(seq_len * 0.8)] = -100
            out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            out.loss.backward()
            peak = torch.cuda.max_memory_allocated()
            row = report(f"3b. seq_len={seq_len}", {"fits": True})
            results.append({**row, "stage": f"3b. seq_len={seq_len}", "seq_len": seq_len})
            best = seq_len
            del ids, attn, lbl, out
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append({"stage": f"3b. seq_len={seq_len}", "seq_len": seq_len, "error": "OOM"})
            print(f"  ❌ seq_len={seq_len} 也放不下")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if best:
        print(f"\n  ➜ 建议 max_seq_length 配为 {best}（或更小）")


def _dump(results: list[dict], name: str, mode: str, args, model_path: str) -> None:
    out = args.out or (ROOT / "reports" / f"vram_probe_{name}_{mode.replace('-', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": name,
        "model_path": model_path,
        "mode": mode,
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": round(gib(torch.cuda.get_device_properties(0).total_memory), 2),
        "seq_len": args.seq_len,
        "micro_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "stages": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细 → {out}")


if __name__ == "__main__":
    sys.exit(main())
