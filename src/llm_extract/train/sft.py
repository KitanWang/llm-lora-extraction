"""LoRA / QLoRA 监督微调（SFT）。

为什么不用 `trl.SFTTrainer`：trl 在 0.13→0.29 之间 `tokenizer=` / `processing_class=`、
`dataset_text_field` / `max_seq_length` 等参数反复改名，锁版本会牺牲可复现性。
这里直接用 `transformers.Trainer` + 手写 label mask，好处是 **SFT 的核心细节全部显式可见**：

    prompt 部分 token 的 label 置为 -100 —— 只对"答案"算 loss，不对"题目"算 loss。
    这是 SFT 与"继续预训练"的本质区别，也是实现时最容易出错的地方。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from llm_extract.prompts import build_messages, build_target, build_user_prompt, SYSTEM_PROMPT
from llm_extract.schema import InquiryRecord

ROOT = Path(__file__).resolve().parents[3]
IGNORE_INDEX = -100


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "outputs/qwen25-1.5b-lora"

    max_seq_length: int = 1024
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"     # QLoRA 时改成 paged_adamw_8bit
    load_in_4bit: bool = False     # True → QLoRA
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    seed: int = 42

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    max_train_samples: int | None = None   # 冒烟测试用
    report_to: str = "none"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"配置里有未知字段：{sorted(unknown)}")
        return cls(**data)


# ──────────────────────────────────────────────────────────────────
# 数据
# ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取数据集，并**从字段重建监督目标**。

    数据集里不冗余存 `target_json`，避免"字段改了但 target 没同步"的隐患——
    监督信号永远由 schema 现算，保证与评测口径完全一致。
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} 为空")
    for r in rows:
        if "target_json" not in r:
            payload = {k: r[k] for k in InquiryRecord.model_fields}
            r["target_json"] = build_target(InquiryRecord.model_validate(payload))
    return rows


def build_supervised_example(
    tokenizer,
    text: str,
    target_json: str,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """把 (输入文本, 目标 JSON) 编码成 input_ids / labels，并把 prompt 段 mask 掉。"""
    messages = build_messages(text)

    # prompt：走 chat template 并追加 assistant 起始标记
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 完整序列：prompt + 答案 + eos
    full_text = prompt_text + target_json + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True,
                     max_length=max_seq_length)
    input_ids = full["input_ids"]
    labels = list(input_ids)
    n_prompt = min(len(prompt_ids), len(input_ids))
    for i in range(n_prompt):
        labels[i] = IGNORE_INDEX

    # 极端情况：答案被截断光了 → 这条样本没有监督信号，跳过
    if all(l == IGNORE_INDEX for l in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def make_dataset(tokenizer, rows: list[dict[str, Any]], max_seq_length: int) -> Dataset:
    encoded = []
    skipped = 0
    for r in rows:
        ex = build_supervised_example(tokenizer, r["text"], r["target_json"], max_seq_length)
        if ex["input_ids"]:
            encoded.append(ex)
        else:
            skipped += 1
    if skipped:
        print(f"[warn] {skipped} 条样本因超长被丢弃")
    return Dataset.from_list(encoded)


class SupervisedCollator:
    """右侧 padding + labels 同步 padding（pad 位置用 -100，不计 loss）。"""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ──────────────────────────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if cfg.load_in_4bit:
        # 关键：把量化模型的 LayerNorm/输出层升到 fp32，并开启输入梯度，
        # 否则 LoRA 的梯度传不进 4bit 主干，loss 不会下降。
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA / QLoRA SFT")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    out_dir = ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 加载模型 {cfg.model_name_or_path}（4bit={cfg.load_in_4bit}）===")
    model, tokenizer = load_model_and_tokenizer(cfg)
    params = count_parameters(model)
    ratio = params["trainable"] / params["total"] * 100
    print(f"可训练参数：{params['trainable']:,} / {params['total']:,} = {ratio:.4f}%")
    if torch.cuda.is_available():
        print(f"显存占用（训练前）：{torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    train_rows = load_jsonl(ROOT / cfg.train_file)
    val_rows = load_jsonl(ROOT / cfg.val_file)
    if cfg.max_train_samples:
        train_rows = train_rows[: cfg.max_train_samples]
    print(f"训练样本 {len(train_rows)} / 验证样本 {len(val_rows)}")

    train_ds = make_dataset(tokenizer, train_rows, cfg.max_seq_length)
    val_ds = make_dataset(tokenizer, val_rows, cfg.max_seq_length)
    print(f"编码后：train={len(train_ds)} val={len(val_ds)}")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
        seed=cfg.seed,
        report_to=cfg.report_to,
        max_steps=args.max_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=SupervisedCollator(tokenizer),
    )

    print("=== 开始训练 ===")
    train_result = trainer.train()

    print("=== 保存 LoRA adapter ===")
    trainer.save_model(str(out_dir / "final_adapter"))
    tokenizer.save_pretrained(str(out_dir / "final_adapter"))

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"训练峰值显存：{peak:.2f} GiB")
    else:
        peak = 0.0

    metrics = {
        "config": asdict(cfg),
        "train_metrics": train_result.metrics,
        "param_total": params["total"],
        "param_trainable": params["trainable"],
        "trainable_ratio_pct": ratio,
        "peak_gpu_gib": round(peak, 3),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成 → {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
