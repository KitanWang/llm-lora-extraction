#!/usr/bin/env bash
# 串行执行项目 A 的全部实验步骤。
#
# 为什么必须串行：本机可用显存 6.87 GiB，7B 训练常驻 5.34 GiB，
# 一旦与另一个进程争抢，WSL 会溢出到主机内存 —— 表现为「不报错但慢几十倍」。
# 实测：1.5B 训练与 7B 评测并发时，297 步耗时 26 小时（正常应约 25 分钟）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/reports/run_all.log"
mkdir -p "$ROOT/reports/preds"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_eval() {
  local model="$1" adapter="$2" dataset="$3" tag="$4" bs="$5" extra="${6:-}"
  # 幂等：已有汇总结果则跳过。7B 单组评测约 30 分钟，断点续跑时必须能跳过
  # 已完成的组，否则每次重跑都要从头再来（这个坑踩过一次）。
  if [ -f "reports/summary_$tag.json" ] && [ "${FORCE:-0}" != "1" ]; then
    log "跳过 $tag（已有结果；FORCE=1 可强制重跑）"
    return 0
  fi
  log "评测 $tag  (model=$(basename "$model") adapter=${adapter:-无} dataset=$dataset bs=$bs)"
  $PY -m llm_extract.evaluate.run_eval \
      --model "$model" ${adapter:+--adapter "$adapter"} \
      --dataset "$dataset" --tag "$tag" \
      --out "reports/preds/$tag.jsonl" --batch-size "$bs" $extra \
      >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "❌ $tag 失败 (exit $rc)"; else log "✅ $tag 完成"; fi
  sleep 5
}

# ── 参数：可选择只跑某一段，便于中断后续跑 ────────────────
STAGE="${1:-all}"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "1.5b" ]; then
  log "===== 1.5B：基座零样本 vs LoRA 微调 ====="
  run_eval models/Qwen2.5-1.5B-Instruct "" \
           data/processed/test_iid.jsonl base_1.5b_iid 16
  run_eval models/Qwen2.5-1.5B-Instruct "" \
           data/processed/test_ood.jsonl base_1.5b_ood 16
  run_eval models/Qwen2.5-1.5B-Instruct outputs/qwen25-1.5b-lora/final_adapter \
           data/processed/test_iid.jsonl lora_1.5b_iid 16
  run_eval models/Qwen2.5-1.5B-Instruct outputs/qwen25-1.5b-lora/final_adapter \
           data/processed/test_ood.jsonl lora_1.5b_ood 16
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "7b-train" ]; then
  # 幂等：已有 adapter 就跳过训练。7B 训练约 1.8 小时，
  # 若因误用 STAGE=all 而重复训练，代价极高（这个坑踩过一次）。
  if [ -f "outputs/qwen25-7b-qlora/final_adapter/adapter_model.safetensors" ] \
     && [ "${FORCE:-0}" != "1" ]; then
    log "跳过 7B 训练（已有 adapter；FORCE=1 可强制重训）"
  else
    log "===== 7B QLoRA 训练 ====="
    $PY -m llm_extract.train.sft --config configs/qwen25-7b-qlora.yaml >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then log "❌ 7B 训练失败"; else log "✅ 7B 训练完成"; fi
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "7b-eval" ]; then
  log "===== 7B：基座零样本 vs QLoRA 微调 ====="
  # 7B 基座必须 4bit 加载（bf16 的 7B 是 15GB，装不下）
  run_eval models/Qwen2.5-7B-Instruct "" \
           data/processed/test_iid.jsonl base_7b_iid 4 "--load-in-4bit"
  run_eval models/Qwen2.5-7B-Instruct "" \
           data/processed/test_ood.jsonl base_7b_ood 4 "--load-in-4bit"
  run_eval models/Qwen2.5-7B-Instruct outputs/qwen25-7b-qlora/final_adapter \
           data/processed/test_iid.jsonl lora_7b_iid 4 "--load-in-4bit"
  run_eval models/Qwen2.5-7B-Instruct outputs/qwen25-7b-qlora/final_adapter \
           data/processed/test_ood.jsonl lora_7b_ood 4 "--load-in-4bit"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "report" ]; then
  log "===== 生成对比报告 ====="
  $PY scripts/make_report.py >> "$LOG" 2>&1 && log "✅ 报告已生成"
fi

log "===== 全部完成 ====="
