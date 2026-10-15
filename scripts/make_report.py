"""把多个 summary_*.json 汇总成最终对比报告。

用法：
    uv run python scripts/make_report.py
读 reports/summary_*.json，产出 reports/RESULTS.md（含对比表、分模板族明细、耗时与显存）。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

FIELD_ORDER = [
    "customer_name", "product_model", "quantity", "unit_price", "currency",
    "delivery_date", "payment_terms", "is_urgent", "contact_phone",
]
FIELD_CN = {
    "customer_name": "客户名称",
    "product_model": "产品型号",
    "quantity": "数量",
    "unit_price": "单价",
    "currency": "币种",
    "delivery_date": "交付日期",
    "payment_terms": "付款条件",
    "is_urgent": "是否加急",
    "contact_phone": "联系电话",
}

# 展示顺序：基座 → 微调后，IID → OOD
ORDER = ["base_iid", "lora_iid", "base_ood", "lora_ood"]


def load_summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(REPORTS.glob("summary_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        out[data.get("tag", p.stem)] = data
    return out


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def main() -> None:
    sums = load_summaries()
    if not sums:
        raise SystemExit("reports/ 下没有 summary_*.json，请先跑评测")

    tags = [t for t in ORDER if t in sums] + [t for t in sums if t not in ORDER]
    lines: list[str] = []
    add = lines.append

    add("# 实验结果：基座模型 vs LoRA 微调后\n")
    add("> 全部指标在同一套评测集、同一套 prompt、**贪心解码**下取得，可复现。")
    add("> 指标定义见 `src/llm_extract/evaluate/metrics.py` 顶部注释。\n")

    # ── 表 1：总体指标 ──
    add("## 一、总体指标对比\n")
    add("| 评测 | 模型 | 数据集 | JSON 合法率 | Schema 合规率 | **全字段准确率** | 字段平均准确率 |")
    add("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    for t in tags:
        s = sums[t]
        avg = sum(s["field_acc"].values()) / len(FIELD_ORDER)
        ds = "IID（同模板族）" if t.endswith("iid") else "OOD（未见模板族）"
        model = Path(s["model"]).name if "/" in s["model"] else s["model"]
        model += " + LoRA" if s.get("adapter") else "（零样本基座）"
        add(f"| `{t}` | {model} | {ds} | {pct(s['json_valid_rate'])} | "
            f"{pct(s['schema_valid_rate'])} | **{pct(s['exact_match_rate'])}** | {pct(avg)} |")

    # ── 表 2：字段级 ──
    add("\n## 二、字段级准确率\n")
    header = "| 字段 | " + " | ".join(f"`{t}`" for t in tags) + " |"
    add(header)
    add("| :--- | " + " | ".join("---:" for _ in tags) + " |")
    for f in FIELD_ORDER:
        row = f"| {FIELD_CN[f]} `{f}` | " + " | ".join(
            pct(sums[t]["field_acc"].get(f)) for t in tags
        ) + " |"
        add(row)

    # ── 表 3：幻觉率与漏抽率 ──
    add("\n## 三、幻觉率（标注为 null 的字段被编造出值的比例）\n")
    add("> 这是抽取任务里最危险的错误：编一个看起来合理的值，比输出 null 危害大得多。\n")
    add("| 字段 | " + " | ".join(f"`{t}`" for t in tags) + " |")
    add("| :--- | " + " | ".join("---:" for _ in tags) + " |")
    for f in FIELD_ORDER:
        vals = [sums[t].get("hallucination_rate", {}).get(f) for t in tags]
        if all(v is None for v in vals):
            continue
        add(f"| {FIELD_CN[f]} | " + " | ".join(pct(v) for v in vals) + " |")

    add("\n## 四、漏抽率（标注非 null 但输出 null 的比例）\n")
    add("| 字段 | " + " | ".join(f"`{t}`" for t in tags) + " |")
    add("| :--- | " + " | ".join("---:" for _ in tags) + " |")
    for f in FIELD_ORDER:
        vals = [sums[t].get("omission_rate", {}).get(f) for t in tags]
        if all(v is None for v in vals):
            continue
        add(f"| {FIELD_CN[f]} | " + " | ".join(pct(v) for v in vals) + " |")

    # ── 表 5：按模板族 ──
    add("\n## 五、按文本模板族细分（看模型在哪种表述上最弱）\n")
    families = sorted({
        fam for t in tags for fam in sums[t].get("by_template_family", {})
    })
    if families:
        add("| 模板族 | " + " | ".join(f"`{t}`" for t in tags) + " |")
        add("| :--- | " + " | ".join("---:" for _ in tags) + " |")
        for fam in families:
            cells = []
            for t in tags:
                d = sums[t].get("by_template_family", {}).get(fam)
                cells.append(f"{pct(d['exact_match_rate'])} (n={d['n']})" if d else "—")
            add(f"| `{fam}` | " + " | ".join(cells) + " |")

    # ── 表 6：枚举错分 ──
    add("\n## 六、枚举字段错分方向（Top 错误）\n")
    for t in tags:
        conf = sums[t].get("enum_confusion", {})
        if not any(conf.values()):
            continue
        add(f"\n**`{t}`**\n")
        add("| 字段 | 错误模式 | 次数 |")
        add("| :--- | :--- | ---: |")
        for f, d in conf.items():
            for k, v in list(d.items())[:6]:
                add(f"| {f} | `{k}` | {v} |")

    # ── 表 7：成本 ──
    add("\n## 七、推理成本\n")
    add("| 评测 | 样本数 | 总耗时(s) | 单条耗时(s) | 峰值显存(GiB) |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    for t in tags:
        s = sums[t]
        add(f"| `{t}` | {s['n']} | {s.get('elapsed_sec', '—')} | "
            f"{s.get('sec_per_sample', '—')} | {s.get('peak_gpu_gib', '—')} |")

    # ── 训练成本 ──
    train_metrics = sorted(REPORTS.glob("train_metrics_*.json"))
    if train_metrics:
        add("\n## 八、训练成本\n")
        add("| 实验 | 基座 | 可训练参数 | 占比 | 峰值显存(GiB) | 训练耗时(s) | 最终 train loss |")
        add("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
        for p in train_metrics:
            d = json.loads(p.read_text(encoding="utf-8"))
            cfg = d.get("config", {})
            tm = d.get("train_metrics", {})
            add(f"| {p.stem.replace('train_metrics_', '')} | {cfg.get('model_name_or_path', '—')} | "
                f"{d.get('param_trainable', 0):,} | {d.get('trainable_ratio_pct', 0):.4f}% | "
                f"{d.get('peak_gpu_gib', '—')} | {tm.get('train_runtime', 0):.0f} | "
                f"{tm.get('train_loss', float('nan')):.4f} |")

    # ── schema 错误举例 ──
    add("\n## 九、Schema 不合规的典型原因\n")
    for t in tags:
        errs = sums[t].get("schema_error_examples", {})
        if not errs:
            continue
        add(f"\n**`{t}`**\n")
        for msg, cnt in errs.items():
            add(f"- 出现 {cnt} 次：`{msg}`")

    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {out}（{len(tags)} 个评测结果）")


if __name__ == "__main__":
    main()
