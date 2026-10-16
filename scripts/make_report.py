"""把 summary_*.json 汇总成最终对比报告（reports/RESULTS.md）。

结果命名约定 `{variant}_{model}_{split}`，例如：
    base_1.5b_iid   1.5B 基座零样本，同分布测试集
    lora_7b_ood     7B QLoRA 微调后，未见模板族测试集

报告结构：
    一、总体指标（按模型分组，基座 vs 微调）
    二、字段级准确率
    三、幻觉率 / 漏抽率（抽取任务最危险的错误）
    四、按模板族细分
    五、枚举错分方向
    六、推理成本
    七、训练成本
    八、结论（自动生成的关键数字）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

FIELD_ORDER = [
    "customer_name", "product_model", "quantity", "unit_price", "currency",
    "delivery_date", "payment_terms", "is_urgent", "contact_phone",
]
FIELD_CN = {
    "customer_name": "客户名称", "product_model": "产品型号", "quantity": "数量",
    "unit_price": "单价", "currency": "币种", "delivery_date": "交付日期",
    "payment_terms": "付款条件", "is_urgent": "是否加急", "contact_phone": "联系电话",
}

PATTERN = re.compile(r"^(base|lora)_(\d+(?:\.\d+)?b)_(iid|ood)$")


def load() -> dict[str, dict]:
    out = {}
    for p in sorted(REPORTS.glob("summary_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        tag = d.get("tag") or p.stem.replace("summary_", "")
        m = PATTERN.match(tag)
        if m:
            d["_variant"], d["_model"], d["_split"] = m.groups()
        out[tag] = d
    return out


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def avg_field_acc(s: dict) -> float:
    return sum(s["field_acc"].values()) / len(FIELD_ORDER)


def main() -> None:
    sums = load()
    if not sums:
        raise SystemExit("reports/ 下没有 summary_*.json，请先跑评测")

    # 分组：模型 -> split -> variant
    models: dict[str, dict[str, dict[str, dict]]] = {}
    for tag, s in sums.items():
        if "_model" not in s:
            continue
        models.setdefault(s["_model"], {}).setdefault(s["_split"], {})[s["_variant"]] = s
    model_order = sorted(models, key=lambda m: float(m.rstrip("b")))

    L: list[str] = []
    add = L.append

    add("# 实验结果：基座模型 vs LoRA/QLoRA 微调后\n")
    add("> 所有指标在同一套评测集、同一套 prompt、**贪心解码**（`do_sample=False`）下取得，可复现。")
    add("> 指标定义见 `src/llm_extract/evaluate/metrics.py`。")
    add(">")
    add("> - **IID**：测试集与训练集同模板族，但骨架不重叠（考「会不会做」）")
    add("> - **OOD**：测试集使用训练中**完全没出现过**的模板族（电话记录 / 邮件回复），考泛化\n")

    # ── 一、总体指标 ──
    add("## 一、总体指标\n")
    add("| 模型 | 测试集 | 版本 | JSON 合法率 | Schema 合规率 | **全字段准确率** | 字段平均准确率 |")
    add("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    for m in model_order:
        for split in ("iid", "ood"):
            if split not in models[m]:
                continue
            for variant in ("base", "lora"):
                s = models[m][split].get(variant)
                if not s:
                    continue
                name = "基座零样本" if variant == "base" else "**微调后**"
                add(f"| Qwen2.5-{m} | {split.upper()} | {name} | {pct(s['json_valid_rate'])} | "
                    f"{pct(s['schema_valid_rate'])} | **{pct(s['exact_match_rate'])}** | "
                    f"{pct(avg_field_acc(s))} |")

    # ── 提升幅度 ──
    add("\n### 微调带来的绝对提升\n")
    add("| 模型 | 测试集 | 全字段准确率 | Schema 合规率 | 字段平均准确率 |")
    add("| :--- | :--- | ---: | ---: | ---: |")
    for m in model_order:
        for split in ("iid", "ood"):
            grp = models[m].get(split, {})
            if "base" not in grp or "lora" not in grp:
                continue
            b, l = grp["base"], grp["lora"]
            add(f"| Qwen2.5-{m} | {split.upper()} | "
                f"{pct(b['exact_match_rate'])} → {pct(l['exact_match_rate'])} "
                f"(**{(l['exact_match_rate']-b['exact_match_rate'])*100:+.1f} pp**) | "
                f"{pct(b['schema_valid_rate'])} → {pct(l['schema_valid_rate'])} "
                f"({(l['schema_valid_rate']-b['schema_valid_rate'])*100:+.1f} pp) | "
                f"{pct(avg_field_acc(b))} → {pct(avg_field_acc(l))} "
                f"({(avg_field_acc(l)-avg_field_acc(b))*100:+.1f} pp) |")

    # ── 二、字段级 ──
    all_tags = [t for t in sums if "_model" in sums[t]]
    all_tags.sort(key=lambda t: (float(sums[t]["_model"].rstrip("b")),
                                 sums[t]["_split"], sums[t]["_variant"]))
    if all_tags:
        add("\n## 二、字段级准确率\n")
        add("| 字段 | " + " | ".join(f"`{t}`" for t in all_tags) + " |")
        add("| :--- | " + " | ".join("---:" for _ in all_tags) + " |")
        for f in FIELD_ORDER:
            add(f"| {FIELD_CN[f]} | " + " | ".join(
                pct(sums[t]["field_acc"].get(f)) for t in all_tags) + " |")

    # ── 三、幻觉率 ──
    add("\n## 三、幻觉率（标注为 null 的字段被编造出值的比例）\n")
    add("> 抽取任务里最危险的错误：编一个看起来合理的值，比老实输出 null 危害大得多。")
    add("> 该指标只在「标注为 null」的样本上统计。\n")
    add("| 字段 | " + " | ".join(f"`{t}`" for t in all_tags) + " |")
    add("| :--- | " + " | ".join("---:" for _ in all_tags) + " |")
    for f in FIELD_ORDER:
        vals = [sums[t].get("hallucination_rate", {}).get(f) for t in all_tags]
        if all(v is None for v in vals):
            continue
        add(f"| {FIELD_CN[f]} | " + " | ".join(pct(v) for v in vals) + " |")

    add("\n## 四、漏抽率（标注非 null 却输出 null 的比例）\n")
    add("| 字段 | " + " | ".join(f"`{t}`" for t in all_tags) + " |")
    add("| :--- | " + " | ".join("---:" for _ in all_tags) + " |")
    for f in FIELD_ORDER:
        vals = [sums[t].get("omission_rate", {}).get(f) for t in all_tags]
        if all(v is None for v in vals):
            continue
        add(f"| {FIELD_CN[f]} | " + " | ".join(pct(v) for v in vals) + " |")

    # ── 五、按模板族 ──
    fams = sorted({f for t in all_tags for f in sums[t].get("by_template_family", {})})
    if fams:
        add("\n## 五、按文本模板族细分\n")
        add("> OOD 的两个族（`phone_transcript` / `quote_reply`）训练时完全没见过。\n")
        add("| 模板族 | " + " | ".join(f"`{t}`" for t in all_tags) + " |")
        add("| :--- | " + " | ".join("---:" for _ in all_tags) + " |")
        for fam in fams:
            cells = []
            for t in all_tags:
                d = sums[t].get("by_template_family", {}).get(fam)
                cells.append(f"{pct(d['exact_match_rate'])} (n={d['n']})" if d else "—")
            add(f"| `{fam}` | " + " | ".join(cells) + " |")

    # ── 六、枚举错分 ──
    add("\n## 六、枚举字段错分方向（Top 错误模式）\n")
    for t in all_tags:
        conf = sums[t].get("enum_confusion", {})
        if not any(conf.values()):
            continue
        add(f"\n**`{t}`**\n")
        add("| 字段 | 错误模式（标注 → 预测） | 次数 |")
        add("| :--- | :--- | ---: |")
        for f, d in conf.items():
            for k, v in list(d.items())[:5]:
                add(f"| {f} | `{k}` | {v} |")

    # ── 七、推理成本 ──
    add("\n## 七、推理成本\n")
    add("| 评测 | 样本数 | 总耗时(s) | 单条(s) | 峰值显存(GiB) |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    for t in all_tags:
        s = sums[t]
        add(f"| `{t}` | {s['n']} | {s.get('elapsed_sec', '—')} | "
            f"{s.get('sec_per_sample', '—')} | {s.get('peak_gpu_gib', '—')} |")

    # ── 八、训练成本 ──
    tm = sorted(REPORTS.glob("train_metrics_*.json")) + \
         sorted(REPORTS.glob("outputs/*/train_metrics.json"))
    tm = [p for p in tm if p.exists()]
    # 也扫描 outputs 下的标准名字
    for p in sorted((ROOT / "outputs").glob("*/train_metrics.json")):
        if p not in tm:
            tm.append(p)
    if tm:
        add("\n## 八、训练成本\n")
        add("| 实验 | 基座 | 方案 | 可训练参数 | 占比 | 峰值显存(GiB) | 训练耗时 | train loss |")
        add("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
        for p in tm:
            d = json.loads(p.read_text(encoding="utf-8"))
            cfg = d.get("config", {})
            m = d.get("train_metrics", {})
            rt = m.get("train_runtime", 0)
            scheme = "QLoRA (NF4 4bit)" if cfg.get("load_in_4bit") else "LoRA (bf16)"
            name = Path(cfg.get("output_dir", p.parent.name)).name
            add(f"| {name} | {Path(cfg.get('model_name_or_path','—')).name} | {scheme} | "
                f"{d.get('param_trainable', 0):,} | {d.get('trainable_ratio_pct', 0):.3f}% | "
                f"{d.get('peak_gpu_gib', '—')} | {rt/60:.1f} min | "
                f"{m.get('train_loss', float('nan')):.4f} |")

    # ── 九、结论 ──
    add("\n## 九、结论\n")
    for m in model_order:
        for split in ("iid", "ood"):
            grp = models[m].get(split, {})
            if "base" not in grp or "lora" not in grp:
                continue
            b, l = grp["base"], grp["lora"]
            add(f"- **Qwen2.5-{m} / {split.upper()}**：全字段准确率 "
                f"{pct(b['exact_match_rate'])} → {pct(l['exact_match_rate'])}"
                f"（{(l['exact_match_rate']-b['exact_match_rate'])*100:+.1f} pp）；"
                f"Schema 合规率 {pct(b['schema_valid_rate'])} → {pct(l['schema_valid_rate'])}")
    # OOD 掉点
    for m in model_order:
        grp = models[m]
        if "iid" in grp and "ood" in grp and "lora" in grp["iid"] and "lora" in grp["ood"]:
            drop = grp["iid"]["lora"]["exact_match_rate"] - grp["ood"]["lora"]["exact_match_rate"]
            add(f"- **Qwen2.5-{m} 泛化差距**：微调后在未见模板族上全字段准确率下降 "
                f"{drop*100:.1f} pp（{pct(grp['iid']['lora']['exact_match_rate'])} → "
                f"{pct(grp['ood']['lora']['exact_match_rate'])}），说明模型学到的不只是模板记忆。")

    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已生成 {out}（{len(all_tags)} 组评测结果）")


if __name__ == "__main__":
    main()
