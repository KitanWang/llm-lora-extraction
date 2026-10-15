"""评测指标：JSON 合法性 / schema 合规 / 字段级准确率 / 幻觉率。

指标定义（写进报告的就是这一套，别处不再重定义）：

| 指标 | 定义 |
| :--- | :--- |
| `json_valid_rate`   | 输出能被解析成 JSON 对象的比例（允许 ```json 围栏） |
| `schema_valid_rate` | 解析成功 **且** 通过 pydantic schema（枚举合法、无多余字段、类型正确） |
| `field_acc.<f>`     | 单字段准确率，仅统计"可比对"的样本；字符串做归一化宽松匹配，数值容差 1e-6 |
| `exact_match_rate`  | 9 个字段全部正确的样本比例（最严苛的指标） |
| `hallucination_rate`| 在"标注为 null"的字段上，模型**编造了非空值**的比例 —— 抽取任务里最危险的错误 |
| `omission_rate`     | 在"标注非 null"的字段上，模型输出 null 的比例 |
| `enum_confusion`    | 枚举字段（currency / payment_terms）的错分方向，用于定位系统性问题 |
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from llm_extract.schema import (
    FIELD_ORDER,
    FIELD_SPECS,
    InquiryRecord,
    normalize_company,
    normalize_digits,
    normalize_text,
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


# ──────────────────────────────────────────────────────────────────
# 解析
# ──────────────────────────────────────────────────────────────────

def extract_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出里尽力抠出一个 JSON 对象。抠不出来返回 None。"""
    if not raw:
        return None
    text = raw.strip()

    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()

    # 直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 退而求其次：取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ──────────────────────────────────────────────────────────────────
# 字段比对
# ──────────────────────────────────────────────────────────────────

def _to_comparable(field_name: str, value: Any) -> Any:
    """把预测值与标注值都转成可比较形式。预测值可能类型不对（如 "100" vs 100）。"""
    if value is None:
        return None
    spec = FIELD_SPECS[field_name]
    kind = spec["kind"]
    try:
        if kind == "number":
            if isinstance(value, bool):
                return None
            if isinstance(value, str):
                v = re.sub(r"[^\d.\-]", "", value)
                return float(v) if v not in ("", "-", ".") else None
            return float(value)
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "是", "1", "yes", "加急"):
                    return True
                if low in ("false", "否", "0", "no", "不加急"):
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            return None
        if kind == "str":
            return str(value).strip() or None
        return value
    except (TypeError, ValueError):
        return None


def values_equal(field_name: str, pred: Any, gold: Any) -> bool:
    spec = FIELD_SPECS[field_name]
    p = _to_comparable(field_name, pred)
    g = _to_comparable(field_name, gold)
    if g is None or p is None:
        return p is None and g is None

    match = spec["match"]
    if match == "numeric_tol":
        return abs(float(p) - float(g)) < 1e-6
    if match == "normalized_str":
        if field_name == "customer_name":
            return normalize_company(str(p)) == normalize_company(str(g))
        return normalize_text(str(p)) == normalize_text(str(g))
    if match == "digits":
        return normalize_digits(str(p)) == normalize_digits(str(g))
    return p == g


# ──────────────────────────────────────────────────────────────────
# 逐条评测
# ──────────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    idx: int
    raw_output: str
    gold: dict[str, Any]
    template_family: str = ""
    pred: dict[str, Any] | None = None
    json_ok: bool = False
    schema_ok: bool = False
    schema_error: str = ""
    field_correct: dict[str, bool] = field(default_factory=dict)
    exact_match: bool = False


def evaluate_one(
    idx: int,
    raw_output: str,
    gold: dict[str, Any],
    template_family: str = "",
) -> PredictionRecord:
    rec = PredictionRecord(idx=idx, raw_output=raw_output, gold=gold, template_family=template_family)

    obj = extract_json_object(raw_output)
    if obj is None:
        rec.field_correct = {f: False for f in FIELD_ORDER}
        return rec

    rec.pred = obj
    rec.json_ok = True

    # schema 校验：多余字段 / 枚举越界 / 类型错误都算不合规
    try:
        validated = InquiryRecord.model_validate(obj)
        rec.schema_ok = True
        normalized_pred = validated.to_target_json()
    except ValidationError as exc:
        rec.schema_error = str(exc).splitlines()[0][:180]
        # schema 不合法时仍做字段级比对（如 {} 里缺字段按错处理）
        normalized_pred = obj

    for f in FIELD_ORDER:
        rec.field_correct[f] = values_equal(f, normalized_pred.get(f), gold.get(f))
    rec.exact_match = all(rec.field_correct.values())
    return rec


# ──────────────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────────────

def summarize(records: list[PredictionRecord]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        raise ValueError("没有评测记录")

    out: dict[str, Any] = {
        "n": n,
        "json_valid_rate": sum(r.json_ok for r in records) / n,
        "schema_valid_rate": sum(r.schema_ok for r in records) / n,
        "exact_match_rate": sum(r.exact_match for r in records) / n,
        "field_acc": {},
        "hallucination_rate": {},
        "omission_rate": {},
    }

    for f in FIELD_ORDER:
        out["field_acc"][f] = sum(r.field_correct.get(f, False) for r in records) / n

        # 幻觉：标注 null，模型给了非空值
        null_gold = [r for r in records if r.gold.get(f) is None]
        if null_gold:
            halluc = sum(
                1 for r in null_gold
                if r.pred is not None and _to_comparable(f, r.pred.get(f)) is not None
            )
            out["hallucination_rate"][f] = halluc / len(null_gold)

        # 漏抽：标注非 null，模型给了 null
        nonnull_gold = [r for r in records if r.gold.get(f) is not None]
        if nonnull_gold:
            omit = sum(
                1 for r in nonnull_gold
                if r.pred is None or _to_comparable(f, r.pred.get(f)) is None
            )
            out["omission_rate"][f] = omit / len(nonnull_gold)

    # 枚举错分方向
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r.pred is None:
            continue
        for f in ("currency", "payment_terms"):
            g, p = r.gold.get(f), r.pred.get(f)
            if not values_equal(f, p, g):
                confusion[f][f"{g} -> {p}"] += 1
    out["enum_confusion"] = {k: dict(v.most_common(8)) for k, v in confusion.items()}

    # 按模板族分组（看 OOD 上掉多少）
    by_family: dict[str, dict[str, float]] = {}
    fams = sorted({r.template_family for r in records if r.template_family})
    for fam in fams:
        sub = [r for r in records if r.template_family == fam]
        by_family[fam] = {
            "n": len(sub),
            "schema_valid_rate": sum(x.schema_ok for x in sub) / len(sub),
            "exact_match_rate": sum(x.exact_match for x in sub) / len(sub),
            "avg_field_acc": sum(
                sum(x.field_correct.values()) / len(FIELD_ORDER) for x in sub
            ) / len(sub),
        }
    out["by_template_family"] = by_family

    # schema 错误样本举例（写报告用）
    errs = Counter(r.schema_error for r in records if r.schema_error)
    out["schema_error_examples"] = dict(errs.most_common(5))
    return out


def format_table(results: dict[str, dict[str, Any]], metric: str = "exact_match_rate") -> str:
    """把多个模型的结果渲染成 markdown 对比表（报告直接贴）。"""
    models = list(results.keys())
    lines = [
        f"| 模型 | JSON 合法率 | Schema 合规率 | 全字段准确率({metric}) | 字段平均准确率 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        s = results[m]
        avg = sum(s["field_acc"].values()) / len(FIELD_ORDER)
        lines.append(
            f"| {m} | {s['json_valid_rate'] * 100:.1f}% | {s['schema_valid_rate'] * 100:.1f}% "
            f"| {s[metric] * 100:.1f}% | {avg * 100:.1f}% |"
        )
    return "\n".join(lines)
