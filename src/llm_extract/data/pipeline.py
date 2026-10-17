"""数据流水线：pandas 清洗 → 校验 → 切分。

每一步都写清楚「为什么做」——这些规则各自对应一种具体的脏数据形态，
改动前需要先理解它要解决的真实场景。

输出（data/processed/）：
    train.jsonl / val.jsonl / test_iid.jsonl / test_ood.jsonl   —— 训练与评测直接消费
    dataset_stats.md                                            —— 数据统计表（写进报告）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from llm_extract.data.synth import (
    IN_DIST_FAMILIES,
    OOD_FAMILIES,
    Sample,
    SynthConfig,
    generate,
)
from llm_extract.prompts import build_messages, build_target
from llm_extract.schema import FIELD_ORDER, InquiryRecord, normalize_text

ROOT = Path(__file__).resolve().parents[3]

# 加急语义的线索词。用于两处：
#   1. is_urgent=True 的样本，正文必须真的含这些词；
#   2. is_urgent=False 的样本，**被引用的历史段落**不能含这些词（见 _has_quoted_urgency）。
URGENT_KW = "急|尽快|越快越好|压缩|催"

# 引用段落的标记：这些行属于「上一轮沟通」而非本次要求
_QUOTE_MARKERS = ("以下为对方原邮件", "历史邮件", "上一轮", "> ", "转发的原邮件")


def _has_quoted_urgency(text: str) -> bool:
    """判断正文的**引用段落**里是否含加急词。

    只检查被标记为引用的行，避免误伤正文里刻意设计的否定表述
    （「不着急」「别急着报」——那些是让模型判 False 的合理干扰项）。
    """
    for line in text.splitlines():
        s = line.strip()
        if s and any(m in s for m in _QUOTE_MARKERS) and re.search(URGENT_KW, s):
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# 1. 合成 → DataFrame
# ──────────────────────────────────────────────────────────────────

def samples_to_frame(samples: list[Sample]) -> pd.DataFrame:
    """把合成样本摊平成 DataFrame：文本一列，每个目标字段一列。

    摊平成宽表的好处：可以用 pandas 直接做「字段级缺失率」「数值分布」检查，
    也能一眼看出某个字段是不是被合成器写坏了。
    """
    rows: list[dict[str, Any]] = []
    for s in samples:
        rec = s.to_record()          # 顺便用 pydantic 校验一遍合成结果
        row: dict[str, Any] = {
            "text": s.text,
            "template_family": s.template_family,
            "skeleton_key": s.skeleton_key,
        }
        for f in FIELD_ORDER:
            row[f] = getattr(rec, f).value if hasattr(getattr(rec, f), "value") else getattr(rec, f)
        row["target_json"] = build_target(rec)
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# 2. 清洗
# ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, min_chars: int = 40, max_chars: int = 900) -> tuple[pd.DataFrame, dict[str, int]]:
    """返回清洗后的 DataFrame 与数据漏斗。

    漏斗的语义统一为「**该步骤执行后剩余多少条**」（键为步骤名），
    淘汰数由相邻两步相减得到——避免"有时记剩余、有时记淘汰"这种混用。
    """
    funnel: dict[str, int] = {"raw": len(df)}
    df = df.copy()

    def step(name: str) -> None:
        funnel[name] = len(df)

    # (a) 空白/换行规范化：合成时随机拼接可能产生连续空行
    #     ⚠️ pandas 3.0 默认用 pyarrow 字符串后端（RE2 正则引擎），RE2 **不支持 `\uXXXX` 转义**，
    #     会直接抛 ArrowInvalid。这里改用非 raw 字符串，让 Python 先把 `\u3000`
    #     和 `\n` 解析成真实字符，正则引擎收到的就是字面量。
    df["text"] = (
        df["text"].str.replace("[ \t\u3000]+", " ", regex=True)
        .str.replace("\n{3,}", "\n\n", regex=True)
        .str.strip()
    )
    df = df[df["text"].str.len() > 0]
    step("非空正文")

    # (b) 正文必须真的包含型号：合成器理论上保证，但这里是最终防线
    df = df[df.apply(lambda r: str(r["product_model"]) in r["text"], axis=1)]
    step("型号在正文中")

    # (c) 文本长度过滤：**放在去重前面**。
    #     真实流水线里也一样——先用最便宜的规则砍掉明显不可用的样本
    #     （空文本、超长邮件线程），再去跑代价更高的字符串比较。
    df["char_len"] = df["text"].str.len()
    df = df[(df["char_len"] >= min_chars) & (df["char_len"] <= max_chars)]
    step("长度在合理区间")

    # (d) 精确去重：完全相同的文本只留一条（不同标签的同文本会污染训练）
    df = df.drop_duplicates(subset=["text"], keep="first")
    step("精确去重后")

    # (e) 近重复去重：同一模板骨架 + 同公司 + 同型号，只保留一条。
    #     严格说这不是"重复"，但同骨架大量重复会让验证集指标虚高。
    df["_near_key"] = (
        df["skeleton_key"] + "##" + df["customer_name"].fillna("-") + "##" + df["product_model"]
    )
    df = df.drop_duplicates(subset=["_near_key"], keep="first")
    step("近重复去重后")

    # (f) 标签自洽性：is_urgent=True 时正文应含加急线索（防止合成器逻辑错位）
    df = df[~(df["is_urgent"] & ~df["text"].str.contains(URGENT_KW, regex=True))]
    step("加急标注自洽")

    # (g) 引用段落污染：is_urgent=False 时，正文里**被引用的历史邮件/历史沟通**
    #     不应含加急词。这类文本属于「上一轮的内容」，不代表本次要求，
    #     但模型无法区分——会让标注变成噪声（模型判 True 反而是对的）。
    #     实测：早期版本 OOD 集有 27 条因此矛盾，is_urgent 误报 18 条（12%）。
    #     注意这里只排查**引用行**，不碰「不着急」这类否定表述——
    #     后者是刻意设计的干扰项，模型本就应该判 False。
    df = df[~(~df["is_urgent"] & df["text"].apply(_has_quoted_urgency))]
    step("无引用加急污染")

    df = df.drop(columns=["_near_key"]).reset_index(drop=True)
    return df, funnel


# ──────────────────────────────────────────────────────────────────
# 3. token 长度统计与过滤
# ──────────────────────────────────────────────────────────────────

def add_token_length(df: pd.DataFrame, model_path: str | None, max_tokens: int) -> pd.DataFrame:
    """用真实 tokenizer 统计 prompt+target 的 token 数。

    没有 tokenizer 时退化为字符数估算（中文约 1 token ≈ 1.2 字符），保证流水线可离线跑通。
    """
    df = df.copy()
    tok = None
    if model_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - 离线时降级，不阻塞流水线
            print(f"[warn] 加载 tokenizer 失败（{exc}），退化为字符估算", file=sys.stderr)
            tok = None

    if tok is not None:
        def _n(row: pd.Series) -> int:
            msgs = build_messages(row["text"])
            prefix = "".join(m["content"] for m in msgs)
            return len(tok(prefix + row["target_json"])["input_ids"])

        df["n_tokens"] = df.apply(_n, axis=1)
        df["token_count_method"] = "tokenizer"
    else:
        df["n_tokens"] = ((df["text"].str.len() + df["target_json"].str.len()) / 1.2).round().astype(int)
        df["token_count_method"] = "char_estimate"

    df["too_long"] = df["n_tokens"] > max_tokens
    return df


# ──────────────────────────────────────────────────────────────────
# 4. 切分（严格隔离）
# ──────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    n_val: int,
    n_test_iid: int,
    n_test_ood: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """切分策略：

    - **OOD 测试集**：只取训练集完全没出现过的模板族（电话记录 / 邮件回复）。
      指标掉多少，直接反映"换个表述方式还能不能抽对"。
    - **IID 测试集**：同模板族、不同 skeleton_key（用 skeleton 做分组，禁止同骨架跨集）。
    - 训练/验证：只从分布内模板族里取。
    """
    in_dist = df[df["template_family"].isin(IN_DIST_FAMILIES)].copy()
    ood_pool = df[df["template_family"].isin(OOD_FAMILIES)].copy()

    # OOD 集：优先挑最长的（长文本更难），保证难度不虚低
    ood = ood_pool.sort_values("n_tokens", ascending=False).head(n_test_ood).copy()

    # 按 skeleton_key 分组，避免同骨架样本同时出现在 train / test。
    # 只留出 test_iid 所需的那部分骨架，其余全部投入 train/val —— 不要凭白浪费数据。
    skeletons = sorted(in_dist["skeleton_key"].unique())
    shuffled = pd.Series(skeletons).sample(frac=1.0, random_state=seed).tolist()
    n_iid_groups = max(1, int(len(shuffled) * 0.15))
    iid_groups = set(shuffled[:n_iid_groups])

    test_iid = in_dist[in_dist["skeleton_key"].isin(iid_groups)].head(n_test_iid).copy()
    rest = in_dist[~in_dist["skeleton_key"].isin(iid_groups)].copy()

    val = rest.sample(n=min(n_val, len(rest) // 10), random_state=seed)
    train = rest.drop(val.index)

    out = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test_iid": test_iid.reset_index(drop=True),
        "test_ood": ood.reset_index(drop=True),
    }

    # 硬校验：模板族与骨架都不允许跨集泄漏
    assert not (set(out["train"]["template_family"]) & set(out["test_ood"]["template_family"])), \
        "OOD 模板族泄漏进训练集"
    assert not (set(out["train"]["skeleton_key"]) & set(out["test_iid"]["skeleton_key"])), \
        "IID 骨架泄漏进训练集"
    return out


# ──────────────────────────────────────────────────────────────────
# 5. 落盘
# ──────────────────────────────────────────────────────────────────

def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame → 记录列表，**并把 NaN 还原成 None、把数值还原成正确的 Python 类型**。

    ⚠️ 两个坑都必须处理：

    1. pandas 会把 Python 的 `None` 统一转成 `NaN`（float），直接 `json.dumps` 会写出
       `NaN`——那不是合法 JSON，`json.loads` 读回来是 `float('nan')` 而不是 `None`。
       后果是评测时「标注为 null 的字段」根本识别不出来，幻觉率会恒等于 0（假指标）。
    2. int 列一旦出现 NaN 就整列升格成 float64，`quantity` 会变成 `20000.0`。
       对下游无伤大雅，但和 schema 声明的 `int` 不符，也会让人怀疑数据质量。
       这里按字段声明类型显式还原。
    """
    obj = df[FIELD_ORDER + ["text", "template_family"]].astype(object)
    obj = obj.where(pd.notna(obj), None)

    records: list[dict[str, Any]] = []
    for row in obj.to_dict("records"):
        rec: dict[str, Any] = {f: row[f] for f in FIELD_ORDER}
        rec["text"] = row["text"]
        rec["template_family"] = row["template_family"]
        # 按 schema 声明的类型还原标量
        if rec["quantity"] is not None:
            rec["quantity"] = int(rec["quantity"])
        if rec["unit_price"] is not None:
            rec["unit_price"] = float(rec["unit_price"])
        if rec["is_urgent"] is not None:
            rec["is_urgent"] = bool(rec["is_urgent"])
        records.append(rec)
    return records


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in frame_to_records(df):
            # allow_nan=False：宁可在这里报错，也不要把 NaN 写进数据集
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")


def write_stats_md(
    path: Path,
    funnel: dict[str, int],
    splits: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    cfg: SynthConfig,
) -> None:
    lines: list[str] = []
    lines.append("# 数据集统计\n")
    lines.append(f"- 随机种子：`{cfg.seed}`")
    lines.append(f"- token 统计方式：`{df['token_count_method'].iloc[0]}`\n")

    lines.append("## 1. 数据漏斗（清洗每一步剩下多少条）\n")
    lines.append("| 步骤 | 剩余样本数 | 本步淘汰 | 说明 |")
    lines.append("| :--- | ---: | ---: | :--- |")
    notes = {
        "raw": "合成器原始产出（已含刻意注入的脏数据）",
        "非空正文": "去除空白文本",
        "型号在正文中": "型号是必填字段，正文里找不到说明标签不可信",
        "长度在合理区间": "超短信息不全 / 超长（历史邮件线程）浪费显存",
        "精确去重后": "完全相同的正文（重复投递），保留首条",
        "近重复去重后": "同骨架 + 同公司 + 同型号（重复录入），保留首条",
        "加急标注自洽": "is_urgent=True 但正文无加急措辞",
        "无引用加急污染": "is_urgent=False 但被引用的历史段落含加急词（标注噪声）",
        "token 长度合规": "prompt+target 超过 max_seq_length",
    }
    keys = list(funnel)
    for i, k in enumerate(keys):
        kept = funnel[k]
        dropped = "" if i == 0 else str(funnel[keys[i - 1]] - kept)
        lines.append(f"| {k} | {kept} | {dropped} | {notes.get(k, '')} |")

    lines.append("\n## 2. 切分结果\n")
    lines.append("| 数据集 | 样本数 | 模板族 | 平均 token | 最大 token |")
    lines.append("| :--- | ---: | :--- | ---: | ---: |")
    for name, sub in splits.items():
        fams = ", ".join(sorted(sub["template_family"].unique()))
        lines.append(
            f"| {name} | {len(sub)} | {fams} | {sub['n_tokens'].mean():.0f} | {sub['n_tokens'].max()} |"
        )

    lines.append("\n## 3. 字段缺失率（可空字段占比）\n")
    lines.append("| 字段 | train | val | test_iid | test_ood |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for f in FIELD_ORDER:
        cells = []
        for name in ["train", "val", "test_iid", "test_ood"]:
            sub = splits[name]
            cells.append(f"{sub[f].isna().mean() * 100:.1f}%")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")

    lines.append("\n## 4. 关键字段取值分布（训练集）\n")
    tr = splits["train"]
    lines.append("| 字段 | 分布 |")
    lines.append("| :--- | :--- |")
    lines.append(f"| currency | {tr['currency'].value_counts().to_dict()} |")
    lines.append(f"| is_urgent | {tr['is_urgent'].value_counts().to_dict()} |")
    lines.append(f"| payment_terms | {tr['payment_terms'].value_counts(dropna=False).to_dict()} |")
    lines.append(f"| quantity | min={tr['quantity'].min()}, median={tr['quantity'].median()}, max={tr['quantity'].max()} |")
    lines.append(f"| unit_price | min={tr['unit_price'].min()}, median={tr['unit_price'].median()}, max={tr['unit_price'].max()} |")

    lines.append("\n## 5. 文本长度分布（字符数）\n")
    lines.append("| 分位 | train | test_ood |")
    lines.append("| :--- | ---: | ---: |")
    for q, label in [(0.0, "min"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (1.0, "max")]:
        lines.append(
            f"| {label} | {tr['char_len'].quantile(q):.0f} | {splits['test_ood']['char_len'].quantile(q):.0f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="合成并清洗结构化抽取数据集")
    ap.add_argument("--n", type=int, default=3000, help="合成样本数（清洗前）")
    ap.add_argument("--seed", type=int, default=20251015)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--tokenizer", type=str, default=None, help="tokenizer 路径或 HF id")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--n-test-iid", type=int, default=200)
    ap.add_argument("--n-test-ood", type=int, default=150)
    args = ap.parse_args()

    cfg = SynthConfig(seed=args.seed)
    print(f"[1/5] 合成 {args.n} 条原始样本 ...")
    samples = generate(args.n, cfg, IN_DIST_FAMILIES + OOD_FAMILIES)
    df = samples_to_frame(samples)

    print("[2/5] pandas 清洗 ...")
    df, funnel = clean(df)
    keys = list(funnel)
    for i, k in enumerate(keys):
        dropped = "" if i == 0 else f"  (-{funnel[keys[i - 1]] - funnel[k]})"
        print(f"      {k:<16} {funnel[k]:>5}{dropped}")

    print("[3/5] token 长度统计 ...")
    df = add_token_length(df, args.tokenizer, args.max_tokens)
    df = df[~df["too_long"]].reset_index(drop=True)
    prev = funnel[list(funnel)[-1]]
    funnel["token 长度合规"] = len(df)
    print(f"      {'token 长度合规':<16} {len(df):>5}  (-{prev - len(df)})")

    print("[4/5] 按模板族 / 骨架严格切分 ...")
    splits = split_dataset(df, args.n_val, args.n_test_iid, args.n_test_ood, args.seed)
    for name, sub in splits.items():
        print(f"      {name}: {len(sub)}")

    print("[5/5] 落盘 ...")
    for name, sub in splits.items():
        write_jsonl(sub, args.out_dir / f"{name}.jsonl")
    write_stats_md(ROOT / "reports" / "dataset_stats.md", funnel, splits, df, cfg)
    print(f"完成 → {args.out_dir} / reports/dataset_stats.md")


if __name__ == "__main__":
    main()
