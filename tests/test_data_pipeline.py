"""数据合成器与清洗管线的单元测试。

合成数据的最大风险是「标签和正文对不上」——那种数据训出来的模型会一本正经地抽错。
这里把最容易出错的几条不变量钉死。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from llm_extract.data.pipeline import (
    clean,
    frame_to_records,
    samples_to_frame,
    split_dataset,
    write_jsonl,
)
from llm_extract.data.synth import IN_DIST_FAMILIES, OOD_FAMILIES, SynthConfig, generate
from llm_extract.schema import FIELD_ORDER, InquiryRecord, normalize_company


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    cfg = SynthConfig(seed=7)
    samples = generate(400, cfg, IN_DIST_FAMILIES + OOD_FAMILIES)
    return samples_to_frame(samples)


class TestSynthInvariants:
    def test_all_samples_pass_schema(self, raw_df):
        """每条合成样本都必须能通过 pydantic 校验，否则标签本身就是错的。

        注意要用 frame_to_records 而不是 df.to_dict()：后者会把 pandas 的 NaN
        原样带进校验，那是 pandas 的表示问题，不是合成器的错误。
        """
        for target in frame_to_records(raw_df):
            InquiryRecord.model_validate({k: target[k] for k in FIELD_ORDER})

    def test_model_string_appears_in_text(self, raw_df):
        """型号必须真的能在正文里找到（它是必填字段）。"""
        bad = raw_df[~raw_df.apply(lambda r: r["product_model"] in r["text"], axis=1)]
        assert bad.empty, f"{len(bad)} 条样本的型号不在正文里"

    def test_phone_is_valid_when_present(self, raw_df):
        phones = raw_df["contact_phone"].dropna()
        assert len(phones) > 0
        for p in phones:
            assert p.isdigit() and len(p) in (10, 11, 12), f"非法电话 {p}"

    def test_quantity_is_positive_int_when_present(self, raw_df):
        q = raw_df["quantity"].dropna()
        assert (q > 0).all()
        assert (q % 1 == 0).all()

    def test_urgent_true_implies_urgent_wording(self, raw_df):
        urgent = raw_df[raw_df["is_urgent"]]
        assert not urgent.empty
        assert urgent["text"].str.contains("急|尽快|优先|越快越好|压缩|催").all()

    def test_nullable_fields_actually_have_nulls(self, raw_df):
        """可空字段必须真的出现 null，否则测不出幻觉率。"""
        for f in ("unit_price", "delivery_date", "payment_terms", "contact_phone"):
            assert raw_df[f].isna().mean() > 0.01, f"{f} 缺失率过低，无法评测幻觉"

    def test_all_template_families_present(self, raw_df):
        assert set(raw_df["template_family"].unique()) == set(IN_DIST_FAMILIES + OOD_FAMILIES)

    def test_generation_is_deterministic(self):
        a = generate(30, SynthConfig(seed=99), IN_DIST_FAMILIES)
        b = generate(30, SynthConfig(seed=99), IN_DIST_FAMILIES)
        assert [s.text for s in a] == [s.text for s in b]


class TestClean:
    def test_drops_duplicate_text(self):
        df = pd.DataFrame({
            "text": ["型号 M1 请报价" * 10, "型号 M1 请报价" * 10, "型号 M2 请报价" * 10],
            "template_family": ["im_chat"] * 3,
            "skeleton_key": ["s1", "s1", "s2"],
            "product_model": ["M1", "M1", "M2"],
            "customer_name": ["C1", "C1", "C2"],
            "target_json": ["{}"] * 3,
            "is_urgent": [False] * 3,
            **{f: [None] * 3 for f in FIELD_ORDER
               if f not in ("product_model", "customer_name", "is_urgent")},
        })
        out, funnel = clean(df, min_chars=10, max_chars=200)
        assert len(out) == 2
        assert funnel["精确去重后"] == 2, "3 条里应有 2 条存活"
        assert funnel["长度在合理区间"] - funnel["精确去重后"] == 1, "应淘汰 1 条重复"

    def test_drops_text_without_model(self):
        df = pd.DataFrame({
            "text": ["这段文本里没有型号" * 5],
            "template_family": ["im_chat"],
            "skeleton_key": ["s1"],
            "product_model": ["HX-999"],
            "customer_name": ["某某公司"],
            "target_json": ["{}"],
            "is_urgent": [False],
            **{f: [None] for f in FIELD_ORDER
               if f not in ("product_model", "customer_name", "is_urgent")},
        })
        out, funnel = clean(df, min_chars=10, max_chars=500)
        assert out.empty
        assert funnel["型号在正文中"] == 0
        assert funnel["非空正文"] - funnel["型号在正文中"] == 1, "应淘汰 1 条"

    def test_drops_label_inconsistent_urgent(self):
        df = pd.DataFrame({
            "text": ["型号 HX-999 数量 100 台，请报价，谢谢配合。" * 2],
            "template_family": ["im_chat"],
            "skeleton_key": ["s1"],
            "product_model": ["HX-999"],
            "customer_name": ["某某公司"],
            "target_json": ["{}"],
            "is_urgent": [True],   # 标了加急，正文却没有加急措辞
            **{f: [None] for f in FIELD_ORDER
               if f not in ("product_model", "customer_name", "is_urgent")},
        })
        out, funnel = clean(df, min_chars=10, max_chars=500)
        assert out.empty
        assert funnel["标签自洽"] == 0
        assert funnel["近重复去重后"] - funnel["标签自洽"] == 1, "应淘汰 1 条"

    def test_length_filter(self):
        # 正文必须包含 product_model，否则会先被「型号不在正文」规则淘汰，
        # 测不到长度过滤这条规则。
        rows = []
        for i, n in enumerate([5, 60, 5000]):
            rows.append({
                "text": "型号 M" + "X" * n,
                "template_family": "im_chat",
                "skeleton_key": f"s{i}",
                "product_model": "M",
                "customer_name": "C",
                "target_json": "{}",
                "is_urgent": False,
                **{f: None for f in FIELD_ORDER
                   if f not in ("product_model", "customer_name", "is_urgent")},
            })
        out, funnel = clean(pd.DataFrame(rows), min_chars=40, max_chars=900)
        assert len(out) == 1
        assert funnel["长度在合理区间"] == 1
        assert funnel["型号在正文中"] - funnel["长度在合理区间"] == 2, "应淘汰 2 条"


class TestSplit:
    def _frame(self, n_skeleton=40, per=5):
        rows = []
        for s in range(n_skeleton):
            fam = IN_DIST_FAMILIES[s % len(IN_DIST_FAMILIES)]
            for k in range(per):
                rows.append({
                    "text": f"样本 {s}-{k} 型号 HX-{s}{k} " + "填充" * 30,
                    "template_family": fam,
                    "skeleton_key": f"{fam}::sk{s}",
                    "product_model": f"HX-{s}{k}",
                    "customer_name": f"客户{s}公司",
                    "target_json": "{}",
                    "is_urgent": False,
                    "n_tokens": 100 + s * 5 + k,
                    "char_len": 80,
                    **{f: None for f in FIELD_ORDER
                       if f not in ("product_model", "customer_name", "is_urgent")},
                })
        # 再补 OOD 家族的样本
        for k in range(30):
            rows.append({
                "text": f"OOD 样本 {k} 型号 GT-{k} " + "填充" * 40,
                "template_family": OOD_FAMILIES[k % len(OOD_FAMILIES)],
                "skeleton_key": f"ood::sk{k}",
                "product_model": f"GT-{k}",
                "customer_name": "客户公司",
                "target_json": "{}",
                "is_urgent": False,
                "n_tokens": 300 + k,
                "char_len": 120,
                **{f: None for f in FIELD_ORDER
                   if f not in ("product_model", "customer_name", "is_urgent")},
            })
        return pd.DataFrame(rows)

    def test_no_template_leakage(self):
        splits = split_dataset(self._frame(), n_val=10, n_test_iid=20, n_test_ood=15, seed=1)
        train_fams = set(splits["train"]["template_family"])
        ood_fams = set(splits["test_ood"]["template_family"])
        assert not (train_fams & ood_fams), "OOD 模板族泄漏进训练集"

    def test_no_skeleton_leakage(self):
        splits = split_dataset(self._frame(), n_val=10, n_test_iid=20, n_test_ood=15, seed=1)
        train_sk = set(splits["train"]["skeleton_key"])
        iid_sk = set(splits["test_iid"]["skeleton_key"])
        assert not (train_sk & iid_sk), "IID 骨架泄漏进训练集"

    def test_all_splits_non_empty(self):
        splits = split_dataset(self._frame(), n_val=10, n_test_iid=20, n_test_ood=15, seed=1)
        for name, sub in splits.items():
            assert len(sub) > 0, f"{name} 为空"


class TestSchemaHelpers:
    def test_normalize_company_strips_suffix(self):
        assert normalize_company("深圳宏昌精密五金有限公司") == normalize_company("深圳宏昌精密五金")

    def test_field_order_is_stable(self):
        assert FIELD_ORDER[0] == "customer_name"
        assert len(FIELD_ORDER) == 9


class TestJsonlRoundTrip:
    """回归测试：pandas 会把 None 变成 NaN，写盘前必须还原成 null。
    否则 json.loads 读回来是 float('nan')，评测时「gold 为 null」判断失效，
    幻觉率会假性恒等于 0。"""

    def test_nan_is_written_as_null(self, tmp_path):
        df = pd.DataFrame([{
            "text": "型号 HX-1 数量 100 台" * 3,
            "template_family": "im_chat",
            "skeleton_key": "s1",
            "product_model": "HX-1",
            "customer_name": None,
            "quantity": 100,
            "unit_price": None,
            "currency": "CNY",
            "delivery_date": None,
            "payment_terms": None,
            "is_urgent": False,
            "contact_phone": None,
        }])
        assert pd.isna(df["customer_name"].iloc[0]), "pandas 应当把 None 变成了 NaN"
        path = tmp_path / "out.jsonl"
        write_jsonl(df, path)

        raw = path.read_text(encoding="utf-8")
        assert "NaN" not in raw, "写出的 JSONL 里不允许出现非法的 NaN 字面量"

        rec = json.loads(raw.strip())
        for f in ("customer_name", "unit_price", "delivery_date",
                  "payment_terms", "contact_phone"):
            assert rec[f] is None, f"{f} 应当是 null 而不是 NaN"
        assert rec["quantity"] == 100
        assert rec["is_urgent"] is False
