"""指标模块的单元测试。

重点测三类容易错的地方：
1. 模型输出带 markdown 围栏 / 前后废话时，解析器还能不能抠出 JSON；
2. 字符串"格式差异"（全半角、空格、公司后缀）不该被算成错；
3. 幻觉率与漏抽率的统计方向不能反。
"""

from __future__ import annotations

from llm_extract.evaluate.metrics import (
    evaluate_one,
    extract_json_object,
    summarize,
    values_equal,
)
from llm_extract.schema import FIELD_ORDER

GOLD = {
    "customer_name": "深圳宏昌精密五金有限公司",
    "product_model": "HX-1234",
    "quantity": 1000,
    "unit_price": 25.5,
    "currency": "CNY",
    "delivery_date": "2025-06-01",
    "payment_terms": "30%定金+70%尾款",
    "is_urgent": True,
    "contact_phone": "13800138000",
}

PERFECT = """{"customer_name":"深圳宏昌精密五金有限公司","product_model":"HX-1234","quantity":1000,"unit_price":25.5,"currency":"CNY","delivery_date":"2025-06-01","payment_terms":"30%定金+70%尾款","is_urgent":true,"contact_phone":"13800138000"}"""


class TestJsonExtraction:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_markdown_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert extract_json_object(raw) == {"a": 1}

    def test_fence_without_lang(self):
        raw = '```\n{"a": 1}\n```'
        assert extract_json_object(raw) == {"a": 1}

    def test_json_with_preamble_and_trailing_text(self):
        raw = '好的，抽取结果如下：\n{"a": 1, "b": "中文"}\n希望对你有帮助。'
        assert extract_json_object(raw) == {"a": 1, "b": "中文"}

    def test_array_is_rejected(self):
        assert extract_json_object("[1, 2, 3]") is None

    def test_garbage_returns_none(self):
        assert extract_json_object("我无法完成这个任务。") is None
        assert extract_json_object("") is None


class TestValueComparison:
    def test_fullwidth_punctuation_ignored(self):
        assert values_equal("product_model", "ＨＸ－1234", "HX-1234")

    def test_company_suffix_ignored(self):
        assert values_equal(
            "customer_name", "深圳宏昌精密五金", "深圳宏昌精密五金有限公司"
        )

    def test_company_whitespace_ignored(self):
        assert values_equal(
            "customer_name", " 深圳宏昌精密五金 有限公司 ", "深圳宏昌精密五金有限公司"
        )

    def test_phone_formatting_ignored(self):
        assert values_equal("contact_phone", "138-0013-8000", "13800138000")

    def test_numeric_string_accepted(self):
        assert values_equal("quantity", "1000", 1000)
        assert values_equal("unit_price", "25.50", 25.5)

    def test_none_matches_none(self):
        assert values_equal("unit_price", None, None)

    def test_none_does_not_match_value(self):
        assert not values_equal("unit_price", None, 25.5)
        assert not values_equal("unit_price", 25.5, None)

    def test_wrong_value_detected(self):
        assert not values_equal("quantity", 999, 1000)
        assert not values_equal("currency", "USD", "CNY")


class TestEvaluateOne:
    def test_perfect_prediction(self):
        r = evaluate_one(0, PERFECT, GOLD)
        assert r.json_ok and r.schema_ok and r.exact_match
        assert all(r.field_correct.values())

    def test_extra_field_breaks_schema_not_fields(self):
        raw = PERFECT[:-1] + ',"extra":"x"}'
        r = evaluate_one(0, raw, GOLD)
        assert r.json_ok and not r.schema_ok
        assert all(r.field_correct.values())      # 字段本身都抽对了
        assert not r.exact_match                  # 但不合规，不算全对

    def test_invalid_enum_breaks_schema(self):
        raw = PERFECT.replace('"CNY"', '"RMB"')
        r = evaluate_one(0, raw, GOLD)
        assert r.json_ok and not r.schema_ok

    def test_unparseable_output_all_wrong(self):
        r = evaluate_one(0, "抱歉，我做不到。", GOLD)
        assert not r.json_ok
        assert sum(r.field_correct.values()) == 0

    def test_partial_credit(self):
        raw = PERFECT.replace('"quantity":1000', '"quantity":999')
        r = evaluate_one(0, raw, GOLD)
        assert not r.exact_match
        assert r.field_correct["quantity"] is False
        assert r.field_correct["product_model"] is True


class TestSummarize:
    def _records(self):
        # 2 条：一条全对，一条把 unit_price 编成了 30（gold 为 null → 幻觉）
        gold_null = dict(GOLD, unit_price=None)
        r1 = evaluate_one(0, PERFECT.replace('"unit_price":25.5', '"unit_price":null'), gold_null)
        r2 = evaluate_one(1, PERFECT.replace('"unit_price":25.5', '"unit_price":30'), gold_null)
        return r1, r2, gold_null

    def test_hallucination_direction(self):
        r1, r2, gold = self._records()
        assert r1.exact_match, "把 null 预测成 null 应该算对"
        assert not r2.exact_match

    def test_hallucination_rate_value(self):
        r1, r2, _ = self._records()
        s = summarize([r1, r2])
        # unit_price 的两条 gold 都是 null，其中一条被编造 → 50%
        assert abs(s["hallucination_rate"]["unit_price"] - 0.5) < 1e-9
        # 这两个字段 gold 都非空且预测都非空 → 漏抽率应为 0
        assert s["omission_rate"]["product_model"] == 0.0

    def test_rates_are_fractions(self):
        r1, r2, _ = self._records()
        s = summarize([r1, r2])
        assert s["n"] == 2
        assert 0.0 <= s["json_valid_rate"] <= 1.0
        assert set(s["field_acc"]) == set(FIELD_ORDER)

    def test_by_template_family_grouping(self):
        r1 = evaluate_one(0, PERFECT, GOLD, "im_chat")
        r2 = evaluate_one(1, "不是JSON", GOLD, "email_formal")
        s = summarize([r1, r2])
        assert s["by_template_family"]["im_chat"]["n"] == 1
        assert s["by_template_family"]["im_chat"]["exact_match_rate"] == 1.0
        assert s["by_template_family"]["email_formal"]["exact_match_rate"] == 0.0
