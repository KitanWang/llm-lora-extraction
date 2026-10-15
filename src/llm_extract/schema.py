"""项目 A：LLM 微调与评测对比实验 —— 抽取目标 schema。

任务：中文 B2B 采购询价文本 → 固定结构 JSON（结构化信息抽取）。

本文件定义抽取目标的 schema、字段元信息与归一化规则。
schema 是数据合成、训练、评测三处的唯一事实来源（single source of truth）。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ──────────────────────────────────────────────────────────────────
# 枚举
# ──────────────────────────────────────────────────────────────────

class Currency(str, Enum):
    CNY = "CNY"
    USD = "USD"
    EUR = "EUR"


class PaymentTerms(str, Enum):
    PREPAID_FULL = "预付全款"
    DEPOSIT_30 = "30%定金+70%尾款"
    CASH_ON_DELIVERY = "货到付款"
    CASH_BEFORE_SHIPMENT = "款到发货"
    MONTHLY_30 = "月结30天"
    MONTHLY_60 = "月结60天"


# ──────────────────────────────────────────────────────────────────
# 抽取结果 schema
# ──────────────────────────────────────────────────────────────────

class InquiryRecord(BaseModel):
    """单条询价文本的抽取目标。

    设计意图（每一项都对应一类真实的抽取难点）：
    - customer_name  字符串 + 干扰项（同一句话里出现多个公司名，只有一个是"客户"）
    - product_model  必填字符串，型号里含数字，容易与订单号/数量混淆
    - quantity       数值，但常以「1.2万」「3 千」等中文数量词出现，需要换算
    - unit_price     浮点，可能带「元/台」「$」等单位；也可能只给总价不给单价 → null
    - currency       枚举，且"元"默认 CNY 需要对英文缩写做映射
    - delivery_date  日期，需与"报价有效期""发布日期"等干扰日期区分
    - payment_terms  枚举，口语化表达（"先打三成定金"）需归一到标准值
    - is_urgent      布尔，且有否定陷阱（"不着急""非加急"）
    - contact_phone  字符串，需与传真号、分机号、订单号区分；缺失时必须输出 null
    """

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = Field(default=None, description="采购方公司全称")
    product_model: str = Field(description="产品型号")
    quantity: int | None = Field(default=None, description="采购数量，单位件/台/套")
    unit_price: float | None = Field(default=None, description="单价，单位元（或对应币种）")
    currency: Currency = Field(default=Currency.CNY, description="币种")
    delivery_date: str | None = Field(default=None, description="要求交付日期 YYYY-MM-DD")
    payment_terms: PaymentTerms | None = Field(default=None, description="付款条件")
    is_urgent: bool = Field(default=False, description="是否加急")
    contact_phone: str | None = Field(default=None, description="联系电话")

    @field_validator("delivery_date")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"delivery_date 必须为 YYYY-MM-DD，得到 {v!r}")
        return v

    @field_validator("contact_phone")
    @classmethod
    def _check_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = re.sub(r"[\s\-]", "", v)
        if not re.fullmatch(r"(1[3-9]\d{9}|0\d{9,11})", cleaned):
            raise ValueError(f"contact_phone 非法：{v!r}")
        return cleaned

    def to_target_json(self) -> dict[str, Any]:
        """训练目标用的紧凑 JSON（字段顺序固定，便于对比）。"""
        return {
            "customer_name": self.customer_name,
            "product_model": self.product_model,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "currency": self.currency.value,
            "delivery_date": self.delivery_date,
            "payment_terms": self.payment_terms.value if self.payment_terms else None,
            "is_urgent": self.is_urgent,
            "contact_phone": self.contact_phone,
        }


# ──────────────────────────────────────────────────────────────────
# 字段元信息：驱动评测（哪些字段可空 / 用哪种匹配方式）
# ──────────────────────────────────────────────────────────────────

FIELD_SPECS: dict[str, dict[str, Any]] = {
    "customer_name": {"kind": "str", "nullable": True, "match": "normalized_str"},
    "product_model": {"kind": "str", "nullable": False, "match": "normalized_str"},
    "quantity": {"kind": "number", "nullable": True, "match": "exact"},
    "unit_price": {"kind": "number", "nullable": True, "match": "numeric_tol"},
    "currency": {"kind": "enum", "nullable": False, "match": "exact"},
    "delivery_date": {"kind": "date", "nullable": True, "match": "exact"},
    "payment_terms": {"kind": "enum", "nullable": True, "match": "exact"},
    "is_urgent": {"kind": "bool", "nullable": False, "match": "exact"},
    "contact_phone": {"kind": "str", "nullable": True, "match": "digits"},
}

FIELD_ORDER: list[str] = list(FIELD_SPECS.keys())
NULLABLE_FIELDS: list[str] = [f for f, s in FIELD_SPECS.items() if s["nullable"]]
REQUIRED_FIELDS: list[str] = [f for f, s in FIELD_SPECS.items() if not s["nullable"]]


# ──────────────────────────────────────────────────────────────────
# 归一化：评测时对字符串做宽松比对，避免"格式差异"被误判为"抽错"
# ──────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[\s\u3000·・\-—_/\\()（）【】\[\]<>《》,，。;；:：!！?？\"'“”‘’]+")

# 公司名后缀与地域前缀的常见简写，用于"简称 vs 全称"的宽松匹配
_COMPANY_TAIL = re.compile(r"(有限公司|股份有限公司|集团有限公司|有限责任公司|公司|集团)$")


def normalize_text(s: str) -> str:
    """去除标点、空白，统一全角/半角与大小写。"""
    s = s.strip()
    # 全角字母数字 → 半角
    s = "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch
        for ch in s
    )
    return _PUNCT.sub("", s).lower()


def normalize_company(s: str) -> str:
    """公司名宽松匹配：去掉空格标点，再剥掉公司后缀。"""
    return _COMPANY_TAIL.sub("", normalize_text(s))


def normalize_digits(s: str) -> str:
    return re.sub(r"\D", "", s)
