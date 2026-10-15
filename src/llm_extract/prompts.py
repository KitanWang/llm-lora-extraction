"""统一 Prompt 模板：训练与评测必须共用同一套，否则对比不成立。"""

from __future__ import annotations

import json

from llm_extract.schema import FIELD_ORDER, InquiryRecord

SYSTEM_PROMPT = (
    "你是一个结构化信息抽取引擎。"
    "从用户给出的采购询价文本中抽取字段，只输出一个 JSON 对象，不要输出任何解释、"
    "markdown 代码块或多余文字。无法确定的字段输出 null（布尔字段输出 false）。"
)

# 字段说明：显式写进 prompt，保证"零样本基座"也拿到同等信息量，
# 这样基座 vs 微调后的差异才归因于微调，而不是归因于"模型不知道要抽什么"。
_FIELD_DOC = """字段定义：
- customer_name: 采购方公司全称（字符串；找不到填 null）
- product_model: 产品型号（字符串，必填）
- quantity: 采购数量（整数；找不到填 null）
- unit_price: 单价（数字，不含单位；只给总价没给单价时填 null）
- currency: 币种，只能取 CNY / USD / EUR 之一（默认 CNY）
- delivery_date: 要求交付日期，格式 YYYY-MM-DD（找不到填 null）
- payment_terms: 付款条件，只能取「预付全款」「30%定金+70%尾款」「货到付款」「款到发货」「月结30天」「月结60天」之一（找不到填 null）
- is_urgent: 是否加急（布尔值 true/false）
- contact_phone: 联系电话（字符串，保留数字；找不到填 null）"""

_OUTPUT_EXAMPLE = json.dumps(
    {
        "customer_name": "示例科技有限公司",
        "product_model": "ABC-1234",
        "quantity": 100,
        "unit_price": 25.5,
        "currency": "CNY",
        "delivery_date": "2025-06-01",
        "payment_terms": "30%定金+70%尾款",
        "is_urgent": True,
        "contact_phone": "13800138000",
    },
    ensure_ascii=False,
    indent=None,
)

USER_TEMPLATE = """{field_doc}

输出 JSON 的字段顺序必须为：{field_order}
输出示例：{example}

待抽取文本：
{text}
"""


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(
        field_doc=_FIELD_DOC,
        field_order=", ".join(FIELD_ORDER),
        example=_OUTPUT_EXAMPLE,
        text=text.strip(),
    )


def build_messages(text: str) -> list[dict[str, str]]:
    """chat 格式消息（训练与推理共用）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]


def build_target(record: InquiryRecord) -> str:
    """训练监督信号：紧凑 JSON，ensure_ascii=False 保证中文不被转义。"""
    return json.dumps(record.to_target_json(), ensure_ascii=False, separators=(",", ":"))
