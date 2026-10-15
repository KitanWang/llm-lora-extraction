"""结构化抽取数据的可控合成器。

设计要点（每一条都对应一个明确的工程取舍）：

1. **模板化合成 + 槽位随机**：文本骨架由「模板族(template_family) × 子句模板」决定，
   公司名/型号/数量/日期等走随机槽位。这样既能造出上千条，又能控制难度分布。
2. **每条样本记录 `template_family` 与 `skeleton_key`**：前者用于构造「模板完全不重叠」的
   OOD 评测集，后者用于近乎重复样本的去重与分组切分。
3. **刻意注入干扰项**：收件方公司名、订单号、传真号、报价有效期、上批货日期……
   目的不是刁难模型，而是让「字段级准确率」这个指标真的有区分度。
4. **可空字段按设定比例注入缺失**：模型必须学会输出 null，而不是编一个值。
   「幻觉率」这个指标就来自这里。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from llm_extract.schema import Currency, InquiryRecord, PaymentTerms

# ──────────────────────────────────────────────────────────────────
# 词表
# ──────────────────────────────────────────────────────────────────

REGIONS = ["深圳", "东莞", "苏州", "宁波", "佛山", "青岛", "杭州", "成都", "无锡",
           "常州", "天津", "厦门", "中山", "嘉兴", "泉州", "合肥", "长沙", "济南"]
BRANDS = ["宏昌", "嘉泰", "鑫源", "瑞德", "华翔", "联创", "正泰", "恒信", "金鹏", "伟业",
          "中科", "远大", "新宇", "利丰", "天成", "博威", "凯盛", "安泰", "百川", "启元"]
INDUSTRIES = ["精密五金", "电子科技", "模具", "自动化设备", "塑胶制品", "机电",
              "智能装备", "仪器仪表", "光电", "新材料"]
COMPANY_SUFFIX = ["有限公司", "股份有限公司", "科技有限公司", "实业有限公司", "制造有限公司"]

MODEL_PREFIX = ["HX", "GT", "SK", "MD", "ZY", "KC", "BL", "TR", "XF", "NW"]
MODEL_MID = ["-", "", "_"]
MODEL_SUFFIX = ["", "A", "B", "Pro", "S", "L"]

UNITS = ["台", "件", "套", "个", "只", "箱", "支", "米"]
UNIT_EN = ["pcs", "sets", "units"]

CITIES_TEL = ["0755", "0769", "0512", "0574", "0757", "0532", "0571", "028", "0510", "0592"]

NOISE_SENTENCES = [
    "冒昧打扰，感谢贵司一直以来的支持。",
    "之前从同行那里了解到贵司的产品口碑不错。",
    "我们这边最近在整理供应链，想多找几家备选供应商。",
    "顺便说一句，贵司网站上的产品手册好像更新得不太及时。",
    "如果有现货的话麻烦优先安排。",
    "麻烦把报价单和产品规格书一起发过来。",
    "我们领导比较看重交付稳定性，价格其次。",
    "另外想问下你们支不支持开专票。",
    "上次合作的那批货质量还可以，所以这次还找你们。",
    "我这边先做个初步询价，具体数量可能还会调整。",
    "邮件可能回得慢，有事直接打我电话。",
    "希望能长期合作。",
]


@dataclass
class SynthConfig:
    """合成配置。"""
    seed: int = 20251015
    # 可空字段的缺失概率（用于训练"该输出 null 就输出 null"）
    null_rate: dict[str, float] = field(default_factory=lambda: {
        "customer_name": 0.12,
        "quantity": 0.10,
        "unit_price": 0.22,      # 只给总价不给单价 —— 高频陷阱
        "delivery_date": 0.18,
        "payment_terms": 0.25,
        "contact_phone": 0.20,
    })
    # 干扰项注入概率
    distractor_rate: dict[str, float] = field(default_factory=lambda: {
        "vendor_company": 0.75,   # 正文里出现"收件方"公司名
        "order_no": 0.55,         # 订单号（与型号形似）
        "fax": 0.35,              # 传真号 / 分机号
        "validity_date": 0.45,    # 报价有效期（干扰交付日期）
        "last_order_date": 0.30,  # 上批货日期
        "total_price": 0.25,      # 总价（干扰单价）
        "calm_wording": 0.35,     # "不着急"类否定表述
        "other_company": 0.40,    # 正文里出现第三方公司名
        "past_quantity": 0.30,    # 上批数量（干扰本次数量）
    })
    noise_rate: float = 0.45
    # ── 脏数据注入 ──────────────────────────────────────────────
    # 真实场景里原始数据从来不是干净的：同一封询价单会被重复投递、
    # 同一需求会被业务员重复录入、邮件底部会拖着整段历史回复。
    # 这里如实模拟这三种情况，让下游的 pandas 清洗规则真正起作用
    # （否则"去重/长度过滤"只是写在代码里、从未被触发过的摆设）。
    exact_dup_rate: float = 0.06    # 完全重复（转发/重复投递）
    near_dup_rate: float = 0.09     # 近重复（同需求重复录入，仅语序不同）
    long_doc_rate: float = 0.06     # 超长（邮件尾部带历史回复线程）


# ──────────────────────────────────────────────────────────────────
# 基础渲染器
# ──────────────────────────────────────────────────────────────────

def gen_company(rng: random.Random) -> str:
    return f"{rng.choice(REGIONS)}{rng.choice(BRANDS)}{rng.choice(INDUSTRIES)}{rng.choice(COMPANY_SUFFIX)}"


def gen_model(rng: random.Random) -> str:
    return (f"{rng.choice(MODEL_PREFIX)}{rng.choice(MODEL_MID)}"
            f"{rng.randint(100, 9999)}{rng.choice(MODEL_SUFFIX)}")


def gen_order_no(rng: random.Random, ref: date) -> str:
    prefix = rng.choice(["PO", "SO", "订单号 ", "合同号 HT", "CG"])
    return f"{prefix}{ref.year}{ref.month:02d}{ref.day:02d}-{rng.randint(1, 99):02d}"


def gen_mobile(rng: random.Random) -> str:
    return f"1{rng.choice('35789')}{rng.randint(0, 999999999):09d}"


def gen_landline(rng: random.Random) -> str:
    return f"{rng.choice(CITIES_TEL)}-{rng.randint(10000000, 89999999)}"


def gen_phone(rng: random.Random) -> str:
    return gen_mobile(rng) if rng.random() < 0.7 else gen_landline(rng)


def fmt_quantity(rng: random.Random, n: int, unit: str) -> str:
    """数量有多种中文写法，模型不能只会读阿拉伯数字。"""
    style = rng.random()
    if style < 0.55:
        return f"{n}{unit}"
    if style < 0.70:
        return f"{n:,}{unit}"
    if style < 0.85 and n >= 1000:
        wan = n / 10000
        return f"{wan:g}万{unit}"
    if n >= 1000:
        return f"{n // 1000}千{unit}"
    return f"{n} {unit}"


def fmt_absolute_date(rng: random.Random, d: date) -> str:
    style = rng.random()
    if style < 0.30:
        return f"{d.year}年{d.month}月{d.day}日"
    if style < 0.55:
        return f"{d.month}月{d.day}号"
    if style < 0.75:
        return f"{d.year}-{d.month:02d}-{d.day:02d}"
    return f"{d.month}/{d.day}"


def fmt_price(rng: random.Random, price: float, currency: Currency, unit: str) -> str:
    """单价的多种写法，含币种线索。"""
    if currency is Currency.USD:
        return rng.choice([f"${price:g}/{rng.choice(UNIT_EN)}", f"USD {price:g}",
                           f"{price:g} 美元"])
    if currency is Currency.EUR:
        return rng.choice([f"€{price:g}", f"EUR {price:g}", f"{price:g} 欧元"])
    return rng.choice([f"单价 {price:g} 元/{unit}", f"单价￥{price:g}", f"每{unit} {price:g} 元",
                       f"报价 {price:g} 元", f"含税单价 {price:g} 元"])


def fmt_quantity_cn_number(n: int) -> str:
    """给「总价」干扰项用的中文金额。"""
    if n >= 10000:
        return f"{n / 10000:g} 万"
    return f"{n:,}"


PAYMENT_PHRASES: dict[PaymentTerms, list[str]] = {
    PaymentTerms.PREPAID_FULL: [
        "全款预付", "预付全款", "合同签订后一次性付清全款", "全额预付再排产",
    ],
    PaymentTerms.DEPOSIT_30: [
        "先打三成定金，货到再付七成", "30%定金，余款发货前结清",
        "定金 30%，剩余 70% 见提单付款", "先付30%定金，尾款70%验收后付",
    ],
    PaymentTerms.CASH_ON_DELIVERY: [
        "货到付款", "收到货验收没问题再打款", "COD", "货到验收后付款",
    ],
    PaymentTerms.CASH_BEFORE_SHIPMENT: [
        "款到发货", "先打款后发货", "打款后安排发货", "货款到了我们就发货",
    ],
    PaymentTerms.MONTHLY_30: [
        "月结30天", "账期30天", "月结一个月", "对账后30天付款",
    ],
    PaymentTerms.MONTHLY_60: [
        "月结60天", "账期60天", "月结两个月", "对账后 60 天内付清",
    ],
}

URGENT_TRUE_PHRASES = [
    "这单比较急", "加急处理", "急单，麻烦优先排产", "客户催得紧，越快越好",
    "希望尽快发货", "交期能压缩就压缩", "务必加急",
]
URGENT_FALSE_PHRASES = [
    "不着急，正常排期就行", "非加急", "不用加急", "时间上不紧张，按你们节奏来",
    "这个不催，慢慢来", "不是急单",
]


# ──────────────────────────────────────────────────────────────────
# 样本构建
# ──────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    text: str
    target: dict[str, Any]
    template_family: str
    skeleton_key: str

    def to_record(self) -> InquiryRecord:
        return InquiryRecord.model_validate(self.target)


class SampleBuilder:
    """一条样本 = 若干子句模板的拼接。子句模板的 id 组合构成 skeleton_key。"""

    def __init__(self, rng: random.Random, cfg: SynthConfig, family: str):
        self.rng = rng
        self.cfg = cfg
        self.family = family
        self.clauses: list[tuple[str, str]] = []  # (clause_id, rendered_text)
        self.values: dict[str, Any] = {}

    # ---- 工具 ----
    def add(self, clause_id: str, text: str) -> None:
        self.clauses.append((clause_id, text))

    def maybe(self, p: float) -> bool:
        return self.rng.random() < p

    def noise(self) -> None:
        if self.maybe(self.cfg.noise_rate):
            self.add("noise", self.rng.choice(NOISE_SENTENCES))

    # ---- 各字段的子句 ----
    def add_customer_name(self) -> None:
        self.noise()
        if self.maybe(self.cfg.null_rate["customer_name"]):
            self.values["customer_name"] = None
            return
        name = gen_company(self.rng)
        style = self.rng.randrange(4)
        if style == 0:
            self.add("cust_self_intro", f"我们是{name}的采购部，想咨询一批货。")
        elif style == 1:
            self.add("cust_self_intro", f"我司全称{name}，需要采购下列物料。")
        elif style == 2:
            self.add("cust_self_intro", f"这边是{name}，想找贵司报个价。")
        else:
            self.add("cust_self_intro", f"采购方：{name}")
        self.values["customer_name"] = name

    def add_vendor_as_distractor(self, vendor: str) -> None:
        if not self.maybe(self.cfg.distractor_rate["vendor_company"]):
            return
        style = self.rng.randrange(3)
        if style == 0:
            self.add("vendor_salutation", f"{vendor}：您好！")
        elif style == 1:
            self.add("vendor_salutation", f"致 {vendor} 销售部")
        else:
            self.add("vendor_salutation", f"收件人：{vendor}")

    def add_other_company_distractor(self) -> None:
        if not self.maybe(self.cfg.distractor_rate["other_company"]):
            return
        other = gen_company(self.rng)
        self.add("other_company",
                 self.rng.choice([
                     f"听{other}的朋友说你们家交期挺稳。",
                     f"我们另一家供应商{other}这次报价偏高，所以来问问你们。",
                     f"顺便帮{other}也问一下同款的价格。",
                 ]))

    def add_order_no(self, ref: date) -> None:
        if not self.maybe(self.cfg.distractor_rate["order_no"]):
            return
        self.add("order_no", f"（参考单号 {gen_order_no(self.rng, ref)}）")

    def add_product_model(self) -> None:
        model = gen_model(self.rng)
        style = self.rng.randrange(3)
        if style == 0:
            self.add("model", f"型号 {model}")
        elif style == 1:
            self.add("model", f"我们要采购的物料是 {model}")
        else:
            self.add("model", f"产品：{model}")
        self.values["product_model"] = model

    def add_quantity(self) -> None:
        n = self.rng.choice([50, 100, 120, 200, 300, 500, 800, 1000, 1200, 1500,
                             2000, 3000, 5000, 8000, 10000, 12000, 20000, 50000])
        unit = self.rng.choice(UNITS)
        if self.maybe(self.cfg.distractor_rate["past_quantity"]):
            self.add("past_quantity",
                     f"上批我们是拿了 {fmt_quantity(self.rng, self.rng.choice([300, 500, 800, 1000]), unit)}。")
        if self.maybe(self.cfg.null_rate["quantity"]):
            self.add("quantity_hint", "数量还没最终定，先按你们的起订量报。")
            self.values["quantity"] = None
            return
        self.add("quantity", f"数量 {fmt_quantity(self.rng, n, unit)}，先按这个报。")
        self.values["quantity"] = n
        self.values["_unit"] = unit

    def add_price_and_currency(self) -> None:
        unit = self.values.get("_unit", self.rng.choice(UNITS))
        currency = self.rng.choices(
            [Currency.CNY, Currency.USD, Currency.EUR], weights=[0.78, 0.15, 0.07]
        )[0]
        price = round(self.rng.uniform(3.5, 980.0), self.rng.choice([0, 0, 1, 1, 2]))
        if self.maybe(self.cfg.distractor_rate["total_price"]) and self.values.get("quantity"):
            total = price * self.values["quantity"]
            self.add("total_price", f"这批总价大概 {fmt_quantity_cn_number(int(total))} 元。")
        if self.maybe(self.cfg.null_rate["unit_price"]):
            self.add("price_hint", "总价我这边有预算，但单价你们先别急着报，我要走个比价流程。")
            self.values["unit_price"] = None
            self.values["currency"] = Currency.CNY
            return
        self.add("price", fmt_price(self.rng, price, currency, unit))
        self.values["unit_price"] = price
        self.values["currency"] = currency

    def add_delivery_date(self, ref: date) -> None:
        if self.maybe(self.cfg.distractor_rate["validity_date"]):
            v = ref + timedelta(days=self.rng.randint(15, 60))
            self.add("validity_date", f"另外这个报价有效期到 {fmt_absolute_date(self.rng, v)}。")
        if self.maybe(self.cfg.distractor_rate["last_order_date"]):
            l = ref - timedelta(days=self.rng.randint(20, 90))
            self.add("last_order_date", f"上一批是 {fmt_absolute_date(self.rng, l)} 交的货，还算及时。")
        if self.maybe(self.cfg.null_rate["delivery_date"]):
            self.add("delivery_hint", "交期你们按常规来就行，我们暂时没卡死日期。")
            self.values["delivery_date"] = None
            return
        target = ref + timedelta(days=self.rng.randint(7, 75))
        style = self.rng.random()
        if style < 0.65:
            self.add("delivery_abs", f"我们要求 {fmt_absolute_date(self.rng, target)} 前到货。")
        else:
            # 相对日期：必须结合正文里给出的"发件日期"才能算出来
            delta_days = (target - ref).days
            self.add("delivery_rel",
                     f"发件日期是 {fmt_absolute_date(self.rng, ref)}，"
                     f"希望 {delta_days} 天内能交货。")
        self.values["delivery_date"] = target.isoformat()

    def add_payment_terms(self) -> None:
        if self.maybe(self.cfg.null_rate["payment_terms"]):
            self.add("payment_hint", "付款方式我们再谈，先把价格和交期确认了。")
            self.values["payment_terms"] = None
            return
        terms = self.rng.choice(list(PaymentTerms))
        phrase = self.rng.choice(PAYMENT_PHRASES[terms])
        self.add("payment", f"付款方式我们希望是{phrase}。")
        self.values["payment_terms"] = terms

    def add_urgency(self) -> None:
        if self.maybe(self.cfg.distractor_rate["calm_wording"]):
            self.add("urgency", self.rng.choice(URGENT_FALSE_PHRASES) + "。")
            self.values["is_urgent"] = False
            return
        if self.maybe(0.45):
            self.add("urgency", self.rng.choice(URGENT_TRUE_PHRASES) + "。")
            self.values["is_urgent"] = True
        else:
            self.values["is_urgent"] = False

    def add_phone(self) -> None:
        phone = gen_phone(self.rng)
        if self.maybe(self.cfg.distractor_rate["fax"]):
            if self.rng.random() < 0.5:
                self.add("fax", f"传真：{rng_tel(self.rng)}")
            else:
                self.add("ext", f"总机 {rng_tel(self.rng)} 转 {self.rng.randint(8000, 8999)}")
        if self.maybe(self.cfg.null_rate["contact_phone"]):
            self.add("phone_hint", "有结果回邮件就行，不用打电话。")
            self.values["contact_phone"] = None
            return
        # 手机与固话的书写习惯不同：手机常见 3-4-4 分段，固话是 区号-号码。
        # 不能拿手机的分段格式去套固话，否则会造出 "051 2-52 900628" 这种假数据。
        is_mobile = phone.startswith("1") and len(phone) == 11
        if is_mobile:
            style = self.rng.randrange(3)
            if style == 0:
                self.add("phone", f"我的电话 {phone}，随时联系。")
            elif style == 1:
                self.add("phone", f"联系方式：{phone}")
            else:
                self.add("phone", f"手机 {phone[0:3]} {phone[3:7]} {phone[7:]}，微信同号。")
        else:
            style = self.rng.randrange(2)
            if style == 0:
                self.add("phone", f"我的电话 {phone}，随时联系。")
            else:
                self.add("phone", f"联系方式：{phone}（工作日 9:00-18:00）")
        self.values["contact_phone"] = phone

    # ---- 收尾 ----
    def finish(self, ref: date) -> Sample:
        target = {
            "customer_name": self.values.get("customer_name"),
            "product_model": self.values["product_model"],
            "quantity": self.values.get("quantity"),
            "unit_price": self.values.get("unit_price"),
            "currency": (self.values.get("currency") or Currency.CNY).value,
            "delivery_date": self.values.get("delivery_date"),
            "payment_terms": (self.values.get("payment_terms").value
                              if self.values.get("payment_terms") else None),
            "is_urgent": bool(self.values.get("is_urgent", False)),
            "contact_phone": self.values.get("contact_phone"),
        }
        skeleton = "|".join(sorted(cid for cid, _ in self.clauses))
        return Sample(
            text=render(self.family, self.rng, self.clauses, ref),
            target=target,
            template_family=self.family,
            skeleton_key=f"{self.family}::{skeleton}",
        )


def rng_tel(rng: random.Random) -> str:
    return f"{rng.choice(CITIES_TEL)}-{rng.randint(10000000, 89999999)}"


# ──────────────────────────────────────────────────────────────────
# 渲染：模板族决定排版风格
# ──────────────────────────────────────────────────────────────────

def render(family: str, rng: random.Random, clauses: list[tuple[str, str]], ref: date) -> str:
    body = [text for cid, text in clauses if cid != "vendor_salutation"]
    rng.shuffle(body)
    salutation = next((t for cid, t in clauses if cid == "vendor_salutation"), "")

    if family == "email_formal":
        subject = rng.choice(["询价函", "关于一批物料的询价", "采购询价（请报价）", "询价：请协助报价"])
        return (
            f"主题：{subject}\n"
            f"{salutation}\n"
            + "\n".join(body)
            + f"\n\n顺祝商祺\n（发件日期 {fmt_absolute_date(rng, ref)}）"
        )

    if family == "im_chat":
        lines = [salutation] if salutation else []
        lines += body
        return "\n".join(lines)

    if family == "bullet_list":
        items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(body))
        return f"【采购询价单】\n{items}"

    if family == "phone_transcript":
        lines = ["通话记录（整理稿）"]
        for t in body:
            speaker = rng.choice(["客户", "客户", "我方", "采购"])
            lines.append(f"{speaker}：{t}")
        return "\n".join(lines)

    if family == "quote_reply":
        quoted = rng.choice([
            "（以下为对方原邮件）贵司上次报价偏高，希望这次能给个更有竞争力的价格。",
            "（以下为对方原邮件）关于交期，我们希望越快越好。",
        ])
        return (
            f"{salutation}\n"
            + "\n".join(body)
            + f"\n\n{quoted}\n（发件日期 {fmt_absolute_date(rng, ref)}）"
        )

    if family == "mixed_informal":
        return "".join(body)

    raise ValueError(f"未知模板族：{family}")


# ──────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────

ALL_FAMILIES = ["email_formal", "im_chat", "bullet_list", "phone_transcript",
                "quote_reply", "mixed_informal"]

# 完全留出、训练集绝不出现的模板族 → 用于 OOD 泛化评测
OOD_FAMILIES = ["phone_transcript", "quote_reply"]
IN_DIST_FAMILIES = [f for f in ALL_FAMILIES if f not in OOD_FAMILIES]


def generate(n: int, cfg: SynthConfig, families: list[str]) -> list[Sample]:
    rng = random.Random(cfg.seed)
    samples: list[Sample] = []
    for i in range(n):
        family = families[i % len(families)] if i < len(families) else rng.choice(families)
        ref = date(2025, 1, 1) + timedelta(days=rng.randint(0, 300))
        vendor = gen_company(rng)   # 收件方（干扰项）
        b = SampleBuilder(rng, cfg, family)
        b.add_vendor_as_distractor(vendor)
        b.add_customer_name()
        b.add_other_company_distractor()
        b.add_order_no(ref)
        b.add_product_model()
        b.add_quantity()
        b.add_price_and_currency()
        b.add_delivery_date(ref)
        b.add_payment_terms()
        b.add_urgency()
        b.add_phone()
        samples.append(b.finish(ref))

    # ── 注入脏数据 ──────────────────────────────────────────────
    samples = _inject_dirty(samples, cfg, rng)
    return samples


def _inject_dirty(samples: list[Sample], cfg: SynthConfig, rng: random.Random) -> list[Sample]:
    """模拟真实原始数据里的三类脏数据（见 SynthConfig 注释）。

    注意两点：
    1. 这里注入的"脏"必须能被 pipeline.clean() 的对应规则识别并剔除，
       否则就只是往数据集里掺噪声；
    2. **必须 append 副本、不能 replace 原样本** —— 复制出一条重复记录才是
       "重复投递"的真实形态；如果把原样本改掉，文本依然全局唯一，
       去重规则永远不会触发（这个坑踩过一次）。
    """
    out: list[Sample] = list(samples)
    for s in samples:
        roll = rng.random()

        if roll < cfg.exact_dup_rate:
            # (1) 完全重复：正文一字不差（同一封询价被重复转发/投递）
            out.append(Sample(text=s.text, target=dict(s.target),
                              template_family=s.template_family,
                              skeleton_key=s.skeleton_key))

        elif roll < cfg.exact_dup_rate + cfg.near_dup_rate:
            # (2) 近重复：同一条需求被重复录入，只是子句顺序不同。
            #     正文不同 → 精确去重抓不到，必须靠"骨架+公司+型号"去重。
            lines = s.text.split("\n")
            if len(lines) > 4:
                head, tail, mid = lines[:2], lines[-1:], lines[2:-1]
                rng.shuffle(mid)
                lines = head + mid + tail
            out.append(Sample(text="\n".join(lines), target=dict(s.target),
                              template_family=s.template_family,
                              skeleton_key=s.skeleton_key))

        elif roll < cfg.exact_dup_rate + cfg.near_dup_rate + cfg.long_doc_rate:
            # (3) 超长：邮件底部拖着整段历史回复线程 → 超出长度预算，需按长度过滤
            quoted = "\n".join(
                f"> 上一轮沟通记录（{i}）：已确认型号与数量，等待贵司正式报价，请勿重复报价。"
                for i in range(60)
            )
            out.append(Sample(text=s.text + "\n\n----- 历史邮件 -----\n" + quoted,
                              target=dict(s.target),
                              template_family=s.template_family,
                              skeleton_key=s.skeleton_key))
    return out
