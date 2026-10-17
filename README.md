# LLM 微调与评测对比实验 —— 中文采购询价结构化抽取

> 一个**可复现**的 LoRA / QLoRA 微调实验：把「一段中文业务文本」抽成「固定结构的 JSON」，
> 并给出**基座模型 vs 微调后**在同一评测集上的逐字段对比。
>
> 本仓库的目标不是刷某个榜单，而是把三件事讲清楚并可复现：
> **① 数据是怎么造的？② 训练是怎么做的？③ 提升是怎么量出来的？**

---

## 1. 任务定义

**输入**：一段中文 B2B 采购询价文本（邮件 / 微信聊天 / 电话记录 / 询价单，共 6 种文体）
**输出**：9 个字段的 JSON

```json
{
  "customer_name": "深圳宏昌精密五金有限公司",
  "product_model": "HX-1234",
  "quantity": 1000,
  "unit_price": 25.5,
  "currency": "CNY",
  "delivery_date": "2025-06-01",
  "payment_terms": "30%定金+70%尾款",
  "is_urgent": true,
  "contact_phone": "13800138000"
}
```

**为什么选这个任务**（而不是"情感分类"这类玩具任务）：

| 难点 | 具体表现 | 考察模型什么能力 |
| :--- | :--- | :--- |
| 多实体混淆 | 正文里同时出现**收件方**公司、**客户**公司、**第三方**公司 | 角色识别，不是见名就抽 |
| 型号 vs 单号 | `HX-1234`（型号）与 `PO20250315-07`（订单号）形似 | 语义区分而非正则匹配 |
| 中文数量词 | 「1.2万台」「3千件」「1,200 台」都要归一到整数 | 语言理解 + 归一化 |
| 只给总价 | 有总价但没单价 → **必须输出 null**，不能拿总价除以数量蒙 | **抑制幻觉** |
| 相对日期 | "发件日期 3/15，希望 30 天内交货" → 要算出 `2025-04-14` | 多步推理 |
| 口语化付款条件 | "先打三成定金，货到再付七成" → `30%定金+70%尾款` | 表达归一化到枚举 |
| 否定陷阱 | "不着急"、"非加急" → `is_urgent: false` | 否定语义 |
| 干扰号码 | 传真号、总机分机号 vs 手机号 | 细粒度实体区分 |

---

## 2. 硬件与本机选型结论

本机实测（`uv run python scripts/check_env.py` 可一键复现）：

| 项 | 配置 |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 5060 Laptop，**8 GB** 显存，Blackwell / `sm_120` |
| CPU / 内存 | AMD Ryzen 9 9955HX 16C32T / 19 GB |
| 系统 | WSL2 Ubuntu 24.04，驱动 596.13 / CUDA 13.2 |

**选型结论**：

| 模型 | 方案 | 显存占用 | 是否采用 |
| :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B-Instruct | LoRA bf16 | ~2 GB | 冒烟测试 |
| 模型 | 方案 | **实测**训练峰值 | 是否采用 |
| :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B-Instruct | LoRA bf16 | 3.60 GiB | 冒烟测试 |
| **Qwen2.5-1.5B-Instruct** | **LoRA bf16，不量化** | **5.41 GiB** | ✅ 主对照（无量化误差，对比最干净） |
| **Qwen2.5-7B-Instruct** | **QLoRA NF4 4bit** | **6.04 GiB** | ✅ 冲 7B 级别 |
| Qwen2.5-14B / 32B | — | >8 GB | ❌ 显存不足 |

> **为什么是 Qwen2.5**：中文能力在同尺寸开源模型里第一梯队；0.5B/1.5B/7B 同一家族
> 便于做尺寸对比；`Instruct` 版本自带 chat template，SFT 不需要额外对齐模板。
>
> **为什么 1.5B 不量化**：LoRA 对比实验里，量化误差会混进"微调带来的收益"里，
> 主对照用 bf16 可以把变量控制到只剩"有没有微调"。
> **7B 走 QLoRA** 则是为了验证「单张 8 GB 消费级显卡也能微调 7B」这一可行性边界
> —— 实测可训练参数占比 **0.92%**。

### 2.1 8 GB 显存训 7B 的三个关键优化

**官方推荐写法在 8 GB 卡上跑不动**，必须做三件事（详见 `reports/EXPERIMENT_REPORT.md` 第四节）：

| # | 优化 | 效果 | 原理 |
| :--- | :--- | :--- | :--- |
| 1 | `fp32_upcast_non_quantized: false` | 常驻 7.38 → **5.34 GiB** | `prepare_model_for_kbit_training` 会把 `embed_tokens`/`lm_head` 升到 fp32；7B 的 `tie_word_embeddings=False` 使两者互不共享，各 152064×3584，升级后合计多占 **2.04 GB** |
| 2 | `logits_to_keep: true` | 峰值降 **58%** | lm_head 输出 `[B, T, 151936]`，而答案只有约 80 token，90% 以上 logits 算完即弃；只对答案位置投影。**与完整计算数学等价**（loss 相对差 9.4e-08） |
| 3 | `bs=1` + 梯度累积 16 + 梯度检查点 + `paged_adamw_8bit` | 撑住 7B | 优化器状态分页到内存 |

> ⚠️ **优化 1 不做的话不会报 OOM**：WSL 允许显存超售，溢出到主机内存的表现是
> 「能跑，但 110 秒/步」。必须用 `scripts/probe_vram.py` 实测才能发现。
>
> ⚠️ **优化 2 有个容易写错的地方**：右侧 padding 使答案不在序列末尾，
> 若照搬「labels 左移后取末尾 k 个」会整体错位 —— loss 照样下降、训练照样收敛，
> 但学的是错位一环的标签。正确公式为 `k = T - min(first_sup) + 1`。

---

## 3. 环境安装

使用 `uv`（Python 3.12）：

```bash
cd projects/llm-lora-extraction
uv venv venv --python 3.12

# ① PyTorch 必须走 cu128 专用索引：RTX 50 系是 sm_120，
#    默认 PyPI 轮子不含该架构内核，装了会报 "no kernel image is available"
uv pip install --python venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 -r requirements-torch.txt

# ② 其余依赖走 PyPI
uv pip install --python venv/bin/python -r requirements.txt

# ③ 把本项目装成可编辑包（src 布局）
uv pip install --python venv/bin/python -e . --no-deps

# ④ 自检
uv run --python venv/bin/python python scripts/check_env.py
```

---

## 4. 复现步骤

```bash
# 1) 合成 + 清洗 + 切分（默认 3000 条，产出 train/val/test_iid/test_ood）
uv run python -m llm_extract.data.pipeline --n 3000 --tokenizer Qwen/Qwen2.5-1.5B-Instruct

# 2) 冒烟测试：60 步跑通全链路
uv run python -m llm_extract.train.sft --config configs/smoke.yaml --max-steps 60

# 3) 正式训练
uv run python -m llm_extract.train.sft --config configs/qwen25-1.5b-lora.yaml   # LoRA  bf16
uv run python -m llm_extract.train.sft --config configs/qwen25-7b-qlora.yaml    # QLoRA NF4

# 4) 评测：基座零样本 vs 微调后，IID 与 OOD 各一遍
uv run python -m llm_extract.evaluate.run_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --dataset data/processed/test_iid.jsonl \
  --tag base_iid --out reports/preds/base_iid.jsonl

uv run python -m llm_extract.evaluate.run_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qwen25-1.5b-lora/final_adapter \
  --dataset data/processed/test_iid.jsonl --tag lora_iid --out reports/preds/lora_iid.jsonl

# 5) 汇总报告（读 reports/summary_*.json → reports/RESULTS.md）
uv run python scripts/make_report.py

# 6) 测试
uv run python -m pytest tests/ -v
```

---

## 5. 目录结构

```
llm-lora-extraction/
├── configs/                     # 训练配置（YAML，dataclass 校验未知字段）
│   ├── smoke.yaml               #   冒烟测试
│   ├── qwen25-1.5b-lora.yaml    #   主力 A：bf16 LoRA
│   └── qwen25-7b-qlora.yaml     #   主力 B：NF4 QLoRA
├── src/llm_extract/
│   ├── schema.py                # 抽取 schema + 字段元信息（单一事实来源）
│   ├── prompts.py               # 训练/评测共用的 prompt 模板
│   ├── data/
│   │   ├── synth.py             # 可控数据合成器（模板族 + 槽位 + 干扰项）
│   │   └── pipeline.py          # pandas 清洗 / token 统计 / 严格切分
│   ├── train/sft.py             # SFT（手写 label mask，不依赖 trl）
│   └── evaluate/
│       ├── metrics.py           # 指标定义
│       └── run_eval.py          # 批量推理 + 评测
├── scripts/
│   ├── check_env.py             # 环境自检 + 选型建议
│   └── make_report.py           # 汇总成 RESULTS.md
├── tests/                       # 单元测试
└── reports/                     # 数据集统计、指标汇总、最终报告
```

---

## 6. 关键设计决策

### 6.1 数据：为什么是"合成"而不是"标注"

- 真实业务文本涉及客户隐私，无法公开；纯手工标注 1500 条的成本又太高。
- 采用**模板化合成 + 槽位随机**：先定义 6 个文本模板族（正式邮件 / 微信聊天 /
  编号列表 / 电话记录 / 邮件回复 / 无格式混排），再往槽位里填随机值。
- **干扰项是刻意注入的**，且每种干扰都对应一个真实业务陷阱（见第 1 节表格）。
  这不是"造难题"，而是让字段级准确率这个指标真的有区分度——如果不放干扰项，
  基座模型也能拿到 90%+，微调的收益就看不出来了。
- 可空字段按设定比例注入缺失，使得**幻觉率**成为可测量指标：
  模型必须学会说"不知道"，而不是编一个看起来合理的值。

### 6.2 切分：为什么必须做 OOD

只做随机切分会有两个问题：

1. **同骨架泄漏**：同一模板骨架的样本被分到训练集和测试集两边，
   测试集本质上是"训练集见过的句式"，指标虚高。
   → 用 `skeleton_key`（子句模板 id 组合）做**分组切分**，同一骨架绝不跨集。
2. **测不出泛化**：随机切分无法回答"换个说法还能不能抽对"。
   → 单独留出 **2 个模板族（电话记录、邮件回复）完全不进训练集**作为 OOD 测试集。

结果就是报告里那张「IID vs OOD」的表——**掉多少，就是泛化差距**。

### 6.3 训练：手写 label mask 而不是直接用 SFTTrainer

`trl.SFTTrainer` 在 0.13 → 0.29 之间把 `tokenizer=` / `processing_class=`、
`dataset_text_field` / `max_seq_length` 反复改名，锁版本会牺牲可复现性。
本项目直接用 `transformers.Trainer` + 手写 collator，换来的是 **SFT 的核心细节完全可见**：

```
input_ids:  <|im_start|>system ... <|im_start|>user ... <|im_start|>assistant {"customer_name":...
labels:     -100 -100 -100 ... -100 -100 -100 ... -100 -100 -100      {"customer_name":...
            └──────────── prompt 段不计 loss ────────────┘            └── 只对答案算 loss ──┘
```

**只对答案算 loss，不对题目算 loss** —— 这是 SFT 与"继续预训练"的本质区别，
也是本项目在实现上最容易出错的地方。

### 6.4 QLoRA：三个必须做对的细节

1. **`prepare_model_for_kbit_training`**：把 LayerNorm 等模块升到 fp32 并开启输入梯度，
   否则梯度传不进 4bit 主干，loss 不会下降（最常见的"训了个寂寞"）。
2. **`paged_adamw_8bit`**：优化器状态分页到 CPU，8 GB 卡上 7B 训练不 OOM 的关键。
3. **梯度检查点 + `use_reentrant=False`**：用时间换显存；`use_reentrant=True`
   在 PEFT + 4bit 组合下会有梯度告警甚至不生效。

### 6.5 评测：为什么必须贪心解码

采样解码（`do_sample=True`）会让同一模型两次评测结果不同，
"提升 30 个百分点"这种结论就站不住脚。本项目统一 `do_sample=False`，
保证任何人跑出来的数都一样。

### 6.6 指标：不只看"抽对多少"

| 指标 | 为什么需要它 |
| :--- | :--- |
| JSON 合法率 | 下游要 `json.loads()`，格式不对等于不可用 |
| Schema 合规率 | 枚举越界、多余字段同样是生产事故 |
| 字段级准确率 | 定位到具体是哪个字段不行 |
| 全字段准确率 | 最严苛：9 个字段全对才算对 |
| **幻觉率** | 标注为 null 却编出值——**比漏抽危险得多** |
| 漏抽率 | 标注有值却输出 null |
| 枚举错分方向 | 看错误是"系统性"还是"随机"的 |

---

## 7. 结果

完整表格见 [`reports/RESULTS.md`](reports/RESULTS.md)（由 `scripts/make_report.py` 自动生成），
分析与工程记录见 [`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md)。

**全字段准确率（9 个字段全对且格式合规）**：

| 模型 | 测试集 | 基座零样本 | **微调后** | 提升 |
| :--- | :--- | ---: | ---: | ---: |
| Qwen2.5-1.5B | IID | 12.5% | **87.5%** | +75.0 pp |
| Qwen2.5-1.5B | OOD | 7.3% | **80.7%** | +73.3 pp |
| **Qwen2.5-7B** | IID | 22.5% | **93.0%** | +70.5 pp |
| **Qwen2.5-7B** | OOD | 15.3% | **91.3%** | +76.0 pp |

**三个关键发现**：

1. **幻觉率归零**：7B 基座在「单价」字段漏抽率高达 **79.8%**（正文有值却判 null），
   微调后降到 **0.0%**。客户名、数量、交付日期、付款条件、联系电话同样全部归零。
2. **泛化差距随规模收窄**：在**训练中完全未出现过**的两种文体上，1.5B 掉 6.8 pp，
   7B 仅掉 **1.7 pp**（91.3%）——说明学到的是抽取能力，而非句式模板记忆。
3. **难点在语义不在格式**：基座 JSON 合法率已是 100%，但全字段准确率仅 12.5%/22.5%；
   Schema 合规率 60.5%（1.5B）/ 93.3%（7B）。差距全部来自语义理解。

**剩余短板**（如实记录，非微调可轻易解决）：

| 字段 | 错误率(IID/OOD) | 表现 |
| :--- | ---: | :--- |
| 交付日期 | 5.0% / 6.0% | 「发件日期 10/26，希望 72 天内交货」跨月运算错 12~20 天 |
| 数量 | 2.0% / 2.7% | 「1千件」换算不稳（答 1500/1200），受「上批 800 只」干扰 |

**成本**：

| 实验 | 方案 | 可训练参数 | 占比 | 峰值显存 | 训练耗时 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | LoRA (bf16) | 18.5 M | 1.182% | 5.41 GiB | 32.7 min |
| **Qwen2.5-7B** | **QLoRA (NF4)** | 40.4 M | **0.919%** | **7.16 GiB** | 108.1 min |

---

## 8. 局限性

- 训练数据是**合成**的，分布与真实业务文本有差距；真实场景需要少量真实样本做校准。
- 只做了 **LoRA / QLoRA**，没有做全参微调对比（8 GB 显存做不了 7B 全参）。
- 评测集同样是合成的，指标绝对值不代表线上表现；**但基座与微调后的相对差异是可信的**，
  因为两者面对的是完全相同的输入。
- 未做量化部署（GPTQ/AWQ）与推理加速（vLLM）对比，属于后续工作。
