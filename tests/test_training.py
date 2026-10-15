"""SFT 监督信号的单元测试。

**这是整个项目最该测的地方**：如果 label mask 写错（比如忘了 mask prompt，
或者把答案也 mask 掉），训练照样能跑完、loss 照样会下降，
但模型学到的东西完全不对——而且从日志里看不出来。

所以这里不测"函数有没有返回"，而是直接验证：
    被监督的那段 token，解码回来必须**精确等于**目标 JSON + EOS。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_extract.prompts import build_target
from llm_extract.schema import InquiryRecord

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "train.jsonl"


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    candidates = [
        ROOT / "models" / "Qwen2.5-0.5B-Instruct",
        ROOT / "models" / "Qwen2.5-1.5B-Instruct",
    ]
    for p in candidates:
        if (p / "tokenizer_config.json").exists():
            return transformers.AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    pytest.skip("本地没有 tokenizer，跳过（需先运行 scripts/download_models.sh）")


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    if not TRAIN_FILE.exists():
        pytest.skip("未生成数据集，跳过")
    with TRAIN_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f][:5]


class TestLabelMasking:
    def test_masked_span_equals_target(self, samples):
        from llm_extract.train.sft import IGNORE_INDEX, build_supervised_example

        tok = _tokenizer()
        for r in samples:
            payload = {k: r[k] for k in InquiryRecord.model_fields}
            target = build_target(InquiryRecord.model_validate(payload))
            ex = build_supervised_example(tok, r["text"], target, max_seq_length=1024)

            ids, labels = ex["input_ids"], ex["labels"]
            assert len(ids) == len(labels)

            supervised = [i for i, l in zip(ids, labels) if l != IGNORE_INDEX]
            assert supervised, "整条样本都被 mask 了，没有监督信号"
            decoded = tok.decode(supervised).strip()
            assert decoded == (target + tok.eos_token).strip(), (
                "被监督的 token 必须精确等于目标 JSON + EOS，"
                f"实际得到：{decoded[:120]!r}"
            )

    def test_prompt_is_masked(self, samples):
        """prompt 段必须全部是 -100，否则等于在教模型复述题目。"""
        from llm_extract.train.sft import IGNORE_INDEX, build_supervised_example

        tok = _tokenizer()
        r = samples[0]
        payload = {k: r[k] for k in InquiryRecord.model_fields}
        target = build_target(InquiryRecord.model_validate(payload))
        ex = build_supervised_example(tok, r["text"], target, max_seq_length=1024)

        labels = ex["labels"]
        # 至少前半段应当全被 mask（prompt 有 ~580 token，答案只有 ~76 token）
        n_masked = sum(1 for l in labels if l == IGNORE_INDEX)
        assert n_masked > len(labels) * 0.7, (
            f"prompt 应占绝大多数 token，实际只 mask 了 {n_masked}/{len(labels)}"
        )

    def test_overlong_sample_yields_no_supervision(self, samples):
        """答案被截断光时，应当返回空样本而不是"只监督半截 JSON"。"""
        from llm_extract.train.sft import build_supervised_example

        tok = _tokenizer()
        r = samples[0]
        payload = {k: r[k] for k in InquiryRecord.model_fields}
        target = build_target(InquiryRecord.model_validate(payload))
        # 把上限压到比 prompt 还短
        ex = build_supervised_example(tok, r["text"], target, max_seq_length=16)
        assert ex["input_ids"] == [], "超长样本应当被丢弃，而不是产生残缺监督"


class TestCollator:
    def test_labels_padded_with_ignore_index(self):
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import IGNORE_INDEX, SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        collator = SupervisedCollator(FakeTok())
        batch = collator([
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 8, 9]},
            {"input_ids": [5], "attention_mask": [1], "labels": [10]},
        ])
        assert batch["input_ids"].shape == (2, 3)
        # 短样本补齐的位置：attention_mask=0，labels 必须是 -100（不能拿 pad token 算 loss）
        assert batch["attention_mask"][1].tolist() == [1, 0, 0]
        assert batch["labels"][1].tolist() == [10, IGNORE_INDEX, IGNORE_INDEX]
        assert batch["input_ids"][1].tolist() == [5, 0, 0]

    def test_dtype_is_long(self):
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        batch = SupervisedCollator(FakeTok())([
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        ])
        for k, v in batch.items():
            if torch.is_tensor(v):
                assert v.dtype == torch.long, f"{k} 必须是 int64"
        # logits_to_keep 是标量 int（不是张量），直接交给模型 forward
        if "logits_to_keep" in batch:
            assert isinstance(batch["logits_to_keep"], int)


class TestTrainConfig:
    def test_yaml_roundtrip(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        from llm_extract.train.sft import TrainConfig

        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump({
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "learning_rate": 1e-4,
            "load_in_4bit": True,
        }), encoding="utf-8")
        cfg = TrainConfig.from_yaml(p)
        assert cfg.learning_rate == 1e-4
        assert cfg.load_in_4bit is True
        # 未指定的字段应当取默认值
        assert cfg.lora_r == 16

    def test_unknown_field_rejected(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        from llm_extract.train.sft import TrainConfig

        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"model_name_or_path": "x", "lerning_rate": 1e-4}),
                     encoding="utf-8")
        with pytest.raises(ValueError, match="未知字段"):
            TrainConfig.from_yaml(p)

    def test_shipped_configs_are_valid(self):
        from llm_extract.train.sft import TrainConfig

        for name in ("smoke.yaml", "qwen25-1.5b-lora.yaml", "qwen25-7b-qlora.yaml"):
            p = ROOT / "configs" / name
            if not p.exists():
                pytest.skip(f"{name} 不存在")
            cfg = TrainConfig.from_yaml(p)
            assert cfg.model_name_or_path
            assert cfg.max_seq_length > 0
            assert 0 < cfg.learning_rate < 1

    def test_qlora_config_actually_enables_4bit(self):
        from llm_extract.train.sft import TrainConfig

        p = ROOT / "configs" / "qwen25-7b-qlora.yaml"
        if not p.exists():
            pytest.skip("配置不存在")
        cfg = TrainConfig.from_yaml(p)
        assert cfg.load_in_4bit, "7B 配置必须开 4bit，否则 8GB 显存放不下"
        assert cfg.per_device_train_batch_size == 1
        assert cfg.gradient_checkpointing
        assert cfg.optim == "paged_adamw_8bit"


class TestLogitsToKeep:
    """`logits_to_keep` 优化：只对答案位置算 logits，是 8GB 卡跑 7B 的关键。

    但这是"看起来对、错了也不报错"的那类改动——如果 shift_labels 与保留的
    logits 区间错位，训练照样收敛、loss 照样下降，模型学的却是错位一环的标签。
    所以必须把对齐关系钉死。
    """

    def test_alignment_with_right_padding(self):
        """右侧 padding 时，答案不在序列末尾 —— 最容易错位的情形。"""
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import IGNORE_INDEX, SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        # 样本 A：真实长度 12，答案 3 个；样本 B：真实长度 10（尾部 2 pad），答案 2 个
        a_ids = list(range(100, 112))
        a_lbl = [IGNORE_INDEX] * 9 + [200, 201, 202]
        b_ids = list(range(200, 210))
        b_lbl = [IGNORE_INDEX] * 8 + [300, 301]

        batch = SupervisedCollator(FakeTok(), use_logits_to_keep=True)([
            {"input_ids": a_ids, "attention_mask": [1] * 12, "labels": a_lbl},
            {"input_ids": b_ids, "attention_mask": [1] * 10, "labels": b_lbl},
        ])

        k = batch["logits_to_keep"]
        t = 12                                   # max_len
        labels = batch["labels"]                 # [2, 12]
        shift = batch["shift_labels"]            # [2, k]

        assert shift.shape == (2, k)
        # 核心不变量：logits 保留的是最后 k 个位置 [t-k, t-1]，
        # 因此 shift[i, j] 必须等于 labels[i, t-k+j+1]（越界则为 ignore）
        for i in range(2):
            for j in range(k):
                pos = t - k + j + 1
                want = int(labels[i, pos]) if pos < t else IGNORE_INDEX
                got = int(shift[i, j])
                assert got == want, (
                    f"样本{i} 第{j}列错位：logits[{t-k+j}] 应配 labels[{pos}]={want}，实际 {got}"
                )

    def test_supervised_token_count_preserved(self):
        """优化不能改变"有多少个 token 参与 loss"。"""
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import IGNORE_INDEX, SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        feats = [
            {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5,
             "labels": [IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9]},
            {"input_ids": [1, 2, 3], "attention_mask": [1] * 3,
             "labels": [IGNORE_INDEX, 4, 5]},
        ]
        base = SupervisedCollator(FakeTok(), use_logits_to_keep=False)(feats)
        opt = SupervisedCollator(FakeTok(), use_logits_to_keep=True)(feats)

        n_base = int((base["labels"] != IGNORE_INDEX).sum())
        n_opt = int((opt["shift_labels"] != IGNORE_INDEX).sum())
        assert n_base == n_opt, f"监督 token 数被改变：{n_base} -> {n_opt}"

    def test_k_covers_all_supervision(self):
        """k 必须覆盖最靠前的监督位置，不能漏算。"""
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import IGNORE_INDEX, SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        feats = [
            {"input_ids": list(range(20)), "attention_mask": [1] * 20,
             "labels": [IGNORE_INDEX] * 5 + list(range(15))},
        ]
        batch = SupervisedCollator(FakeTok(), use_logits_to_keep=True)(feats)
        # first_sup = 5 (位置5的标签预测位置6...)，
        # 需要 logits[4] 来预测 labels[5]，故 k 至少为 20-4=16
        assert batch["logits_to_keep"] >= 16

    def test_all_ignored_batch_does_not_crash(self):
        """全 -100 的 batch（理论上不该出现）不能让 collator 崩掉。"""
        torch = pytest.importorskip("torch")
        from llm_extract.train.sft import IGNORE_INDEX, SupervisedCollator

        class FakeTok:
            pad_token_id = 0
            eos_token_id = 1

        batch = SupervisedCollator(FakeTok(), use_logits_to_keep=True)([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
             "labels": [IGNORE_INDEX] * 3},
        ])
        assert batch["input_ids"].shape[0] == 1
