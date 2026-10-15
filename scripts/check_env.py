"""环境自检：一键确认本机能否跑 LoRA / QLoRA，并打印选型结论。

换一台机器时先跑这个脚本，再决定用哪个模型与量化方案。
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check_env() -> int:
    """返回 0 表示 LoRA 可用，1 表示 QLoRA 可用但 bf16 LoRA 受限，2 表示 CPU only。"""
    section("1. 系统")
    print(f"平台      : {platform.platform()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"可执行文件: {sys.executable}")

    section("2. 关键依赖")
    try:
        import torch
        import transformers
        import peft
        import datasets
        import pandas as pd
        import numpy as np
    except ImportError as exc:
        print(f"[FAIL] 依赖缺失：{exc}")
        print("       请先执行 README 里的两步安装")
        return 3

    for mod in (torch, transformers, peft, datasets, pd, np):
        print(f"{mod.__name__:<14}: {getattr(mod, '__version__', 'n/a')}")

    section("3. GPU / CUDA")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda 不可用 —— 只能 CPU 训练（不推荐）")
        return 2

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    total_gib = props.total_memory / 1024**3
    print(f"设备名    : {torch.cuda.get_device_name(dev)}")
    print(f"计算能力  : sm_{props.major}{props.minor}")
    print(f"显存      : {total_gib:.2f} GiB")
    print(f"CUDA 版本 : torch 编译于 CUDA {torch.version.cuda}")

    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list:
        print(f"已编译架构: {', '.join(arch_list)}")
        want = f"sm_{props.major}{props.minor}"
        if want not in arch_list:
            print(f"[WARN] 当前 torch 未包含 {want} 内核，可能回退到 JIT 编译或报 no kernel image")

    section("4. 算力自检（bf16 矩阵乘，含预热）")
    try:
        import time

        torch.backends.cuda.matmul.allow_tf32 = True
        n = 4096
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        # 首次调用要初始化 cuBLAS 句柄并做内核选择，必须预热后再计时，
        # 否则会得到一个低一个数量级的假结果。
        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 50
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tflops = 2 * n**3 / dt / 1e12
        print(f"bf16 {n}³ 矩阵乘: {dt * 1000:.2f} ms/次 ≈ {tflops:.1f} TFLOPS")
        del a, b
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] GPU 矩阵乘失败：{exc}")
        return 2

    section("5. bitsandbytes（QLoRA 4bit 依赖）")
    bnb_ok = False
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes: {bnb.__version__}")
        from transformers import BitsAndBytesConfig  # noqa: F401

        print("[OK] 4bit 量化配置可导入")
        bnb_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] bitsandbytes 不可用：{exc}")
        print("       → 只能走 bf16 LoRA（1.5B 可以，7B 显存不够）")

    section("6. 选型结论")
    print(f"可用显存 {total_gib:.1f} GiB：")
    if total_gib >= 6:
        print("  ✅ Qwen2.5-1.5B-Instruct + LoRA (bf16)   —— 推荐作为主对照（无量化误差）")
    if total_gib >= 7.5 and bnb_ok:
        print("  ⚠️  Qwen2.5-7B-Instruct   + QLoRA (NF4)   —— 可行但紧张，须 bs=1 + 梯度检查点")
    if total_gib >= 16 and bnb_ok:
        print("  ✅ Qwen2.5-7B-Instruct   + bf16 LoRA     —— 显存充裕时的更优解")
    if total_gib < 8:
        print("  ⚠️  7B QLoRA 会非常紧张，建议只用 1.5B/3B")

    sec = shutil.which("nvidia-smi")
    if sec:
        out = subprocess.run([sec, "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        print(f"\nnvidia-smi: {out.stdout.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(check_env())
