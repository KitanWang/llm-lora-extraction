# 待验证：7B 训练速度差异

## 现象
- `scripts/sweep_train_config.py` 实测：1.88 s/micro-batch（bs=1, seq_len=832）
- 实际训练（bs=1, grad_accum=16）：63.5 s/step ÷ 16 = **3.97 s/micro-batch**

即实际比 sweep 慢 **2.1 倍**，且样本长度相近（652 vs 667 token）。

## 已排除
- 样本长度差异（实测相近）
- 数据集版本差异
- 其他进程争抢（GPU 利用率 99%，无并发任务）

## 候选原因
1. **显存分配器碎片化**：训练时显存 7550/8151 MiB（94%），
   接近上限时 native 分配器会频繁 cudaMalloc/cudaFree 并同步。
   sweep 时余量更充裕（6.04 GiB 峰值），未触发该问题。
   候选解法：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. **gradient checkpointing 的 eager 重算开销**：sweep 与训练都开了，
   但 sweep 只跑 3-6 步，未进入稳态。
3. **paged_adamw_8bit 的分页开销**：每 16 个 micro-batch 才 step 一次，
   影响应被摊薄，但首次分页可能有成本。

## 验证方法（需在无并发任务时执行）
1. 基线：现有配置跑 20 步，记录 s/step
2. 对照组：加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 重跑
3. 若有效，写入配置与 README

## 影响评估
- 只影响**训练时长**，不影响结果正确性（loss 曲线正常，梯度有限）
- 200 步 × 63.5s ≈ 3.5 小时，在可接受范围
- 但「提升 2 倍训练速度」值得作为后续优化项
