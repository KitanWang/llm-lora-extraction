# 数据集统计

- 随机种子：`20251015`
- token 统计方式：`tokenizer`

## 1. 数据漏斗（清洗每一步剩下多少条）

| 步骤 | 剩余样本数 | 本步淘汰 | 说明 |
| :--- | ---: | ---: | :--- |
| raw | 3628 |  | 合成器原始产出（已含刻意注入的脏数据） |
| 非空正文 | 3628 | 0 | 去除空白文本 |
| 型号在正文中 | 3628 | 0 | 型号是必填字段，正文里找不到说明标签不可信 |
| 长度在合理区间 | 3472 | 156 | 超短信息不全 / 超长（历史邮件线程）浪费显存 |
| 精确去重后 | 3217 | 255 | 完全相同的正文（重复投递），保留首条 |
| 近重复去重后 | 3000 | 217 | 同骨架 + 同公司 + 同型号（重复录入），保留首条 |
| 加急标注自洽 | 3000 | 0 | is_urgent=True 但正文无加急措辞 |
| 无引用加急污染 | 3000 | 0 | is_urgent=False 但被引用的历史段落含加急词（标注噪声） |
| token 长度合规 | 3000 | 0 | prompt+target 超过 max_seq_length |

## 2. 切分结果

| 数据集 | 样本数 | 模板族 | 平均 token | 最大 token |
| :--- | ---: | :--- | ---: | ---: |
| train | 1590 | bullet_list, email_formal, im_chat, mixed_informal | 649 | 796 |
| val | 120 | bullet_list, email_formal, im_chat, mixed_informal | 649 | 726 |
| test_iid | 200 | bullet_list, email_formal, im_chat, mixed_informal | 649 | 746 |
| test_ood | 150 | phone_transcript, quote_reply | 712 | 768 |

## 3. 字段缺失率（可空字段占比）

| 字段 | train | val | test_iid | test_ood |
| :--- | ---: | ---: | ---: | ---: |
| customer_name | 11.3% | 5.0% | 14.5% | 1.3% |
| product_model | 0.0% | 0.0% | 0.0% | 0.0% |
| quantity | 11.1% | 8.3% | 11.0% | 9.3% |
| unit_price | 19.7% | 24.2% | 27.0% | 37.3% |
| currency | 0.0% | 0.0% | 0.0% | 0.0% |
| delivery_date | 17.4% | 11.7% | 14.5% | 10.7% |
| payment_terms | 24.2% | 26.7% | 28.0% | 23.3% |
| is_urgent | 0.0% | 0.0% | 0.0% | 0.0% |
| contact_phone | 20.6% | 26.7% | 22.5% | 4.0% |

## 4. 关键字段取值分布（训练集）

| 字段 | 分布 |
| :--- | :--- |
| currency | {'CNY': 1318, 'USD': 180, 'EUR': 92} |
| is_urgent | {False: 1152, True: 438} |
| payment_terms | {nan: 384, '预付全款': 216, '月结30天': 212, '货到付款': 201, '款到发货': 194, '30%定金+70%尾款': 192, '月结60天': 191} |
| quantity | min=50.0, median=1200.0, max=50000.0 |
| unit_price | min=4.0, median=482.31, max=979.9 |

## 5. 文本长度分布（字符数）

| 分位 | train | test_ood |
| :--- | ---: | ---: |
| min | 112 | 266 |
| p25 | 192 | 294 |
| p50 | 220 | 306 |
| p75 | 252 | 320 |
| max | 386 | 367 |
