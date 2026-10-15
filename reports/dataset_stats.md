# 数据集统计

- 随机种子：`20251015`
- token 统计方式：`tokenizer`

## 1. 数据漏斗（清洗每一步剩下多少条）

| 步骤 | 剩余样本数 | 本步淘汰 | 说明 |
| :--- | ---: | ---: | :--- |
| raw | 3635 |  | 合成器原始产出（已含刻意注入的脏数据） |
| 非空正文 | 3635 | 0 | 去除空白文本 |
| 型号在正文中 | 3635 | 0 | 型号是必填字段，正文里找不到说明标签不可信 |
| 长度在合理区间 | 3471 | 164 | 超短信息不全 / 超长（历史邮件线程）浪费显存 |
| 精确去重后 | 3219 | 252 | 完全相同的正文（重复投递），保留首条 |
| 近重复去重后 | 3000 | 219 | 同骨架 + 同公司 + 同型号（重复录入），保留首条 |
| 标签自洽 | 3000 | 0 | is_urgent=True 但正文无加急措辞 |
| token 长度合规 | 3000 | 0 | prompt+target 超过 max_seq_length |

## 2. 切分结果

| 数据集 | 样本数 | 模板族 | 平均 token | 最大 token |
| :--- | ---: | :--- | ---: | ---: |
| train | 1584 | bullet_list, email_formal, im_chat, mixed_informal | 648 | 796 |
| val | 120 | bullet_list, email_formal, im_chat, mixed_informal | 650 | 719 |
| test_iid | 200 | bullet_list, email_formal, im_chat, mixed_informal | 649 | 751 |
| test_ood | 150 | phone_transcript, quote_reply | 711 | 768 |

## 3. 字段缺失率（可空字段占比）

| 字段 | train | val | test_iid | test_ood |
| :--- | ---: | ---: | ---: | ---: |
| customer_name | 11.7% | 13.3% | 14.5% | 1.3% |
| product_model | 0.0% | 0.0% | 0.0% | 0.0% |
| quantity | 10.0% | 12.5% | 8.0% | 8.7% |
| unit_price | 22.7% | 24.2% | 17.5% | 36.0% |
| currency | 0.0% | 0.0% | 0.0% | 0.0% |
| delivery_date | 17.0% | 17.5% | 16.5% | 8.7% |
| payment_terms | 23.9% | 25.0% | 23.0% | 32.0% |
| is_urgent | 0.0% | 0.0% | 0.0% | 0.0% |
| contact_phone | 19.9% | 19.2% | 18.5% | 6.7% |

## 4. 关键字段取值分布（训练集）

| 字段 | 分布 |
| :--- | :--- |
| currency | {'CNY': 1312, 'USD': 185, 'EUR': 87} |
| is_urgent | {False: 1159, True: 425} |
| payment_terms | {nan: 379, '月结30天': 232, '货到付款': 205, '款到发货': 201, '30%定金+70%尾款': 198, '预付全款': 185, '月结60天': 184} |
| quantity | min=50.0, median=1200.0, max=50000.0 |
| unit_price | min=6.0, median=486.0, max=979.9 |

## 5. 文本长度分布（字符数）

| 分位 | train | test_ood |
| :--- | ---: | ---: |
| min | 99 | 265 |
| p25 | 190 | 293 |
| p50 | 220 | 308 |
| p75 | 251 | 325 |
| max | 386 | 367 |
