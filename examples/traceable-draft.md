# 可回查草稿示例

> **研究草稿 · 正式交付门禁待通过 · 语义核验尚未开始**

本示例使用仓库的自制 PDF 测试素材。正文中的数字属于测试数据，来源与核验状态随正文列出。

## 正文

双栏测试的 PDF 内容流先写入右栏，版面候选按左栏、右栏顺序输出。［1］

表格测试中，First 方法的 F1 为 86%，Accuracy 为 91%。［2］

## 逐句回查

| 正文引用 | 可回查来源 | 核验状态 |
| --- | --- | --- |
| ［1］ | 在来源中查看 `test_layout_reorders_two_columns_without_merging_page_offsets` 的写入顺序与两版输出断言。 | 自动回归核对顺序，语义核验未开始 |
| ［2］ | 在来源中查看 `test_layout_table_retains_row_and_column_identity` 的表头、First 行和网格相等断言。 | 自动回归核对行列，语义核验未开始 |

来源与复现命令见 [PDF 版面对照评测](../docs/validation/pdf-layout.md) 和 [PDF 测试素材](../tests/test_document_ingestion.py)。
