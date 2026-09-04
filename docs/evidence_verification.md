# Evidence Card 与生成后引用验证

## 目标

系统在原有检索、排序和综述生成流程上增加两项能力：

1. 将论文摘要或开放获取全文转换为带来源位置的 Evidence Card。
2. 对生成综述逐句执行 Claim–Evidence 支持性验证。

主链路为：

```text
论文元数据 / PDF
→ Evidence Card
→ 综述生成
→ 事实性句子识别
→ 引用论文证据召回
→ supported / partially_supported / unsupported
→ 修改建议与 QLoRA 数据导出
```

## Evidence Card

`PaperCard` 新增：

- `evidence_spans`：原始证据片段、章节、页码、字符位置和来源等级。
- `field_evidence`：研究问题、方法、结果、局限等字段到 `evidence_id` 的映射。
- `relation_type`：为后续 Related Work 关系建模预留。

未开启 PDF 流程时，系统从摘要和元数据生成证据；开启 `ENABLE_PDF_PIPELINE=true`
后，解析器会保留逐页文本，证据片段可回溯到页码。

默认不逐篇调用 LLM，避免大规模检索时产生额外成本。需要更高质量字段抽取时可设置：

```dotenv
ENABLE_LLM_CARD_EXTRACTION=true
```

## Claim–Evidence 验证

验证器检查：

- 事实性句子是否包含引用；
- 引用的 `paper_id` 是否真实存在；
- 句子与证据片段的词项覆盖程度；
- 数值是否能够在证据中找到；
- “首次、显著、最先进”等强措辞是否获得证据支持；
- 标题或元数据级证据是否被错误用于支撑具体实验结论。

API 返回 `claim_verification`，其中包含统计指标和逐句结果。Streamlit 前端的
“引用验证”标签页可查看原句、证据位置、问题和修改建议。

## QLoRA 数据接口

规则验证器是基线，不应直接当作最终训练标签。可先导出候选记录：

```powershell
python scripts/export_claim_verification_data.py agent_output.json data/claim_verification.jsonl
```

正式训练前需要：

1. 人工复标 `supported / partially_supported / unsupported`；
2. 按论文或主题切分 train/dev/test，避免同论文泄漏；
3. 补充修改数值、替换数据集、强化措辞、跨论文嫁接等困难负样本；
4. 报告 Macro-F1、Unsupported Recall 和混淆矩阵；
5. 对比规则基线、通用大模型和 QLoRA 验证模型。

也可以从 Evidence Card 构造可控正负样本，并按 `paper_id` 切分，避免同论文泄漏：

```powershell
python scripts/build_claim_verifier_dataset.py agent_output.json data/claim_verifier
```

脚本会生成 `train.jsonl`、`dev.jsonl` 和 `test.jsonl`。其中包含原字段—证据正样本、
加入“首次/显著”措辞的部分支持样本，以及跨论文证据错配的不支持样本。自动标签均带有
`label_source=controlled_generation_requires_review`，训练前应人工复核。
