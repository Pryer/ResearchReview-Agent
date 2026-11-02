# 字段证据审查与独立验收

审查材料的标注完成、AI复核和独立人工验收是不同状态。
浏览器导出仍保留human_gold=false；正式状态由校验器从来源和签署重新计算，不信任status或quote_exact。

## 审查和签署

1. 打开本地REVIEW_21_FIELDS.html，可导入同版本AI预审或之前导出的进度。
2. 逐字段阅读完整原文块；相关项复制连续原文，不翻译改写。可选relevant、irrelevant、uncertain，未读保持not_reviewed。
3. 人工复核时选择human，实际对照原始PDF版式，逐条勾选独立确认，填写审查者及独立核对声明。
4. 导出JSON并运行独立校验；仅修改审查者类型或总声明不能把导入的AI标注签成人审。

```powershell
conda activate rragent
python scripts/validate_field_review.py --template <review_template.json> --review <field-evidence-review.json> --output <validation.json>
```

校验器检查模板字节哈希、PDF来源哈希、任务/块身份、重复或遗漏、原文摘录、审查者类型、
带时区的签署时间、原PDF核对声明与逐条人工确认。
annotated complete只说明标注覆盖；AI标注始终不增加human_reviewed_tasks。
部分人审可计入人审覆盖率，但所有任务完成且全部明确判断前，human_gold仍为false。
uncertain不会被删除或自动转换为相关。

该流程是本地内容与自声明签署校验，不验证现实身份、不构成密码学签名，也不能证明审查者实际阅读。
实际人审仍由阅读者负责，模型和Agent不得代签。

## 解析版本变化后的AI预审

新解析、过滤或切块可能改变chunk_id，旧文件必须保留，不能凭ID直接回填新模板。
以下命令仅迁移AI预审，不迁移人工签署：

```powershell
python scripts/reanchor_ai_field_review.py --old-template <old_template.json> --new-template <new_template.json> --review <ai_review.json> --output-dir <new_directory>
```

只在PDF版本、页码、Offset和原文全部一致且唯一时迁移，原文区间不一致或已排除的行另行归档。
所有迁移行保持reviewer_type=ai、human_gold=false，逐条人工确认重置，不重写原文件。

新解析版本可先生成审查包，不复用旧块排名或旧参考标签来计算新指标：

```powershell
python scripts/diagnose_evidence_funnel.py --source-dir <new_parsed_sources> --evaluation-dir <old_evaluation> --output-dir <new_review_directory> --review-only
```

新版运行须加载更新后的代码，索引按新parser/chunker处理指纹重建；旧快照和旧会话不自动重判为新版验收结果。
