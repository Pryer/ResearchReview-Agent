# 自动HTML运行报告

每次研究执行结束后生成独立报告，包含Main每轮目标、决策、上下文变化，专业Agent收到的输入和角色记忆，以及执行后返回Main的结果。报告默认按中文时间线阅读，原始字段折叠展示，支持搜索、角色筛选和字号调整。电脑正文整页滚动，手机自动调整为单列。

## 使用方式

重启后端和Streamlit进程以加载新代码。新任务默认开启：

```dotenv
EXECUTION_REPORT_ENABLED=true
EXECUTION_REPORT_DIR=./data/execution_reports
```

默认文件位置（相对路径统一以`ResearchReview-Agent`仓库根目录为基准，与启动终端目录无关；绝对路径遵守显式配置）：

```text
data/execution_reports/<report_id>/
  report.html          # 可直接双击，离线阅读
  report.json.gz       # 结构化记录
  main-01.json         # 各轮Main记录
  subagent-01.json     # 各次专业委派
  bundle.zip           # 上述文件的完整下载包
```

直接Python调用返回的`execution_report.html_path`可以定位页面。会话/API调用返回`execution_report`描述，使用`html_url`查看、`download_url`下载。Streamlit在任务消息下提供“运行报告”下载入口，由服务端携带现有API密钥读取，不把密钥放进URL或HTML。

`scripts/run_real_e2e.py`会在当前E2E目录额外保存`execution_report.html`和`execution_report.zip`，不需要同时启动API服务。

## 页面阅读顺序

1. **按时间顺着读**：本轮为何选择这个动作，专业Agent做了什么，资料、正文或发布状态发生了什么变化。
2. **Main看到了什么**：本轮目标、状态、关键证据、已有决策、未解决问题，及相对前次上下文的差异。
3. **子Agent如何执行**：Main分发的补充材料、操作契约输入、同角色上次结果、依赖变化、工具步骤、结构化汇报和更新后的工作摘要。
4. **结果与检查**：本次请求用量、累计预算、首稿时间、终态和上下文归档范围。

“模型决策轮数”与“请求尝试数”采用不同口径：同一轮纠错或重试可能发出多次请求，实际用量仍以共享预算账本为准。代码策略交接单独标记，不记作额外Main模型调用。专业工具内部的LLM调用也不冒充多个Agent并行自治。

## 接口与隔离

```text
GET /api/reviews/sessions/{session_id}/reports
GET /api/reviews/sessions/{session_id}/reports/{report_id}/files/report.html
GET /api/reviews/sessions/{session_id}/reports/{report_id}/files/bundle.zip
```

接口沿用现有API密钥保护；读取同时匹配会话和报告ID，文件必须在登记名单中。HTML响应限制网络连接、表单和嵌入。报告资料类型仅供service读取，不进入Main或专业Agent的证据授权。

每次执行有独立报告ID，查询范围冻结在本次执行的起止版本，不读取同会话后续运行。生成报告不调用模型，不修改正文、证据门禁、请求预算或研究终态。文件/观察器故障会返回`execution_report.state=failed`并记录原因类型，原研究结果与取消异常保持。任务终态刚提交而报告尚未发布时，下载入口显示待整理并可重试。

## 覆盖范围

- 正常完成、草稿、阻断、失败、取消与等待澄清均有报告；未进入Main的任务可有空轨迹。
- 新执行直接捕获策略交接的五字段快照，避免将上一轮模型输入当成这一轮上下文。
- 专业委派继承同角色摘要与依赖增量；完整原文等大字段保存带长度、哈希和资料引用的预览，避免重复PDF撑大页面。
- 进程崩溃后，重启恢复会为中断/取消任务生成说明报告；无法恢复的本轮内存轨迹明确为空。
- 业务模型每次HTTP完整Prompt此前并未逐次持久化，本功能不事后伪造，也不输出内部思考协议。
- 报告含任务上下文，属于诊断资料；`data/`沿用Git忽略规则，正文导出与报告分开。

实现入口：`execution_report_capture.py`负责执行局部观察，`execution_report_collector.py`从冻结检查点和归档组合轨迹，`execution_report_service.py`负责文件及登记；会话装饰器、直接图入口和后台任务终态分别接入同一服务，没有新增研究调度器或模型汇报循环。

## 项目内文件位置

| 内容 | 位置 |
| --- | --- |
| 运行报告实现 | `app/services/execution_report_*.py` |
| HTML模板 | `app/services/templates/execution_report.html` |
| 回归测试 | `tests/test_execution_reports.py` |
| 历史任务回放工具 | `scripts/export_execution_report.py` |
| 生成报告 | `data/execution_reports/<report_id>/` |
| 本地验证日志 | `data/validation/20261111_execution_reports/` |
| 开发历史快照 | `.local-docs/implementation/20261111_auto_reports/` |

已完成任务的报告可只读回放，不修改研究数据库、不调用模型：

```powershell
python scripts/export_execution_report.py --run-dir data/e2e_runs/20261110_draft_delivery/accepted/DRAFT5_20261111_002405_85448b1da8
```

默认导出到项目配置的报告目录；`--output-dir`可指定其它目录，相对参数均按仓库根目录解析。仅回放已结束且没有执行租约的会话。该命令只写报告文件，不向原研究库补写登记。

本地样例：`data/execution_reports/648eefd1f884f29e184f6f5822f208a9/report.html`。原E2E目录也提供`execution_report.html`与`execution_report.zip`。生成物、日志和开发快照沿用Git忽略规则。
