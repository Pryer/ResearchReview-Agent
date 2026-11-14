# Embedding适配后真实E2E复跑

记录时间：2026-11-09 22:59:42 +08:00。

## 修复与回归

补全本地原生Embedding URL；app/clients/embedding_client.py同时接受text_index和index，双字段冲突拒绝，原有数量、整数、重复、范围、维度、有限值及范数检查保留。tests/test_hybrid_retrieval.py新增8项回归。20项定向通过；完整1892 passed、17 skipped，78.39秒。未放宽证据或发布门禁，未更改Agent角色及控制轮数。

## 真实执行

rragent Conda环境：D:/Anaconda/envs/rragent/python.exe，Python 3.11.15。相同T06：2020—2023年RAG开放域问答研究背景，中文约600字，至少3篇有效唯一引用，best_effort_on_failure=false。新隔离数据库/缓存/向量租户。1024次总模型请求，检索模型独立上限128；10M Token、1800秒、每篇8块。

- 运行：T06_20261109_225209_a10d653f0e；耗时320.906秒；终态blocked。
- 请求：123/1024，其中LLM83次、检索模型40次；账本Token806693。
- 候选177条；5张partial PaperCard。
- 原生pgvector查询21/21成功，检索降级0条；Embedding和重排实际调用成功。
- 内容证据原文位置回读156/156匹配。此项不代表独立人工语义验收。
- 上下文ID拒绝1次，在有界纠错后继续运行；不是终态错误。
- 主张规划、生成、独立核验、缺口诊断、改写和再规划已执行。最终错误agent_no_progress，候选隔离，公开正文无、公开引用0。
- 终态generation_readiness.ready=true，5篇可用证据、3篇授权和实际候选引用；仍有pending_result_validation=true且当前候选存在。不能据此宣称复验或公开交付通过。
- 运行前后源码身份一致，留档错误为空；没有达到总请求或Token上限。

## 产物

原始目录：data/e2e_runs/20261109_native_embedding/execute/T06_20261109_225209_a10d653f0e。run.json、result.json、observations.jsonl、research.db保留配置、源码身份、调用和检查点。历史运行及组件诊断未覆盖。

面试资料库09_真实运行测试/20261109_hybrid_e2e包含完整报告、原文回读CSV、卡片质量CSV、回归日志、兼容修复补丁与产物哈希。没有导出密钥、全文提示词或隔离候选稿。

本轮验证了新服务下真实原生检索恢复；尚未完成最终正文公开交付。剩余阻塞位于核验与恢复调度，不能再归因为此前Embedding地址或字段不兼容。
