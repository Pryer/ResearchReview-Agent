# Conda 原生 pgvector 安装与验收（2026-10-28）

范围为当前工作区、Windows 64 位 rragent 环境，不将安装结果等同于真实研究 E2E。

## 安装与隔离

Conda 安装 PostgreSQL 16.15、pgvector 服务端扩展 0.8.6 及必要库。
实际依赖解析显示 Windows pgvector 包要求 PostgreSQL 16；初始 PostgreSQL 17 约束不兼容，
改用解析成功的 16 系列后执行安装，没有强制忽略依赖。
更新 OpenSSL 至 3.6.5 和 CA 证书；原 Python 3.11.15、NumPy 2.4.6 和已有 Python 包版本未变。
另安装 psycopg/psycopg-binary 3.3.6、pgvector Python 适配包 0.5.1；pip check 通过。

新建项目专用 PostgreSQL 数据目录，启用密码认证并限制本机监听。
应用库和专用测试库分离，业务 SQLite 未修改；连接凭据仅保存在被忽略的私有配置中。
应用库已通过现有显式迁移创建运行表和扩展；未使用测试数据库冒充应用库。

## 验证

| 验证项 | 实际结果 |
| --- | --- |
| 服务 | 原生 PostgreSQL 16.15，Windows Visual C++ 构建，64 位 |
| 扩展 | SQL 查询确认 vector 0.8.6 |
| Python 驱动 | rragent 中可导入，固定版本符合依赖清单 |
| 现有 Python 包 | 安装前后比对，无旧包移除或版本变化 |
| 依赖一致性 | pip check 通过 |
| 应用向量库迁移 | 成功 |
| 项目启动检查 | ready=true，vector_connection=true，vector_schema=true |
| 数据库/启动迁移回归 | 34 passed，4.43 秒，无跳过 |

测试命令为 rragent Python 执行：

```text
python -X utf8 -m pytest tests/test_pgvector_store.py tests/test_rag_runtime_migration.py -q --tb=short --show-capture=no
```

运行时设置 RRA_VECTOR_TEST_DATABASE_URL 为专用测试库连接，RRA_VECTOR_TEST_NATIVE=1。
17 项数据库测试和 17 项启动/迁移测试全部通过，包括原生多连接竞争；不同于此前单连接
PGlite 验证。用例仍是有界正确性回归，不是生产并发规模、吞吐或延迟基准。

## 启动边界及限制

受限工具令牌下 pg_ctl 无法创建自己的受限令牌，普通用户执行上下文启动成功。
后台启动改用文件日志和关闭多余继承句柄，避免捕获输出管道随服务存活而使安装检测不返回。
没有关闭数据库认证、放开外部监听、注册管理员服务或修改生产代码来绕过这些边界。

最终数据库后台运行；操作说明见 [Windows Conda 部署](../windows-conda-pgvector.md)。
原始数据、安装计划、依赖前后快照及测试输出仅保存在本地被忽略的 data/ 下。
未提交连接凭据，没有新的真实 Provider 请求；真实论文抽取、检索效果和综述成功仍须单独验收。
