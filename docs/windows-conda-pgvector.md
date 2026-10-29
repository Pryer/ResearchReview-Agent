# Windows Conda 原生 PostgreSQL / pgvector

项目可直接在 Conda 环境中运行原生 PostgreSQL，无需 Docker。数据库仍是独立进程，
项目 Python 客户端通过 psycopg 连接，不会把 PostgreSQL 当成 Python 内嵌数据库。

## rragent 当前安装

2026-10-28 安装并验证的 Windows 64 位组合：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15，保留原版本 |
| PostgreSQL 服务 | 16.15 |
| pgvector 服务端扩展 | 0.8.6 |
| psycopg / psycopg-binary | 3.3.6 |
| pgvector Python 适配包 | 0.5.1 |

Conda 的 pgvector 扩展包与同名 Python 适配包不是同一个组件，二者均已安装。
Windows 包依赖解析选择 PostgreSQL 16；不要直接套用其他平台的 PostgreSQL 17 版本约束。
安装期间按依赖计划更新了 OpenSSL 和 CA 证书；Python、NumPy 及已有 Python 包版本没有改变，
pip check 通过。

## 从头安装

在 Anaconda Prompt 中执行：

```bat
conda install -n rragent --override-channels -c conda-forge --freeze-installed "postgresql>=16,<17" "pgvector>=0.8,<0.9"
conda activate rragent
python -m pip install -r requirements-vectorstore.txt
```

这些命令只安装组件；首次部署还需初始化数据目录、配置密码认证、创建数据库和执行显式迁移。
本项目当前实例已经完成初始化，不要对现有目录重新运行 initdb。

## 已配置的项目实例

- 数据目录：项目根目录下 data/postgresql_rragent。
- 仅监听本机，使用密码认证。
- 应用向量库和专用测试库分离，业务 SQLite 保持不变。
- 自动生成的连接凭据已保存到私有 .env，不在文档或输出中展示。
- .env、数据目录和原始诊断均被 Git 忽略。

该实例是普通用户后台进程，没有注册 Windows 服务；重启电脑后需手动启动。
在项目根目录的 Anaconda Prompt 中：

```bat
conda activate rragent
pg_ctl -D "data/postgresql_rragent" status
```

未运行时启动：

```bat
pg_ctl -D "data/postgresql_rragent" -l "data/postgresql_rragent/server.log" -w start
```

需要停止时：

```bat
pg_ctl -D "data/postgresql_rragent" -m fast -w stop
```

使用 rragent 的 Python 检查，不要误用未安装驱动的 base 环境：

```bat
python scripts/manage_vectorstore.py health
python scripts/check_rag_runtime.py
```

已有应用服务需要重启，才能重新读取私有连接配置。启动检查不消耗 Provider Token。

## 验证结果

原生服务健康检查返回 PostgreSQL 16.15 / vector 0.8.6，项目启动检查 ready=true。
在独立测试库启用 RRA_VECTOR_TEST_NATIVE=1 后执行数据库集成和启动迁移回归，
34 passed，4.43 秒；包含此前跳过的多连接竞争用例。详细记录见
[原生数据库验收](validation/2026-10-28-conda-native-pgvector.md)。

本次没有调用真实 LLM、Embedding 或 Reranker；依赖和数据库就绪不代表真实模型效果已验收。
