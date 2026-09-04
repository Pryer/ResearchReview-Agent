"""自定义异常类。

定义 ResearchReview-Agent 各层使用的领域异常，
便于在 API 层统一捕获并转换为对应的 HTTP 响应。
"""


class AgentBaseError(Exception):
    """Agent 基础异常。

    status_code 供 API 层映射 HTTP 状态：默认 400（请求侧问题）；
    上游可用性类错误应覆盖为 503，数据库类覆盖为 500。
    """

    status_code = 400

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------- 论文检索相关 ----------
class PaperSearchError(AgentBaseError):
    """论文检索失败（上游数据源不可用）。"""

    status_code = 503


class MetadataFetchError(AgentBaseError):
    """论文详情获取失败（上游数据源不可用）。"""

    status_code = 503


# ---------- PDF 相关 ----------
class PDFDownloadError(AgentBaseError):
    """PDF 下载失败。"""


class PDFParseError(AgentBaseError):
    """PDF 解析失败。"""


# ---------- PaperCard 相关 ----------
class PaperCardExtractionError(AgentBaseError):
    """PaperCard 抽取失败。"""


# ---------- 综述生成相关 ----------
class ReviewGenerationError(AgentBaseError):
    """综述生成失败。"""


# ---------- 引用相关 ----------
class CitationValidationError(AgentBaseError):
    """引用校验失败。"""


class CitationGenerationError(AgentBaseError):
    """参考文献生成失败。"""


# ---------- Agent 执行相关 ----------
class AgentExecutionError(AgentBaseError):
    """Agent 节点执行失败。"""


class IntentRecognitionError(AgentBaseError):
    """意图识别失败。"""


class SlotExtractionError(AgentBaseError):
    """槽位抽取失败。"""


# ---------- LLM 相关 ----------
class LLMInvocationError(AgentBaseError):
    """LLM 调用失败。"""


# ---------- 数据库相关 ----------
class DatabaseError(AgentBaseError):
    """数据库操作失败。"""

    status_code = 500
