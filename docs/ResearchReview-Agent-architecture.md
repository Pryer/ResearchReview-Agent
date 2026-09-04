# ResearchReview-Agent 架构与函数设计文档

> 面向 Coding Agent 的实现说明  
> 目标：生成一个「根据用户指令检索开放论文，并基于论文生成文献综述」的科研 Agent。  
> 核心关键词：Intent Recognition、Slot Filling、Paper Search、Agentic RAG、PaperCard、Literature Review、Citation Validation。

---

## 1. 项目定位

### 1.1 项目名称

**ResearchReview-Agent：面向计算机视觉论文的自动化文献综述智能体**

### 1.2 项目目标

用户输入自然语言请求，例如：

```text
帮我调研近五年少样本动作识别相关论文，并生成中文文献综述，引用不少于 15 篇。
```

系统应自动完成：

```text
意图识别
→ 槽位抽取
→ 查询规划
→ 多源论文检索
→ 去重与排序
→ 获取论文元数据
→ 下载开放获取 PDF
→ 解析 PDF 或摘要
→ 抽取 PaperCard
→ 文献聚类
→ 生成文献综述
→ 生成参考文献
→ 校验引用
```

### 1.3 合规原则

Coding Agent 必须遵守：

1. 只检索公开论文元数据。
2. 只下载开放获取 PDF。
3. 不绕过付费墙。
4. 不编造论文、作者、年份、venue、DOI、引用量、实验结果。
5. 如果只有摘要，`PaperCard.evidence_source = "abstract"`。
6. 如果解析到全文，`PaperCard.evidence_source = "full_text"`。
7. 综述中的引用必须来自已检索到的论文。

---

## 2. 总体架构

```text
User Query
   ↓
Intent Recognition
   ↓
Slot Extraction
   ↓
Task Planner
   ↓
Task Router
   ↓
Paper Search Tools
   ↓
Paper Ranking / Deduplication
   ↓
Metadata Fetcher
   ↓
Open Access PDF Downloader
   ↓
PDF Parser
   ↓
PaperCard Extractor
   ↓
Vector Store / Local Library
   ↓
Paper Clustering
   ↓
Review Generator
   ↓
Citation Generator + Citation Validator
   ↓
Final Answer
```

---

## 3. 技术栈建议

### MVP

```text
Python
FastAPI
Pydantic
SQLite
SQLAlchemy
requests
arXiv API
Semantic Scholar API
OpenAlex API
PyMuPDF
Streamlit
```

### 增强版

```text
LangGraph
LangChain / LlamaIndex
FAISS / Chroma / Milvus
GROBID
PostgreSQL
Celery / RQ
OpenAI / DeepSeek / Qwen / GLM
Docker
```

---

## 4. 目录结构

```text
research_review_agent/
├── README.md
├── requirements.txt
├── .env.example
├── run_api.py
├── run_streamlit.py
│
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes_paper.py
│   │   ├── routes_review.py
│   │   └── routes_library.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── exceptions.py
│   │
│   ├── schemas/
│   │   ├── paper_schema.py
│   │   ├── review_schema.py
│   │   ├── agent_schema.py
│   │   └── common_schema.py
│   │
│   ├── agent/
│   │   ├── intent.py
│   │   ├── slot_extractor.py
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── prompts.py
│   │
│   ├── tools/
│   │   ├── search_papers.py
│   │   ├── fetch_metadata.py
│   │   ├── download_pdf.py
│   │   ├── parse_pdf.py
│   │   ├── extract_paper_card.py
│   │   ├── rank_papers.py
│   │   ├── cluster_papers.py
│   │   ├── generate_review.py
│   │   └── generate_citation.py
│   │
│   ├── clients/
│   │   ├── arxiv_client.py
│   │   ├── semantic_scholar_client.py
│   │   ├── openalex_client.py
│   │   └── crossref_client.py
│   │
│   ├── services/
│   │   ├── paper_service.py
│   │   ├── review_service.py
│   │   ├── library_service.py
│   │   └── citation_service.py
│   │
│   ├── database/
│   │   ├── db.py
│   │   ├── models.py
│   │   └── repositories.py
│   │
│   ├── vectorstore/
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   └── retriever.py
│   │
│   ├── utils/
│   │   ├── text_cleaner.py
│   │   ├── pdf_utils.py
│   │   ├── deduplicate.py
│   │   ├── date_utils.py
│   │   └── file_utils.py
│   │
│   └── frontend/
│       └── streamlit_app.py
│
├── data/
│   ├── pdfs/
│   ├── parsed/
│   ├── reviews/
│   └── vector_index/
│
└── tests/
    ├── test_intent.py
    ├── test_slot_extractor.py
    ├── test_search.py
    ├── test_rank.py
    ├── test_extract.py
    ├── test_review.py
    └── test_citation.py
```

---

## 5. Schema 设计

### 5.1 `app/schemas/paper_schema.py`

```python
from typing import List, Optional
from pydantic import BaseModel


class PaperSearchRequest(BaseModel):
    query: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    max_results: int = 20
    sources: List[str] = ["arxiv", "semantic_scholar", "openalex"]


class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source: str = "unknown"


class PaperCard(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    research_problem: str
    method: str
    dataset: Optional[str] = None
    metrics: List[str] = []
    results: Optional[str] = None
    contributions: List[str] = []
    limitations: List[str] = []
    relevance_reason: str
    evidence_source: str  # abstract / full_text / metadata
```

### 5.2 `app/schemas/review_schema.py`

```python
from typing import List, Optional
from pydantic import BaseModel
from app.schemas.paper_schema import PaperCard


class ReviewRequest(BaseModel):
    topic: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    max_papers: int = 20
    language: str = "zh"
    citation_style: str = "gbt7714"
    review_type: str = "survey"


class ReviewSection(BaseModel):
    title: str
    content: str
    citations: List[str] = []


class LiteratureReview(BaseModel):
    topic: str
    sections: List[ReviewSection]
    references: List[str]
    paper_cards: List[PaperCard]
```

### 5.3 `app/schemas/agent_schema.py`

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reason: str
    slots: Dict[str, Any] = {}


class AgentStep(BaseModel):
    step_name: str
    tool_name: Optional[str] = None
    input_data: Dict[str, Any] = {}
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"


class AgentOutput(BaseModel):
    answer: str
    steps: List[AgentStep] = []
    references: Optional[List[str]] = None
```

---

## 6. Agent 模块设计

### 6.1 `app/agent/intent.py`

职责：识别用户意图。

支持意图：

```python
class IntentType(str, Enum):
    SEARCH_PAPERS = "search_papers"
    READ_PAPER = "read_paper"
    GENERATE_REVIEW = "generate_review"
    COMPARE_PAPERS = "compare_papers"
    GENERATE_REFERENCES = "generate_references"
    EXTRACT_PAPER_CARD = "extract_paper_card"
    FIND_DATASETS = "find_datasets"
    FIND_TRENDS = "find_trends"
    GENERAL_QA = "general_qa"
```

函数：

```python
def recognize_intent(user_query: str, llm=None) -> IntentResult:
    """先规则识别，低置信度时再用 LLM。"""


def rule_based_intent_recognition(user_query: str) -> IntentResult:
    """基于关键词规则判断意图。"""


def llm_based_intent_recognition(user_query: str, llm) -> IntentResult:
    """调用 LLM 输出 JSON 格式意图结果。"""
```

规则建议：

```text
包含「综述 / 文献综述 / 研究现状 / survey / review」→ generate_review
包含「找论文 / 搜索 / 检索 / 推荐论文 / papers」→ search_papers
包含「总结这篇 / 读一下 / 解析论文」→ read_paper
包含「对比 / 比较 / 区别 / compare」→ compare_papers
包含「参考文献 / bibtex / apa / gb/t / 引用格式」→ generate_references
包含「数据集 / dataset / benchmark」→ find_datasets
包含「趋势 / 发展方向 / 未来方向 / 研究热点」→ find_trends
```

---

### 6.2 `app/agent/slot_extractor.py`

职责：抽取用户请求中的参数。

函数：

```python
def extract_slots(user_query: str, intent: IntentType, llm=None) -> dict:
    """统一抽取槽位。"""


def extract_topic(user_query: str) -> str:
    """抽取研究主题。"""


def extract_year_range(user_query: str) -> tuple[int, int]:
    """抽取年份范围；近五年、本年、2021-2025 等。"""


def extract_max_papers(user_query: str) -> int:
    """抽取论文数量；默认 20。"""


def extract_language(user_query: str) -> str:
    """中文 zh，英文 en。"""


def extract_citation_style(user_query: str) -> str:
    """gbt7714 / apa / ieee / bibtex。"""


def llm_extract_slots(user_query: str, intent: IntentType, llm) -> dict:
    """规则无法稳定提取时调用 LLM。"""
```

输出示例：

```python
{
    "topic": "课堂行为识别",
    "start_year": 2021,
    "end_year": 2025,
    "max_papers": 20,
    "language": "zh",
    "citation_style": "gbt7714"
}
```

---

### 6.3 `app/agent/planner.py`

职责：生成执行计划。

函数：

```python
def build_search_plan(user_query: str, llm=None) -> dict:
    """整合意图识别和槽位抽取，生成计划。"""


def choose_workflow(intent: IntentType) -> str:
    """根据意图选择工作流。"""


def generate_search_keywords(topic: str) -> list[str]:
    """把中文主题扩展为中英文检索关键词。"""
```

工作流映射：

```text
search_papers → paper_search_workflow
generate_review → literature_review_workflow
read_paper → single_paper_reading_workflow
compare_papers → paper_comparison_workflow
generate_references → citation_generation_workflow
find_datasets → dataset_search_workflow
find_trends → trend_analysis_workflow
general_qa → general_qa_workflow
```

---

### 6.4 `app/agent/router.py`

职责：路由和条件判断。

函数：

```python
def route_task(state: ResearchAgentState) -> str:
    """返回当前 state 对应的 workflow。"""


def should_parse_pdf(state: ResearchAgentState) -> bool:
    """判断是否需要解析 PDF。"""


def should_generate_review(state: ResearchAgentState) -> bool:
    """判断是否进入综述生成节点。"""


def should_retry_search(state: ResearchAgentState) -> bool:
    """候选论文不足时是否扩展关键词重试。"""
```

---

### 6.5 `app/agent/state.py`

职责：定义 Agent 全局状态。

```python
from typing import TypedDict, List, Dict, Any


class ResearchAgentState(TypedDict, total=False):
    user_query: str
    intent: str
    confidence: float
    topic: str
    keywords: List[str]
    start_year: int
    end_year: int
    max_papers: int
    language: str
    citation_style: str
    workflow: str

    candidate_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    paper_details: List[Dict[str, Any]]
    pdf_paths: Dict[str, str]
    parsed_papers: Dict[str, Dict[str, Any]]
    paper_cards: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    review: str
    references: List[str]

    errors: List[str]
    steps: List[Dict[str, Any]]
```

---

### 6.6 `app/agent/nodes.py`

职责：定义 Agent 工作流节点。

函数：

```python
def append_step(state: ResearchAgentState, step_name: str, status: str, extra=None):
    """记录 Agent 执行步骤。"""


def plan_node(state: ResearchAgentState) -> ResearchAgentState:
    """解析用户需求，生成检索计划。"""


def search_node(state: ResearchAgentState) -> ResearchAgentState:
    """调用论文检索工具，获取候选论文。"""


def rank_node(state: ResearchAgentState) -> ResearchAgentState:
    """候选论文去重、排序、筛选。"""


def fetch_detail_node(state: ResearchAgentState) -> ResearchAgentState:
    """补全论文详情。"""


def download_pdf_node(state: ResearchAgentState) -> ResearchAgentState:
    """下载开放获取 PDF；失败则降级为摘要分析。"""


def parse_pdf_node(state: ResearchAgentState) -> ResearchAgentState:
    """解析 PDF，抽取摘要、引言、方法、实验、结论、参考文献。"""


def extract_card_node(state: ResearchAgentState) -> ResearchAgentState:
    """生成 PaperCard。"""


def cluster_node(state: ResearchAgentState) -> ResearchAgentState:
    """按方法、任务、数据集、年份聚类论文。"""


def generate_review_node(state: ResearchAgentState) -> ResearchAgentState:
    """生成文献综述。"""


def citation_check_node(state: ResearchAgentState) -> ResearchAgentState:
    """生成并校验参考文献。"""


def final_answer_node(state: ResearchAgentState) -> ResearchAgentState:
    """组装最终输出。"""
```

---

### 6.7 `app/agent/graph.py`

职责：组织整体工作流。

MVP 可先使用顺序函数，不强依赖 LangGraph。

```python
def run_research_agent(user_query: str) -> dict:
    """运行完整 Agent 流程。"""
```

建议流程：

```text
plan_node
→ search_node
→ rank_node
→ fetch_detail_node
→ download_pdf_node
→ parse_pdf_node
→ extract_card_node
→ cluster_node
→ generate_review_node
→ citation_check_node
→ final_answer_node
```

分支规则：

```text
只查论文：plan → search → rank → final
生成综述：完整流程
读单篇论文：fetch → pdf/abstract parse → card → final
论文对比：fetch → card → compare → final
参考文献：fetch → citation → final
```

---

### 6.8 `app/agent/prompts.py`

集中管理提示词。

变量：

```python
INTENT_RECOGNITION_PROMPT = """..."""
SLOT_EXTRACTION_PROMPT = """..."""
PAPER_CARD_EXTRACTION_PROMPT = """..."""
CLUSTER_PROMPT = """..."""
LITERATURE_REVIEW_PROMPT = """..."""
CITATION_CHECK_PROMPT = """..."""
```

提示词要求：

1. 所有 LLM 结构化输出都要求 JSON。
2. PaperCard 抽取不得编造字段。
3. 综述生成只能引用输入论文。
4. 引用校验需要输出 `valid / missing_citations / unused_references / suggestions`。

---

## 7. Tools 模块设计

### 7.1 `app/tools/search_papers.py`

职责：统一论文检索入口。

函数：

```python
def search_papers(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
    sources: list[str],
) -> list[PaperMetadata]:
    """从多个开放论文数据源检索论文。"""


def merge_search_results(results: list[list[PaperMetadata]]) -> list[PaperMetadata]:
    """合并不同数据源结果。"""


def normalize_paper_metadata(raw_paper: dict, source: str) -> PaperMetadata:
    """统一不同来源的元数据格式。"""


def filter_by_year(
    papers: list[PaperMetadata],
    start_year: int,
    end_year: int,
) -> list[PaperMetadata]:
    """按年份过滤。"""
```

---

### 7.2 `app/tools/rank_papers.py`

职责：去重、相关性打分、质量打分。

函数：

```python
def normalize_title(title: str) -> str:
    """标题归一化。"""


def title_similarity(a: str, b: str) -> float:
    """计算标题相似度。"""


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """按 DOI、arXiv ID、标题去重。"""


def compute_relevance_score(paper: dict, topic: str) -> float:
    """计算论文与主题的相关性。"""


def compute_quality_score(paper: dict) -> float:
    """综合 citation_count、venue、abstract、pdf_url、year 计算质量。"""


def rank_papers(papers: list[dict], topic: str, top_k: int) -> list[dict]:
    """综合排序并返回 top_k。"""


def explain_ranking_reason(paper: dict, topic: str) -> str:
    """生成入选理由。"""
```

建议打分：

```text
rank_score = 0.7 * relevance_score + 0.3 * quality_score
```

---

### 7.3 `app/tools/fetch_metadata.py`

职责：补全论文详情。

函数：

```python
def fetch_paper_detail(paper: dict) -> dict:
    """根据 DOI、arXiv ID、Semantic Scholar ID 补全详情。"""


def enrich_with_citation_count(paper: dict) -> dict:
    """补充引用量。"""


def enrich_with_pdf_url(paper: dict) -> dict:
    """补充开放获取 PDF 链接。"""


def fetch_batch_details(papers: list[dict]) -> list[dict]:
    """批量补全论文详情。"""
```

---

### 7.4 `app/tools/download_pdf.py`

职责：只下载开放获取 PDF。

函数：

```python
def is_open_access(paper: dict) -> bool:
    """判断是否有开放 PDF。"""


def download_open_access_pdf(paper: dict, save_dir: str) -> str | None:
    """下载开放 PDF，返回 pdf_path；失败返回 None。"""


def batch_download_pdfs(papers: list[dict], save_dir: str) -> dict[str, str | None]:
    """批量下载开放 PDF。"""


def validate_pdf_file(pdf_path: str) -> bool:
    """检查 PDF 文件是否有效。"""
```

异常处理要求：

1. 下载失败不能中断整个工作流。
2. 单篇失败后返回 `None`。
3. 不得尝试构造非开放获取下载链接。

---

### 7.5 `app/tools/parse_pdf.py`

职责：解析 PDF。

函数：

```python
def parse_pdf(pdf_path: str) -> dict:
    """解析 PDF，返回结构化文本。"""


def extract_pdf_text(pdf_path: str) -> str:
    """提取全文文本。"""


def split_paper_sections(text: str) -> dict:
    """粗略划分 abstract / introduction / method / experiment / conclusion / references。"""


def extract_abstract(text: str) -> str:
    """提取摘要。"""


def extract_references(text: str) -> list[str]:
    """提取参考文献列表。"""


def clean_parsed_text(text: str) -> str:
    """清洗 PDF 解析文本。"""
```

---

### 7.6 `app/tools/extract_paper_card.py`

职责：抽取结构化论文卡片。

函数：

```python
def extract_paper_card(
    paper: dict,
    parsed_text: dict | None,
    llm=None,
) -> PaperCard:
    """优先基于全文抽取；无全文时基于摘要抽取。"""


def extract_from_abstract(paper: dict, llm=None) -> PaperCard:
    """基于摘要生成 PaperCard。"""


def extract_from_full_text(
    paper: dict,
    parsed_text: dict,
    llm=None,
) -> PaperCard:
    """基于全文生成 PaperCard。"""


def validate_paper_card(card: PaperCard) -> bool:
    """检查 PaperCard 必要字段。"""


def batch_extract_paper_cards(
    papers: list[dict],
    parsed_texts: dict[str, dict],
    llm=None,
) -> list[PaperCard]:
    """批量抽取 PaperCard。"""
```

PaperCard 必填字段：

```text
paper_id
title
research_problem
method
relevance_reason
evidence_source
```

---

### 7.7 `app/tools/cluster_papers.py`

职责：文献分类。

函数：

```python
def cluster_papers_by_method(cards: list[dict]) -> list[dict]:
    """按方法路线分类。"""


def cluster_papers_by_task(cards: list[dict]) -> list[dict]:
    """按研究任务分类。"""


def cluster_papers_by_dataset(cards: list[dict]) -> list[dict]:
    """按数据集分类。"""


def cluster_papers_by_year(cards: list[dict]) -> list[dict]:
    """按年份分类。"""


def generate_cluster_summary(clusters: list[dict], llm=None) -> str:
    """总结每类文献特点。"""
```

分类结果格式：

```python
{
    "cluster_name": "Transformer-based methods",
    "description": "基于 Transformer 的视觉理解方法",
    "paper_ids": ["..."],
    "representative_papers": ["..."]
}
```

---

### 7.8 `app/tools/generate_review.py`

职责：生成综述。

函数：

```python
def generate_literature_review(
    topic: str,
    paper_cards: list[dict],
    clusters: list[dict],
    language: str,
    citation_style: str,
    llm=None,
) -> str:
    """生成完整文献综述。"""


def generate_background_section(topic: str, cards: list[dict], llm=None) -> str:
    """研究背景。"""


def generate_method_taxonomy_section(clusters: list[dict], llm=None) -> str:
    """方法分类。"""


def generate_representative_work_section(cards: list[dict], llm=None) -> str:
    """代表性工作。"""


def generate_dataset_metric_section(cards: list[dict], llm=None) -> str:
    """数据集与评价指标。"""


def generate_challenge_section(cards: list[dict], llm=None) -> str:
    """问题与挑战。"""


def generate_future_trend_section(cards: list[dict], llm=None) -> str:
    """未来趋势。"""


def assemble_review(sections: list[str], references: list[str]) -> str:
    """拼接综述正文。"""
```

综述结构必须包含：

```text
1. 研究背景
2. 发展脉络
3. 方法分类
4. 代表性工作
5. 数据集与评价指标
6. 当前问题
7. 未来趋势
8. 总结
9. 参考文献
```

---

### 7.9 `app/tools/generate_citation.py`

职责：参考文献生成和引用校验。

函数：

```python
def generate_reference(paper: dict, citation_style: str) -> str:
    """生成单条参考文献。"""


def generate_references(papers: list[dict], citation_style: str) -> list[str]:
    """批量生成参考文献。"""


def generate_in_text_citation(paper: dict, citation_style: str) -> str:
    """生成正文引用标记。"""


def validate_citations(review_text: str, references: list[str]) -> dict:
    """检查正文引用与参考文献是否匹配。"""


def generate_bibtex_key(paper: dict) -> str:
    """生成 BibTeX key。"""
```

支持格式：

```text
gbt7714
apa
ieee
bibtex
```

---

## 8. Clients 模块设计

### 8.1 `app/clients/arxiv_client.py`

函数：

```python
def search_arxiv(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
) -> list[PaperMetadata]:
    """调用 arXiv API 检索论文。"""


def parse_arxiv_response(
    response_text: str,
    start_year: int,
    end_year: int,
) -> list[PaperMetadata]:
    """解析 arXiv Atom XML。"""


def get_arxiv_pdf_url(arxiv_id: str) -> str:
    """生成 arXiv PDF URL。"""
```

---

### 8.2 `app/clients/semantic_scholar_client.py`

函数：

```python
def search_semantic_scholar(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
) -> list[PaperMetadata]:
    """调用 Semantic Scholar 检索论文。"""


def get_semantic_scholar_detail(paper_id: str) -> dict:
    """获取论文详情。"""


def parse_semantic_scholar_response(response_json: dict) -> list[PaperMetadata]:
    """解析 Semantic Scholar JSON。"""
```

---

### 8.3 `app/clients/openalex_client.py`

函数：

```python
def search_openalex(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
) -> list[PaperMetadata]:
    """调用 OpenAlex 检索论文。"""


def get_openalex_detail(work_id: str) -> dict:
    """获取 OpenAlex work 详情。"""


def parse_openalex_response(response_json: dict) -> list[PaperMetadata]:
    """解析 OpenAlex JSON。"""
```

---

### 8.4 `app/clients/crossref_client.py`

函数：

```python
def search_crossref(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
) -> list[PaperMetadata]:
    """调用 Crossref 检索论文元数据。"""


def get_crossref_detail(doi: str) -> dict:
    """根据 DOI 获取详情。"""


def parse_crossref_response(response_json: dict) -> list[PaperMetadata]:
    """解析 Crossref JSON。"""
```

---

## 9. Services 模块设计

### 9.1 `app/services/paper_service.py`

职责：论文相关业务流程。

类与方法：

```python
class PaperService:
    def search(self, request: PaperSearchRequest) -> list[PaperMetadata]:
        """检索论文。"""

    def search_and_rank(self, request: PaperSearchRequest) -> list[dict]:
        """检索并排序。"""

    def fetch_details(self, papers: list[dict]) -> list[dict]:
        """补全元数据。"""

    def download_and_parse(self, papers: list[dict]) -> dict[str, dict]:
        """下载并解析开放 PDF。"""

    def build_paper_cards(self, papers: list[dict]) -> list[PaperCard]:
        """构建 PaperCard。"""
```

---

### 9.2 `app/services/review_service.py`

职责：综述生成业务流程。

```python
class ReviewService:
    def generate_review(self, request: ReviewRequest) -> LiteratureReview:
        """根据主题生成完整综述。"""

    def generate_review_from_cards(
        self,
        topic: str,
        cards: list[dict],
    ) -> LiteratureReview:
        """基于已有 PaperCard 生成综述。"""

    def compare_papers(self, paper_ids: list[str]) -> str:
        """对比多篇论文。"""

    def generate_summary_table(self, cards: list[dict]) -> list[dict]:
        """生成论文对比表。"""
```

---

### 9.3 `app/services/library_service.py`

职责：本地论文库管理。

```python
class LibraryService:
    def add_paper(self, paper: PaperMetadata) -> None:
        """保存论文元数据。"""

    def add_paper_card(self, card: PaperCard) -> None:
        """保存 PaperCard。"""

    def search_local_library(self, query: str, top_k: int) -> list[PaperCard]:
        """检索本地论文库。"""

    def get_paper_by_id(self, paper_id: str) -> PaperMetadata | None:
        """根据 ID 查询论文。"""

    def list_papers(self, limit: int = 50) -> list[PaperMetadata]:
        """列出本地论文。"""
```

---

### 9.4 `app/services/citation_service.py`

职责：引用生成与校验。

```python
class CitationService:
    def generate_references(self, papers: list[dict], style: str) -> list[str]:
        """生成参考文献。"""

    def validate_review_citations(self, review_text: str, references: list[str]) -> dict:
        """校验引用。"""

    def convert_citation_style(self, references: list[str], target_style: str) -> list[str]:
        """转换引用格式。"""
```

---

## 10. API 设计

### 10.1 `app/api/routes_paper.py`

接口：

```python
@router.post("/papers/search")
def search_papers_api(request: PaperSearchRequest):
    """检索论文列表。"""


@router.get("/papers/{paper_id}")
def get_paper_detail_api(paper_id: str):
    """获取论文详情。"""


@router.post("/papers/{paper_id}/card")
def generate_paper_card_api(paper_id: str):
    """生成单篇论文卡片。"""


@router.get("/papers")
def list_papers_api(limit: int = 50):
    """列出本地论文库。"""
```

---

### 10.2 `app/api/routes_review.py`

接口：

```python
@router.post("/reviews/generate")
def generate_review_api(request: ReviewRequest):
    """根据结构化参数生成文献综述。"""


@router.post("/reviews/agent")
def generate_review_by_agent_api(user_query: str):
    """根据自然语言用户请求运行 Agent。"""


@router.post("/reviews/from_papers")
def generate_review_from_papers_api(paper_ids: list[str]):
    """基于指定论文生成综述。"""


@router.get("/reviews/{review_id}")
def get_review_api(review_id: int):
    """查看历史综述。"""


@router.get("/reviews")
def list_reviews_api():
    """列出历史综述。"""
```

---

### 10.3 `app/api/routes_library.py`

接口：

```python
@router.post("/library/import_pdf")
def import_pdf_api(file_path: str):
    """导入本地 PDF。"""


@router.post("/library/search")
def search_library_api(query: str, top_k: int = 5):
    """搜索本地论文库。"""


@router.post("/library/rebuild_index")
def rebuild_index_api():
    """重建向量索引。"""
```

---

## 11. 数据库设计

### 11.1 `app/database/models.py`

建议三张表：

```python
class Paper(Base):
    __tablename__ = "papers"
    id: str
    title: str
    authors: str
    year: int
    venue: str
    abstract: str
    doi: str
    arxiv_id: str
    url: str
    pdf_url: str
    citation_count: int
    source: str
    created_at: datetime


class PaperCardModel(Base):
    __tablename__ = "paper_cards"
    id: int
    paper_id: str
    research_problem: str
    method: str
    dataset: str
    metrics: str
    results: str
    contributions: str
    limitations: str
    relevance_reason: str
    evidence_source: str
    created_at: datetime


class ReviewModel(Base):
    __tablename__ = "reviews"
    id: int
    topic: str
    review_text: str
    references: str
    paper_ids: str
    citation_style: str
    created_at: datetime
```

### 11.2 `app/database/repositories.py`

```python
class PaperRepository:
    def save(self, paper: PaperMetadata) -> None: ...
    def get_by_id(self, paper_id: str) -> PaperMetadata | None: ...
    def find_by_title(self, title: str) -> PaperMetadata | None: ...
    def list(self, limit: int = 50) -> list[PaperMetadata]: ...


class PaperCardRepository:
    def save(self, card: PaperCard) -> None: ...
    def get_by_paper_id(self, paper_id: str) -> PaperCard | None: ...
    def search(self, query: str, limit: int = 10) -> list[PaperCard]: ...


class ReviewRepository:
    def save(self, review: LiteratureReview) -> None: ...
    def list(self, limit: int = 20) -> list[LiteratureReview]: ...
```

---

## 12. 向量库设计

### 12.1 `app/vectorstore/embeddings.py`

```python
def get_embedding_model():
    """加载 embedding 模型。"""


def embed_text(text: str) -> list[float]:
    """单段文本向量化。"""


def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量向量化。"""
```

### 12.2 `app/vectorstore/index.py`

```python
def build_index(cards: list[PaperCard]) -> None:
    """基于 PaperCard 构建向量索引。"""


def add_card_to_index(card: PaperCard) -> None:
    """添加单个 PaperCard。"""


def save_index(path: str) -> None:
    """保存索引。"""


def load_index(path: str):
    """加载索引。"""
```

### 12.3 `app/vectorstore/retriever.py`

```python
def retrieve_relevant_cards(query: str, top_k: int = 5) -> list[PaperCard]:
    """按语义检索相关 PaperCard。"""


def retrieve_by_method(method_name: str, top_k: int = 10) -> list[PaperCard]:
    """按方法检索。"""


def retrieve_by_dataset(dataset_name: str, top_k: int = 10) -> list[PaperCard]:
    """按数据集检索。"""
```

---

## 13. Core 和 Utils

### 13.1 `app/core/config.py`

```python
class Settings(BaseSettings):
    app_name: str = "ResearchReview-Agent"
    debug: bool = False
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    database_url: str = "sqlite:///./data/research_review.db"
    vectorstore_path: str = "./data/vector_index"
    pdf_save_dir: str = "./data/pdfs"
    parsed_save_dir: str = "./data/parsed"
    review_save_dir: str = "./data/reviews"


def get_settings() -> Settings:
    """读取配置。"""
```

### 13.2 `app/core/logger.py`

```python
def setup_logger(name: str):
    """创建 logger。"""


def get_logger(name: str):
    """获取 logger。"""
```

### 13.3 `app/core/exceptions.py`

```python
class PaperSearchError(Exception): ...
class PDFDownloadError(Exception): ...
class PDFParseError(Exception): ...
class CitationValidationError(Exception): ...
class AgentExecutionError(Exception): ...
```

### 13.4 `app/utils/text_cleaner.py`

```python
def normalize_whitespace(text: str) -> str: ...
def remove_references_noise(text: str) -> str: ...
def truncate_text(text: str, max_chars: int) -> str: ...
```

### 13.5 `app/utils/file_utils.py`

```python
def ensure_dir(path: str) -> None: ...
def safe_filename(name: str) -> str: ...
```

### 13.6 `app/utils/date_utils.py`

```python
def current_year() -> int: ...
def default_year_range(offset: int = 5) -> tuple[int, int]: ...
```

---

## 14. 前端

### 14.1 `app/frontend/streamlit_app.py`

页面功能：

1. 输入自然语言研究需求。
2. 点击按钮运行 Agent。
3. 展示最终综述。
4. 展示检索论文列表。
5. 展示 PaperCard 表格。
6. 展示 Agent 执行步骤。
7. 展示参考文献。

核心函数：

```python
def render_query_input() -> str:
    """渲染输入框。"""


def render_agent_result(result: dict) -> None:
    """展示 Agent 输出。"""


def render_steps(steps: list[dict]) -> None:
    """展示执行步骤。"""


def render_references(references: list[str]) -> None:
    """展示参考文献。"""
```

---

## 15. 启动文件

### 15.1 `run_api.py`

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 15.2 `run_streamlit.py`

```python
import os

if __name__ == "__main__":
    os.system("streamlit run app/frontend/streamlit_app.py")
```

---

## 16. `requirements.txt`

MVP：

```text
fastapi
uvicorn
pydantic
pydantic-settings
requests
feedparser
pymupdf
sqlalchemy
streamlit
python-dotenv
```

可选增强：

```text
langchain
langgraph
openai
faiss-cpu
chromadb
sentence-transformers
pdfplumber
```

---

## 17. `.env.example`

```text
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=

DATABASE_URL=sqlite:///./data/research_review.db
VECTORSTORE_PATH=./data/vector_index
PDF_SAVE_DIR=./data/pdfs
```

---

## 18. MVP 实现顺序

### 阶段 1：项目骨架

- [ ] 创建目录结构
- [ ] 创建 `requirements.txt`
- [ ] 创建 `.env.example`
- [ ] 创建 `app/main.py`
- [ ] 创建 API 路由
- [ ] 创建 Schema
- [ ] 创建配置与日志模块

### 阶段 2：意图识别和槽位抽取

- [ ] 实现 `intent.py`
- [ ] 实现 `slot_extractor.py`
- [ ] 实现 `planner.py`
- [ ] 实现 `router.py`
- [ ] 编写测试：`test_intent.py`
- [ ] 编写测试：`test_slot_extractor.py`

### 阶段 3：论文检索

- [ ] 实现 `arxiv_client.py`
- [ ] 实现 `semantic_scholar_client.py`
- [ ] 实现 `openalex_client.py`
- [ ] 实现 `search_papers.py`
- [ ] 实现 `rank_papers.py`
- [ ] 编写测试：`test_search.py`
- [ ] 编写测试：`test_rank.py`

### 阶段 4：论文解析和 PaperCard

- [ ] 实现 `download_pdf.py`
- [ ] 实现 `parse_pdf.py`
- [ ] 实现 `extract_paper_card.py`
- [ ] 编写测试：`test_extract.py`

### 阶段 5：综述生成

- [ ] 实现 `cluster_papers.py`
- [ ] 实现 `generate_review.py`
- [ ] 实现 `generate_citation.py`
- [ ] 编写测试：`test_review.py`
- [ ] 编写测试：`test_citation.py`

### 阶段 6：Agent 串联

- [ ] 实现 `nodes.py`
- [ ] 实现 `graph.py`
- [ ] 实现 `/api/reviews/agent`
- [ ] 实现 Streamlit 前端
- [ ] 跑通端到端 Demo

---

## 19. 端到端验收标准

### 19.1 测试输入

```text
帮我调研近五年课堂行为识别相关论文，并生成中文文献综述，引用不少于 10 篇。
```

### 19.2 预期行为

1. 意图识别为 `generate_review`。
2. 抽取主题为 `课堂行为识别`。
3. 抽取年份为近五年。
4. 抽取论文数量不少于 10。
5. 检索 arXiv / Semantic Scholar / OpenAlex。
6. 对候选论文去重和排序。
7. 尽量下载开放获取 PDF。
8. PDF 不可用时基于摘要生成 PaperCard。
9. 对 PaperCard 分类。
10. 生成中文综述。
11. 输出参考文献。
12. 不编造任何论文和引用。

### 19.3 最终输出结构

```text
# 文献综述标题

## 1. 研究背景

## 2. 发展脉络

## 3. 方法分类

## 4. 代表性工作

## 5. 数据集与评价指标

## 6. 当前问题

## 7. 未来趋势

## 8. 总结

## 参考文献
```

---

## 20. 简历描述

### 项目名称

**ResearchReview-Agent：面向计算机视觉论文的自动化文献综述智能体**

### 项目描述

构建面向计算机视觉领域的论文调研 Agent，支持根据用户研究主题自动检索开放论文资源，获取论文元数据和摘要，生成结构化论文卡片，并按照研究任务、方法路线和数据集进行分类，最终生成带引用的中文文献综述。

### 项目亮点

1. 设计 Agentic RAG 工作流，实现从研究主题解析、论文检索、文献筛选、论文卡片抽取到综述生成的自动化流程。
2. 接入 arXiv、Semantic Scholar、OpenAlex 等开放论文数据源，实现多源论文检索与元数据融合。
3. 构建 PaperCard 结构，抽取研究问题、核心方法、数据集、评价指标、贡献与局限。
4. 基于相关性、年份、引用量和开放获取状态对论文进行排序与筛选。
5. 设计引用校验机制，降低综述生成中的论文幻觉和伪引用问题。
6. 支持计算机视觉、课堂视频理解、行为识别、多模态学习等方向的快速文献调研。

---

## 21. 后续增强方向

1. 使用 LangGraph 实现条件分支、重试和状态恢复。
2. 接入 GROBID 提高论文结构化解析质量。
3. 接入 LLM 做高质量 PaperCard 抽取。
4. 引入向量库，对论文卡片和全文段落做 RAG 检索。
5. 支持上传本地 PDF。
6. 支持导出 Markdown / Word / LaTeX。
7. 支持论文对比表格导出。
8. 支持关键词共现图和年份趋势图。
9. 支持针对计算机视觉论文的字段：模型结构、backbone、数据集、评价指标、SOTA 对比。
10. 支持引用强校验：正文引用和参考文献双向匹配。

---

## 22. Coding Agent 注意事项

1. 优先实现 MVP，不要一开始过度工程化。
2. 外部 API 调用必须加 timeout。
3. 外部 API 失败时必须降级处理。
4. PDF 下载失败不能导致整体失败。
5. PaperCard 字段缺失时不要编造。
6. 综述生成只能引用已检索到的论文。
7. 测试中尽量 mock 外部 API。
8. 端到端 Demo 优先级高于复杂优化。
9. 代码中所有函数都要有 docstring。
10. 错误信息要记录到 `state["errors"]` 和日志中。

---

## 23. 路线证据恢复闭环

研究现状类任务在首次路线验证后执行受控恢复：

```text
Route–Evidence Validation
→ Evidence Gap Diagnosis（确定性指标 + LLM 语义解释）
→ Deterministic Recovery Controller
→ Targeted Search / Query Revision / Scope Revision / Degrade
→ Incremental Metadata + Evidence Cards
→ Route Re-validation
→ Claim Planning
→ Claim Evidence Gate
→ Writing
```

控制规则：

- LLM 只能解释缺口、生成新查询或提出范围修订建议，不能增加循环预算；
- 检索轮数、每路线尝试次数、范围修订次数和边际收益阈值来自配置；
- 数据源全部失败时不得把连接失败误判为关键词问题；
- 恢复轮对累积候选池使用确定性重排，LLM 仅负责语义缺口诊断和新查询规划，避免每轮重复全池 LLM 排序；
- 补搜只为新增论文获取详情并建立 Evidence Card，原有证据不重复抽取；
- Claim Gate 默认删除无证据的可选主张或降低语言强度，不让证据无限追逐主张；
- `recovery_history` 保留每轮查询、新增证据、覆盖率变化和停止原因。

---

## 24. Route Validator v2

路线验证不再对中文路线文本和英文论文全文做单一 Jaccard 判定。每条路线先形成受边界约束的语义锚点集合：

```text
Route definition
→ semantic / method / task / negative anchors
→ anchor provenance + drift guard
→ Route–Paper feature matrix
→ independent Route Validity
→ independent Evidence Sufficiency
→ KEEP / WEAK / DROP + action
```

Route–Paper feature matrix 保留：

- `semantic_similarity`
- `concept_coverage`
- `lexical_anchor_score`
- `evidence_claim_match`
- `method_compatibility`
- `evidence_role_score`
- `negative_anchor_conflict`

`route_validity` 只依赖路线定义、锚点来源、边界和内部一致性；`evidence_sufficiency` 只依赖核心/支持证据数量与质量。结构合理但证据不足时输出 `WEAK + TARGETED_SEARCH`。多路线同时无 KEEP 时记录 `route_validator_recheck`，Recovery 耗尽且仍无证据支撑时才进入动态聚类回退。

冻结基线位于 `tests/test_route_recovery_gold.py`，当前覆盖跨语言匹配、同义表达、稀疏证据、错误路线定义、系统性低存活、补搜状态迁移、恢复耗尽与 Claim 弱化。

---

## 25. Global Evidence Gate

路线级验证回答"每条路线证据够不够"，本层回答"整篇综述是否满足用户的显式要求"。插入在 Route Recovery 之后、Claim Planning 之前：

```text
Route Evidence Recovery（局部修复完成后的最终状态）
→ Global Evidence Gate（综述级评估与推荐）
→ Claim Planning → Claim Evidence Gate → Writing
→ derive_result_status（显式约束未满足 → partial）
→ final_answer（blocking 缺口横幅）
```

评估维度与阈值（全部确定性，LLM 不参与决策）：

| 维度 | 数据来源 | 阈值（配置） | blocking 条件 |
|---|---|---|---|
| `citation_count` | `required_reference_count` vs `len(paper_details)` | 差量 > 0 | 用户显式指定数量（`max_papers_explicit`） |
| `recency` | `paper_details[*].year` vs `[start_year, end_year]` | 窗口内占比 < `global_gate_min_recency_ratio`（0.9） | 用户显式指定年份（`year_range_explicit`） |
| `route_coverage` | KEEP 路线 `paper_ids` 数量均衡（min/avg） | < `global_gate_route_balance_min_ratio`（0.25） | 恒 blocking（但**不**改变结果状态） |
| `quality` | `paper_cards[*].peer_review_status` 占比 | < `global_gate_peer_review_ratio`（0.8） | 用户显式提及同行评审/期刊/SCI/EI |
| `claim_support` | v1 DEFERRED（代理指标 = 各 KEEP 路线 `evidence_sufficiency` 均值） | — | 由 `claim_evidence_gate` 在写作前评估 |

v1 冻结范围：只测量与推荐，不执行任何恢复动作（不自动扩年份、不改 scope、不降低用户约束、不重分配论文）。推荐动作优先级：`REBALANCE_ROUTE`（路线失衡）→ `TARGETED_GLOBAL_SEARCH`（其余阻断）→ `ASK_USER`（路线级恢复已耗尽，需用户决策）→ `CONTINUE`。

`evidence_debt` 汇总各维度缺口（{维度: missing}），是后续 Evidence Budget Optimization（按 ROI 分配补搜预算）的输入。行为语义为 RECORD AND CONTINUE：不达标不阻断写作，最终答复呈现 blocking 缺口横幅；显式约束未满足时 `derive_result_status` 返回 `partial`。节点异常时写 `status: FAILED` 而非 `passed: False`，避免误报证据不足。
