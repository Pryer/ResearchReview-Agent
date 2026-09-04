"""分类轴约束解析器。

这里只根据上游研究模式提供抽象约束；具体分类轴、主题名称和比较维度必须
由 LLM 根据本次论文证据动态归纳，不能在代码里维护领域词表或模型清单。
"""

from typing import Dict, Any


class TaxonomyStrategyResolver:
    """根据 research_mode 下发无领域预设的分类约束。"""

    @classmethod
    def resolve(cls, research_mode: str) -> Dict[str, Any]:
        """
        解析并返回对应的分类策略配置。

        Args:
            research_mode: 上游语义解析得到的研究模式。

        Returns:
            Dict 包含：
            - axis_instruction: 提示词中对分类轴的指导建议
            - example_axes: 保留兼容字段；始终为空，避免硬编码领域示例
        """
        mode = str(research_mode).lower().strip()

        if mode in {"technical", "technology_oriented"}:
            return {
                "mode": "technical",
                "axis_instruction": (
                    "当前研究以方法或技术问题为主要对象。请仅根据输入论文中反复出现、"
                    "且有明确证据支持的核心建模思想、问题设定或研究路线归纳分类轴。"
                    "不得预置具体架构、模型、任务、数据模态或应用领域；如果样本显示其他"
                    "维度更有解释力，应服从样本证据。"
                ),
                "example_axes": [],
            }

        if mode in {"empirical", "domain_oriented"}:
            return {
                "mode": "empirical",
                "axis_instruction": (
                    "当前研究以领域问题或经验现象为主要对象。请从输入论文实际采用的研究"
                    "问题、解释视角、研究设计或证据类型中归纳最有解释力的分类轴。不得预置"
                    "具体变量、理论、干预或学科框架。"
                ),
                "example_axes": [],
            }

        if mode in {
            "technology_applied_to_domain",
            "technology_assisted_domain_analysis",
        }:
            return {
                "mode": "applied_or_assisted",
                "axis_instruction": (
                    "当前研究同时包含方法手段与领域目标。请根据输入证据判断，分类轴应围绕"
                    "方法路线、任务链阶段、领域问题或它们之间的作用关系展开。必须区分手段"
                    "和最终研究目标，不得把代码中预设的技术类别或应用场景当作分类答案。"
                ),
                "example_axes": [],
            }

        # 默认、mixed 或未知模式：让样本证据决定分类轴。
        return {
            "mode": "mixed",
            "axis_instruction": (
                "不要假定固定研究范式。请比较输入论文的研究问题、证据类型、研究设计和"
                "主要结论，自主选择一个最能解释样本差异的分类轴。主题必须可由论文内容"
                "直接验证，不得使用预设领域词表或仅按年份分组。"
            ),
            "example_axes": [],
        }
