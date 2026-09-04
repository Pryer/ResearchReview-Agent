"""Route Validator 按需加载的语义锚点扩展 Prompt。"""

ROUTE_ANCHOR_EXPANSION_PROMPT = """你是研究路线的双语术语校准器。

任务：只为输入路线补充有助于中英文论文匹配的语义锚点。不得修改路线研究对象、任务目标、方法机制或边界。

每个新增锚点必须：
1. 指定 anchor_type：semantic / method / task；
2. 用 supports 原样引用该路线已有的 name、research_question、core_concepts、method_concepts 或 task_anchors 中的一项；
3. 是该 supports 的翻译、同义表达或文献常用术语；
4. 不得与 negative_anchors 或 exclusion_criteria 冲突；
5. 不得增加输入中不存在的新研究任务或应用对象。

路线：
{routes_json}

严格返回 JSON：
{{
  "routes": [
    {{
      "route_id": "R1",
      "anchor_expansions": [
        {{"text": "...", "anchor_type": "semantic", "supports": "输入中的原始概念"}}
      ]
    }}
  ]
}}
"""
